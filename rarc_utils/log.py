"""log.py, implement custom loggers for CLI programs.

supports: indented formatting, colored formatting, save to file, save all log records to pandas, and more
"""

import copy
import json
import logging
import os
import re
import sys
import textwrap
from collections import defaultdict
from datetime import datetime
from functools import partial, partialmethod
from itertools import chain
from typing import Any, Dict, List, Tuple

import coloredlogs
import lz4.frame
import numpy as np
import pandas as pd
from humanfriendly.compat import coerce_string
from humanfriendly.terminal import ansi_wrap
from pythonjsonlogger import jsonlogger

JSON_LOGS = "json_logs"


class MsgCountHandler(logging.Handler):
    """Count number of log calls per level type.

    Additionally alsos store every emitted message to a dataframe, for later inspection
    """

    def __init__(self, *args, savePandas=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.__levelCount = defaultdict(int)
        self.__df = None

        if savePandas:
            self.dtypes = dtypes = np.dtype(
                [
                    ("name", str),
                    ("level", str),
                    ("funcName", str),
                    ("lineno", int),
                ]
            )
            data = np.empty(0, dtype=dtypes)
            self._df_cols = df_cols = dtypes.names
            df = pd.DataFrame(data, columns=df_cols)
            self.__df = df

    @property
    def levelCount(self) -> Dict[str, int]:
        lc = dict(self.__levelCount)
        # add total field
        lc["total"] = 0
        if len(lc) > 0:
            lc["TOTAL"] = sum(lc.values())

        return lc

    @property
    def df(self) -> pd.DataFrame:
        # turn string columns into category types. this makes only sense to do it when retrieving the object, after a concat the dtypes are not preserved
        df = self.__df
        for col in self._df_cols:
            if self.dtypes[col] == np.dtype("<U"):
                df[col] = df[col].astype("category")

        return df

    @df.setter
    def df(self, df: pd.DataFrame) -> None:
        assert isinstance(df, pd.DataFrame)
        self.__df = df

    def emit(self, record: logging.LogRecord) -> None:
        """overrides logging.Handler.emit"""

        self.__levelCount[record.levelname] += 1
        if self.__df is not None:
            dt = datetime.fromtimestamp(record.created)
            # print(f"{record.asctime=} {dt=} {dir(record)=}")
            new_row = pd.DataFrame(
                [[record.name, record.levelname, record.funcName, record.lineno]],
                columns=self._df_cols,
                index=[dt],
            )
            self.df = pd.concat([self.__df, new_row], ignore_index=False)


class MultiLineFormatter1(logging.Formatter):
    """Format multiline log message with same indentation."""

    def format(self, record):
        # print('using format')
        message = record.msg
        record.msg = ""
        header = super().format(record)
        msg = textwrap.indent(message, " " * len(header)).lstrip()
        record.msg = message
        return header + msg


class MultiLineFormatter2(logging.Formatter):
    """Multi-line formatter."""

    def __init__(self, width, **kwargs):

        self.width = width
        self.header_len = None
        super().__init__(**kwargs)

    def get_header_length(self, record) -> int:
        """Get the header length of a given record."""
        record = copy.copy(record)
        record.msg = ""
        record.exc_info = None
        record.exc_text = None
        record.stack_info = None
        header = super().format(record)
        # tiny problem: the color characters are part of the header, how to get pure formatted string length?
        # --> strip the ASCII color chars using regex
        header = re.sub("\033\\[([0-9]+)(;[0-9]+)*m", "", header)
        return len(header)

    def format(self, record) -> str:
        """Format a record with added indentation."""
        message = record.msg
        # For now we only indent string typed messages
        # Other message types like list or bytes won't be touched
        if isinstance(message, str):
            # wrap text into fixed width block
            # msgs = '\n'.join()
            # msgs = textwrap.wrap(message, width=self.width, replace_whitespace=False, subsequent_indent='  ') # default subsequent_indent == ''
            # msgs = msgs.splitlines(True)
            texts = message.split("\n")
            # I use itertools to repeatedly wrap the sub texts, this is the only to honour any '\n' in a log message
            msgs = list(
                chain(
                    *[
                        textwrap.wrap(t, width=self.width, subsequent_indent="  ")
                        for t in texts
                    ]
                )
            )
            msgs = [
                f"{msg}\n" if i != len(msgs) else msg
                for i, msg in enumerate(msgs, start=1)
            ]
            # print(f'{msgs=}')
            if len(msgs) > 1:
                self.header_len = header_len = self.get_header_length(record)

                # Indent lines (except the first line)
                indented_message = msgs[0] + "".join(
                    map(lambda m: " " * header_len + m if m != "\n" else m, msgs[1:])
                )
                # Use the original formatter since it handles exceptions well
                record.msg = indented_message
                formatted_text = super().format(record)
                # Revert to keep the msg field untouched
                # As other modules may capture the log for further processing
                record.msg = message
                return formatted_text

        return super().format(record)


def loggingLevelNames() -> Tuple[Any, ...]:
    """Return logging level names."""
    return tuple(
        logging.getLevelName(x)
        for x in range(1, 101)
        if not logging.getLevelName(x).startswith("Level")
    )


def set_log_level(logger, fmt, level=logging.DEBUG) -> None:
    """Set log level and reinstall coloredlogs.

    unfortunetaly, you have to reinstall coloredlogs
    """
    logger.setLevel(level)
    coloredlogs.install(logger=logger, level=level, fmt=fmt, milliseconds=1)


# add_log_level('URGENT', 25)
def add_log_level(name: str, level: int) -> None:
    """Add log level 'name' will to all loggers, instantiated from logging.getLogger()."""
    setattr(logging, name, level)
    newLevelAttr = getattr(logging, name)
    logging.addLevelName(newLevelAttr, name)
    # logging.Logger.urgent = partialmethod(logging.Logger.log, newLevelAttr)
    setattr(
        logging.Logger, name.lower(), partialmethod(logging.Logger.log, newLevelAttr)
    )
    # logging.urgent = partial(logging.log, logging.URGENT)
    setattr(logging, name.lower(), partial(logging.log, newLevelAttr))


class Empty:
    """An empty class used to copy :class:`~logging.LogRecord` objects without reinitializing them."""

    def __init__(self):
        pass


def save_json_log_redis(
    json_lines: List[Dict[str, Any]], ls, key=JSON_LOGS, rs=None
) -> None:
    """Save json log to redis."""
    assert rs is not None
    uuid = getattr(ls, "id")
    json_logs = json.dumps(json_lines).encode()
    compr_logs: bytes = lz4.frame.compress(json_logs)

    rs.hset(key, str(uuid), compr_logs)


def read_json_log_redis(uuid: str, key=JSON_LOGS, rs=None) -> List[Dict[str, Any]]:
    """Read json log from redis."""
    assert rs is not None
    compr_log: bytes = rs.hget(key, uuid)
    assert compr_log is not None, f"cannot find log {uuid}, try different uuid"

    res = lz4.frame.decompress(compr_log)
    lines = json.loads(res)

    assert isinstance(lines, list)

    return lines


def read_json_log_file(log_file: str) -> List[Dict[str, Any]]:
    """Read json log file, every line contains a json log message."""
    renameDict = dict(message="msg")

    lines = []
    with open(log_file) as f:
        for line in f:
            d = json.loads(line)
            for k_old, k_new in renameDict.items():
                d[k_new] = d.pop(k_old)
            lines.append(d)

    return lines


def setup_logger(
    cmdLevel=logging.INFO,
    brokers=(),
    saveFileLogLevel=logging.INFO,
    saveFile=False,
    savePandas=False,
    addUrgent=1,
    fmt="%(asctime)s - %(name)10s - %(funcName)-19s - %(levelname)6s - %(message)s",
    jsonLogger=False,
    multiLine=False,
    msgWidth=80,
    color=False,
):
    """See https://docs.python.org/2/library/logging.html#logrecord-attributes for a list of LogRecord attributes.

    usage:
        logger = log.setup_logger(cmdLevel=getattr(logging, verbosity), brokers=brokers, save_to_file=1, savePandas=1, fmt=)
        log_fmt = "%(asctime)s - %(name)-10s - %(lineno)5s - %(funcName)-10s - %(levelname)6s - %(message)s"
        logger = log.setup_logger(cmdLevel=getattr(logging, verbosity), brokers=brokers, save_to_file=1, savePandas=1, fmt=log_fmt)
    """
    assert isinstance(msgWidth, int), f"{type(msgWidth)} is not int"

    if addUrgent:
        add_log_level("URGENT", 25)

    # Prints logger info to terminal
    logger = logging.getLogger()
    logger.setLevel(cmdLevel)  # Change this to DEBUG if you want a lot more info

    # prevents duplicated logging output
    if logger.hasHandlers():
        logger.handlers.clear()

    console_handler = logging.StreamHandler()
    # create formatter
    # formatter = logging.Formatter("%(asctime)s - %(name)10s - %(levelname)6s - %(message)s")
    if not multiLine:
        console_formatter = logging.Formatter(
            fmt
        )  # same width for module and levelname
    else:
        # console_formatter = MultiLineFormatter1(msgWidth)
        console_formatter = MultiLineFormatter2(msgWidth, fmt=fmt)
    # add formatter to ch

    console_handler.setFormatter(console_formatter)

    # save errors to file (or slack/rabbitmq)
    if saveFile:
        dt_str = datetime.utcnow().strftime("%Y_%m_%dT%H_%M_%S")
        # sym_str = '_'.join(map(str, trading_symbols))i
        logname = f"logs/{dt_str}_{'_'.join(brokers)}.txt"

        fh = logging.FileHandler(logname, mode="a")
        fh.setLevel(saveFileLogLevel)
        file_formatter = logging.Formatter(fmt)
        fh.setFormatter(file_formatter)  # same format as regular log
        logger.addHandler(fh)

    logger.addHandler(MsgCountHandler(savePandas=savePandas))
    logger.addHandler(console_handler)

    # last try: monkey patch the format method to the colored logs stream handler formatter?
    # myformatter = MultiLineFormatter2(80, fmt=fmt)
    myformatter = console_formatter
    # needed for monkeypatching this into coloredlogs ColoredFormatter
    def myformat(self, record):
        self.width = 80

        style = self.nn.get(self.level_styles, record.levelname)
        if style and Empty is not None:
            copy = Empty()
            copy.__class__ = record.__class__
            copy.__dict__.update(record.__dict__)
            copy.msg = ansi_wrap(coerce_string(record.msg), **style)
            record = copy

        # was:
        # return logging.Formatter.format(self, record)
        # print(f'{record=}')
        # return myformatter.format(self, record)
        return MultiLineFormatter2.format(self, record)

    # todo: monkey-patch it into the module
    # coloredlogs.ColoredFormatter.format = myformat

    # todo: coloredlogs removes all handlers, also MultiLineFormatter, how to have both functioning?
    # solution: set formatter AFTER installing coloredlogs
    if color:
        coloredlogs.install(
            reconfigure=1, logger=logger, level=cmdLevel, fmt=fmt, milliseconds=1
        )
        logger.warning("using colored logs")

    if jsonLogger:
        # overwrite console_formatter, so use jsonLogger only inside Docker
        cwd = os.getcwd()
        json_file_dir = cwd + "/json_lines.log"
        logger.json_file_dir = json_file_dir
        logger.warning(f"using jsonLogger. {json_file_dir=}")

        def json_translate(obj):
            # if isinstance(obj, MyClass):
            #     return {"special": obj.special}
            pass

        # format_str = '%(message)%(levelname)%(filename)%(asctime)%(funcName)%(lineno)'
        format_str = fmt  # use same format as console_handler
        json_formatter = jsonlogger.JsonFormatter(
            format_str, json_default=json_translate, json_encoder=json.JSONEncoder
        )

        json_file_handler = logging.FileHandler(json_file_dir, mode="w")  # mode='a'
        json_file_handler.setLevel(
            logging.DEBUG
        )  # save all messages, filtering can happen later
        json_file_handler.setFormatter(json_formatter)
        # json_stream_handler = logging.StreamHandler()
        # json_stream_handler.setFormatter(json_formatter)

        logger.addHandler(json_file_handler)

    # assuming console_handler is the last handler!
    stream_handler = logger.handlers[-1]
    assert issubclass(
        (tsh := type(stream_handler)), logging.StreamHandler
    ), f"{tsh=} is not logging.StreamHandler"  # coloredlogs StreamHandler inherits from logging.StreamHandler, so this is a good safety check

    # pff, coloredlogs returns a new Formatter, so this hack does not work either
    # stream_handler.setFormatter(console_formatter)

    return logger


def setupCCXTTradeLogger(
    LOG_FILE,
    session_id,
    useFluent=1,
    saveRotatingFile=1,
    saveFile=0,
    savePandas=0,
    cmdLevel=logging.INFO,
    addUrgent=1,
    fmt="%(asctime)-15s.%(msecs)03d - %(name)10s - %(levelname)7s - %(funcName)-19s - %(message)s",
    multiLine=False,
    msgWidth=80,
    color=False,
):
    """See https://docs.python.org/2/library/logging.html#logrecord-attributes for a list of LogRecord attributes."""
    assert not (saveRotatingFile and saveFile)

    # ugly code, don't reset logger in this way
    # import logging
    # logging.shutdown()
    # importlib.reload(logging)
    if addUrgent:
        add_log_level("URGENT", 25)

    logger = logging.getLogger()  # 'main' # __name__

    if saveFile:
        file_handler = logging.FileHandler(LOG_FILE)

    elif saveRotatingFile:
        from logging.handlers import RotatingFileHandler

        file_handler = RotatingFileHandler(
            filename=LOG_FILE,
            mode="a",
            maxBytes=1024**3,
            backupCount=2,
            encoding=None,
            delay=0,
        )

    if useFluent:
        from fluent import handler

        fluent_handler = handler.FluentHandler(
            "app.follow", host="77.249.149.174", port=24224
        )
        fluent_format = {
            "host": "%(hostname)s",
            "where": "%(module)s.%(funcName)s",
            "type": "%(levelname)s",
            "stack_trace": "%(exc_text)s",
            "session_id": str(session_id),
        }
        formatter = handler.FluentRecordFormatter(fluent_format)
        fluent_handler.setFormatter(formatter)

    if not multiLine:
        formatter = logging.Formatter(fmt)  # same width for module and levelname
    else:
        logger.warning("using multiLine formatter")
        formatter = MultiLineFormatter2(msgWidth, fmt=fmt)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(cmdLevel)
    # - %(name)10s -
    handlers = []

    if useFluent:
        handlers += [fluent_handler]

    if saveFile:
        handlers += [file_handler]

    if savePandas:
        handlers += [MsgCountHandler(savePandas=savePandas)]

    handlers += [console_handler]

    # handlers += [MsgCountHandler(savePandas=savePandas)]

    logging.basicConfig(
        format=fmt, level=logging.DEBUG, datefmt="%Y-%m-%d %H:%M:%S", handlers=handlers
    )  #  INFO

    if color:
        coloredlogs.install(logger=logger, level=cmdLevel, fmt=fmt, milliseconds=1)
        logger.warning("using colored logs")

    # re setting formatter, since coloredlogs removes them
    # assuming console_handler is the last handler!
    stream_handler = logger.handlers[-1]
    assert issubclass(
        (tsh := type(stream_handler)), logging.StreamHandler
    ), f"{tsh=} is not logging.StreamHandler. {logger.handlers=}"  # coloredlogs StreamHandler inherits from logging.StreamHandler, so this is a good safety check

    # logger.handlers[-1].setFormatter(formatter)
    # stream_handler.setFormatter(formatter)

    return logger
