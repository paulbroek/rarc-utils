"""misc.py, miscellaneous utility methods."""

import asyncio
import configparser
import hashlib
import importlib
import itertools
import logging
import os
import random
import re
import signal
import subprocess
import sys
from collections import ChainMap
from datetime import datetime
from functools import partial, wraps
from importlib.metadata import version
from pathlib import Path
from time import sleep
from typing import (Any, Callable, Deque, Dict, List, Optional, Set, Tuple,
                    Union)

import numpy as np
import pandas as pd
import psutil
import requests
import timeago
import yaml

logger = logging.getLogger(__name__)

cfg = configparser.ConfigParser()


def load_yaml(filepath: Union[str, Path]):
    """Import YAML config file."""
    with open(filepath, "r", encoding="utf-8") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return config


def plural(int_var: int, word: str) -> str:
    """Correctly print message in plural.

    E.g.:
        downloaded 5 items
    vs  downloaded 1 item

    """
    assert isinstance(int_var, int)
    assert isinstance(word, str)

    if int_var != 1:
        return word + "s"

    return word


def hash_accross_the_same(item):
    """Use hash algorithm to produce same hash.

    so that objects send from different sources have the same hash when the dicts are equal in values
    """
    h = hashlib.new("ripemd160")
    h.update(item.encode())
    return h.hexdigest()


def reload_submodule():
    """Replace all reload_XX methods below, pass a str module name.

    this method assumes the structures of the loaded module is: df_methods.df_methods

    Usage:
        cm = reload_submodule()(df_methods)
        cm = reload_submodule()(ccxt_methods)
    """

    def reload(module):

        importlib.reload(module)

        return getattr(module, module.__name__)

    # this works in jupyter notebook:

    # import importlib
    # import rarc.tf_methods
    # importlib.reload(rarc.tf_methods)
    # tf_methods = rarc.tf_methods.tf_methods

    return reload


# def reload_cm():
#     """Return reload method, so it can be executed in the right context / path."""

#     def reload():
#         import ccxt_methods

#         importlib.reload(ccxt_methods)
#         from ccxt_methods import ccxt_methods as cm

#         return cm

#     return reload


# def reload_dm():
#     """Return reload method, so it can be executed in the right context / path.

#     can import directly from folder or as a module
#     """

#     def reload():
#         as_module = False
#         try:
#             import df_methods
#         except ModuleNotFoundError:
#             import rarc.df_methods as df_methods

#             as_module = True

#         importlib.reload(df_methods)

#         if as_module:
#             from rarc.df_methods import df_methods as dm
#         else:
#             from df_methods import df_methods as dm

#         return dm

#     return reload


# def reload_fm():
#     """returns reload method, so it can be executed in the right context / path"""

#     def reload():
#         import feature_methods

#         importlib.reload(feature_methods)
#         from feature_methods import feature_methods as fm

#         return fm

#     return reload


# def reload_pm():
#     """Return reload method, so it can be executed in the right context / path."""

#     def reload():
#         import plot_methods

#         importlib.reload(plot_methods)
#         from plot_methods import plot_methods as pm

#         return pm

#     return reload


def run_in_executor(f):
    """Run function in executor."""

    @wraps(f)
    def inner(*args, **kwargs):
        loop = asyncio.get_running_loop()
        return loop.run_in_executor(None, partial(f, *args, **kwargs))

    return inner


def validate_url(url: str) -> bool:
    regex = re.compile(
        r"^(?:http|ftp)s?://"  # http:// or https://
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"  # domain...
        r"localhost|"  # localhost...
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
        r"(?::\d+)?"  # optional port
        r"(?:/?|[/?]\S+)$",
        re.IGNORECASE,
    )
    return re.match(regex, url) is not None


class AttrDict(dict):
    """Emulate js object: dict.attr === dict['attr']."""

    def __init__(self, d=None):
        super().__init__()
        if d:
            for k, v in d.items():
                self.__setitem__(k, v)

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            value = AttrDict(value)
        super().__setitem__(key, value)

    def __getattr__(self, item):
        return self.__getitem__(item)

    def __missing__(self, key):
        return AttrDict()

    __setattr__ = __setitem__


# dict of signal codes
SIGD = {getattr(signal, n): n for n in dir(signal) if n.startswith("SIG")}


def handle_exit(sigNum, stackFrame, addDate=0, andPrint=0):
    """Handle exiting of program."""
    sigName = SIGD.get(sigNum, "undefined")
    dash = "------------------"
    dtstr = ""
    if addDate:
        dtstr = datetime.utcnow().strftime("%Y:%m:%d %H:%M:%S ")
    msg = f"{dash} {dtstr}EXITING ({sigNum=} {sigName=}) {dash}"  #  {stackFrame=}
    # lenmsg = len(msg)
    # msgs = ['', lenmsg * '-', msg, lenmsg * '-']
    # msgs = [lenmsg * '-', msg, lenmsg * '-']
    msgs = [msg]

    # logger.info(msg)
    for msg_ in msgs:
        logger.warning(msg_)

    if andPrint:
        for msg_ in msgs:
            print(msg_)

    sys.exit()


def get_random_df(arcres: Dict[str, pd.DataFrame]) -> Tuple[str, pd.DataFrame]:
    """Get random dataframe from dictionary of dataframes."""
    return random.choice(list(arcres.items()))


def get_sample(d: dict, n=3) -> dict:
    """Take sample of dictionary, for printing.

    n   unique random elements to choose, if n > len(d),
        return the original dict
    """
    assert isinstance(d, dict)

    if n > len(d):
        return d

    return dict(random.sample(d.items(), n))


du_dir_ = lambda path: sum(f.stat().st_size for f in path.glob("**/*") if f.is_file())


def du_dir(path: Path) -> int:
    """Return total disk usage of path."""
    return du_dir_(path)


def dir_(obj=None, not_starts=("__", "_")) -> List[str]:
    """Alternative dir() method that only shows public attributes."""
    # calling to see children of object
    return [a for a in dir(obj) if not a.startswith(not_starts)]


def dir__(not_starts=("__", "_")) -> Callable:
    """Call dir() to see local variables, without (semi) private mmethods."""

    def inner():
        # todo: does not work, still calls dir right now
        return [a for a in dir() if not a.startswith(not_starts)]

    return inner


def map_list(l: list, d: dict, noNan=False) -> list:
    """Map a dictionary over a list."""
    res = list(map(d.get, l))
    if noNan:
        return [r for r in res if r is not None]

    return res


def mem_usage(prec=2, power=2) -> int:
    """Get memory usage of this process, and therefore this python session.

    power   1024**2 is mb, 1024**3 is gb, etc.

    alternative way is running bash directly in ipython, but only works when using only one session:
    !ps aux | grep python | awk '{sum=sum+$6}; END {print sum/1024 " MB"}'
    """
    process = psutil.Process(os.getpid())
    bytes_usage = process.memory_info().rss
    fmt_usage = bytes_usage / 1024**power
    logger.info(f"{fmt_usage:.{prec}f} MB")

    return bytes_usage


def colab_auto_refresh():
    """
    Google Colab automatically disconnects the notebook if we leave it idle for more than 30 minutes.

    Open your Chrome DevTools by pressing F12 or ctrl+shift+i on Linux and enter the following JavaScript snippet in your console:

    function KeepClicking(){
    console.log("Clicking");
    document.querySelector("colab-connect-button").click()
    }
    setInterval(KeepClicking,60000)

    """
    raise NotImplementedError


def get_module_versions(modules: List[str]) -> Dict[str, str]:
    """Get module versions for a list of module strings."""
    return {mod: version(mod) for mod in modules}


def log_module_versions(modules: List[str]) -> None:
    """Log module version for a list of module strings."""
    for mod, v in get_module_versions(modules).items():
        logger.info(f"{mod} version={v}")


def check_version(package: str, lib, reqVersion=False, askUser=True):
    """Compare local version of a package with that listed on pypi.org. for ccxt lib it's crucial that you always update before running this bot."""
    # fetch latest PyPi version
    url = f"https://pypi.org/pypi/{package}/json"
    try:
        response = requests.get(url)
    except:
        logger.error(f"could not reach {url}")
        raise Exception

    # and fetch YOUR latest version
    my_version = version(package)

    logger.info(f"{reqVersion=} {my_version=}")
    if my_version == reqVersion:
        logger.info(f"versions match, nothing to do")
        return lib

    if reqVersion != "latest" and reqVersion and reqVersion != my_version:
        update_command = f"pip install --upgrade {package}=={reqVersion}"
        logger.info(f"{update_command=}")
        os.system(update_command)
        # lib = __import__(package)
        importlib.reload(lib)
    else:

        latest_version = response.json()["info"]["version"]

        # warn user if versions don't match, automatically update if desired. returns updated and reloaded lib
        if my_version != latest_version:
            update_command = f"pip install --upgrade {package}"
            msg = f"{my_version=} != {latest_version=} for {package=}. "
            msg += (
                f"please update using: {update_command}" if askUser else "updating now."
            )
            logger.warning(msg)

            if not (askUser) or input(f"update {package=} inline? Press 'y'") == "y":
                os.system(update_command)
                # lib = __import__(package)
                importlib.reload(lib)
                new_version = version(package)
                logger.info(f"Upgraded {package}, using version: {new_version}")
        else:
            logger.urgent(f"Using latest version of {package}: {latest_version}")

    return lib


def load_keys(
    brokers: List[str],
    capitalize=False,
    secretsDir="/home/paul/repos/rarc-secrets",
    file="/rarc/config/keys.cfg",
):
    """Load cfg keys to dictionary.

    capitalize  capitalize dict keys
    """
    file = secretsDir + file
    if not capitalize:
        brokers = [b.lower() for b in brokers]

    def load():
        read_res = cfg.read(file)
        assert (
            lreadres := len(read_res)
        ) == 1, f"{lreadres=} != 1, cannot load config. {file=}"

        print(f"cfg keys: {list(cfg.keys())}")
        # keys = defaultdict(dict) # dangerous, any read will give a return value
        keys = {b: {} for b in brokers}
        for broker in brokers:
            for what in ("apiKey", "secret", "password"):
                if (info := cfg.get(broker.lower(), what, fallback="")) != "":
                    keys[broker][what] = info
                    # print(f'{info=}')
                    # print(f"loaded {what} for {broker}")
                else:
                    print(f"cannot load {what} cfg for {broker}")

        return keys

    return load


def chainGatherRes(res: List[dict], uniqueKeys=True) -> dict:
    """Chain asyncio.gather results.

    asyncio.gather(*cors) returns a list of results per coroutine
    his methods chains the results, and checks if there are duplicate keys
    since ChainMap would overwrite the values
    """
    assert isinstance(res, list), f"{type(res)=} is not list"

    if uniqueKeys:
        allkeys = [d.keys() for d in res]
        alls = list(itertools.chain(*allkeys))
        uniques: Set[Any] = set().union(*allkeys)

        assert (lalls := len(alls)) == (luniq := len(uniques)), f"{lalls} != {luniq}"

    return dict(ChainMap(*res))


def format_time(t) -> str:
    """Return a formatted time string 'HH:MM:SS'.

    based on a numeric time() value
    """
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    return f"{h:0>2.0f}:{m:0>2.0f}:{s:0>2.0f}"


class MultipleTimeSeriesCV:
    """Generate tuples of train_idx, test_idx pairs.

    Assumes the MultiIndex contains levels 'symbol' and 'date'
    purges overlapping outcomes
    """

    def __init__(
        self,
        n_splits=3,
        train_period_length=126,
        test_period_length=21,
        lookahead=None,
        date_idx="date",
        shuffle=False,
    ):
        self.n_splits = n_splits
        self.lookahead = lookahead
        self.test_length = test_period_length
        self.train_length = train_period_length
        self.shuffle = shuffle
        self.date_idx = date_idx

    def split(self, X, y=None, groups=None):

        unique_dates = X.index.get_level_values(self.date_idx).unique()
        days = sorted(unique_dates, reverse=True)
        split_idx = []

        for i in range(self.n_splits):
            test_end_idx = i * self.test_length
            test_start_idx = test_end_idx + self.test_length
            train_end_idx = test_start_idx + self.lookahead - 1
            train_start_idx = train_end_idx + self.train_length + self.lookahead - 1
            split_idx.append(
                [train_start_idx, train_end_idx, test_start_idx, test_end_idx]
            )

        dates = X.reset_index()[[self.date_idx]]

        for train_start, train_end, test_start, test_end in split_idx:

            train_idx = dates[
                (dates[self.date_idx] > days[train_start])
                & (dates[self.date_idx] <= days[train_end])
            ].index
            test_idx = dates[
                (dates[self.date_idx] > days[test_start])
                & (dates[self.date_idx] <= days[test_end])
            ].index
            if self.shuffle:
                np.random.shuffle(list(train_idx))
            yield train_idx.to_numpy(), test_idx.to_numpy()

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits


def round_to_power(num: Union[float, int], power: int) -> Union[float, int]:
    """Round number to nearest power.

    round 59_354 -> 59_000, power=3
    round 59_354 -> 59_300, power=2

    never returns number greater than 'num'
    """
    if power == 0:
        return num

    res = round(num, -power)
    if res > num:
        res -= 10**power

    return int(res)


class FlexibleTimeSeriesCV:
    """Generate tuples of train_idx, test_idx.

    uses datetimeindex with split, so it always
    return valid tuples

    supports TICK datasets with many (million) rows

    powern  round split_len, train_len and test_len
    to this power. So with round_to_power=3 59_354 -> 59_000, etc.
    """

    # from: https://towardsdatascience.com/dont-use-k-fold-validation-for-time-series-forecasting-30b724aaea64
    def __init__(
        self, n_splits=4, powern=3, train_test_ratio=0.8, date_idx="date", shuffle=False
    ):

        self.n_splits = n_splits
        self.powern = powern
        self.train_test_ratio = train_test_ratio
        self.date_idx = date_idx
        self.shuffle = shuffle

    def split(self, X, y=None, groups=None):

        # ugly import here, since you don't know in what environment misc.py will run
        from sktime.forecasting.model_selection import SlidingWindowSplitter

        split_len = round_to_power(
            len(np.array_split(X.index, self.n_splits)[0]), self.powern
        )

        train_len = round_to_power(split_len * self.train_test_ratio, self.powern)
        test_len = split_len - train_len

        split_msg = "split_len, train_len, test_len = {:_}, {:_}, {:_}".format(
            split_len, train_len, test_len
        )
        # logger.info(split_msg)
        print(split_msg)

        # unique_dates = X.index.get_level_values(self.date_idx).unique()
        # dates = sorted(unique_dates, reverse=True)
        # split_idx = []

        # self.n_splits = cv.get_n_splits(X.index)

        cv = SlidingWindowSplitter(
            window_length=train_len,
            fh=np.arange(1, test_len + 1),
            step_length=split_len,
        )

        # for train_ix, test_ix in cv.split(X.index):
        #     yield train_ix, test_ix

        return cv.split(X.index)

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits


def getPublicIP(url="https://api.ipify.org", n=0, max_retry=5) -> Optional[str]:
    """Get public IP address from API endpoint."""
    try:
        resp = requests.get(url)
        res = resp.content.decode("utf8")

        # retry if result is html page instead of host url, max 5 retries
        if len(res) > 25 and n < max_retry:
            n += 1
            sleep(0.2)
            return getPublicIP(url, n=n)

        if n > 0:
            logger.warning(f"{n} retries")
        return res

    except Exception as e:
        logger.error(f"cannot fetch public IP from API. {str(e)=}")

    return None


def get_key_or_none(d: Optional[dict], key: str) -> Optional[Any]:
    if isinstance(d, dict):
        return d.get(key, None)

    return None


def unnest_dict(
    df, col: str, key: str, renameCol=None, fmt="{}_{}", assign=True, debug=False
) -> Union[str, Tuple[str, pd.Series]]:
    """Unnest a dataframe column of dictionaries to a new column with dict values for key `key`.

    df          dframe  dataframe to unnest
    col         str     dataframe column that contains dictionaries
    key         str     key of dict to unnest
    renameCol   str     optional name to give to new col
    fmt         str     format to use for new col name
    assign      bool    assigns new column to passed dataframe (violates pure function restriction)
    debug       bool    print df.shape after assigning

    examples:
        unnest_dict(df, 'meta', 'field', assign=False, debug=False)

    unnests
        project  {'id': 'ob-train', 'number': '653959998010'}

    to
        project_id:         'ob_train'
        project_number:     '653959998010'

    """
    assert isinstance(df, pd.DataFrame)

    if len(df) > 0:
        assert col in df.columns, f"{col=} not in cols={list(df.columns)}"
        some_row = df.iloc[0]
        assert isinstance(
            some_row[col], dict
        ), f"{col=} {type(some_row[col])=}, should be dict"
        # assert key in some_row[col], f"{key=} not in {some_row[col]}"

    # newColName = f'{col}_{key}'
    if renameCol is None:
        renameCol = col
    newColName = fmt.format(renameCol, key)

    newCol = df[col].map(
        lambda x: get_key_or_none(x, key)
    )  # lambda x: x.get(key, None)

    if assign:
        # assign created new df, original df is not updated
        # df = df.assign(**{newColName: newCol})
        df[newColName] = newCol

        if debug:
            print(f"{df.shape=}")

        return newColName

    return newColName, newCol


def unnest_assign_cols(
    df,
    unnestList=None,
    col: str = None,
    fmt="{}_{}",
    renameCol=None,
    multiindexCols=False,
    debug=False,
) -> Tuple[List[str], pd.DataFrame]:
    """Convert list of ('col','key') tuples to list of new col names and assigned dataframe.

    unnestList  if None, will try to unnest all keys in dict
    col         if passed, unnest this col

    usage:
        unnestList = [('project','id'), ('service','description'), ('sku','description'), ]
        addedCols, df = unnest_assign_cols(df, unnestList)
    """
    assert isinstance(df, pd.DataFrame)
    assert isinstance(unnestList, (list, type(None)))

    if col is not None:
        assert isinstance(col, str), f"{type(col)=} not `str`"
        # substract all possible keys from this col
        d = df[col].map(lambda x: x.keys()).values
        all_keys = set(itertools.chain.from_iterable(d))

        unnestList = [(col, key) for key in all_keys]

    assert col or unnestList, f"pass `col` or `unnestList`"

    nameColTuples = list(
        map(
            lambda x: unnest_dict(
                df, *x, renameCol=renameCol, fmt=fmt, assign=False, debug=debug
            ),
            unnestList,
        )
    )

    addedCols = [n[0] for n in nameColTuples]

    # optionally create multiindex columns
    if multiindexCols and col is not None:
        df.columns = pd.MultiIndex.from_product([["global"], df.columns])

        # now create a new dataframe, with upper level 'col'
        # print(dict(nameColTuples))
        to_join = pd.DataFrame(dict(nameColTuples))
        to_join.columns = pd.MultiIndex.from_product([[col], addedCols])

        # multiindexCols = pd.MultiIndex.from_product([['global', col], [globCols, addedCols]])

        # join dfs horizontally into one
        df = pd.concat([df, to_join], axis=1)

    else:
        df = df.assign(**dict(nameColTuples))

    return addedCols, df


def fmt_shape(shape: tuple) -> str:
    """Format shape of dataframe, so that big numbers are displayed in 10_000, 250_000, etc."""
    assert isinstance(shape, tuple)
    inside = ", ".join([f"{x:_}" for x in shape])
    return "({})".format(inside)


def get_git_revision_hash() -> str:
    """Get git revision hash."""
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def get_git_revision_short_hash() -> str:
    """Get git revision hash (short)."""
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )


def elaps_df(
    call_hist: Deque[tuple] = None, byFunc=None, fmtPrec=None, drop=True
) -> pd.DataFrame:
    """Get elapsed dataframe.

    precision follows from the namedtuple field names, is it either time_ms or time_ns

    byFunc      filter df by function name and drop 'func' col
    fmtPrec     format datetimeindex precision: 's', ms', 'ns'
    drop        drop intermediary columns 'time' and 'dindex'

    usage:
        df = elaps_df(zp.call_hist, byFunc='calc_talib')
        df = elaps_df(zp.call_hist, byFunc='calc_talib', fmtPrec='ms')
        elaps_df(zp.call_hist).groupby('func').describe()
        elaps_df(zp.call_hist).groupby('func').describe(percentiles=(.25, .5, .7, .8, .9, .95)).T
    """
    assert call_hist is not None
    assert len(call_hist) > 0, "empty call history"

    PRECS = ("s", "ms", "ns")
    funcCol = "func"

    try:
        # df = pd.DataFrame(copy.deepcopy(call_hist))
        df = pd.DataFrame(call_hist)
    except Exception as e:
        logger.error(f"cannot parse list to dataframe. {e=!r}")
        raise

    if byFunc is not None:
        funcNames = list(df[funcCol].unique())
        assert byFunc in funcNames, f"{byFunc=} not in {funcNames=}"
        df = df[df[funcCol] == byFunc]
        del df[funcCol]

    timeCol = df.filter(regex=r"^time.+").columns[0]  # returns 'time_ms' or 'time_ns'

    precision = timeCol.split("_")[-1]
    assert precision in PRECS, f"{precision=} not in {PRECS=}"
    # set ns time columns as datetimeindex
    df[timeCol].values.astype(dtype=f"datetime64[{precision}]")
    df.index = pd.DatetimeIndex(df[timeCol])
    df["dindex"] = df.index

    # does this assumptionhold? or sort
    assert df.index.is_monotonic_increasing

    # convenient way of making the index display less or more decimal digits
    if fmtPrec is not None:
        assert fmtPrec in PRECS, f"{fmtPrec=} not in {PRECS=}"
        df.index = df.dindex.dt.ceil(freq=fmtPrec)

    if drop:
        del df[timeCol], df["dindex"]

    return df


def timeago_series(s: pd.Series) -> pd.Series:
    """Apply timeago to pandas.Series object."""
    return s.map(lambda x: timeago.format(x, datetime.utcnow()))


def trunc_msg(msg: Optional[str], maxlen=125, pfx="...") -> str:
    """Truncate a long msg, e.g. json string to {'a':2, ...}."""
    assert isinstance(msg, (str, type(None))), f"{type(msg)=}, is not str"

    if msg is None:
        msg = "None"

    if len(msg) <= maxlen:
        return msg

    # only truncate when maxlen is exceeded
    return f"{msg[:maxlen]}{pfx}"


def size_mb(obj: Any, precision=2) -> float:
    """Get size in mb for any python object, including nested dicts."""
    size = 0
    if isinstance(obj, dict):
        for _, v in obj.items():
            siz = 0
            try:
                siz = sys.getsizeof(v)
            except Exception as e:
                logger.error(f"cannot get size of {type(v)=} \n{v=} {e=!r}")
                raise

            size += siz
    else:
        size = sys.getsizeof(obj)

    return round(size / 1024**2, precision)
