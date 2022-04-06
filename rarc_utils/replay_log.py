"""
    replay_log.py

    replay json log files as if they occured in current session

    todo:
        - load log files from redis, create interface that quickly selects last log per broker, instance, datetime, etc.
        - allow to replay debug OR info level

    example usage:
        %run ~/repos/rarc-utils/rarc_utils/replay_log.py --how uuid --uuid '5f78dbdb-efe3-4170-b980-ce23c3999e00'
        ipy ~/repos/rarc-utils/rarc_utils/replay_log.py -i -- --how uuid --uuid '5f78dbdb-efe3-4170-b980-ce23c3999e00'
        ipy ~/repos/rarc-utils/rarc_utils/replay_log.py -i -- --how redis
        %run ~/repos/rarc-utils/rarc_utils/replay_log.py --how redis
        ipy ~/repos/rarc-utils/rarc_utils/replay_log.py -i -- --how file
"""

from typing import List, Optional
import argparse
import logging
import configparser
from datetime import datetime
from pathlib import Path
from enum import Enum, auto

import redis
import pandas as pd

from sqlalchemy.orm import Session

from rarc.config.redis import redis as rk
import rarc.config.redis
from rarc_utils.log import setup_logger, read_json_log_file, read_json_log_redis
from rarc_utils.misc import AttrDict
from rarc_utils.sqlalchemy_base import get_session

log_fmt = "%(asctime)s - %(module)-16s - %(lineno)-4s - %(funcName)-16s - %(levelname)-7s - %(message)s"  # name
logger = setup_logger(cmdLevel=logging.INFO, saveFile=0, savePandas=1, jsonLogger=0, color=1, fmt=log_fmt) # URGENT WARNING
log_file = 'json_lines.log'
log_handler = logger.handlers[-1] # last handler is colored logs console handler

# ugly way of retrieving postgres cfg file
p = Path(rarc.config.redis.__file__)
p.with_name('aatPostgres.cfg')
cfgFile = p.with_name('aatPostgres.cfg')
parser  = configparser.ConfigParser()
parser.read(cfgFile)
psql  = AttrDict(parser['psql'])
psession = get_session(psql)()

class read_options(Enum):
    """ log read options """
    FILE = auto()
    UUID = auto()
    REDIS = auto()

read_options_str = ', '.join(read_options.__members__.keys())

session_query = """
        SELECT * 
        FROM 
            last_sessions;
    """
    
log_query = """         
        SELECT 
            DISTINCT ON (instrum_broker) strategy_name, log_id, instrums, brokers, success, platform, ccxt_version, nerror, ntotal, updated
        FROM 
            (
            SELECT 
                *, CONCAT(instrums, ' ', brokers) AS instrum_broker 
            FROM 
                last_log_sessions
            ORDER BY 
                updated
            ) AS nested

        ORDER BY instrum_broker, updated DESC;
    """

# get_last_log_sessions(get_session(psql)())
def get_last_log_sessions(session: Session) -> pd.DataFrame:
    # this now happens automatically through trigger functions
    # refresh_view = "REFRESH MATERIALIZED VIEW last_log_sessions;"
    # psession.execute(refresh_view)

    res = session.execute(log_query)

    df = pd.DataFrame(res.mappings().fetchall())
    df['success'] = df['success'].astype(int)
    df['updated_ago'] = datetime.utcnow() - df.updated
    df = df.sort_values('updated', ascending=False).reset_index(drop=True)

    session.close()

    return df

def select_log_session(df: pd.DataFrame) -> Optional[pd.Series]:
    """ user can select log file to load by typing the index of dataframe """

    print(f'last_log_sessions: \n{df} \n')
    input_ = input('select index to read: ')
    input_ = int(input_)

    try:
        assert input_ in df.index
    except AssertionError:
        raise
    finally:
        psession.close()

    return df.loc[input_, :]

if __name__ == "__main__":

    CLI=argparse.ArgumentParser()
    CLI.add_argument(
      "--how",
      type=str,         
      default='file',
      help=f"select read option: {read_options_str}"
    )
    CLI.add_argument(
      "--uuid",
      type=str,         
      default='',
      help="UUID of log to fetch from redis"
    )
    CLI.add_argument(
      "-v", "--verbosity", 
      type=str,         
      default='info',
      help="choose debug log level: DEBUG, INFO, WARNING, ERROR, CRITICAL"
    )

    args = CLI.parse_args()

    lines: List[dict] = []

    how: str = args.how.upper()
    
    if how == 'FILE':
        lines = read_json_log_file(log_file)

    elif how in ('REDIS', 'UUID'):
        rs = redis.Redis(host=rk.REDIS_HOST, port=rk.REDIS_PORT, password=rk.REDIS_PASS, db=6)

        if how == 'REDIS':
            # get last logs per broker, instrument from postgres, and let user select what log file to load
            item = select_log_session(get_last_log_sessions(psession))
            uuid = item.log_id

        elif how == 'UUID':

            assert args.uuid, f"please pass `uuid` flag to read from redis"
            uuid = args.uuid
    
        lines = read_json_log_redis(str(uuid), rs=rs)

    else:
        raise NotImplementedError(f"{args.how=} not implemented. Available: {read_options_str}")

    assert isinstance(lines, list)

    # set log verbosity
    verbosity = args.verbosity.upper()
    log_level   = getattr(logging, verbosity)
    logger.setLevel(log_level)

    # replay log
    for line in lines:
        # logFunc = getattr(logger, line['levelname'].lower())
        # logFunc(*msg)

        print(log_handler.formatter.format(logging.makeLogRecord(line)))
