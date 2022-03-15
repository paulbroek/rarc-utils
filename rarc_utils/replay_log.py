"""
    replay_log.py

    replay json log files as if they occured in current session

    todo:
        load log files from redis, create interface that quickly selects last log per broker, instance, datetime, etc.

    example usage:
        %run ~/repos/rarc-utils/rarc_utils/replay_log.py --how redis --uuid '5f78dbdb-efe3-4170-b980-ce23c3999e00'
        ipy ~/repos/rarc-utils/rarc_utils/replay_log.py -i -- --how redis --uuid '5f78dbdb-efe3-4170-b980-ce23c3999e00'
"""

from typing import List
import argparse
import logging
import configparser
from pathlib import Path
from enum import Enum

import redis
import pandas as pd

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
    FILE = 0
    UUID = 1
    REDIS = 2

read_options_str = ', '.join(read_options.__members__.keys())

def get_last_log_sessions() -> pd.DataFrame:
    log_query = """         
        SELECT 
            DISTINCT ON (instrum_broker) log_id, instrums, brokers, platform, ccxt_version, nerror, ntotal, updated, updated_ago 
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

    res = psession.execute(log_query)

    df = pd.DataFrame(res.mappings().fetchall())
    df = df.sort_values('updated', ascending=False).reset_index(drop=True)

    return df

def select_log_session(df: pd.DataFrame):

    print(f'last_log_sessions: \n{df} \n')
    input_ = input('select index to read: ')
    input_ = int(input_)

    assert input_ in df.index

    return df.loc[input_, :]

if __name__ == "__main__":

    CLI=argparse.ArgumentParser()
    CLI.add_argument(
      "--how",    # read log from file or from redis
      type=str,         
      default='file',
      help=f"select read option: {read_options_str}"
    )
    CLI.add_argument(
      "--uuid",    # uuid of log to reply from redis
      type=str,         
      default='',
    )

    args = CLI.parse_args()

    lines: List[dict] = []

    how = args.how.upper()
    
    if how == 'FILE':
        lines = read_json_log_file(log_file)

    elif how in ('REDIS', 'UUID'):
        rs = redis.Redis(host=rk.REDIS_HOST, port=rk.REDIS_PORT, password=rk.REDIS_PASS, db=6)

        if how == 'REDIS':
            # get last logs per broker, instrument from postgres, and let user select what log file to load
            df = get_last_log_sessions()

            item = select_log_session(df)
            uuid = item.log_id

        elif how == 'UUID':

            assert args.uuid, f"please pass `uuid` flag to read from redis"
            uuid = args.uuid
    
        lines = read_json_log_redis(str(uuid), rs=rs)

    else:
        raise NotImplementedError(f"{args.how=} not implemented. Available: {read_options_str}")

    assert isinstance(lines, list)

    # replay log
    for line in lines:
        # logFunc = getattr(logger, line['levelname'].lower())
        # logFunc(*msg)

        print(log_handler.formatter.format(logging.makeLogRecord(line)))
