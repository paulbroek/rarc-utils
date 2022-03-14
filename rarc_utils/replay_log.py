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

import redis
from rarc.config.redis import redis as rk
from rarc_utils.log import setup_logger, read_json_log_file, read_json_log_redis

log_fmt = "%(asctime)s - %(module)-16s - %(lineno)-4s - %(funcName)-16s - %(levelname)-7s - %(message)s"  # name
logger = setup_logger(cmdLevel=logging.INFO, saveFile=0, savePandas=1, jsonLogger=0, color=1, fmt=log_fmt) # URGENT WARNING
log_file = 'json_lines.log'
log_handler = logger.handlers[-1] # last handler is colored logs console handler


if __name__ == "__main__":

    CLI=argparse.ArgumentParser()
    CLI.add_argument(
      "--how",    # read log from file or from redis
      type=str,         
      default='file',
    )
    CLI.add_argument(
      "--uuid",    # uuid of log to reply from redis
      type=str,         
      default='',
    )

    args = CLI.parse_args()

    lines: List[dict] = []
    # read log file
    if args.how == 'file':

        lines = read_json_log_file(log_file)
    elif args.how == 'redis':

        assert args.uuid, f"please pass `uuid` flag to read from redis"
        rs = redis.Redis(host=rk.REDIS_HOST, port=rk.REDIS_PORT, password=rk.REDIS_PASS, db=6)
        lines = read_json_log_redis(args.uuid, rs=rs)

    else:
        raise NotImplementedError(f"{args.how=} not implemented")

    assert isinstance(lines, list)

    # replay log
    for line in lines:
        # logFunc = getattr(logger, line['levelname'].lower())
        # logFunc(*msg)

        print(log_handler.formatter.format(logging.makeLogRecord(line)))
