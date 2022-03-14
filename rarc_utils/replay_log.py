"""
    replay_log.py

    replay json log files as if they occured in current session

    todo:
        load log files from redis, create interface that quickly selects last log per broker, instance, datetime, etc.
"""


import logging
from yapic import json

from rarc_utils.log import setup_logger

log_fmt = "%(asctime)s - %(module)-16s - %(lineno)-4s - %(funcName)-16s - %(levelname)-7s - %(message)s"  # name
logger = setup_logger(cmdLevel=logging.INFO, saveFile=0, savePandas=1, jsonLogger=0, color=1, fmt=log_fmt) # URGENT WARNING
log_file = 'json_lines.log'
log_handler = logger.handlers[-1] # last handler is colored logs console handler

renameDict = dict(message='msg')

# read log file
lines = []
with open(log_file) as f:
    for line in f:
        d = json.loads(line)
        for k_old, k_new in renameDict.items():
            d[k_new] = d.pop(k_old)
        lines.append(d)

# replay log
for line in lines:
    # logFunc = getattr(logger, line['levelname'].lower())
    # logFunc(*msg)

    print(log_handler.formatter.format(logging.makeLogRecord(line)))
