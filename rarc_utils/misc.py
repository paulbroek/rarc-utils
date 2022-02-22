""" 
    misc.py

    miscellaneous methods used by rarc 
"""

from typing import Dict, List, Tuple, Union, Optional, Any, Deque
import traceback
import hashlib
import uuid
import importlib
from time import time, time_ns
import os
from datetime import datetime
import logging
from collections import defaultdict, deque, namedtuple, ChainMap
from functools import wraps, partial
import configparser
import signal
import itertools
import sys
import random
import platform
import copy
import yaml
import subprocess
import requests
from pathlib import Path
import psutil
import asyncio

from yapic import json
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

cfg = configparser.ConfigParser()

# using hash algorithm that produces the same hash, so that objects send from different sources have the same hash when the dicts are equal in values
def hash_accross_the_same(item):
    h = hashlib.new('ripemd160')
    h.update(item.encode())
    return h.hexdigest()

class KeyDict(defaultdict):
    def __missing__(self, key):
        return key

class CapKeyDict(defaultdict):
    def __missing__(self, key):
        return key.capitalize()

def reload_by_str():
    """ replaces all reload_XX methods below, pass a str module name and 
        this method assumes the structures of the loaded module is: df_methods.df_methods
    """

    def reload():
        import ccxt_methods
        importlib.reload(ccxt_methods)
        from ccxt_methods import ccxt_methods as cm
        return cm


    # this works in jupyter notebook:

    # import importlib
    # import rarc.tf_methods
    # importlib.reload(rarc.tf_methods)
    # tf_methods = rarc.tf_methods.tf_methods

    return reload

def reload_cm():
    """ returns reload method, so it can be executed in the right context / path """

    def reload():
        import ccxt_methods
        importlib.reload(ccxt_methods)
        from ccxt_methods import ccxt_methods as cm
        return cm

    return reload

def reload_dm():
    """ returns reload method, so it can be executed in the right context / path 

        can import directly from folder or as a module
    """
    
    def reload():
        as_module = False 
        try:
            import df_methods
        except ModuleNotFoundError:
            import rarc.df_methods as df_methods

            as_module = True

        importlib.reload(df_methods)
        
        if as_module:
            from rarc.df_methods import df_methods as dm
        else:
            from df_methods import df_methods as dm

        return dm

    return reload

def reload_fm():
    """ returns reload method, so it can be executed in the right context / path """
    def reload():
        import feature_methods
        importlib.reload(feature_methods)
        from feature_methods import feature_methods as fm
        return fm

    return reload

def reload_pm():
    """ returns reload method, so it can be executed in the right context / path """
    def reload():
        import plot_methods
        importlib.reload(plot_methods)
        from plot_methods import plot_methods as pm
        return pm

    return reload

def run_in_executor(f):
    @wraps(f)
    def inner(*args, **kwargs):
        loop = asyncio.get_running_loop()
        return loop.run_in_executor(None, partial(f, *args, **kwargs))

    return inner

class AttrDict(dict):
    """ emulate js object: dict.attr equals dict['attr'] """
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
sigd = {getattr(signal, n) : n for n in dir(signal) if n.startswith('SIG')}

def handle_exit(sigNum, stackFrame, addDate=0, andPrint=0):
    sigName = sigd.get(sigNum, 'undefined')
    dash = '------------------'
    dtstr = ''
    if addDate:
        dtstr = datetime.utcnow().strftime('%Y:%m:%d %H:%M:%S ')
    msg = f'{dash} {dtstr}EXITING ({sigNum=} {sigName=}) {dash}' #  {stackFrame=}
    lenmsg = len(msg)
    #msgs = ['', lenmsg * '-', msg, lenmsg * '-']
    #msgs = [lenmsg * '-', msg, lenmsg * '-']
    msgs = [msg]

    #logger.info(msg)
    for msg_ in msgs:
        logger.warning(msg_)

    if andPrint:
        for msg_ in msgs:
            print(msg_)

    sys.exit()

def get_random_df(arcres: Dict[str, pd.DataFrame]) -> Tuple[str, pd.DataFrame]:
    return random.choice(list(arcres.items()))

def get_sample(d: dict, n=3) -> dict:
    """ take sample of dictionary, for printing purposes 
        n   unique random elements to choose, if n > len(d), 
            return the original dict
    """
    assert isinstance(d, dict)

    if n > len(d):
        return d
    
    return dict(random.sample(d.items(), n))

du_dir_ = lambda path: sum(f.stat().st_size for f in path.glob('**/*') if f.is_file())

def du_dir(path: Path) -> int:
    """ return total disk usage of path """

    return du_dir_(path)

def dir_(obj, not_starts=('__','_')) -> List[str]:
    """ alternative dir() method that only shows public attributse """
    
    return [a for a in dir(obj) if not a.startswith(not_starts)]
    
def mem_usage(prec=2, power=2) -> int:
    """ get memory usage of this process, and therefore this python session 
        
        power   1024**2 is mb, 1024**3 is gb, etc.

        alternative way is running bash directly in ipython, but only works when using only one session:
        !ps aux | grep python | awk '{sum=sum+$6}; END {print sum/1024 " MB"}'
    """

    process     = psutil.Process(os.getpid())
    bytes_usage = process.memory_info().rss
    fmt_usage   = bytes_usage / 1024**power
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

def check_version(package: str, lib, logger, reqVersion=False, askUser=True):
    """ compares local version of a package with that listed on pypi.org. for ccxt lib it's crucial that you always update before running this bot. """
    import requests
    from importlib.metadata import version

    # fetch latest PyPi version
    url = f'https://pypi.org/pypi/{package}/json'
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

    if reqVersion != 'latest' and reqVersion and reqVersion != my_version: 
        update_command = f'pip install --upgrade {package}=={reqVersion}'
        logger.info(f'{update_command=}')
        os.system(update_command)
        #lib = __import__(package)
        importlib.reload(lib)
    else:
            
        latest_version = response.json()['info']['version']

        # warn user if versions don't match, automatically update if desired. returns updated and reloaded lib
        if my_version != latest_version:
            update_command = f'pip install --upgrade {package}'
            msg = f"{my_version=} != {latest_version=} for {package=}. "
            msg += f"please update using: {update_command}" if askUser else "updating now."
            logger.warning(msg)

            if not(askUser) or input(f"update {package=} inline? Press 'y'") == 'y':
                os.system(update_command)
                #lib = __import__(package)
                importlib.reload(lib)
                new_version = version(package)
                logger.info(f"Upgraded {package}, using version: {new_version}")
        else:
            logger.urgent(f"Using latest version of {package}: {latest_version}")

    return lib

def load_keys(brokers: dict, capitalize=False, secretsDir='/home/paul/repos/rarc-secrets', file='/rarc/config/keys.cfg'): # file='config/keys.cfg'
    """ load cfg keys to dictionary
    
        capitalize  capitalize dict keys
    """

    file = secretsDir + file
    if not capitalize:
        brokers = [b.lower() for b in brokers]
    #print(f'{brokers=}')
    def load():
        read_res = cfg.read(file) # returns ['filename']
        assert (lreadres := len(read_res)) == 1, f"{lreadres=} != 1, cannot load config. {file=}"
        print(f'cfg keys: {list(cfg.keys())}')
        #keys = defaultdict(dict) # dangerous, any read will give a return value
        keys = {b: dict() for b in brokers}
        for broker in brokers:
            for what in ('apiKey','secret','password'):
                if (info := cfg.get(broker.lower(), what, fallback='')) != '':
                    keys[broker][what] = info
                    # print(f'{info=}')
                    # print(f"loaded {what} for {broker}")
                else: 
                    print(f'cannot load {what} cfg for {broker}')

        return keys

    return load

def chainGatherRes(res: List[dict], uniqueKeys=True) -> dict:
    """ 
        chain asyncio.gather results

        asyncio.gather(*cors) returns a list of results per coroutine
        his methods chains the results, and checks if there are duplicate keys
        since ChainMap would overwrite the values
    """

    assert isinstance(res, list), f"{type(res)=} is not list"

    if uniqueKeys: 
        allkeys = [d.keys() for d in res]
        alls =  list(itertools.chain(*allkeys))
        uniques = set().union(*allkeys)

        assert (lalls := len(alls)) == (luniq := len(uniques)), f"{lalls} != {luniq}"

    return dict(ChainMap(*res))

def format_time(t) -> str:
    """Return a formatted time string 'HH:MM:SS
    based on a numeric time() value"""
    
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    return f'{h:0>2.0f}:{m:0>2.0f}:{s:0>2.0f}'

class MultipleTimeSeriesCV:
    """Generates tuples of train_idx, test_idx pairs
    Assumes the MultiIndex contains levels 'symbol' and 'date'
    purges overlapping outcomes"""

    def __init__(self,
                 n_splits=3,
                 train_period_length=126,
                 test_period_length=21,
                 lookahead=None,
                 date_idx='date',
                 shuffle=False):
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
            split_idx.append([train_start_idx, train_end_idx,
                              test_start_idx, test_end_idx])

        dates = X.reset_index()[[self.date_idx]]

        for train_start, train_end, test_start, test_end in split_idx:

            train_idx = dates[(dates[self.date_idx] > days[train_start])
                              & (dates[self.date_idx] <= days[train_end])].index
            test_idx = dates[(dates[self.date_idx] > days[test_start])
                             & (dates[self.date_idx] <= days[test_end])].index
            if self.shuffle:
                np.random.shuffle(list(train_idx))
            yield train_idx.to_numpy(), test_idx.to_numpy()

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits

def round_to_power(num: Union[float, int], power: int) -> Union[float, int]:
    """ 
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
    """ Generates tuples of train_idx, test_idx
    uses datetimeindex with split, so it always 
    return valid tuples 

    supports TICK datasets with many (million) rows

    powern  round split_len, train_len and test_len 
    to this power. So with round_to_power=3 59_354 -> 59_000, etc.
    """

    # from: https://towardsdatascience.com/dont-use-k-fold-validation-for-time-series-forecasting-30b724aaea64
    def __init__(self,
                 n_splits=4,
                 powern=3,
                 train_test_ratio=0.8,
                 date_idx='date',
                 shuffle=False):

        self.n_splits = n_splits
        self.powern = powern
        self.train_test_ratio = train_test_ratio
        self.date_idx = date_idx
        self.shuffle = shuffle

    def split(self, X, y=None, groups=None):

        # ugly import here, since you don't know in what environment misc.py will run
        from sktime.forecasting.model_selection import SlidingWindowSplitter

        split_len = round_to_power(len(np.array_split(X.index, self.n_splits)[0]), self.powern)

        train_len = round_to_power(split_len * self.train_test_ratio, self.powern)
        test_len = split_len - train_len

        split_msg = 'split_len, train_len, test_len = {:_}, {:_}, {:_}'.format(split_len, train_len, test_len)
        # logger.info(split_msg)
        print(split_msg)

        # unique_dates = X.index.get_level_values(self.date_idx).unique()
        # dates = sorted(unique_dates, reverse=True)
        # split_idx = []

        # self.n_splits = cv.get_n_splits(X.index)

        cv = SlidingWindowSplitter(window_length=train_len, fh=np.arange(1, test_len+1), step_length=split_len)

        # for train_ix, test_ix in cv.split(X.index):
        #     yield train_ix, test_ix

        return cv.split(X.index)

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits

def loadConfig(configFile=None):
    assert configFile is not None

    with open(configFile, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        logger.info(f'loaded {configFile=}')

    return config

def getPublicIP(url='https://api.ipify.org') -> Optional[str]:

    try:
        res = requests.get(url)
        return res.content.decode('utf8')

    except Exception as e:
        logger.error(f'cannot fetch public IP from API. {str(e)=}')

    return None

def get_key_or_none(d: Optional[dict], key: str) -> Optional[Any]:
    if isinstance(d, dict):
        return d.get(key, None)

def unnest_dict(df, col: str, key: str, renameCol=None, fmt='{}_{}', assign=True, debug=False) -> Union[str, Tuple[str, pd.Series]]:
    """ 
        unnest a dataframe column of dictionaries to a new column with dict values for key 'key'

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
        assert isinstance(some_row[col], dict), f"{col=} {type(some_row[col])=}, should be dict"
        # assert key in some_row[col], f"{key=} not in {some_row[col]}"

    # newColName = f'{col}_{key}'
    if renameCol is None:
        renameCol = col
    newColName = fmt.format(renameCol, key)

    newCol = df[col].map(lambda x: get_key_or_none(x, key)) # lambda x: x.get(key, None)

    if assign:
        # assign created new df, original df is not updated
        # df = df.assign(**{newColName: newCol})
        df[newColName] = newCol

        if debug:
            print(f'{df.shape=}')

        return newColName

    return newColName, newCol
    
# unnestList = [('project','id'), ('service','description'), ('sku','description'), ]
# addedCols, df = unnest_assign_cols(df, unnestList)
def unnest_assign_cols(df, unnestList=None, col: str=None, fmt='{}_{}', renameCol=None, multiindexCols=False, debug=False) -> Tuple[List[str], pd.DataFrame]:
    """
        convert list of ('col','key') tuples to list of new col names and assigned dataframe

        unnestList  if None, will try to unnest all keys in dict
        col         if passed, unnest this col
    """

    assert isinstance(df, pd.DataFrame)
    assert isinstance(unnestList, (list, type(None)))

    if col is not None:
        assert isinstance(col, str), f"{type(col)=} not `str`"
        # substract all possible keys from this col
        d = df[col].map(lambda x: x.keys()).values
        all_keys = set(itertools.chain.from_iterable(d))

        unnestList = [(col, key) for key in all_keys]

    nameColTuples = list(map(lambda x: unnest_dict(df, *x, renameCol=renameCol, fmt=fmt, assign=False, debug=debug), unnestList))

    addedCols = [n[0] for n in nameColTuples]

    # optionally create multiindex columns
    if multiindexCols and col is not None:
        df.columns = pd.MultiIndex.from_product([['global'], df.columns])

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
    """ format shape of dataframe, so that big numbers are displayed in 10_000, 250_000, etc. """
    assert isinstance(shape, tuple)
    inside = ', '.join([f"{x:_}" for x in shape])
    return '({})'.format(inside)

def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

def get_git_revision_short_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()

# df = elaps_df(zp.call_hist, byFunc='calc_talib')
# df = elaps_df(zp.call_hist, byFunc='calc_talib', fmtPrec='ms')
# elaps_df(zp.call_hist).groupby('func').describe()
# elaps_df(zp.call_hist).groupby('func').describe(percentiles=(.25, .5, .7, .8, .9, .95)).T
def elaps_df(call_hist: Deque[tuple]=None, byFunc=None, fmtPrec=None, drop=True) -> pd.DataFrame:
    """ get elapsed dataframe  

        precision follows from the namedtuple field names, is it either time_ms or time_ns

        byFunc      filter df by function name and drop 'func' col
        fmtPrec     format datetimeindex precision: 's', ms', 'ns'
        drop        drop intermediary columns 'time' and 'dindex'
    """

    assert call_hist is not None
    assert len(call_hist) > 0, 'empty call history'

    PRECS = ('s','ms','ns')
    funcCol = 'func'

    try:
        # df = pd.DataFrame(copy.deepcopy(call_hist))
        df = pd.DataFrame(call_hist)
    except Exception as e:
        logger.error(f'cannot parse list to dataframe. {e=!r}')
        raise 

    if byFunc is not None:
        funcNames = list(df[funcCol].unique())
        assert byFunc in funcNames, f"{byFunc=} not in {funcNames=}"
        df = df[df[funcCol] == byFunc]
        del df[funcCol]

    timeCol = df.filter(regex=r'^time.+').columns[0] # returns 'time_ms' or 'time_ns'

    precision = timeCol.split('_')[-1]
    assert precision in PRECS, f"{precision=} not in {PRECS=}"
    # set ns time columns as datetimeindex
    df[timeCol].values.astype(dtype=f'datetime64[{precision}]')
    df.index = pd.DatetimeIndex(df[timeCol])
    df['dindex'] = df.index

    # does this assumptionhold? or sort
    assert df.index.is_monotonic_increasing

    # convenient way of making the index display less or more decimal digits
    if fmtPrec is not None:
        assert fmtPrec in PRECS, f"{fmtPrec=} not in {PRECS=}"
        df.index = df.dindex.dt.ceil(freq=fmtPrec)

    if drop:
        del df[timeCol], df['dindex']

    return df

def trunc_msg(msg: str, maxlen=125, pfx='...') -> str:
    """ truncate a long msg, e.g. json string to {'a':2, ...} """

    assert isinstance(msg, str), f"{type(msg)=}, is not str"
    return f"{msg[:maxlen]}{pfx}"
