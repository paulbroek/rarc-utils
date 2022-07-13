"""decorators.py, decorator utility functions."""

import asyncio
import copy
import inspect
import logging
import platform
import traceback
from collections import deque, namedtuple
from datetime import datetime
from functools import wraps
from time import sleep, time, time_ns
from typing import Callable, Tuple

import pandas as pd
from yapic import json  # type: ignore[import]

logger = logging.getLogger(__name__)

# __all__ = ['check_running', 'timeit']


def check_running(method):
    """Check if `is_running` flag is true.
    
    decorator
    """
    func_name = method.__name__

    @wraps(method)
    def check_run(*args, **kw):

        # make sure to run it BEFORE the method call
        self = args[0]
        if self.is_running:
            msg = f"{self.__class__.__name__} already running, {func_name=}"
            logger.error(msg)
            raise Exception(msg)

        logger.info(f"call {method.__name__}")
        result = method(*args, **kw)

        return result

    return check_run


def save_ncall(log_once_per_n=None):
    """Show `log_once_per_n` the number of times a method was called.

    decorator
    """
    def decorator(method):
        ncall = 0
        lastcall = 0

        @wraps(method)
        def ncalld(*args, **kw):
            nonlocal ncall, lastcall

            result = method(*args, **kw)
            te = time()
            ncall += 1

            if log_once_per_n is not None:
                lastAgo = (te - lastcall) if lastcall > 0 else log_once_per_n + 1

            if not log_once_per_n or lastAgo >= log_once_per_n:
                msg = "%r was called %d times in %d secs" % (
                    method.__name__,
                    ncall,
                    log_once_per_n,
                )

                logger.info(msg)

                lastcall = te
                ncall = 0

            return result

        return ncalld

    return decorator


def wait_for_lock(variable_name, every=1, max_=8):
    """Wait for a flag to become true, and then run the method.

    todo: or make async, and schedule the task?

    assumes:
        args[0] is the 'self' object

    uses:
        self.lock
    """
    def decorator(method):
        @wraps(method)
        def wait_for(*args, **kw):

            self = args[0]
            cnt = 0
            while self.lock[variable_name] and cnt < max_:
                logger.warning(f"waiting for '{variable_name}' lock to resolve")
                sleep(every)
                cnt += 1

            if cnt >= max_:
                logger.warning(f"reached {max_=} tries")
                # read out type annotation return type, and return an empty instance
                return_type = method.__annotations__.get("return", None)
                if return_type is not None:  # the method has an annotated return type
                    # create instance of flexible types? pd.Index can only instantiated like: pd.Index([])
                    annotation_instance = None
                    if return_type is pd.Index:
                        return pd.Index([])

                    try:
                        annotation_instance = return_type()
                    except Exception as e:
                        logger.error(
                            f"cannot create default annotation instance to return, will return None. {e=!r}"
                        )

                    return annotation_instance

                return None

            result = method(*args, **kw)

            return result

        return wait_for

    return decorator


def items_per_sec(f):
    """Display number of received/saved items per second.
    
    Applies to returning value first, if missing, it uses the first argument
    """
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        how = "got"
        if result is None:
            result = args[0]
            how = "saved"

        elapsed = time() - ts
        nitem = len(result)
        item_per_sec = int(nitem / elapsed)
        ITEMS = "item" if nitem == 1 else "items"
        msg = f"{how} {nitem:,} {ITEMS} in {elapsed:.2f} seconds. ({item_per_sec:,} items/sec) `{f.__name__}`"
        logger.info(msg)
        return result

    return wrap


def timet(f):
    """Simpler decorator than timeit."""
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        logger.debug(
            "func:%r args:[%r, %r] took: %2.4f sec" % (f.__name__, args, kw, te - ts)
        )
        return result

    return wrap


def timeit(log_once_per_n=None, reset_hist=False, save_hist=False):
    """Timeit decorator.

    make sure not to reset logger BEFORE importing this file

    supports both nanosecond and 'normal' millisecond, not sure if ns works under windows

    reset_hist      if True, resets elaps history every 'log_once_per_n'
    save_hist       if True, saves (func, time, elaps) namedtuple to self.call_hist
    """
    PRECISION = "ns"  # 'ms'
    time_func = time_ns if PRECISION == "ns" else time

    TIME_KEY = f"time_{PRECISION}"
    elaps_record = namedtuple("elaps_record", f"func {TIME_KEY} elaps_ms")

    MAX_HIST_LEN = 1_000_000

    dq = lambda: deque(
        maxlen=MAX_HIST_LEN
    )  # safer than list, for long running programs

    def decorator(method):
        lastcall = 0
        elaps_hist = dq()

        func_name = method.__name__

        # does not seem to run when imported as module
        @wraps(method)
        def timed(*args, **kw):
            nonlocal lastcall, elaps_hist

            ts = time_func()
            result = method(*args, **kw)
            te = time_func()

            elaps = te - ts
            if PRECISION == "ns":
                elaps /= 10**9
            elaps_ms = elaps * 1000

            elaps_hist.append(elaps)

            # assuming, that when this flag is True, the decorator is attached to an instance method, and it is NOT static
            if save_hist:
                self = args[0]
                # namedtuple faster?
                assert hasattr(self, "call_hist")
                self.call_hist.append(elaps_record(func_name, te, elaps_ms))
                # self.call_hist.append(elaps_record(func=func_name, TIME_KEY=te, elaps_ms=elaps_ms))
                # self.call_hist.append(dict(func=func_name, time=te, elaps=elaps))

            # what is this?
            if "log_time" in kw:
                name = kw.get("log_name", func_name.upper())
                kw["log_time"][name] = int(elaps_ms)

            else:
                # don't log when log_once_per_n was passed
                if log_once_per_n is not None:
                    if lastcall > 0:
                        lastAgo = te - lastcall
                        if PRECISION == "ns":
                            lastAgo /= 10**9
                    else:
                        lastAgo = log_once_per_n + 1

                if not log_once_per_n or lastAgo >= log_once_per_n:
                    min_elaps = min(elaps_hist)
                    max_elaps = max(elaps_hist)

                    # msg = '%r  %2.2f ms (min=%2.2f, max=%2.2f, n=%d)' % (method.__name__, elaps_ms, min_elaps, max_elaps, len(elaps_hist))
                    msg = (
                        "{!r:<25} {:<7.2f} ms (min={:.2f}, max={:.2f}, n={:,})".format(
                            func_name, elaps_ms, min_elaps, max_elaps, len(elaps_hist)
                        )
                    )

                    logger.info(msg)

                    lastcall = te

                    if reset_hist:
                        elaps_hist = dq()

                else:
                    pass

            return result

        return timed

    return decorator


def atimeit(func):
    async def process(func, *args, **params):
        if asyncio.iscoroutinefunction(func):
            # print('this function is a coroutine: {}'.format(func.__name__))
            return await func(*args, **params)

        # print('this is not a coroutine')
        return func(*args, **params)

    async def helper(*args, **params):
        print("{}.time".format(func.__name__))
        start = time()
        result = await process(func, *args, **params)

        # Test normal function route...
        # result = await process(lambda *a, **p: print(*a, **p), *args, **params)

        print(">>>", time() - start)
        return result

    return helper


# general decorator for try/except cases. generic_feed cannot have any uncaught exceptions
def get_try_catch_decorator(errors=(Exception,), default_value=""):
    def decorator(func):
        # @wraps(func)
        def new_func(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                import logging

                logger = logging.getLogger(__name__)
                logger.error(f"in {func.__name__}(): {repr(e)=} \n {args=}")
                if 1:
                    logger.error(traceback.format_exc())

                return default_value

        return new_func

    return decorator


# try_catch_key_name = get_decorator((KeyError, NameError), default_value='default')
try_catch_any = get_try_catch_decorator((Exception), default_value="")


def reconnect_mysql(func: Callable) -> Callable:
    """Reconnect to MySQL if disconnected.

    decorator
    """

    @wraps(func)
    def _reconnect(*args, **kwargs):

        self = args[0]

        assert hasattr(self, "con")

        # decorator functionality
        if not self.con.is_connected():
            logger.info(f"reconnected to MySQL")
            self.con.reconnect()

        # return original func
        return func(*args, **kwargs)

    return _reconnect


def reconnect(func: Callable) -> Callable:
    """Reconnect to redis if selected_db does not match 'db'.

    decorator
    """

    @wraps(func)
    def _reconnect(*args, **kwargs):

        # look for 'db' kwarg at binding time
        bound = inspect.signature(func).bind(*args, **kwargs)
        bound.apply_defaults()

        # print(f'{func.__name__} called with {bound}')
        # print(f'{dir(bound)=}')
        # print(f'{bound.arguments=}')
        # print(f'{type(bound.arguments)=}')

        assert (
            "db" in bound.arguments.keys()
        ), f"decorator cannot be applied to method '{func.__name__}', misses 'db' kwarg"
        self = args[0]

        # decorator functionality
        if (db := bound.arguments["db"]) != self.selected_db:
            self._connect(db=db)

        # return original func
        return func(*args, **kwargs)

    return _reconnect


def elapsed_broker(func: Callable) -> Callable:
    """Save the time elapsed for a function call.

    decorator

    TO-DO: doesn't work now since the coroutine is entered from start, resulting in large elapsed times.
    """
    @wraps(func)
    async def elaps_broker(self, *args, **kwargs):

        t0 = time()

        try:
            if asyncio.iscoroutinefunction(func):
                return await func.__call__(self, *args, **kwargs)
            else:
                return func.__call__(self, *args, **kwargs)

        except Exception as e:

            # reraise exception, so it can be caught in main flow
            raise

        finally:

            elaps = time() - t0

            # print(f'{kwargs.keys()=}')
            # how to reliably fetch exchange object and name?
            exchange = kwargs.get("exchange", "")
            exchange_name = getattr(exchange, "name", "")
            # self = args[0]
            self.brokerCalls[exchange_name].append(elaps)

    return elaps_broker


def register_with_redis(func: Callable) -> Callable:
    """Register a job/function with redis so that other processes can not run jobs simultaneously.

    decorator

    TO-DO (✓): permanently save jobs so that you can see when the last run was. this means converting key:value to key:sorted_set (done)
    """
    db = 2

    @wraps(func)
    async def reg_redis(self, *args, **kwargs):

        job_key = f"job-{func.__name__}"

        # rsel = kwargs.get('rsel', None)
        rsel = kwargs["rsel"]
        # remove star from rsel, to be able to compare if string is contained in the other rsel later
        rsel_str = rsel if "*" not in rsel else rsel.split("*")[0]
        # job_pars = dict(rsel=rsel)

        # not run if match is found in redis. Or use separate decorator for this?
        job_matches = await self.get_jobs(active=1)

        # job_ns = [int(s.split('-')[-1]) for s in job_matches.keys()]
        # max_n = max(job_ns, default=0)
        # job_key_redis = job_key
        # job_key_redis = f"{job_key}-{rsel}"
        # job_key_redis = f"{job_key}-{max_n+1}" # add a number so that multiple jobs can run with same function name

        me = platform.node()

        rerunning_old_job = False
        # rerunning_succes = False
        match_time_key = None

        if len(job_matches) > 0:

            # logger.debug(f"{job_matches=}")

            for job_key_, job_list in job_matches.items():
                for job_pars_ in job_list:
                    if not isinstance(job_pars_["rsel"], str):
                        continue

                    # parse 'job-compr_book-1' to 'job-compr_book'
                    # job_key_ = job_key_.rsplit('-', maxsplit=1)[0]
                    # job_key_ = job_key_.split('-')[-1]
                    # logger.warning(f'{job_pars_=}')
                    match_rsel = job_pars_["rsel"]

                    match_rsel_str = (
                        match_rsel
                        if "*" not in match_rsel
                        else match_rsel.split("*")[0]
                    )

                    # print(f'{job_key=} {job_key_=} {rsel_str=} {match_rsel_str=}')

                    # find similar job
                    if (
                        not kwargs.get("recursive", False)
                        and job_key_ == job_key
                        and rsel_str == match_rsel_str
                    ):

                        # find unfinished job from same 'user'
                        if job_pars_.get("who", None) == me:
                            msg = f"a similar job ran on this node but did not finish. will try to run in smaller batches for: {match_rsel}"
                            logger.warning(msg)
                            kwargs["recursive"] = True
                            rerunning_old_job = True
                            match_time_key = job_pars_.get("ts", None)
                            logger.warning(f"{match_time_key=}")

                        else:
                            msg = f"a similar job is running with similar job_pars, rsel: {match_rsel} "  # {job_pars_}

                            logger.error(msg)
                            raise Exception(msg)

        # register the job_pars for the current job
        job_kwargs = copy.deepcopy(kwargs)
        job_kwargs.pop("arc", None)  # pop connection, is not serializable
        job_kwargs["when"] = datetime.utcnow()
        job_kwargs["active"] = True
        job_kwargs["who"] = me

        # self.r.set(job_key_redis, json.dumps(job_kwargs), ex=4*60**2) # expires after 4 hours, assuming any job will have finished within that horizon
        time_key = time()  # should remain unique
        time_keys = [time_key]
        job_kwargs["ts"] = time_key
        # self.r.zadd(job_key, {json.dumps(job_kwargs): time_key})
        await self.aior[db].zadd(job_key, {json.dumps(job_kwargs): time_key})

        # question: does this also work with coroutines. When are they 'finished'?
        try:
            # func(self, *args, **kwargs)

            if asyncio.iscoroutinefunction(func):
                return await func.__call__(self, *args, **kwargs)

            return func.__call__(self, *args, **kwargs)

        except Exception as e:

            # reraise exception, so it can be caught in main flow
            raise

        finally:

            # this assumes a smaller job always succeeds.. dangerous assumption
            if rerunning_old_job:
                # rerunning_succes = True
                time_keys.append(match_time_key)

            # delete key after leaving function
            # self.r.delete(job_key_redis)
            # TO-DO: hmm, but now you don't keep a history of jobs. maybe it's better to just turn 'active' to False.
            # logger.warning(f'{time_keys=} {rerunning_old_job=}')
            for time_key_ in time_keys:
                await self.aior[db].zremrangebyscore(job_key, time_key_, time_key_)
                job_kwargs["active"] = False
                await self.aior[db].zadd(job_key, {json.dumps(job_kwargs): time_key_})

    return reg_redis


def saveLastCall(func: Callable) -> Callable:
    """Save last function calling time to `self.lastFuncCalls`.

    decorator
    """
    @wraps(func)
    def saveCall(self, *args, **kwargs):
        res = func(self, *args, **kwargs)
        self.lastFuncCalls[func.__name__] = datetime.utcnow()
        return res

    return saveCall


def colsAdded(x_or_func=None, return_cols=0, print_less_than=5, debug=0) -> Callable:
    """Return the columns that have been added to a dataframe, during a dataframe operation.

    decorator

    assumes that first or second argument dataframe type is the dataframe being manipulated (so can be used with classmethods)

    return_cols         return list of added columns along with wrapped functions return value. if false, returns only wrapped functions return value
    print_less_than     if less than 'print_less_than' columns are added, all names are logged

    """
    def _colsAdded(func):
        @wraps(func)
        def wrapper(*args, **kwargs) -> Tuple[pd.DataFrame, pd.Index]:
            df = args[0]
            if not isinstance(df, pd.DataFrame):
                df = args[1]

            assert isinstance(df, pd.DataFrame), f"{args=}"

            cols_before = df.columns
            func_res = func(*args, **kwargs)
            # assuming the result is a dataframe, or try the first item of a tuple rest
            df = func_res[0] if not isinstance(func_res, pd.DataFrame) else func_res
            assert isinstance(df, pd.DataFrame), f"{type(df)=}, should be pd.DataFrame"

            cols_after = df.columns

            added = cols_after.difference(
                cols_before, sort=False
            )  # this however, does not return columns in the order they were added to df. how to fix? --> Use sort=False

            if debug:
                print(f"{added=}")

            # try to join multiindex cols
            if isinstance(added, pd.MultiIndex):
                added = ["".join([*ky]) for ky in added]

            if len(added) > 0:
                COLS = "cols" if len(added) > 1 else "col"
                COLS_ADDED = (
                    "" if len(added) > print_less_than else f": {', '.join(added)}"
                )
                logger.debug(f"{func.__name__}: {len(added)} {COLS} added{COLS_ADDED}")

            if return_cols:
                return func_res, added

            return func_res

        return wrapper

    return _colsAdded(x_or_func) if callable(x_or_func) else _colsAdded


def rowsReduced(func):
    """Return the number of rows that were removed from a dataframe, during a dataframe operation.

    decorator
    """
    @wraps(func)
    def _rowsReduced(*args, **kwargs):

        # df = kwargs['df']
        df = args[0]
        assert isinstance(df, pd.DataFrame)

        rows_before = len(df)

        res_df = func(*args, **kwargs)

        rows_dropped = rows_before - len(res_df)
        pct_rows_dropped = rows_dropped / rows_before

        if rows_dropped > 0:
            ROWS = "row" if rows_dropped == 1 else "rows"
            logger.info(
                f"reduced df by {rows_dropped:,} {ROWS} ({pct_rows_dropped:.2%})"
            )

        return res_df

    return _rowsReduced
