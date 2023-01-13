"""sqlqalchemy_base.py, utility functions that are frequently used when working with SQLAlchemy.

e.g.: creating async or blocking sessions, creating all models, getting all str models, get_or_create methods, ...
"""

import asyncio
import configparser
import inspect
import logging
import math
import os
from pathlib import Path
from pprint import pprint
from typing import (Any, AsyncGenerator, Callable, Dict, List, Optional, Set,
                    Tuple, Union)

import numpy as np
from fastapi import HTTPException
from sqlalchemy import create_engine
# from sqlalchemy import inspect as inspect_sqlalchemy
# from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession  # type: ignore[import]
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.future import select  # type: ignore[import]
from sqlalchemy.future.engine import Engine  # type: ignore[import]
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.orm.exc import NoResultFound
from tqdm import tqdm  # type: ignore[import]

# from .misc import AttrDict
from .models.psql_config import psqlConfig

logger = logging.getLogger(__name__)


# def load_config(db_name=None, cfg_file=None, config_dir=None, starts_with=False):
#     """Load config.

#     ugly way of retrieving postgres cfg file
#     """
#     assert db_name is not None
#     assert cfg_file is not None
#     assert config_dir is not None

#     # take from secrets dur if running in production: kubernetes
#     releaseMode = os.environ.get("RELEASE_MODE", "DEVELOPMENT")
#     cfgPath = (
#         Path(config_dir.__file__).with_name(cfg_file)
#         if releaseMode == "DEVELOPMENT"
#         else Path("/run/secrets") / cfg_file
#         # else Path("/run/secrets") / cfg_file / "secret.file"
#     )

#     parser = configparser.ConfigParser()
#     parser.read(cfgPath)
#     assert "psql" in parser, f"'psql' not in {cfgPath=}"
#     psql = AttrDict(parser["psql"])

#     # do not overwrite existing other db
#     if starts_with:
#         assert psql["db"].startswith(db_name)
#     else:
#         assert psql["db"] == db_name

#     return psql


class UtilityBase:
    """Adds helper methods to SQLAlchemy `Base` class."""

    def as_dict(self) -> Dict[str, Any]:
        """Format model as dictionary."""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

    def json(self) -> dict:
        """Alias for `as_dict`."""
        return self.as_dict()

    def as_big_dict(self) -> Dict[str, Any]:
        """Format model as dictionary, including methods."""
        return {
            c: getattr(self, c)
            for c in dir(self)
            if not c.startswith(("_", "__", "registry"))
        }


async def async_main(psql: psqlConfig, base, force=False, dropFirst=False) -> None:
    """Create async engine for sqlalchemy."""
    # port = getattr(psql, "port", 5432)
    engine = create_async_engine(
        f"postgresql+asyncpg://{psql.PG_USER}:{psql.PG_PASSWD}@{psql.PG_HOST}:{psql.PG_PORT}/{psql.PG_DB}",
        echo=True,
    )

    print(f"{psql=}")
    # TO-DO: make sure to check for a backup file first, as it deletes all psql data
    if dropFirst:
        if not force:
            if (
                input(
                    "are you sure you want to recreate all models? Use alembic for migrations, \
                    and run this function only for a total model reset. Pres 'y' to confirm: "
                )
                != "y"
            ):
                print("leaving")
                return

        async with engine.begin() as conn:
            await conn.run_sync(base.metadata.drop_all)
            # does not work with association tables.. --> use DROP DATABASE "enabler"; CREATE DATABASE "enabler"; for now

    async with engine.begin() as conn:
        await conn.run_sync(base.metadata.create_all)


def fmt_connection_url(psql: psqlConfig, async_=False) -> str:
    """Format connection url."""
    # pghost = os.environ.get("POSTGRES_HOST", psql.host)
    # pgport = os.environ.get("POSTGRES_PORT", psql.port) or 5432

    pfx = "postgresql"
    if async_:
        pfx += "+asyncpg"

    return f"{pfx}://{psql.PG_USER}:{psql.PG_PASSWD}@{psql.PG_HOST}:{psql.PG_PORT}/{psql.PG_DB}"


def get_engine(psql: psqlConfig, pool_size=20) -> Engine:
    """Create engine."""
    # default config can be overriden by passing pg host env var

    # read https://docs.sqlalchemy.org/en/14/core/engines.html#configuring-logging
    # `echo=True` shows duplicate log output
    engine = create_engine(
        fmt_connection_url(psql),
        echo=False,
        echo_pool=False,
        future=True,
        pool_size=pool_size,
    )

    return engine


def get_session(psql: psqlConfig, pool_size=20) -> sessionmaker:
    """Create normal (bocking) connection."""
    engine = get_engine(psql, pool_size=pool_size)

    session = sessionmaker(
        engine,
        expire_on_commit=False,
    )

    return session


def get_async_session(psql: psqlConfig, pool_size=20) -> sessionmaker:
    """Create async connection."""
    engine = create_async_engine(
        fmt_connection_url(psql, async_=True),
        echo=False,
        echo_pool=False,
        future=True,
        pool_size=pool_size,
    )

    async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    return async_session


# Dependency
def get_async_db(psql: psqlConfig) -> Callable:
    """Create async db."""

    async def make_db() -> AsyncGenerator:
        async_session = get_async_session(psql)

        async with async_session() as session:
            try:
                yield session
                await session.commit()
            except SQLAlchemyError as sql_ex:
                await session.rollback()
                raise sql_ex
            except HTTPException as http_ex:
                await session.rollback()
                raise http_ex
            finally:
                await session.close()

    return make_db


async def run_in_session(async_session, func, **kwargs):
    """Run in async session."""
    async with async_session() as session:
        return await func(session, **kwargs)


def inject_str_attributes(parentInstance, itemDict, attrModel, session):
    """Inject string attributes for a model instance.

    Example:
        book.genres = ['genre1', 'fiction', 'fantasy']
        book = inject_str_attributes(book, Genre)
        book.genres
        # [Genre(name=genre1), Genre(name=fiction), Genre(name=fantasy)]
    """
    # assert hasattr(parentInstance, attrName)
    assert set([c.name for c in attrModel.__table__.columns]) == set(
        ["id", "name"]
    ), "only works for attr models with columns `id` and `name`: str"
    attrModelName = attrModel.__table__.name
    attrName = attrModelName.lower() + "s"
    for attr in itemDict.get(attrName, []):
        attr = session.merge(attrModel(name=attr))
        if hasattr(parentInstance, attrName):
            setattr(parentInstance, attrName, [attr])
        else:
            getattr(parentInstance, attrName).append(attr)

    return parentInstance


async def aget_str_mappings(psql: psqlConfig, models=None) -> Dict[str, Any]:
    """Get string mappings asynchronously."""
    assert models is not None

    # not much faster than blocking version below
    async_sess = get_async_session(psql)
    async with async_sess() as session:
        queries = {
            model.__tablename__: session.execute(select(model)) for model in models
        }

        # str_mappings = {model.__tablename__: None for model in models}

        res = dict(zip(queries.keys(), await asyncio.gather(*queries.values())))
        str_mappings = {
            tablename: {g.name: g for g in res_.scalars()}
            for tablename, res_ in res.items()
        }

    return str_mappings


def get_str_mappings(session: Session, models=None) -> Dict[str, Any]:
    """Get string mappings."""
    assert models is not None
    assert isinstance(session, Session), f"{type(session)=} is not Session"

    str_mappings = {}
    for model in models:
        # print(f"{model=}")
        modelName = model.__tablename__
        str_mappings[modelName] = session.execute(select(model))
        str_mappings[modelName] = {g.name: g for g in str_mappings[modelName].scalars()}

    return str_mappings


def get_or_create(session: Session, model, item=None, filter_by=None, **kwargs):
    """Get item from postgres, if it exists by `filter_by` OR `kwargs`.

    if it doesn't exist, create and return it
    """
    if item is not None:
        kwargs = item.as_dict()

    filter_by = filter_by or kwargs
    instance = session.query(model).filter_by(**filter_by).first()

    if instance:
        return instance

    # else:
    logger.info(f"creating {model}")
    instance = model(**kwargs)
    session.add(instance)
    session.commit()

    return instance


def get_one_or_create(
    session, model, create_method="", create_method_kwargs=None, **kwargs
):
    """Safer version of get_or_create.

    Does not commit, user should do this after the transaction.
    From:
        https://stackoverflow.com/questions/2546207/does-sqlalchemy-have-an-equivalent-of-djangos-get-or-create
    """
    try:
        return session.query(model).filter_by(**kwargs).one(), False
    except NoResultFound:
        kwargs.update(create_method_kwargs or {})
        created = getattr(model, create_method, model)(**kwargs)
        try:
            session.add(created)
            session.flush()
            return created, True
        except IntegrityError:
            session.rollback()
            return session.query(model).filter_by(**kwargs).one(), False


async def aget_one_or_create(
    session, model, create_method="", create_method_kwargs=None, **kwargs
) -> Tuple[Any, bool]:
    """Async version of get_one_or_create.

    Does not commit, user should do this after the transaction.
    From:
        https://stackoverflow.com/questions/2546207/does-sqlalchemy-have-an-equivalent-of-djangos-get-or-create
    """
    try:
        q = select(model).filter_by(**kwargs)
        res = await session.execute(q)
        # logger.warning(f"{dir(res.scalars())=}")
        return res.scalars().first(), False
    except NoResultFound:
        kwargs.update(create_method_kwargs or {})
        created = getattr(model, create_method, model)(**kwargs)
        try:
            await session.add(created)
            await session.flush()
            return created, True
        except IntegrityError:
            await session.rollback()
            q = select(model).filter_by(**kwargs)
            res = await session.execute(q)
            return q.scalars().first(), False


async def aget(
    session: AsyncSession, model, filter_by: Optional[dict] = None
) -> Optional["model"]:
    """Get model instance asynchronously."""
    assert filter_by is not None

    query = select(model).filter_by(**filter_by)

    res = await session.execute(query)
    instance = res.scalars().first()

    return instance


async def aaget_or_create(session: AsyncSession, model, **kwargs):
    """Is similar to aget_or_create, but also returns bool flag if item was created."""
    query = select(model).filter_by(**kwargs)

    res = await session.execute(query)
    instance = res.scalars().first()

    did_create = False

    if not instance:
        instance = model(**kwargs)
        session.add(instance)
        await session.commit()

        did_create = True

    assert isinstance(instance, model), f"{type(instance)=}, should be {model}"

    return instance, did_create


async def aget_or_create(session: AsyncSession, model, **kwargs):
    """Create model instance asynchronously."""
    instance, _ = await aaget_or_create(session, model, **kwargs)

    return instance


def create_instance(model, item: Union[dict, Any]):
    """Create model instance."""
    # assert model is not None

    if isinstance(item, model):
        return item

    try:
        return model.__call__(**item)
    except Exception as e:
        logger.warning(f"cannot create '{model.__tablename__}'. {str(e)=} \n{item=}")
        raise


async def new_names(
    session: AsyncSession, model, items: Dict[Union[str, int], dict], nameAttr="name"
) -> Set[Union[str, int]]:
    """Return names that are missing for model."""
    selAttr = getattr(model, nameAttr)
    namesSet = set(items.keys())
    # nope, cannot be used for more than 32K arguments
    # existingNames = await session.execute(select(selAttr).filter(selAttr.in_(namesSet)))
    existingNames = await session.execute(select(selAttr))
    newNames = namesSet - set(existingNames.scalars())

    return newNames


async def add_many(
    session: AsyncSession,
    model,
    items: Dict[Union[str, int], dict],
    nameAttr,
) -> Dict[str, Any]:

    # ugly: asyncio cannot handle so many coroutines
    cors = (aget_one_or_create(session, model, **item) for item in items.values())

    res: List[Tuple[Any, bool]] = await asyncio.gather(*cors)
    instances = [i[0] for i in res]

    return {getattr(item, nameAttr): item for item in instances if item is not None}


# TODO: rewrite and simplify
async def create_many(
    session: AsyncSession,
    model,
    items: Dict[Union[str, int], dict],
    nameAttr="name",
    debug=False,
    many=True,
    nbulk=1,
    autobulk=False,
    returnExisting=False,
    mergeExisting=False,
    printCondition=None,
    tqdmFrom=5_000,
    commit=True,
) -> Dict[str, Any]:
    """Create many instances of a model.

    printCondition  print all items when this condition is met
    autobulk        add items in bulk to db, but determine automatically
                    how many bulk inserts to apply
    mergeExisting   safer than returnExisting, it fetches existing items singly, and
                    returns it together with newly created instances
    """
    assert isinstance(session, AsyncSession)
    assert isinstance(items, dict)
    # first check if names exist
    newNames = await new_names(session, model, items, nameAttr=nameAttr)
    # logger.info(f"{len(newNames)=:,}")
    disable = len(items) < tqdmFrom
    itemsDict = {
        name: create_instance(model, item)
        for name, item in tqdm(items.items(), disable=disable)
        if name in newNames
    }
    existingItems = {name: i for name, i in items.items() if name not in newNames}

    newItems = list(itemsDict.values())

    logger.info(f"{model.__tablename__}s to add: {len(newItems):,}")

    if printCondition is not None and printCondition(model, newItems):
        # fmt_items = ', '.join([i.title for i in items.values()])
        fmt_items = ", ".join([i.title for i in newItems])
        logger.info(fmt_items)

    if debug:
        logger.info(f"{newItems[:3]=}")

    if len(newItems) > 0:
        if not many:
            for item in items:
                if debug:
                    logger.info(f"{item=}")
                    if hasattr(item, "_as_big_dict"):
                        logger.info(f"big dict: \n")
                        pprint(getattr(item, "_as_big_dict")())
                    # logger.info(f"{item._as_big_dict()=}")
                try:
                    session.add(item)
                except Exception as e:
                    logger.warning(f"cannot add {item=}. {str(e)=}")
                    raise

                if commit:
                    await session.commit()

        else:
            # use bulks of at least 20K items, and max 20 bulks
            bulksize = 20_000
            maxbulk = 20
            if autobulk and len(newItems) > bulksize:
                nbulk = int(len(newItems) / bulksize)
                nbulk = min(nbulk, maxbulk)

            logger.debug(f"{nbulk=}")
            disable = len(newItems) < bulksize
            for chunk in tqdm(np.array_split(newItems, nbulk), disable=disable):

                session.add_all(list(chunk))
                # session.bulk_save_objects(newItems) # not available for async sessions
                if commit:
                    await session.commit()

    # return all existing items for items.keys() ids
    if returnExisting:
        attr = getattr(model, nameAttr)
        # logger.info(f"{attr=}")

        reqNames = list(items.keys())
        # cannot be used for more than 32K arguments
        query = select(model).where(attr.in_(reqNames))
        res = await session.execute(query)

        instances = res.scalars().fetchall()

        # return a dict
        itemsDict = {getattr(i, nameAttr): i for i in instances}

    elif mergeExisting:
        existingItemsDict = await add_many(
            session, model, existingItems, nameAttr=nameAttr
        )

        itemsDict = {**itemsDict, **existingItemsDict}

    return itemsDict


def did_change(item_existing, item_new, key: str) -> bool:
    """Check if `key` changed `item_new`."""
    # raise NotImplementedError

    if item_new is None:
        return False

    if (
        key in dir(item_existing)
        and key in dir(item_new)
        and getattr(item_new, key) != getattr(item_existing, key)
    ):
        return True

    return False


def did_increase(item_existing, item_new, key: str) -> bool:
    """Check if `key` increased in `item_new`."""
    if item_new is None:
        return False

    if (
        key in dir(item_existing)
        and key in dir(item_new)
        and (getattr(item_new, key) or 0) > (getattr(item_existing, key) or 0)
    ):
        return True

    return False


def did_increase_len(item_existing, item_new, key: str) -> bool:
    """Check if len(item_new.key) increased in `item_new`."""
    # no new item or a None attribute
    if item_existing is None:
        return False

    # key does not exist, always update the item
    # if getattr(item_existing, key) is None:
    #     return True

    if (
        key in dir(item_existing)
        and key in dir(item_new)
        and len(getattr(item_new, key) or []) > len(getattr(item_existing, key) or [])
    ):
        return True

    return False


def upsert_many(
    session,
    model,
    items: Dict[Union[str, int], dict],
    nameAttr="name",
    bulksize=20_000,
    updateCondition: Optional[Callable] = None,
    debug=False,
):
    """Upsert many instances of a model.

    Also increments instance.nupdate, if present

    Multiple options:
        - use stmt.on_conflit_do_update
        - OR fetch all objects first (if dataset not too large), and only update when a field is >= ,
            this means that you always update with newer data
    """
    # stmt = insert(model).values(user_email='a@b.com', data='inserted data')
    # stmt = stmt.on_conflict_do_update(
    #     index_elements=[my_table.c.user_email],
    #     index_where=my_table.c.user_email.like('%@gmail.com'),
    #     set_=dict(data=stmt.excluded.data)
    # )
    # conn.execute(stmt)

    list_items = list(items.items())
    nbulk = math.ceil(len(items) / bulksize)

    disable = len(items) < bulksize
    instances = []
    attr = getattr(model, nameAttr)
    for chunk in tqdm(np.array_split(list_items, nbulk), disable=disable):
        reqNames = [c[0] for c in chunk]
        query = select(model).where(attr.in_(reqNames))
        instances += session.execute(query).scalars().fetchall()

        # logger.info(f"{inspect_sqlalchemy(instances[0]).session=}")

    instances_dict = {getattr(i, nameAttr): i for i in instances}
    # calculate number of items to be updates
    new_keys = items.keys() - set(getattr(i, nameAttr) for i in instances)
    nnew = len(new_keys)

    existing_to_new: Dict[Union[str, int], Tuple[dict, dict]] = {
        k: (instances_dict.get(k, None), i) for k, i in items.items()
    }

    if updateCondition is not None:
        assert callable(updateCondition)
        assert (
            narg := len(inspect.signature(updateCondition).parameters)
        ) <= 3, f"{narg=} > 3, did you forget to pass a partial function?"
        toupdate = [
            item_new
            for k, (item_existing, item_new) in existing_to_new.items()
            if updateCondition(item_existing, item_new)
        ]
    else:
        toupdate = instances

    if debug:
        raise Exception("debugging")

    ntoupdate = len(toupdate)

    logger.info(f"{nnew=:,} {ntoupdate=:,}")

    toupdate_keys = set(getattr(i, nameAttr) for i in toupdate)
    disable = len(toupdate_keys) < 0.7 * bulksize
    nupdated = 0
    nfield_updated = 0
    query = select(model).where(attr.in_(toupdate_keys))
    # for item in tqdm(session.execute(query).scalars().fetchall(), disable=disable):
    for key in tqdm(toupdate_keys, disable=disable):
        existing, new = existing_to_new[key]
        existing.nupdate += 1
        new = existing_to_new[key][-1]
        nfield_updated += existing.update_from_json(new.as_dict())
        nupdated += 1

    # for k, (existing, new) in tqdm(existing_to_new.items(), disable=disable):
    #     if k not in toupdate_keys:
    #         continue
    #     # increase nupdate before updating
    #     if 'nupdate' in dir(new):
    #         existing.nupdate += 1

    #     # todoupdate items based on dict of new item
    #     # fetch items again?
    #     existing.update_from_json(new.as_dict())
    #     nupdated += 1

    # logger.info(f"{inspect_sqlalchemy(existing).session=}")

    if nupdated > 0:
        logger.info(f"updating {nupdated=:,} items ({nfield_updated=:,})")

    session.commit()

    # todo: reset nupdate for failed attemps to max 1
    #

    # add new items
    if nnew > 0:
        logger.info(f"adding {nnew:,} items")
        session.add_all([v for i, v in items.items() if i in new_keys])
        session.commit()

    return toupdate_keys, existing_to_new
