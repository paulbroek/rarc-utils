"""sqlqalchemy_base.py, utility functions that are frequently used when working with SQLAlchemy.

e.g.: creating async or blocking sessions, creating all models, getting all str models, get_or_create methods, ...
"""

import asyncio
import logging
import os
from pprint import pprint
from typing import Any, AsyncGenerator, Callable, Dict, Optional, Union

import numpy as np
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession  # type: ignore[import]
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.future import select  # type: ignore[import]
from sqlalchemy.orm import Session, sessionmaker
from tqdm import tqdm  # type: ignore[import]

from .misc import AttrDict

logger = logging.getLogger(__name__)


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


async def async_main(psql, base, force=False, dropFirst=False) -> None:
    """Create async engine for sqlalchemy."""
    engine = create_async_engine(
        f"postgresql+asyncpg://{psql.user}:{psql.passwd}@{psql.host}/{psql.db}",
        echo=True,
    )

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


def get_session(psql: AttrDict) -> sessionmaker:
    """Create normal (bocking) connection."""
    # default config can be overriden by passing pg host env var
    pghost = os.environ.get("POSTGRES_HOST", psql.host)
    # read https://docs.sqlalchemy.org/en/14/core/engines.html#configuring-logging
    # `echo=True` shows duplicate log output
    engine = create_engine(
        f"postgresql://{psql.user}:{psql.passwd}@{pghost}/{psql.db}",
        echo=False,
        echo_pool=False,
        future=True,
    )

    session = sessionmaker(
        engine,
        expire_on_commit=False,
    )

    return session


def get_async_session(psql: AttrDict) -> sessionmaker:
    """Create async connection."""
    engine = create_async_engine(
        f"postgresql+asyncpg://{psql.user}:{psql.passwd}@{psql.host}/{psql.db}",
        echo=False,
        echo_pool=False,
        future=True,
    )

    async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    return async_session


# Dependency
def get_async_db(psql: AttrDict) -> Callable:
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


async def aget_str_mappings(psql: AttrDict, models=None) -> Dict[str, Any]:
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
    printCondition=None,
    pushOneByOne=True,
    tqdmFrom=1_000,
) -> Dict[str, Any]:
    """Create many instances of a model.

    printCondition  print all items when this condition is met
    autobulk        add items in bulk to db, but determine automatically
                    how many bulk inserts to apply
    """
    assert isinstance(items, dict)
    # first check if names exist
    selAttr = getattr(model, nameAttr)
    namesSet = set(items.keys())
    # nope, cannot be used for more than 32K arguments
    # existingNames = await session.execute(select(selAttr).filter(selAttr.in_(namesSet)))
    existingNames = await session.execute(select(selAttr))
    newNames = namesSet - set(existingNames.scalars())
    # logger.info(f"{len(newNames)=:,}")
    # todo: slow for large lists?
    disable = len(items) < tqdmFrom
    itemsDict = {
        name: create_instance(model, item)
        for name, item in tqdm(items.items(), disable=disable)
        if name in newNames
    }

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

    return itemsDict
