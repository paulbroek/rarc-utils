"""sqlqalchemy_base.py, utility functions that are frequently used when working with SQLAlchemy.

e.g.: creating async or blocking sessions, creating all models, getting all str models, get_or_create methods, ...
"""

import asyncio
import logging
from abc import ABCMeta, abstractmethod
from pprint import pprint
from typing import Any, AsyncGenerator, Callable, Dict, Optional, Union

from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.future import select
from sqlalchemy.orm import Session, sessionmaker

from .misc import AttrDict

logger = logging.getLogger(__name__)


class UtilityBase:
    """Adds helper methods to SQLAlchemy `Base` class."""

    def as_dict(self) -> Dict[str, Any]:
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

    def as_big_dict(self) -> Dict[str, Any]:
        return {
            c: getattr(self, c)
            for c in dir(self)
            if not c.startswith(("_", "__", "registry"))
        }


class AbstractBase(metaclass=ABCMeta):  # ABC
    @abstractmethod
    def __repr__(self):
        """force every child to have a __repr__ method
        todo:
            still don't know how to inherit from Base, UtilityBase and this AbstractBase, it throws error:
            TypeError: metaclass conflict: the metaclass of a derived class must be a (non-strict) subclass of the metaclasses of all its bases
        """
        pass


async def async_main(psql, base, force=False, dropFirst=False) -> None:
    engine = create_async_engine(
        f"postgresql+asyncpg://{psql.user}:{psql.passwd}@{psql.host}/{psql.db}",
        echo=True,
    )

    """ TO-DO: make sure to check for a backup file first, as it deletes all psql data """

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
            await conn.run_sync(
                base.metadata.drop_all
            )  # does not work with association tables.. --> use DROP DATABASE "enabler"; CREATE DATABASE "enabler"; for now

    async with engine.begin() as conn:
        await conn.run_sync(base.metadata.create_all)


def get_session(psql: AttrDict) -> sessionmaker:
    """Create normal (bocking) connection."""
    engine = create_engine(
        f"postgresql://{psql.user}:{psql.passwd}@{psql.host}/{psql.db}",
        echo=False,  # shows double log output
        echo_pool=False,  # read https://docs.sqlalchemy.org/en/14/core/engines.html#configuring-logging
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
        echo=False,  # shows double log output
        echo_pool=False,  # read https://docs.sqlalchemy.org/en/14/core/engines.html#configuring-logging
        future=True,
    )

    async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    return async_session


# Dependency
def get_async_db(psql: AttrDict) -> Callable:
    """Create async db."""
    # todo: add dep to repos like scrape_goodreads
    from fastapi import HTTPException

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
    async with async_session() as session:
        return await func(session, **kwargs)


async def aget_str_mappings(psql: AttrDict, models=None) -> Dict[str, Any]:

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

    instance, _ = await aaget_or_create(session, model, **kwargs)

    return instance


def create_instance(model, item: Union[dict, Any]):
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
    items: Dict[str, dict],
    nameAttr="name",
    debug=False,
    many=True,
    returnExisting=False,
    printCondition=None,
) -> Dict[str, Any]:
    """Create many instances of a model.

    printCondition  print all items when this condition is met
    """
    assert isinstance(items, dict)
    # async with async_session() as session:
    # first check if character names exist
    # todo: do not query for all names!
    existingNames = await session.execute(select(getattr(model, nameAttr)))
    namesSet = set(items.keys())
    names = list(namesSet - set(existingNames.scalars()))
    itemsDict = {
        name: create_instance(model, item)
        for name, item in items.items()
        if name in names
    }
    newItems = list(itemsDict.values())

    logger.info(f"{model.__tablename__}s to add: {len(newItems)}")

    if printCondition is not None and printCondition(model, newItems):
        # fmt_items = ', '.join([i.title for i in items.values()])
        fmt_items = ", ".join([i.title for i in newItems])
        logger.info(fmt_items)

    if debug:
        logger.info(f"{newItems[:3]=}")

    # print(f"{dir(session)=}")
    # print(f"{help(session.add_all)=}")

    # session.add(c)
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
            session.add_all(newItems)
            await session.commit()

    # return all existing items for items.keys() ids
    if returnExisting:
        attr = getattr(model, nameAttr)
        # logger.info(f"{attr=}")

        reqNames = list(items.keys())
        query = select(model).where(attr.in_(reqNames))
        res = await session.execute(query)

        instances = res.scalars().fetchall()

        # return a dict
        itemsDict = {getattr(i, nameAttr): i for i in instances}

    return itemsDict
