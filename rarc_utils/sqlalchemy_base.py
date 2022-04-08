""" slqalchemy.py
    
    utility functions that are used a lot when working with SQLAlchemy

    Like: creating async or blocking sessions, creating all models, ...
"""

from typing import Dict, Any, Union, Callable, AsyncGenerator # List, 
from abc import ABCMeta, abstractmethod # , ABC
import logging
import asyncio
from pprint import pprint

from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine
# from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.future import select

from .misc import AttrDict

logger = logging.getLogger(__name__)

# Base = declarative_base()


class UtilityBase(object):
    """ adds helper methods to SQLAlchemy `Base` class """

    def as_dict(self) -> Dict[str, Any]:
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}  

    def as_big_dict(self) -> Dict[str, Any]:
        return {c: getattr(self, c) for c in dir(self) if not c.startswith(('_','__','registry'))}
 
class AbstractBase(metaclass=ABCMeta): # ABC
   
    @abstractmethod
    def __repr__(self):
        """ force every child to have a __repr__ method 
            todo: still don't know how to inherit from Base, UtilityBase and this AbstractBase, it throws error:
                TypeError: metaclass conflict: the metaclass of a derived class must be a (non-strict) subclass of the metaclasses of all its bases
        """
        pass

async def async_main(psql, base, force=False, dropFirst=False) -> None:
    engine = create_async_engine(
        f"postgresql+asyncpg://{psql.user}:{psql.passwd}@{psql.host}/{psql.db}",
        echo=True,
    )

    """ TO-DO: make sure to check for a backup file first, as it deletes all psql data """

    if not force:
        if input("are you sure you want to recreate all models? Use alembic for migrations, and run this function only for a total model reset. Pres 'y' to confirm: ") != 'y':
            print("leaving")
            return 

    if dropFirst:
        async with engine.begin() as conn:
            await conn.run_sync(base.metadata.drop_all) # does not work with association tables.. --> use DROP DATABASE "enabler"; CREATE DATABASE "enabler"; for now

    async with engine.begin() as conn:
        await conn.run_sync(base.metadata.create_all)

def get_session(psql: AttrDict) -> sessionmaker:
    """ create normal (bocking) connection """

    engine = create_engine(
        f"postgresql://{psql.user}:{psql.passwd}@{psql.host}/{psql.db}",
        echo=False,  # shows double log output
        echo_pool=False,  # read https://docs.sqlalchemy.org/en/14/core/engines.html#configuring-logging
        future=True
        )

    session = sessionmaker(
        engine, 
        expire_on_commit=False,
    )

    return session

def get_async_session(psql: AttrDict) -> sessionmaker:
    engine = create_async_engine(
        f"postgresql+asyncpg://{psql.user}:{psql.passwd}@{psql.host}/{psql.db}",
        echo=False,  # shows double log output
        echo_pool=False,  # read https://docs.sqlalchemy.org/en/14/core/engines.html#configuring-logging
        future=True
        )

    async_session = sessionmaker(
        engine, 
        expire_on_commit=False, class_=AsyncSession
    )

    return async_session

# Dependency
def get_async_db(psql: AttrDict) -> Callable:

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

    return make_db()

async def aget_str_mappings(psql: AttrDict, models=None) -> Dict[str, Any]:

    assert models is not None

    # not much faster than blocking version below
    async_sess = get_async_session(psql)
    async with async_sess() as session:
        queries = {model.__tablename__: session.execute(select(model)) for model in models}

        str_mappings = {model.__tablename__: None for model in models}

        res = dict(zip(queries.keys(), await asyncio.gather(*queries.values())))
        str_mappings = {tablename: {g.name: g for g in res_.scalars()} for tablename, res_ in res.items()}

    return str_mappings

def get_str_mappings(session: Session, models=None) -> Dict[str, Any]:

    assert models is not None
    assert isinstance(session, Session), f"{type(session)=} is not Session"

    str_mappings = dict()
    for model in models:
        # print(f"{model=}")
        modelName = model.__tablename__
        str_mappings[modelName] = session.execute(select(model))
        str_mappings[modelName] = {g.name: g for g in str_mappings[modelName].scalars()}

    return str_mappings

def get_or_create(session: Session, model, item=None, **kwargs):

    if item is not None:
        kwargs = item.as_dict()
    
    instance = session.query(model).filter_by(**kwargs).first()

    if instance:
        return instance

    # else:
    logger.info(f"creating {model}")
    instance = model(**kwargs)
    session.add(instance)
    session.commit()

    return instance

async def aget_or_create(session, model, **kwargs):

    # instance = session.query(model).filter_by(**kwargs).first()
    query = select(model).filter_by(**kwargs)
    # print(f"{type(query)=} {dir(query)=}")
    # .first()

    res = await session.execute(query)
    instance = res.scalars().first()
    if instance:
        # print(f"{instance=} {dir(instance)=}")
        # return instance.scalars()
        pass

    else:
        instance = model(**kwargs)
        session.add(instance)
        await session.commit()

    assert isinstance(instance, model), f"{type(instance)=}, should be {model}"

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

async def create_many(session, model, items: Dict[str, dict], nameAttr='name', debug=False, many=True, printCondition=None) -> Dict[str, Any]:
    """ create many of any model type: Character, Genre, Places, Author, ... 
       
        printCondition  print all items when this condition is met

        Todo:   filtering out existing names only works for Category models, author is unique by name + birth_date,
                so it needs a different implementation
    """

    assert isinstance(items, dict)
    # async with async_session() as session:
    # first check if character names exist
    existingNames = await session.execute(select(getattr(model, nameAttr)))
    names = set(items.keys())
    names = list(names - set(existingNames.scalars()))
    itemsDict = {name: create_instance(model, item) for name, item in items.items() if name in names}
    items = list(itemsDict.values())

    logger.info(f"{model.__tablename__}s to add: {len(items)}")

    if printCondition is not None and printCondition(model, items):
        # fmt_items = ', '.join([i.title for i in items.values()])
        fmt_items = ', '.join([i.title for i in items])
        logger.info(fmt_items)

    if debug:
        logger.info(f"{items[:3]=}")

    # print(f"{dir(session)=}")
    # print(f"{help(session.add_all)=}")

    # session.add(c)
    if len(items) > 0:
        if not many:
            for item in items:
                if debug:
                    logger.info(f"{item=}")
                    if hasattr(item, '_as_big_dict'):
                        logger.info(f"big dict: \n")
                        pprint(item._as_big_dict())
                    # logger.info(f"{item._as_big_dict()=}")
                try:
                    session.add(item)
                except Exception as e:
                    logger.warning(f"cannot add {item=}. {str(e)=}")
                    raise

                await session.commit()
        else:
            session.add_all(items)
            await session.commit()

    return itemsDict
