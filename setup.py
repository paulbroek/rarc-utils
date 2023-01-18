# import versioneer # https://github.com/python-versioneer/python-versioneer/blob/master/INSTALL.md
from setuptools import find_packages, setup

requires = [
    "yapic.json",
    "pandas",
    "aioredis",
    "redis",
    "pyzmq",
    "timeago",
    "tqdm",
    "psutil",
    "sqlalchemy",
    "PyYAML",
    "python-json-logger",
    "lz4",
    "coloredlogs",
    "fastapi",
    "asyncpg",
    "psycopg2",
    "requests",
]

setup(
    name="rarc_utils",
    version="0.2.1",
    description="Rarc utility functions",
    url="git@github.com:paulbroek/rarc-utils.git",
    author="Paul Broek",
    author_email="pcbroek@paulbroek.nl",
    license="unlicense",
    install_requires=requires,
    package_data={"rarc_utils": ["py.typed"]},
    packages=find_packages(exclude=["tests", "logs"]),
    python_requires=">=3.9",
    zip_safe=False,
)
