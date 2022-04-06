# import versioneer # https://github.com/python-versioneer/python-versioneer/blob/master/INSTALL.md
from setuptools import setup, find_packages

requires = [
    "yapic.json>=1.7.0",
    "websockets>=9.1",
    "pandas>=1.0.3",
    "aioredis>=2.0.0",
    "redis>=3.5.3",
    "pyzmq>=22.2.1",
    "timeago>=1.0.15",
    "tqdm>=4.62.1",
    "psutil>=5.8.0",
    "sqlalchemy>=1.4.23",
    "PyYAML>=5.4.1",
    "python-json-logger>=2.0.2",
]

setup(
    name='rarc_utils',
    version='0.0.8',
    description='Rarc utility functions',
    url='git@github.com:paulbroek/rarc-utils.git',
    author='Paul Broek',
    author_email='pcbroek@paulbroek.nl',
    license='unlicense',
    install_requires=requires,
    #packages=['rarc'],
    packages=find_packages(exclude=['tests','pymt5adapter','logs']),
    python_requires='>=3.7', # 3.8 # remove 3.8 requirement so it can be installed in google colab
    zip_safe=False
)
