"""psql_config.py.

postgresql configuration class to enforce strong typing on configurations 
passed from .db.{test,prod}.env files
"""

# from attrs import asdict, field, fields_dict, frozen, validators
from pydantic import BaseSettings

# @frozen
# class psqlConfig:
#     """Psql configuration."""

#     HIDDEN_FIELD = "PG_PASSWD"
#     PG_HOST: str = field(validator=[validators.instance_of(str)])
#     PG_PORT: int = field(converter=int)
#     PG_USER: str = field(validator=[validators.instance_of(str)])
#     PG_PASSWD: str = field(validator=[validators.instance_of(str)])
#     PG_DB: str = field(validator=[validators.instance_of(str)])

#     def __repr__(self):
#         """Do not display `HIDDEN_FIELD` class attribute."""
#         fields = asdict(self)
#         fields = {
#             k: "..." if k == psqlConfig.HIDDEN_FIELD else v for k, v in fields.items()
#         }
#         return str(fields)


# psqlKeys = fields_dict(psqlConfig)


# turn attrs model into pydantic baseSettings
class psqlConfig(BaseSettings):
    pg_host: str
    pg_port: int
    pg_user: str
    pg_passwd: str
    pg_db: str
