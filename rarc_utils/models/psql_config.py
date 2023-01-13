"""psql_config.py.

postgresql configuration class to enforce strong typing on configurations
passed from .db.{test,prod}.env files
"""

from attrs import field, fields_dict, frozen, validators


@frozen
class psqlConfig:
    """Psql configuration."""
    PG_HOST: str = field(validator=[validators.instance_of(str)])
    PG_PORT: int = field(converter=int)
    PG_USER: str = field(validator=[validators.instance_of(str)])
    PG_PASSWD: str = field(validator=[validators.instance_of(str)])
    PG_DB: str = field(validator=[validators.instance_of(str)])


psqlKeys = fields_dict(psqlConfig)
