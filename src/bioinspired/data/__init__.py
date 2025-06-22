"""Data module for bioinspired package.
This module contains the database management and data handling
for the bioinspired package. It includes functions to initialize the database,
create tables, and manage data storage.
It is designed to work with PostgreSQL and uses SQLAlchemy for ORM.
"""

from .init_db import init_database as init_db

__all__ = ["init_db"]
