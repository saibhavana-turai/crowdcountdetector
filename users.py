from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String

# Base class for our database models.
# This is the standard way to define models in SQLAlchemy without a framework.
Base = declarative_base()

class User(Base):
    """
    Defines the User model for the database.
    This version is framework-agnostic and works directly with SQLAlchemy.
    """
    __tablename__ = 'user'  # The name of the database table

    id = Column(Integer, primary_key=True)
    email = Column(String(100), unique=True, nullable=False)
    password = Column(String(200), nullable=False)