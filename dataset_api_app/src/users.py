from __future__ import annotations

# Third-party imports
import sys
import pathlib
from datetime import datetime, timezone
from sqlalchemy import create_engine, select
from sqlalchemy.orm import (
    DeclarativeBase,
    MappedAsDataclass,
    Mapped,
    mapped_column,
    sessionmaker,
)

# First-party imports
file = pathlib.Path(__file__).resolve()
sys.path.append(str(file.parents[0]))
from paths import USERS_DB_PATH
from github import request_github_user


# Create SQLAlchemy engine and session objects
engine = create_engine(f"sqlite:///{USERS_DB_PATH}")
Session = sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase, MappedAsDataclass):
    pass


class User(Base):
    """Users who are allowed to access data via the API."""

    __tablename__ = "user"

    id: Mapped[int] = mapped_column(init=False, primary_key=True)
    github_id: Mapped[int] = mapped_column(index=True, unique=True)
    github_username: Mapped[str]
    email: Mapped[str | None]
    name: Mapped[str | None]
    affiliation: Mapped[str | None]
    added_at: Mapped[str] = mapped_column(
        default_factory=lambda: datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )

    @staticmethod
    def add(
        github_username: str,
        email: str | None = None,
        name: str | None = None,
        affiliation: str | None = None,
    ) -> User:
        """
        Add a user to the database, allowing them to access data via the API.

        :param github_username: Used to look up the user's GitHub ID, which will be used
            to identify them for authentication purposes. We use their GitHub ID number
            since usernames can be changed.
        :param email: Can be added for information purposes. If not provided, this will
            be pulled from the user's GitHub profile.
        :param name: Can be added for information purposes. If not provided, this will
            be pulled from the user's GitHub profile.
        :param affiliation: Can be added for information purposes. If not provided, this will
            be pulled from the user's GitHub profile.
        """
        response = request_github_user(github_username)
        if response.ok:
            user_info = response.json()
            user = User(
                github_id=user_info["id"],
                github_username=user_info["login"],
                email=email or user_info["email"],
                name=name or user_info["name"],
                affiliation=affiliation or user_info["company"],
            )
            with Session.begin() as session:
                session.add(user)
            return user
        raise RuntimeError(f"GitHub user {github_username} not found")

    @staticmethod
    def get(github_id: int) -> User | None:
        """
        Get a user in the database.

        :param github_id: The user's GitHub ID.
        :returns: The ``User`` with ``github_id``, or ``None`` if there is no such
            ``User``.
        """
        stmt = select(User).where(User.github_id == github_id)
        with Session.begin() as session:
            user = session.scalars(stmt).one_or_none()
        return user

    @staticmethod
    def get_by_github_username(github_username: str) -> list[User]:
        """
        Get the user(s) with the given ``github_username``.

        :param github_username: The GitHub username to search for.
        :returns: A list of ``User`` objects, which may be empty if there are no such
            users. (The database does not enforce uniqueness among GitHub usernames
            because they can in theory be changed on the GitHub side, so a list is
            returned.)
        """
        stmt = select(User).where(User.github_username == github_username)
        with Session.begin() as session:
            users = session.scalars(stmt).all()
        return list(users)

    @staticmethod
    def get_all() -> list[User]:
        """Get all users."""
        stmt = select(User)
        with Session.begin() as session:
            users = session.scalars(stmt).all()
        return list(users)

    @staticmethod
    def print_all() -> list[User]:
        """Print the names and GitHub user link of all users."""
        all_users = User.get_all()
        for user in all_users:
            print(f"{user.name}: https://api.github.com/user/{user.github_id}")


# Create all tables if they do not already exist
Base.metadata.create_all(engine)
