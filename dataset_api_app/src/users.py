from __future__ import annotations

# Third-party imports
import sys
import pathlib
from datetime import datetime, timedelta, timezone
from sqlalchemy import create_engine, select, update, delete, func
from sqlalchemy.dialects.sqlite import insert
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

SQLITE_UNIXEPOCH = func.unixepoch("now", "subsec")
"""SQLite ``unixepoch('now', 'subsec')`` function, for use in rate limiting."""


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
    def remove(github_id_or_username: int | str) -> None:
        """
        Remove a user from the database.

        :param github_id_or_username: The user's GitHub ID or username.
        """
        user_property = (
            User.github_id
            if isinstance(github_id_or_username, int)
            else User.github_username
        )
        stmt = delete(User).where(user_property == github_id_or_username)
        with Session.begin() as session:
            session.execute(stmt)

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
            print(
                f"{user.name} ({user.github_username}): https://api.github.com/user/{user.github_id}"
            )


class RateLimit(Base):
    __tablename__ = "rate_limit"

    key: Mapped[str] = mapped_column(primary_key=True)
    github_id: Mapped[int] = mapped_column(primary_key=True)
    interval_start: Mapped[float] = mapped_column(server_default=SQLITE_UNIXEPOCH)
    interval_requests: Mapped[int] = mapped_column(default=0)

    @staticmethod
    def update_and_check(
        key: str, github_id: int, interval_length: timedelta, requests_per_interval: int
    ) -> bool:
        """
        Update the rate limit corresponding to the given key and user, and check if the
        limit has been reached.

        :param key: Key corresponding to a particular rate limit.
        :param github_id: GitHub ID of the user making the request.
        :param interval_length: Length of the interval for the rate limit in question.
        :param requests_per_interval: Number of requests per interval allowed for the
            rate limit in question.
        """
        with Session.begin() as session:
            # Create a new `RateLimit` entry if one does not already exist for the
            # current key and GitHub ID.
            session.execute(
                insert(RateLimit)
                .values(key=key, github_id=github_id)
                .on_conflict_do_nothing()
            )

            # Reset the `RateLimit` entry if the `interval_length` has passed. We do
            # this in a single UPDATE command to avoid race conditions.
            session.execute(
                update(RateLimit)
                .where(
                    RateLimit.key == key,
                    RateLimit.github_id == github_id,
                    SQLITE_UNIXEPOCH
                    > RateLimit.interval_start + interval_length.total_seconds(),
                )
                .values(interval_start=SQLITE_UNIXEPOCH, interval_requests=0)
            )

            # Increment and return the number of requests. We do this in a single UPDATE
            # command to avoid race conditions.
            interval_requests = session.scalar(
                update(RateLimit)
                .where(RateLimit.key == key, RateLimit.github_id == github_id)
                .values(interval_requests=RateLimit.interval_requests + 1)
                .returning(RateLimit.interval_requests)
            )

            return interval_requests > requests_per_interval


# Create all tables if they do not already exist
Base.metadata.create_all(engine)
