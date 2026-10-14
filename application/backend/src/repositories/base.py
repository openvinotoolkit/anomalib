# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import abc
from collections.abc import AsyncGenerator, Callable, Sequence
from typing import Any, TypeVar
from uuid import UUID

from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.sql import expression
from sqlalchemy.sql.selectable import Select, and_

from db.schema import Base
from pydantic_models.base import BaseIDModel

ModelType = TypeVar("ModelType", bound=BaseIDModel)
SchemaType = TypeVar("SchemaType", bound=Base)


class BaseRepository[ModelType, SchemaType](metaclass=abc.ABCMeta):
    """Base repository class for database operations."""

    def __init__(self, db: AsyncSession, schema: type[SchemaType]):
        self.db = db
        self.schema = schema

    @property
    @abc.abstractmethod
    def to_schema(self) -> Callable[[ModelType], SchemaType]:
        """to_schema mapper callable"""

    @property
    @abc.abstractmethod
    def from_schema(self) -> Callable[[SchemaType], ModelType]:
        """from_schema mapper callable"""

    @property
    def base_filters(self) -> dict:
        """Base filter expression for the repository"""
        return {}

    def _get_filter_query(self, extra_filters: dict | None = None, expressions: list[Any] | None = None) -> Select:
        """Build query with filters and expressions combined with AND."""
        query = expression.select(self.schema)

        # Apply keyword filters (column=value)
        if extra_filters is None:
            extra_filters = {}
        combined_filters = extra_filters | self.base_filters
        if combined_filters:
            query = query.filter_by(**combined_filters)

        # Apply additional expressions with AND
        if expressions:
            query = query.where(and_(*expressions))

        return query

    async def get_by_id(self, obj_id: str | UUID) -> ModelType | None:
        return await self.get_one(extra_filters={"id": self._id_to_str(obj_id)})

    async def get_one(
        self,
        extra_filters: dict | None = None,
        expressions: list[Any] | None = None,
        order_by: Any | None = None,
        ascending: bool = False,
    ) -> ModelType | None:
        query = self._get_filter_query(extra_filters=extra_filters, expressions=expressions)
        if order_by is not None:
            query = query.order_by(order_by.asc() if ascending else order_by.desc())
        result = await self.db.execute(query)
        first_result = result.scalars().first()
        if first_result:
            return self.from_schema(first_result)
        return None

    async def get_all_count(self, extra_filters: dict | None = None, expressions: list[Any] | None = None) -> int:
        query = self._get_filter_query(extra_filters=extra_filters, expressions=expressions)
        count_query = expression.select(expression.func.count()).select_from(query.subquery())
        result = await self.db.execute(count_query)
        return result.scalar_one()

    async def get_all_pagination(
        self,
        limit: int,
        offset: int,
        extra_filters: dict | None = None,
        expressions: list[Any] | None = None,
    ) -> list[ModelType]:
        """
        Get a paginated list of records.

        Args:
            limit: Maximum number of records to return.
            offset: Number of records to skip.
            extra_filters: Dictionary of equality filters (column=value).
            expressions: List of SQLAlchemy expressions to filter by.

        Returns:
            List of ModelType instances.
        """
        query = self._get_filter_query(extra_filters=extra_filters, expressions=expressions)
        query = query.limit(limit).offset(offset)
        results = await self.db.execute(query)
        scalars: Sequence[SchemaType] = results.scalars().all()
        return [self.from_schema(result) for result in scalars]

    async def get_all_streaming(
        self,
        extra_filters: dict | None = None,
        expressions: list[Any] | None = None,
        batch_size: int = 100,
    ) -> AsyncGenerator[ModelType]:
        """
        Stream all records via a generator.

        Args:
            extra_filters: Dictionary of equality filters (column=value).
            expressions: List of SQLAlchemy expressions to filter by.
            batch_size: Number of records to fetch from the DB at a time.

        Yields:
            ModelType instances one by one.
        """
        query = self._get_filter_query(extra_filters=extra_filters, expressions=expressions)
        result = await self.db.stream(query.execution_options(yield_per=batch_size))
        async for row in result.scalars():  # type: ignore[var-annotated]
            yield self.from_schema(row)

    async def save(self, item: ModelType) -> ModelType:
        schema_item: SchemaType = self.to_schema(item)
        self.db.add(schema_item)
        await self.db.commit()
        return item

    async def update(self, item: ModelType, partial_update: dict) -> ModelType:
        # note: model_copy does not validate the model, so we need to validate explicitly
        to_update = item.model_copy(update=partial_update, deep=True)  # type: ignore[attr-defined]
        item.__class__.model_validate(to_update.model_dump())  # type: ignore[attr-defined]
        schema_item: SchemaType = self.to_schema(to_update)
        await self.db.merge(schema_item)
        await self.db.commit()
        updated = await self.get_by_id(item.id)  # type: ignore[attr-defined]
        if updated is None:
            raise ValueError(f"{item.__class__} with ID `{item.id}` doesn't exist")  # type: ignore[attr-defined]
        return updated

    async def delete_by_id(self, obj_id: str | UUID) -> None:
        if not hasattr(self.schema, "id"):
            raise AttributeError(f"Delete by ID is not supported by schema: `{self.schema}`")

        obj_id = self._id_to_str(obj_id)
        where_expression = [
            self.schema.id == obj_id,  # type: ignore[attr-defined]
            *[self.schema.__table__.c[k] == v for k, v in self.base_filters.items()],  # type: ignore[attr-defined]
        ]
        query = expression.delete(self.schema).where(*where_expression)
        await self.db.execute(query)
        await self.db.commit()

    async def delete_all(self, commit: bool, extra_filters: dict | None = None) -> None:
        """Delete all records matching the filters.

        Does NOT commit the transaction.
        """
        if extra_filters is None:
            extra_filters = {}

        combined_filters = extra_filters | self.base_filters

        query = expression.delete(self.schema)
        if combined_filters:
            query = query.filter_by(**combined_filters)

        await self.db.execute(query)
        if commit:
            await self.db.commit()

    @staticmethod
    def _id_to_str(obj_id: str | UUID) -> str:
        return str(obj_id)


class ProjectBaseRepository(BaseRepository[ModelType, SchemaType], metaclass=abc.ABCMeta):
    def __init__(self, db: AsyncSession, project_id: str | UUID, schema: type[SchemaType]):
        super().__init__(db, schema)
        self.project_id = self._id_to_str(project_id)

    @property
    def base_filters(self) -> dict:
        return {"project_id": self.project_id}
