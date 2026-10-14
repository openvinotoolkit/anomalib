# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_serializer
from pydantic.json_schema import SkipJsonSchema


class BaseIDModel(ABC, BaseModel):
    """Base model with an id field."""

    id: UUID = Field(default_factory=uuid4)

    @field_serializer("id")
    def serialize_id(self, id: UUID, _info: Any) -> str:
        return str(id)


class BaseIDNameModel(BaseIDModel):
    """Base model with id and name fields."""

    name: str = "Default Name"


class NoRequiredIDs(BaseModel):
    """
    Mixin class for create models (POST request schemas) that makes id and project_id optional.

    This class is used as a base class for create models (e.g., WebcamSourceConfigCreate,
    FolderSinkConfigCreate) to override required id and project_id fields from base classes.
    Via Method Resolution Order (MRO), this class's fields take precedence, making these
    fields optional and excluded from the JSON schema.

    The id field is auto-generated if not provided, and project_id is typically injected
    from the URL path parameter in the endpoint handler.
    """

    project_id: SkipJsonSchema[UUID] = Field(exclude=True, default=UUID("00000000-0000-0000-0000-000000000000"))
    id: SkipJsonSchema[UUID] = Field(exclude=True, default_factory=uuid4)


class Pagination(BaseModel):
    """Pagination model."""

    offset: int  # index of the first item returned (0-based)
    limit: int  # number of items requested per page
    count: int  # number of items actually returned (can be less than the limit if at the end)
    total: int  # total number of items available
