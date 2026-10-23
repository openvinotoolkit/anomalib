# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Sequence

from loguru import logger

from pydantic_models import Sink, SinkType
from services.dispatchers import Dispatcher, FolderDispatcher, MqttDispatcher, WebhookDispatcher


class DispatchService:
    _dispatcher_registry: dict[SinkType, Callable[[Sink], Dispatcher | None]] = {
        SinkType.DISCONNECTED: lambda _: None,
        SinkType.FOLDER: lambda config: FolderDispatcher(output_config=config),  # type: ignore[union-attr, arg-type]
        SinkType.MQTT: lambda config: MqttDispatcher(output_config=config),  # type: ignore[union-attr, arg-type]
        SinkType.ROS: lambda _: _raise_not_implemented("ROS output is not implemented yet"),
        SinkType.WEBHOOK: lambda config: WebhookDispatcher(output_config=config),  # type: ignore[union-attr, arg-type]
    }

    @classmethod
    def _get_destination(cls, output_config: Sink) -> Dispatcher | None:
        factory = cls._dispatcher_registry.get(output_config.sink_type)
        if factory is None:
            logger.warning(f"Unrecognized sink type: {output_config.sink_type}")
            return None

        try:
            return factory(output_config)
        except Exception:
            logger.opt(exception=True).warning(
                f"Failed to initialize dispatcher for sink type: {output_config.sink_type}"
            )
            return None

    @classmethod
    def get_destinations(cls, output_configs: Sequence[Sink]) -> list[Dispatcher]:
        """
        Get a list of dispatchers based on the provided output configurations.

        Args:
            output_configs (Sequence[OutputConfig]): A sequence of output configurations.
        """
        return [dispatcher for config in output_configs if (dispatcher := cls._get_destination(config)) is not None]


def _raise_not_implemented(message: str) -> None:
    raise NotImplementedError(message)
