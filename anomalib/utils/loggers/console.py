"""Console Logger."""

# Original Copyright (c) OpenMMLab.
# Modified Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import logging
from logging import Handler, Logger, StreamHandler
from typing import Dict, List, Optional

import torch.distributed as dist

INITIALIZED_LOGGERS: Dict[str, bool] = {}


def get_console_logger(
    name: str, log_level: int = logging.INFO, filename: Optional[str] = None, mode: str = "w"
) -> Logger:
    """Initialize and get a logger by name.

    Args:
        name (str): Logger name.
        log_level (int, optional): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.. Defaults to logging.INFO.
        filename (Optional[str]): The log filename. If specified, a FileHandler
            will be added to the logger. Defaults to None.
        mode (str, optional): The file mode used in opening log file.
            Defaults to 'w'.

    Returns:
        Logger: The expected logger.
    """

    logger = logging.getLogger(name)
    if name in INITIALIZED_LOGGERS:
        return logger

    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    for logger_name in INITIALIZED_LOGGERS:
        if name.startswith(logger_name):
            return logger

    # handle duplicate logs to the console
    # Starting in 1.8.0, PyTorch DDP attaches a StreamHandler <stderr> (NOTSET)
    # to the root logger. As logger.propagate is True by default, this root
    # level handler causes logging messages from rank>0 processes to
    # unexpectedly show up on the console, creating much unwanted clutter.
    # To fix this issue, we set the root logger's StreamHandler, if any, to log
    # at the ERROR level.
    for handler in logger.root.handlers:
        if isinstance(handler, StreamHandler):
            handler.setLevel(logging.ERROR)

    stream_handler = logging.StreamHandler()
    handlers: List[Handler] = [stream_handler]

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # only rank 0 will add a FileHandler
    if rank == 0 and filename is not None:
        # Here, the default behaviour of the official logger is 'a'. Thus, we
        # provide an interface to change the file mode to the default
        # behaviour.
        file_handler = logging.FileHandler(filename, mode)
        handlers.append(file_handler)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    # Apply the same to Pytorch Lightning logs.
    for handler in logging.getLogger("pytorch_lightning").handlers:
        handler.setFormatter(formatter)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    INITIALIZED_LOGGERS[name] = True

    # Do not pass to the handlers of ancestor loggers.
    # Otherwise, logs are duplicated.
    logger.propagate = False

    return logger
