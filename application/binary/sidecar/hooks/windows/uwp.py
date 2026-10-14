# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import ctypes
import os
import shutil
from pathlib import Path

GetCurrentPackagePath = ctypes.windll.kernel32.GetCurrentPackagePath  # type: ignore[attr-defined]
GetCurrentPackagePath.argtypes = [ctypes.POINTER(ctypes.c_uint), ctypes.c_wchar_p]
GetCurrentPackagePath.restype = ctypes.c_long


def _get_current_package_path() -> str | None:
    length = ctypes.c_uint(256)
    buffer = ctypes.create_unicode_buffer(256)

    result = GetCurrentPackagePath(ctypes.byref(length), buffer)
    return buffer.value if result == 0 else None


def _copy_initial_data(app_data_folder: Path) -> None:
    package_path = _get_current_package_path()
    print(f"Setup Hook: Application package path: {package_path}")
    if not package_path:
        return
    initial_data_path = Path(package_path) / "InitialData"
    if not os.path.exists(initial_data_path):
        return
    for item in os.listdir(initial_data_path):
        destination_path = app_data_folder / item
        if os.path.exists(destination_path):
            continue
        print(f"Setup Hook: Copying initial data: {item}. Destination: {destination_path}")
        try:
            shutil.copytree(initial_data_path / item, destination_path)
        except OSError as e:
            print(f"Setup Hook: Failed to copy initial data {item}", e)


def _main() -> None:
    local_app_data = os.getenv("LOCALAPPDATA")
    if not local_app_data:
        raise OSError("LOCALAPPDATA environment variable is not set.")

    app_data_folder = Path(local_app_data) / "Intel" / "AnomalibStudio"
    os.makedirs(app_data_folder, exist_ok=True)

    data_folder = app_data_folder / "data"
    os.makedirs(data_folder, exist_ok=True)

    logs_folder = app_data_folder / "logs"
    os.makedirs(logs_folder, exist_ok=True)

    print(f"Setup Hook: Using folder: {data_folder} for data and {logs_folder} for logs")
    os.environ["DATA_DIR"] = str(data_folder)
    os.environ["LOG_DIR"] = str(logs_folder)

    _copy_initial_data(app_data_folder)


_main()
