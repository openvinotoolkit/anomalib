"""Pretrained weights downloader for DSR model implementation."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import hashlib
from os import remove
from os.path import basename
from zipfile import ZipFile

import requests


class DsrWeightDownloader:
    """Pretrained weights downloader of the DSR model. This class will download the pretrained encoder and general
    object decoder weights, as well as the codebook weights. THe pretraining has been done on ImageNet.

    Args:
        md5: Expected hash value from the downloaded zip file.
    """

    def __init__(self, md5: str = "5fcf5c557a2ffe3366c6e8f90163d6ae"):
        self.md5 = md5

    def download(
        self,
        url: str = "https://docs.google.com/uc?export=download",
        file_id: str = "15plhikrUjYCcx23JVxxBKb-HBwKAb8UK",
        dest: str = "./src/anomalib/models/dsr/",
        temp: str = "./src/anomalib/models/dsr/checkpoints.zip",
    ) -> None:
        """Download the pretrained checkpoint file.

        Args:
            url (str, optional): URL to download from. Defaults to "https://docs.google.com/uc?export=download".
            file_id (str, optional): File ID to download (Google Drive). Defaults to "15plhikrUjYCcx23JVxxBKb-HBwKAb8UK".
            dest (str, optional): Folder to extract the ckpt file to. Defaults to "./src/anomalib/models/dsr/".
            temp (str, optional): Temporary zip file location. Defaults to "./src/anomalib/models/dsr/checkpoints.zip".
        """
        session = requests.Session()

        response = session.get(url, params={"id": file_id, "confirm": 1}, stream=True)
        token = self._get_confirm_token(response)

        if token:
            params = {"id": file_id, "confirm": token}
            response = session.get(url, params=params, stream=True)

        self._save_response_content(response, temp)

        self._extract_weights(temp, dest)

        self._delete_zip(temp)

    def _get_confirm_token(self, response: requests.Response) -> str | None:
        """Get confirmation token from Google Drive.

        Args:
            response (Response): GET query response.

        Returns:
            str | None: Confirmation token if success, nothing otherwise.
        """
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value

        return None

    def _save_response_content(self, response: requests.Response, destination: str):
        """Download and save response content.

        Args:
            response (Response): GET query response.
            destination (str): Local filename.
        """
        CHUNK_SIZE = 32768

        print("Downloading weights...")
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

        # MD5 checking
        with open(destination, "rb") as hash_file:
            assert (
                hashlib.md5(hash_file.read()).hexdigest() == self.md5
            ), f"Downloaded file {basename(destination)} does not match the required hash."

        print("Weights downloaded.")

    def _extract_weights(self, source: str, destination: str):
        """Extract the weights file from the archive.

        Args:
            source (str): Source file.
            destination (str): Destination folder.
        """
        print("Extracting weights...")
        with ZipFile(source) as file:
            checkpoint = file.getinfo("checkpoints/vq_model_pretrained_128_4096.pckl")
            checkpoint.filename = basename(checkpoint.filename)
            file.extract(checkpoint, destination)
        print("Weights extracted.")

    def _delete_zip(self, source: str):
        """Deleted temporary zip archive.

        Args:
            source (str): Source file.
        """
        print("Deleting temporary files...")
        remove(source)
        print("Files deleted.")


if __name__ == "__main__":
    DsrWeightDownloader()
