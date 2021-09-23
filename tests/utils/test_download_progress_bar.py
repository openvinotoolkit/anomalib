import os
import tempfile
from urllib.request import urlretrieve

from anomalib.utils.download_progress_bar import DownloadProgressBar


def test_output_on_download(capfd):
    """Test whether progress bar is shown"""
    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/SIPI_Jelly_Beans_4.1.07.tiff/lossy-page1-256px-SIPI_Jelly_Beans_4.1.07.tiff.jpg"
    with tempfile.TemporaryDirectory() as dir_loc:
        destination = os.path.join(dir_loc, "jelly.jpg")
        with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]) as p_bar:
            urlretrieve(url, filename=destination, reporthook=p_bar.update_to)

        assert os.path.exists(destination), "Failed retrieving the file"
        _, err = capfd.readouterr()
        assert "lossy-page1-256px-SIPI_Jelly_Beans_4.1.07.tiff.jpg" in err, "Failed showing progress bar in terminal"
