import torch

from anomalib.utils.metrics.perimg.common import _perimg_boxplot_stats


def test__perimg_boxplot_stats():
    data = torch.arange(100) / 7 - 4  # arbitrary data
    img_cls = (torch.arange(100) % 3 == 0).to(torch.long)

    stats = _perimg_boxplot_stats(data, img_cls)
    assert len(stats) > 0
    statdic = stats[0]
    assert "statistic" in statdic
    assert "value" in statdic
    assert "nearest" in statdic
    assert "imgidx" in statdic

    _perimg_boxplot_stats(data, img_cls, only_class=0)
    assert len(stats) > 0

    _perimg_boxplot_stats(data, img_cls, only_class=1)
    assert len(stats) > 0
