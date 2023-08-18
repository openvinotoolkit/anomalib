import torch

from anomalib.utils.metrics.perimg.common import perimg_boxplot_stats


def test_perimg_boxplot_stats():
    data = torch.arange(100) / 7 - 4  # arbitrary data
    img_cls = (torch.arange(100) % 3 == 0).to(torch.long)

    stats = perimg_boxplot_stats(data, img_cls)
    assert len(stats) > 0
    statdic = stats[0]
    assert "statistic" in statdic
    assert "value" in statdic
    assert "nearest" in statdic
    assert "imgidx" in statdic

    perimg_boxplot_stats(data, img_cls, only_class=0)
    assert len(stats) > 0

    perimg_boxplot_stats(data, img_cls, only_class=1)
    assert len(stats) > 0
