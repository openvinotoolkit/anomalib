""" Test comparison of multiple models.
There is no such module, the functions tested can come from any module in `anomalib.utils.metrics.perimg`.
"""
import torch

from anomalib.utils.metrics.perimg.common import compare_models_nonparametric, compare_models_parametric
from anomalib.utils.metrics.perimg.plot import (
    compare_models_perimg,
    compare_models_perimg_rank,
)


def pytest_generate_tests(metafunc):
    num_images = 100
    aucs1 = 0.8 * torch.ones(num_images)
    aucs2 = torch.linspace(0.5, 0.9, num_images)
    aucs3 = torch.sin(torch.linspace(0, torch.pi, num_images)).clip(0, 1)

    # add `nan`s (normal images)
    for aucs in (aucs1, aucs2, aucs3):
        aucs[30:45] = torch.nan

    def get_similar(aucs):
        # a signal oscilating around a's signal to provoke the non-parametric plot to show
        # the two are within the tolerance
        factor = 1 + 0.005 * torch.sin(torch.linspace(0, 2 * torch.pi, len(aucs)))
        aucs_bis = (aucs * factor).clip(0, 1)
        aucs_bis[torch.isnan(aucs)] = torch.nan
        return aucs_bis

    if "aucs_a" in metafunc.fixturenames:
        metafunc.parametrize(
            ("aucs_a",),
            [
                (aucs1,),
                (aucs2,),
            ],
        )
        metafunc.parametrize(("aucs_b", "aucs_c"), [(aucs2, get_similar(aucs2)), (aucs3, get_similar(aucs3))])


def test_compare_models_perimg(aucs_a, aucs_b, aucs_c):
    models = {"a": aucs_a, "b": aucs_b}
    fig, ax = compare_models_perimg(models, metric_name="auc", random_model_score=0.5)
    assert fig is not None
    assert ax is not None

    compare_models_perimg(models, metric_name="auc", ax=ax)

    models["c"] = aucs_c
    compare_models_perimg(models, metric_name="auc", ax=ax)


def test_compare_models_perimg_rank(aucs_a, aucs_b, aucs_c):
    models = {"a": aucs_a, "b": aucs_b}
    fig, ax = compare_models_perimg_rank(models, metric_name="auc", higher_is_better=True, atol=None)
    assert fig is not None
    assert ax is not None

    compare_models_perimg_rank(models, metric_name="auc", ax=ax, higher_is_better=False)

    models["c"] = aucs_c
    compare_models_perimg_rank(models, metric_name="auc", ax=ax, atol=0.05)


def test_compare_models_parametric(aucs_a, aucs_b, aucs_c):
    models = {"a": aucs_a, "b": aucs_b}
    table = compare_models_parametric(models, higher_is_better=True)
    assert table is not None
    assert table.data.shape == (2, 2)

    models["c"] = aucs_c
    sorted_models, test_results = compare_models_parametric(models, higher_is_better=False, return_test_results=True)
    assert len(sorted_models) == 3
    assert len(test_results) == 3
    for modelpair in (sorted_models[0:2], sorted_models[1:3]):
        assert tuple(modelpair) in test_results


def test_compare_models_nonparametric(aucs_a, aucs_b, aucs_c):
    models = {"a": aucs_a, "b": aucs_b}

    table = compare_models_nonparametric(models, higher_is_better=True, atol=None)
    assert table is not None
    assert table.data.shape == (2, 2)

    table = compare_models_nonparametric(models, higher_is_better=False)
    assert table is not None
    assert table.data.shape == (2, 2)

    models["c"] = aucs_c
    sorted_models, test_results = compare_models_nonparametric(models, return_test_results=True, atol=0.05)
    assert len(sorted_models) == 3
    assert len(test_results) == 3
    for modelpair in (sorted_models[0:2], sorted_models[1:3]):
        assert tuple(modelpair) in test_results
