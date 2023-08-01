# plot_piwith pytest.raisesmo_curves

import pytest
import torch

from anomalib.utils.metrics.perimg.pimo import AUPImO, PImO


def pytest_generate_tests(metafunc):
    """
    all functions are parametrized with the same arguments
    they have 2 normal and 2 anomalous images
    """

    # {0, 1, 2, 3}
    thresholds = torch.arange(4, dtype=torch.float32)

    # --- normal ---

    pred_norm1 = torch.ones(4, dtype=torch.float32).reshape(2, 2)
    mask_norm1 = torch.zeros(4, dtype=torch.int32).reshape(2, 2)
    # expected_fpr_norm1 = torch.tensor([1, 1, 0, 0])
    expected_tpr_norm1 = torch.full((4,), torch.nan, dtype=torch.float32)
    expected_aupimo_norm1 = torch.tensor(torch.nan)

    pred_norm2 = 2 * torch.ones(4, dtype=torch.float32).reshape(2, 2)
    mask_norm2 = torch.zeros(4, dtype=torch.int32).reshape(2, 2)
    # expected_fpr_norm2 = torch.tensor([1, 1, 1, 0])
    expected_tpr_norm2 = torch.full((4,), torch.nan, dtype=torch.float32)
    expected_aupimo_norm2 = torch.tensor(torch.nan)

    expected_fpr_mean = torch.tensor([1, 1, 0.5, 0])

    # --- anomalous ---

    pred_anom1 = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    mask_anom1 = torch.ones(4, dtype=torch.int32).reshape(2, 2)
    expected_fpr_anom1 = torch.tensor([4, 3, 2, 1]) / 4
    expected_aupimo_anom1 = torch.tensor((0.25 + 0.75) / 2)

    pred_anom2 = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    mask_anom2 = torch.concatenate([torch.ones(2), torch.zeros(2)]).to(torch.int32).reshape(2, 2)
    expected_fpr_anom2 = torch.tensor([2, 1, 0, 0]) / 2
    expected_aupimo_anom2 = torch.tensor((0.0 + 0.25) / 2)

    anomaly_maps = torch.stack([pred_norm1, pred_norm2, pred_anom1, pred_anom2], axis=0)
    masks = torch.stack([mask_norm1, mask_norm2, mask_anom1, mask_anom2], axis=0)
    expected_fpr = expected_fpr_mean
    expected_tprs = torch.stack(
        [expected_tpr_norm1, expected_tpr_norm2, expected_fpr_anom1, expected_fpr_anom2], axis=0
    )
    expected_image_classes = torch.tensor([0, 0, 1, 1])
    expected_aupimos = torch.stack(
        [expected_aupimo_norm1, expected_aupimo_norm2, expected_aupimo_anom1, expected_aupimo_anom2], axis=0
    )

    if "expected_aupimos" in metafunc.fixturenames:
        metafunc.parametrize(
            argnames=(
                "anomaly_maps",
                "masks",
                "expected_thresholds",
                "expected_fpr",
                "expected_tprs",
                "expected_image_classes",
                "expected_aupimos",
            ),
            argvalues=[
                (anomaly_maps, masks, thresholds, expected_fpr, expected_tprs, expected_image_classes, expected_aupimos)
            ],
        )
    else:
        metafunc.parametrize(
            argnames=(
                "anomaly_maps",
                "masks",
            ),
            argvalues=[(anomaly_maps, masks)],
        )


def test_pimo(anomaly_maps, masks):
    # the expected values are already tested in test_aupimo
    pimo = PImO()
    pimo.update(anomaly_maps, masks)
    pimo.compute()
    pimo.plot()


def test_aupimo(
    anomaly_maps, masks, expected_thresholds, expected_fpr, expected_tprs, expected_image_classes, expected_aupimos
):
    aupimo = AUPImO(num_thresholds=expected_thresholds.shape[0])
    aupimo.update(anomaly_maps, masks)
    computed_thresholds, computed_fpr, computed_tprs, image_classes, computed_aupimos = aupimo.compute()

    assert (computed_thresholds == expected_thresholds).all()
    assert (computed_fpr == expected_fpr).all()
    assert computed_tprs[:2].isnan().all()
    assert (computed_tprs[2:] == expected_tprs[2:]).all()
    assert (image_classes == expected_image_classes).all()
    assert computed_aupimos[:2].isnan().all()
    assert (computed_aupimos[2:] == expected_aupimos[2:]).all()


def test_aupimo_plot(
    anomaly_maps, masks, expected_thresholds, expected_fpr, expected_tprs, expected_image_classes, expected_aupimos
):
    aupimo = AUPImO(num_thresholds=expected_thresholds.shape[0])
    aupimo.plot_pimo_curves()  # should not break
    aupimo.update(anomaly_maps, masks)
    aupimo.plot_pimo_curves()
