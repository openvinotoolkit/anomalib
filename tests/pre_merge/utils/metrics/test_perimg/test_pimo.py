import torch

from anomalib.utils.metrics.perimg.pimo import AULogPImO, AUPImO, PImO


def pytest_generate_tests(metafunc):
    """
    all functions are parametrized with the same arguments
    they have 2 normal and 2 anomalous images
    """

    if "log" not in metafunc.function.__name__:
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
        expected_tpr_anom1 = torch.tensor([4, 3, 2, 1]) / 4
        # trapezoid surface
        expected_aupimo_anom1 = torch.tensor((0.25 + 0.75) / 2)
        expected_aupimo_anom1_ubound05 = torch.tensor((0.25 + 0.50) / 2)

        pred_anom2 = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        mask_anom2 = torch.concatenate([torch.ones(2), torch.zeros(2)]).to(torch.int32).reshape(2, 2)
        # trapezoid surface
        expected_tpr_anom2 = torch.tensor([2, 1, 0, 0]) / 2
        expected_aupimo_anom2 = torch.tensor((0.0 + 0.25) / 2)
        expected_aupimo_anom2_ubound05 = torch.tensor((0.0 + 0.0) / 2)

        anomaly_maps = torch.stack([pred_norm1, pred_norm2, pred_anom1, pred_anom2], axis=0)
        masks = torch.stack([mask_norm1, mask_norm2, mask_anom1, mask_anom2], axis=0)
        expected_fpr = expected_fpr_mean
        expected_tprs = torch.stack(
            [expected_tpr_norm1, expected_tpr_norm2, expected_tpr_anom1, expected_tpr_anom2], axis=0
        )
        expected_image_classes = torch.tensor([0, 0, 1, 1])
        expected_aupimos = torch.stack(
            [expected_aupimo_norm1, expected_aupimo_norm2, expected_aupimo_anom1, expected_aupimo_anom2], axis=0
        )
        expected_aupimos_ubound05 = torch.stack(
            [
                expected_aupimo_norm1,
                expected_aupimo_norm2,
                expected_aupimo_anom1_ubound05,
                expected_aupimo_anom2_ubound05,
            ],
            axis=0,
        )

        metafunc.parametrize(
            argnames=(
                "anomaly_maps",
                "masks",
            ),
            argvalues=[(anomaly_maps, masks)],
        )

        if "expected_aupimos" in metafunc.fixturenames:
            metafunc.parametrize(
                argnames=(
                    "expected_thresholds",
                    "expected_fpr",
                    "expected_tprs",
                    "expected_image_classes",
                    "expected_aupimos",
                    "ubound",
                ),
                argvalues=[
                    (thresholds, expected_fpr, expected_tprs, expected_image_classes, expected_aupimos, 1),
                    (thresholds, expected_fpr, expected_tprs, expected_image_classes, expected_aupimos_ubound05, 0.5),
                ],
            )

        elif "ubound" in metafunc.fixturenames:
            metafunc.parametrize(
                argnames=("ubound",),
                argvalues=[
                    (1,),
                    (0.5,),
                ],
            )

    else:
        # {0, 1, 2, 3}
        thresholds = torch.arange(4, dtype=torch.float32)
        shape = (40, 25)  # 1000 pixels

        # --- normal ---

        def get_pred():
            """
            pred (content | nb pixels): (1 | 3), (9 | 2), (90 | 1), (900 | 0)
                                     (1 | >= 3), (10 | >= 2), (100 | >=1), (1000 | >= 0)
            """
            pred = torch.zeros(1000, dtype=torch.float32)
            pred[:100] += 1
            pred[:10] += 1
            pred[:1] += 1
            return pred

        pred_norm = get_pred().reshape(shape)
        mask_norm = torch.zeros(1000, dtype=torch.int32).reshape(shape)
        expected_fpr = torch.tensor([1, 0.1, 0.01, 0.001], dtype=torch.float64)
        expected_tpr_norm = torch.full((4,), torch.nan, dtype=torch.float64)

        # --- anomalous ---
        pred_anom1 = get_pred().reshape(shape)
        mask_anom1 = torch.ones(1000, dtype=torch.int32).reshape(shape)
        expected_tpr_anom1 = torch.tensor([1, 0.1, 0.01, 0.001], dtype=torch.float64)

        # only 100 pixels are anom
        # (1 | 3), (9 | 2), (90 | 1), (0 | 0)
        # (1 | >= 3), (10 | >= 2), (100 | >=1), (100 | >= 0)
        pred_anom2 = get_pred().reshape(shape)
        mask_anom2 = torch.concatenate([torch.ones(100), torch.zeros(900)]).to(torch.int32).reshape(shape)
        expected_tpr_anom2 = torch.tensor([1, 1, 0.1, 0.01], dtype=torch.float64)

        anomaly_maps = torch.stack([pred_norm, pred_anom1, pred_anom2], axis=0)
        masks = torch.stack([mask_norm, mask_anom1, mask_anom2], axis=0)
        expected_tprs = torch.stack([expected_tpr_norm, expected_tpr_anom1, expected_tpr_anom2], axis=0)
        expected_image_classes = torch.tensor([0, 1, 1])

        metafunc.parametrize(
            argnames=(
                "anomaly_maps",
                "masks",
            ),
            argvalues=[(anomaly_maps, masks)],
        )

        if "expected_aulogpimos" in metafunc.fixturenames:
            metafunc.parametrize(
                argnames=(
                    "expected_thresholds",
                    "expected_fpr",
                    "expected_tprs",
                    "expected_image_classes",
                ),
                argvalues=[
                    (thresholds, expected_fpr, expected_tprs, expected_image_classes),
                ],
            )

            metafunc.parametrize(
                argnames=(
                    "lbound",
                    "ubound",
                    "expected_aulogpimos",  # trapezoid surfaces
                ),
                argvalues=[
                    (
                        0.001,
                        1,
                        torch.as_tensor(
                            [
                                torch.nan,
                                sum([0.011 / 2, 0.11 / 2, 1.1 / 2]) / 3,
                                sum([0.11 / 2, 1.1 / 2, 1]) / 3,
                            ],
                            dtype=torch.float64,
                        ),
                    ),
                    (
                        0.001,
                        0.1,
                        torch.as_tensor(
                            [
                                torch.nan,
                                sum([0.011 / 2, 0.11 / 2]) / 2,
                                sum([0.11 / 2, 1.1 / 2]) / 2,
                            ],
                            dtype=torch.float64,
                        ),
                    ),
                    (
                        0.1,
                        1,
                        torch.as_tensor(
                            [
                                torch.nan,
                                1.1 / 2,
                                1.0,
                            ],
                            dtype=torch.float64,
                        ),
                    ),
                ],
            )
        else:
            metafunc.parametrize(
                argnames=(
                    "lbound",
                    "ubound",
                ),
                argvalues=[
                    (0.001, 1),
                    (0.001, 0.1),
                    (0.1, 1),
                ],
            )


def test_pimo(anomaly_maps, masks):
    # the expected values are already tested in test_aupimo
    pimo = PImO()
    pimo.update(anomaly_maps, masks)
    pimoresult = pimo.compute()
    assert pimoresult.thresholds.ndim == 1
    assert pimoresult.fprs.ndim == 2
    assert pimoresult.shared_fpr.ndim == 1
    assert pimoresult.tprs.ndim == 2
    assert pimoresult.image_classes.ndim == 1
    pimo.plot()


def test_aupimo(
    anomaly_maps,
    masks,
    expected_thresholds,
    expected_fpr,
    expected_tprs,
    expected_image_classes,
    expected_aupimos,
    ubound,
):
    aupimo = AUPImO(num_thresholds=expected_thresholds.shape[0], ubound=ubound)
    str(aupimo)
    aupimo.update(anomaly_maps, masks)
    # `com` stands for `computed`
    pimoresult, com_aulogpimos = aupimo.compute()
    (com_thresholds, com_fprs, com_shared_fpr, com_tprs, com_image_classes) = pimoresult
    assert pimoresult.thresholds.ndim == 1
    assert pimoresult.fprs.ndim == 2
    assert pimoresult.shared_fpr.ndim == 1
    assert pimoresult.tprs.ndim == 2
    assert pimoresult.image_classes.ndim == 1
    assert com_aulogpimos.ndim == 1
    assert (com_thresholds == expected_thresholds).all()
    assert (com_shared_fpr == expected_fpr).all()
    assert com_tprs[:2].isnan().all()
    assert (com_tprs[2:] == expected_tprs[2:]).all()
    assert (com_image_classes == expected_image_classes).all()
    assert com_aulogpimos[:2].isnan().all()
    assert (com_aulogpimos[2:] == expected_aupimos[2:]).all()

    stats = aupimo.boxplot_stats()
    assert len(stats) > 0


def test_aupimo_plots(anomaly_maps, masks, ubound):
    aupimo = AUPImO(num_thresholds=1000, ubound=ubound)
    aupimo.update(anomaly_maps, masks)
    aupimo.compute()

    fig, axes = aupimo.plot()
    assert fig is not None
    assert axes is not None
    aupimo.plot(axes=axes)

    fig, ax = aupimo.plot_all_pimo_curves()
    assert fig is not None
    assert ax is not None
    aupimo.plot_all_pimo_curves(ax=ax)

    fig, ax = aupimo.plot_boxplot()
    assert fig is not None
    assert ax is not None
    aupimo.plot_boxplot(ax=ax)

    fig, ax = aupimo.plot_boxplot_pimo_curves()
    assert fig is not None
    assert ax is not None
    aupimo.plot_boxplot_pimo_curves(ax=ax)

    fig, axes = aupimo.plot_perimg_fprs()
    assert fig is not None
    assert ax is not None
    aupimo.plot_perimg_fprs(axes=axes)


def test_aulogpimo(
    anomaly_maps,
    masks,
    expected_thresholds,
    expected_fpr,
    expected_tprs,
    expected_image_classes,
    expected_aulogpimos,
    lbound,
    ubound,
):
    aulogpimo = AULogPImO(num_thresholds=expected_thresholds.shape[0], lbound=lbound, ubound=ubound)
    str(aulogpimo)
    assert 0 < aulogpimo.max_primitive_auc
    assert 0 < aulogpimo.random_model_primitive_auc < aulogpimo.max_primitive_auc
    assert 0 < aulogpimo.random_model_auc < 1

    aulogpimo.update(anomaly_maps, masks)
    # `com` stands for `computed`
    pimoresult, com_aulogpimos = aulogpimo.compute()
    (com_thresholds, com_fprs, com_shared_fpr, com_tprs, com_image_classes) = pimoresult
    assert pimoresult.thresholds.ndim == 1
    assert pimoresult.fprs.ndim == 2
    assert pimoresult.shared_fpr.ndim == 1
    assert pimoresult.tprs.ndim == 2
    assert pimoresult.image_classes.ndim == 1
    assert com_aulogpimos.ndim == 1
    assert (com_thresholds == expected_thresholds).all()
    assert (com_shared_fpr == expected_fpr).all()
    assert com_tprs[:1].isnan().all()
    assert (com_tprs[1:] == expected_tprs[1:]).all()
    assert (com_image_classes == expected_image_classes).all()
    assert com_aulogpimos[:1].isnan().all()
    assert torch.allclose(com_aulogpimos[1:], expected_aulogpimos[1:], atol=1e-6)

    stats = aulogpimo.boxplot_stats()
    assert len(stats) > 0


def test_aulogpimo_plots(anomaly_maps, masks, lbound, ubound):
    aulogpimo = AULogPImO(num_thresholds=1000, lbound=lbound, ubound=ubound)
    aulogpimo.update(anomaly_maps, masks)
    aulogpimo.compute()

    fig, axes = aulogpimo.plot()
    assert fig is not None
    assert axes is not None
    aulogpimo.plot(axes=axes)

    fig, ax = aulogpimo.plot_all_logpimo_curves()
    assert fig is not None
    assert ax is not None
    aulogpimo.plot_all_logpimo_curves(ax=ax)

    fig, ax = aulogpimo.plot_boxplot()
    assert fig is not None
    assert ax is not None
    aulogpimo.plot_boxplot(ax=ax)

    fig, ax = aulogpimo.plot_boxplot_logpimo_curves()
    assert fig is not None
    assert ax is not None
    aulogpimo.plot_boxplot_logpimo_curves(ax=ax)

    fig, axes = aulogpimo.plot_perimg_fprs()
    assert fig is not None
    assert ax is not None
    aulogpimo.plot_perimg_fprs(axes=axes)
