import numpy as np
import pytest

from anomalib.post_processing import anomaly_map_to_color_map, superimpose_anomaly_map


def pytest_generate_tests(metafunc):
    resolution = (256, 256)
    image = np.zeros(resolution + (3,)).astype(np.uint8)
    rng = np.random.default_rng()
    anomaly_map = rng.uniform(0, 1, resolution)

    if "anomaly_map" in metafunc.fixturenames:
        metafunc.parametrize(
            argnames=("anomaly_map",),
            argvalues=[
                (anomaly_map,),
            ],
        )

    if "image" in metafunc.fixturenames:
        metafunc.parametrize(
            argnames=("image",),
            argvalues=[
                (image,),
            ],
        )


def test_anomaly_map_to_color_map(anomaly_map):
    resolution = anomaly_map.shape
    color_map_shape = resolution + (3,)

    color_map = anomaly_map_to_color_map(anomaly_map)
    assert color_map.shape == color_map_shape

    # `anomaly_map` must be \in [0, 1] when `normalize=False`
    anomaly_map[0, 0] = -1
    with pytest.raises(ValueError):
        anomaly_map_to_color_map(anomaly_map, normalize=False)

    color_map = anomaly_map_to_color_map(anomaly_map, normalize=True)
    assert color_map.shape == color_map_shape

    color_map = anomaly_map_to_color_map(anomaly_map, normalize=(0.3, 0.5))
    assert color_map.shape == color_map_shape

    color_map = anomaly_map_to_color_map(anomaly_map, normalize=(0.3, 0.5), saturation_colors=((0, 0, 0), (1, 1, 1)))
    assert color_map.shape == color_map_shape


def test_superimpose_anomaly_map(anomaly_map, image):
    resolution = anomaly_map.shape
    color_map_shape = resolution + (3,)

    sup_img = superimpose_anomaly_map(anomaly_map, image)
    assert sup_img.shape == color_map_shape

    # `anomaly_map` must be \in [0, 1] when `normalize=False`
    anomaly_map[0, 0] = -1
    with pytest.raises(ValueError):
        superimpose_anomaly_map(anomaly_map, image, normalize=False)

    sup_img = superimpose_anomaly_map(anomaly_map, image, normalize=True)
    assert sup_img.shape == color_map_shape

    sup_img = superimpose_anomaly_map(anomaly_map, image, normalize=(0.3, 0.5))
    assert sup_img.shape == color_map_shape

    sup_img = superimpose_anomaly_map(
        anomaly_map, image, normalize=(0.3, 0.5), saturation_colors=((0, 0, 0), (1, 1, 1))
    )
    assert sup_img.shape == color_map_shape

    sup_img = superimpose_anomaly_map(
        anomaly_map, image, normalize=(0.3, 0.5), saturation_colors=((0, 0, 0), (1, 1, 1)), alpha=0.5, gamma=0.5
    )
    assert sup_img.shape == color_map_shape

    sup_img = superimpose_anomaly_map(
        anomaly_map, image, normalize=(0.3, 0.5), saturation_colors=((0, 0, 0), (1, 1, 1)), ignore_low_scores=True
    )
    assert sup_img.shape == color_map_shape

    sup_img = superimpose_anomaly_map(
        anomaly_map, image, normalize=(0.3, 0.5), saturation_colors=((0, 0, 0), (1, 1, 1)), ignore_low_scores=False
    )
    assert sup_img.shape == color_map_shape

    # `ignore_low_scores` should be ignored when `normalize` is boolean
    sup_img1 = superimpose_anomaly_map(anomaly_map, image, normalize=True, ignore_low_scores=False)
    sup_img2 = superimpose_anomaly_map(anomaly_map, image, normalize=True, ignore_low_scores=True)
    assert (sup_img1 == sup_img2).all()
