#

def tiling_test():
    # Import the required modules
    from anomalib.data import MVTec
    from anomalib.engine import Engine
    from anomalib.models import Padim, Stfpm
    from anomalib.callbacks import TilerConfigurationCallback

    # Initialize the datamodule and model
    datamodule = MVTec(num_workers=0, image_size=(128, 128))
    model = Stfpm()

    # prepare tiling configuration callback
    tiler_config_callback = TilerConfigurationCallback(enable=True, tile_size=[128, 64], stride=64)

    # pass the tiling configuration callback to engine
    engine = Engine(image_metrics=["AUROC"], pixel_metrics=["AUROC"], callbacks=[tiler_config_callback])

    # train the model (tiling is seamlessly utilized in the background)
    engine.fit(datamodule=datamodule, model=model)


if __name__ == '__main__':
    tiling_test()
