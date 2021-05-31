from nncf.initialization import InitializingDataLoader, DefaultInitializingDataLoader


def criterion_fn(outputs, criterion):
    return criterion(outputs)


class InitLoader(InitializingDataLoader):

    def __next__(self):
        loaded_item = next(self.data_loader_iter)
        return loaded_item["image"]

    def get_inputs(self, dataloader_output):
        return (dataloader_output,), {}

    def get_target(self, dataloader_output):
        return None
