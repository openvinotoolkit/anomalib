import torch.nn.functional as F
from einops import reduce
from torch import nn

from anomalib.models import AnomalyModule


class DummyAnomalibModule(AnomalyModule):
    def validation_step(self, batch, *args, **kwargs):
        batch["anomaly_maps"] = reduce(batch["images"], "b c h w -> b 1 h w", "sum")
        return batch

    def configure_optimizers(self):
        return None


class DummyModel(nn.Module):
    """Creates a very basic CNN model to fit image data for classification task
    The test uses this to check if this model is converted to OpenVINO IR."""

    def __init__(
        self,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 5)
        self.conv3 = nn.Conv2d(32, 1, 7)
        self.fc1 = nn.Linear(400, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        batch_size, _, _, _ = x.size()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = F.dropout(x, p=0.2)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x
