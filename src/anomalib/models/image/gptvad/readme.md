# GptVad: Zero-/Few-Shot Anomaly Classification

This repository contains the implementation of the `OpenAI VLM`, a model designed for zero-shot and few-shot anomaly detection using OpenAI's GPT-4 for image analysis.

## Description

The `OpenAI VLM` is an anomaly detection model that leverages OpenAI's GPT-4 to identify anomalies in images. It supports both zero-shot and few-shot modes:

- **Zero-Shot Mode**: Direct anomaly detection without any prior examples of normal images.
- **Few-Shot Mode**: Anomaly detection using a small set of normal reference images to improve accuracy.

The model operates by encoding images into base64 format and passing them to the GPT-4 API. In zero-shot mode, the model analyzes the image directly. In few-shot mode, the model compares the target image with a set of reference images to detect anomalies.

## Features

- **Zero-/Few-Shot Learning**: Capable of performing anomaly detection without training (zero-shot) or with a few normal examples (few-shot).
- **OpenAI GPT-4 Integration**: Utilizes the latest advancements in natural language processing and image understanding for anomaly detection.

## Usage

### Zero-Shot Anomaly Detection

In zero-shot mode, the model does not require any reference images:

```python
from anomalib.data import MVTec
from anomalib.engine import Engine
from anomalib.models import ChatGPTVision
from dotenv import load_dotenv

# Load the environment variables from the .env file
# The implementation searchs for an environment variable OPENAI_API_KEY
# that will contain the key of OpenAI.

# load from .env to an environment variable.
load_dotenv()

model = ChatGPTVision(k_shot=0)
engine = Engine(task=TaskType.VISUAL_PROMPTING)
datamodule = MVTec(
    category=bottle,
    train_batch_size=1,
    eval_batch_size=1,
    num_workers=0,
    )
engine.test(model=model, datamodule=datamodule)
```

### Few-Shot Anomaly Detection

In few-shot mode, the model uses a small set of normal reference images:

```python
from anomalib.data import MVTec
from anomalib.engine import Engine
from anomalib.models import ChatGPTVision
from dotenv import load_dotenv

# Load the environment variables from the .env file
# load_dotenv(dotenv_path=env_path)
load_dotenv()

model = ChatGPTVision(k_shot=2)
engine = Engine(task=TaskType.VISUAL_PROMPTING)
datamodule = MVTec(
    category=bottle,
    train_batch_size=1,
    eval_batch_size=1,
    num_workers=0,
    )
engine.test(model=model, datamodule=datamodule)
```

## Parameters

| Parameter    | Type | Description                                                                                     | Default                    |
| ------------ | ---- | ----------------------------------------------------------------------------------------------- | -------------------------- |
| `k_shot`     | int  | Number of normal reference images used in few-shot mode.                                        | `0`                        |
| `model_name` | str  | The OpenAI VLM for the image detection.                                                         | `"gpt-4o-mini-2024-07-18"` |
| `detail`     | bool | The detail level of the input in the VLM for image detection: 'high' (`true`), 'low' (`false`). | `True`                     |

## Example Outputs

The model returns a response indicating whether an anomaly is detected:

- **Zero-Shot/Few-Shot Example**:

  ```plaintext
  "NO"
  ```

  ![GptVad result no anomaly](/docs/source/images/gptvad/good.png "GptVad without anomaly result")

  ```plaintext
  "YES: Description of the detected anomaly."
  ```

  ![GptVad result with anomaly](/docs/source/images/gptvad/broken.png "GptVad with Anomaly result")
