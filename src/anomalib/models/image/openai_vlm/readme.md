# OpenAI VLM: Zero-/Few-Shot Anomaly Classification

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
from openai_vlm import OpenaiVlm

model = OpenaiVlm(openai_key="your-openai-api-key")
result = model.api_call(image="path/to/image.jpg")
print(result)
```

### Few-Shot Anomaly Detection

In few-shot mode, the model uses a small set of normal reference images:

```python
from anomalib.data import MVTec
from anomalib.engine import Engine
from anomalib.models import OpenaiVlm
from dotenv import load_dotenv

# Load the environment variables from the .env file
# load_dotenv(dotenv_path=env_path)
load_dotenv()

                                                                                                                          # Access the secret key
secret_key = os.getenv("OPEN_AI_KEY")
model = OpenaiVlm(k_shot=0, openai_key=secret_key)
engine = Engine(task=TaskType.EXPLANATION)
datamodule = MVTec(
    category=bottle,
    train_batch_size=1,
    eval_batch_size=1,
    num_workers=0,
    )
engine.test(model=model, datamodule=datamodule)
```

## Parameters

| Parameter    | Type | Description                                               | Default |
| ------------ | ---- | --------------------------------------------------------- | ------- |
| `k_shot`     | int  | Number of normal reference images used in few-shot mode.  | `0`     |
| `openai_key` | str  | API key for OpenAI. Required for accessing the GPT-4 API. | `None`  |

## Example Outputs

The model returns a response indicating whether an anomaly is detected:

- **Zero-Shot Example**:

  ```plaintext
  "NO"
  ```

  ![Openai result no anomaly](/docs/source/images/openai_vlm/good.png "Openai without anomaly result")

- **Few-Shot Example**:

  ```plaintext
  "YES: Description of the detected anomaly."
  ```

  ![Openai result with anomaly](/docs/source/images/winclip/architecture.png "Openai with Anomaly result")
