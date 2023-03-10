# Guide to Setting up the CI using the Docker images

## Steps

1. Build the docker image using the Dockerfile in the .ci directory.
   Make sure you are in the root directory of `anomalib`.

   ```bash
   sudo docker build --build-arg HTTP_PROXY="$http_proxy" --build-arg \
   HTTPS_PROXY="$https_proxy" --build-arg NO_PROXY="$no_proxy" \
   . -t anomalib-ci -f .ci/cuda11.4.Dockerfile
   ```

   Here, `anomalib-ci` is the name of the image.

1. Create and start a container

   ```bash
   sudo docker run --gpus all \
   --shm-size=2G\
    -i -t --mount type=bind,source=<path-to-datasets>,target=/home/user/datasets,readonly\
    -d --name anomalib-ci-container anomalib-ci
   ```

   Note: `--gpus all` is required for the container to have access to the GPUs.
   `-d` flag ensure that the container is detached when it is created.
   `mount` is required to ensure that tests have access to the dataset.

1. Enter the container by

   ```bash
   sudo docker exec -it  anomalib-ci-container /bin/bash
   ```

1. Install github actions runner in the container by navigating to [https://github.com/openvinotoolkit/anomalib/settings/actions/runners/new](https://github.com/openvinotoolkit/anomalib/settings/actions/runners/new)

   For example:

   ```bash
   mkdir actions-runner && cd actions-runner

   curl -o actions-runner-linux-x64-2.296.1.tar.gz -L https://github.com/actions/runner/releases/download/v2.296.1/actions-runner-linux-x64-2.296.1.tar.gz

   tar xzf ./actions-runner-linux-x64-2.296.1.tar.gz

   rm actions-runner-linux-x64-2.296.1.tar.gz

   ./config.sh --url https://github.com/openvinotoolkit/anomalib --token <enter-your-token-here>
   ```

   Follow the instructions on the screen to complete the installation.

1. Now the container is ready. Type `exit` to leave the container.

1. Start github actions runner in detached mode in the container and set the
   the anomalib dataset environment variables.

   ```bash
   sudo docker exec -d anomalib-ci-container /bin/bash -c \
   "export ANOMALIB_DATASET_PATH=/home/user/datasets && /home/user/actions-runner/run.sh"
   ```
