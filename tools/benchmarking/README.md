# Performance Benchmarking tools

These bash scripts will assist in measuring the training performance of the anomalib library.

The python script (`benchmark.py`) will assist in computing metrics for all the models in the repository.

## Usage
Run the train.sh with the same args as the tools/train.py. Refer to [`../README.md`](https://github.com/openvinotoolkit/anomalib/blob/development/README.md) for those details.

Note: To collect memory read/write numbers, run the script with sudo privileges. Otherwise, those values will be blank.

```
sudo -E ./train.sh    # Train STFPM on MVTec AD leather

sudo -E ./train.sh --model_config_path <path/to/model/config.yaml>

sudo -E ./train.sh --model stfpm
```

The training script will create an output directory in this location, and inside it will have a time stamped directory for each training run you do. You can find the raw logs in there, as well as any errors captured in the train.log file.

For post processing, run the post-process.sh script with the results directory you want to post process.

```
./post-process.sh ./output/2021Aug31_2351
```

---

To use the python script, run it from the root directory.

```
python tools/benchmarking/benchmark.py
```

The output will be generated in results folder and a csv file for each model.
