# Migrating from 0.\* to 1.0

## Overview

The 1.0 release of the Anomaly Detection Library (AnomalyLib) introduces several
changes to the library. This guide provides an overview of the changes and how
to migrate from 0.\* to 1.0.

## Installation

For installation instructions, refer to the [installation guide](anomalib.md).

## Changes to the CLI

### Upgrading the Configuration

There are several changes to the configuration of Anomalib. The configuration
file has been updated to include new parameters and remove deprecated parameters.
In addition, some parameters have been moved to different sections of the
configuration.

Anomalib provides a python script to update the configuration file from 0.\* to 1.0.
To update the configuration file, run the following command:

```bash
python tools/upgrade/config.py \
    --input_config <path_to_0.*_config> \
    --output_config <path_to_1.0_config>
```

This script will ensure that the configuration file is updated to the 1.0 format.

In the following sections, we will discuss the changes to the configuration file
in more detail.

### Changes to the Configuration File

🚧 To be updated.
