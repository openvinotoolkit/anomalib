#!/usr/bin/env python
# coding: utf-8

# In[]:
# Setup (pre-args)

from __future__ import annotations

import os

os.environ["WANDB_NOTEBOOK_NAME"] = __file__

from IPython import get_ipython
from IPython.core.interactiveshell import InteractiveShell

get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")
# make a cell print all the outputs instead of just the last one
InteractiveShell.ast_node_interactivity = "all"


from main import (  # noqa: E402
    INPUT_IMAGE_RESOLUTION,
    evaluate,
    get_datamodule,
    get_model_dir,
    parser,
)

# In[]:

# Args
# The mechanism below allows the 'notebook mode' and the cli to
# eventually executed a subset of all dataset/category choices.

# `dc` stands for "dataset-category"
argstr = "--dc mvtec/hazelnut"  # hazelnut is the largest dataset
# argstr = "--dc " + " ".join(DATASET_CATEGORY_CHOICES)
print(f"{argstr=}")
cliargs = parser.parse_args(argstr.split())
print(f"{cliargs=}")
datasets_categories = cliargs.dataset_category

# In[]:

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from progressbar import progressbar  # noqa: E402

MODELDIR = get_model_dir("padim_r18")
print(f"{MODELDIR=}")
assert MODELDIR.exists()

for ds_cat in progressbar(datasets_categories):
    dataset, category = ds_cat.split("/")
    print(f"{dataset=} {category=}")

    loaddir = MODELDIR / dataset / category
    assert loaddir.exists()

    datamodule = get_datamodule(
        dataset,
        category,
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=8,
        seed=0,
        input_image_resolution=INPUT_IMAGE_RESOLUTION,
    )
    datamodule.setup()

    masks = torch.concatenate([batch["mask"].cpu().int() for batch in datamodule.test_dataloader()], dim=0)

    asmaps = torch.load(loaddir / "asmaps.pt")
    df_preds = pd.read_csv(loaddir / "predictions.csv")

    ascores = torch.as_tensor(df_preds["ascore"].values)
    imgclass = torch.as_tensor(df_preds["imgclass"].values)

    img_metrics, pix_set_metrics, times, aupimos = evaluate(
        None, ascores, imgclass, asmaps, masks, savedir=None, logger=None, debug=True
    )

# %%
norm_scores = asmaps[masks == 0]
anom_scores = asmaps[masks == 1]
min_score = min(norm_scores.min(), anom_scores.min())
max_score = max(norm_scores.max(), anom_scores.max())

# %%
import matplotlib.pyplot as plt
import numpy as np

bins = np.linspace(min_score, max_score, 300)
cdf0, _, __ = plt.hist(norm_scores, bins=bins, label="norm", density=True, cumulative=True, histtype="step")
cdf1, _, __ = plt.hist(anom_scores, bins=bins, label="anom", density=True, cumulative=True, histtype="step")

# %%
plt.plot(bins[:-1], cdf0, label="norm")
plt.plot(bins[:-1], 1 - cdf1, label="anom")
th_star = bins[np.argmin(np.abs(cdf0 - (1 - cdf1)))]
plt.axvline(th_star, color="k", linestyle="--", label="th*")

# %%


def T0(th_star, ascores, t):
    assert t >= 0 and t <= 1
    mi = ascores.min()
    ma = ascores.max()
    assert th_star <= ma
    return mi + (ascores - mi) * (1 - (ma - th_star) / (ma - mi) * t)


def T1(th_star, ascores, t):
    assert t >= 0 and t <= 1
    mi = ascores.min()
    ma = ascores.max()
    assert th_star >= mi
    return ma - (ma - ascores) * (1 - (th_star - mi) / (ma - mi) * t)


cdf0t0, _, __ = plt.hist(
    T0(th_star, norm_scores, 0),
    bins=bins,
    label="norm",
    density=True,
    cumulative=True,
    histtype="step",
    color="blue",
    linestyle="-",
)
cdf0t05, _, __ = plt.hist(
    T0(th_star, norm_scores, 0.5),
    bins=bins,
    label="norm",
    density=True,
    cumulative=True,
    histtype="step",
    color="blue",
    linestyle="-.",
)
cdf0t1, _, __ = plt.hist(
    T0(th_star, norm_scores, 1),
    bins=bins,
    label="norm",
    density=True,
    cumulative=True,
    histtype="step",
    color="blue",
    linestyle="--",
)

cdf1t0, _, __ = plt.hist(
    T1(th_star, anom_scores, 0),
    bins=bins,
    label="anom",
    density=True,
    cumulative=True,
    histtype="step",
    color="red",
    linestyle="-",
)
cdf1t05, _, __ = plt.hist(
    T1(th_star, anom_scores, 0.5),
    bins=bins,
    label="anom",
    density=True,
    cumulative=True,
    histtype="step",
    color="red",
    linestyle="-.",
)
cdf1t1, _, __ = plt.hist(
    T1(th_star, anom_scores, 1),
    bins=bins,
    label="anom",
    density=True,
    cumulative=True,
    histtype="step",
    color="red",
    linestyle="--",
)

plt.axvline(th_star, color="k", linestyle="--", label="th*")

# %%

metrics = []

for t in np.linspace(0, 1, 5):
    asmaps_t = asmaps.clone()
    asmaps_t[masks == 0] = T0(th_star, asmaps[masks == 0], t)
    asmaps_t[masks == 1] = T1(th_star, asmaps[masks == 1], t)

    img_metrics, pix_set_metrics, times, aupimos = evaluate(
        None, ascores, imgclass, asmaps_t, masks, savedir=None, logger=None, debug=False
    )
    metrics.append(
        {
            "t": t,
            "img_metrics": img_metrics,
            "pix_set_metrics": pix_set_metrics,
            "times": times,
            "aupimos": aupimos,
        }
    )


# %%

ts = [m["t"] for m in metrics]
aupros = [m["pix_set_metrics"]["pix_set_aupro"] for m in metrics]
aurocs = [m["pix_set_metrics"]["pix_set_auroc"] for m in metrics]
avgaupimos = [m["aupimos"].nanmean().item() for m in metrics]

plt.plot(ts, aupros, label="aupro")
plt.plot(ts, aurocs, label="auroc")
plt.plot(ts, avgaupimos, label="avgaupimo")
plt.legend()

# %%

df = pd.DataFrame(
    {
        "t": ts,
        "aupro": aupros,
        "auroc": aurocs,
        "avgaupimo": avgaupimos,
    }
)
df.plot.line(x="t", y=["aupro", "auroc", "avgaupimo"])
import seaborn as sns

sns.pairplot(df, vars=["aupro", "auroc", "avgaupimo"])
# %%
aurocs
