# Use the DVC api for loading the YAML parameters and results
import dvc.api
# Loss and curve fitting data as numpy arrays
import numpy as np
# Pandas for handling experiment results as data frames
import pandas as pd

# For plotting of loss and fitted curves
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# High-level interface to matplotlib plotting
import seaborn as sns

# Aggregate statistics to collect when summarizing results
AGGREGATE = ["mean", "median", "std", "min", "max"]

# Configuration of plotting accuracy vs. quantization steps
ACC_VS_QUANT_PLOT = {"x": "model.bits", "y": "top-1", "errorbar": "sd"}

# Cutoff number of quantization steps, sharpness factor alpha and accuracy to
# skip over the boring and/or messy parts...
CUTOFF_ACCURACY = 0.35

# Maximum accuracy drop accepted
ACC_DROP = 0.02


# Summarizes and plot the results aggregated from the DVC experiments
def summarize(model, tag=None, repo=None, revs=None):
    # Load the DVC-tracked experiment results into a pandas data frame
    results = pd.DataFrame(dvc.api.exp_show(repo=repo, revs=revs))

    # Throw away the current workspace and any baselines, i.e., the parent
    # branches/commits of a set of experiments
    results = results[results["rev"] != "workspace"]
    results = results[results["typ"] != "baseline"]

    # Lift all "<model>/params.yaml:"-prefixed parameters to the top level
    for key in results.columns:
        if key.startswith(f"{model}/params.yaml:"):
            results[key.removeprefix(f"{model}/params.yaml:")] = results[key]

    # Lift all "outputs/<model>>/accuracy.yaml:"-prefixed to the top level
    for key in results.columns:
        if key.startswith(f"outputs/{model}/accuracy.yaml:"):
            results[key.removeprefix(f"outputs/{model}/accuracy.yaml:")] = \
                results[key]

    # Optionally filter by the experiment tag
    if tag is not None:
        results = results[results["tag"] == tag]

    # Filter out seed 12 which is not used by experiments
    results = results[results["seed"] != 12]

    # Select baseline and quantization experiments
    baseline = results[results["tag"] == "baseline"]
    quantized = results[results["tag"] == "quantized"]

    # If there is no best top-1 accuracy (if there are no baseline experiments),
    # assume 70% top-1 accuracy (reasonable according to previous experiments)
    best_top_1 = 0.70

    # If there are baseline experiments, summarize by aggregating over the
    # random seeds
    if not baseline.empty:
        # Summarize the results by aggregating over the random seeds and sorting
        # by average top-1 accuracy
        baseline = baseline[["top-1", "top-5"]].mean()
        # Print the resulting data as markdown
        print(baseline.to_markdown())
        # Remember the best performing model top-1 accuracy as a reference for
        # the other experiments
        best_top_1 = np.round(baseline["top-1"].item(), decimals=2)

    # Create a figure with a single subplot
    fig, ax = plt.subplots(ncols=1)

    # 500x300 pixel image (when saved as PNG)
    fig.set_figheight(3)
    fig.set_figwidth(5)

    # Collect all unique remaining quantization bits after filtering to label
    # the shared x-axis
    bits = np.unique(quantized["model.bits"])

    # Plotting top-1 accuracy [%] vs. number of quantization bits
    ax.set_ylabel("Top-1 Accuracy")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0, 1))
    ax.set_xlabel("Quantization Bit-Width")
    ax.xaxis.set_major_locator(ticker.FixedLocator(bits.tolist()))

    # Lowest acceptable accuracy
    min_top_1 = best_top_1 - ACC_DROP

    # Add a line marking the -X% drop of top-1 accuracy compared to the best
    # baseline model from above
    ax.axhline(min_top_1, ls='dotted', color="black")
    ax.text(
        8, best_top_1 - ACC_DROP - 0.02,  # -0.01 to not occlude the lines
        f"Baseline -{ACC_DROP:.1%}\n{min_top_1:.1%}",
        ha="center", va="center",
        backgroundcolor="white",

    )

    # If there are quantization experiments, summarize by grouping by model
    # configuration, aggregating over the random seeds and plotting accuracy vs.
    # number of quantization steps per configuration group
    if not quantized.empty:
        # Sort by activation function (grouping) to have same color ordering for
        # all plots
        quantized = quantized.sort_values(["model.activation"])
        # Plot the top-1 accuracy vs. quantization steps grouped by model
        # activation function aggregating over seeds to add standard deviation
        sns.lineplot(quantized, **ACC_VS_QUANT_PLOT, color="red", ax=ax)

        # Hyperparameters varied across the quantization experiments
        QUANT_PARAMS = ["model.bits"]

        # Group by hyperparameters and select only the task specific metrics
        quantized = quantized.groupby(QUANT_PARAMS)[["top-1"]]
        # Summarize the results by aggregating over the random seeds and sorting
        # by average top-1 accuracy
        quantized = quantized.mean().reset_index().sort_values(["top-1"])
        # Print (bits, top-1) selection sorted by accuracy
        print(quantized.to_markdown())

        # Remove all rows below the -X% accuracy drop threshold
        quantized = quantized[quantized["top-1"] >= min_top_1]
        # Find the last quantization steps before falling below the -X% accuracy
        # drop threshold
        _min = quantized["model.bits"].min()

        # Plot vertical line marking the best quantization achieved
        ax.axvline(_min.min(), ls='dotted', color="black", zorder=0)

    # Tighten the layout, save as scalable vector graphics and show the result
    plt.tight_layout()
    plt.savefig(f"outputs/{model}/summary.svg")
    plt.show()


# Script entrypoint
if __name__ == "__main__":
    summarize(model="vision", revs=["19680d2"])
    summarize(model="radioml", revs=["19680d2"])
