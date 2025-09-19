"""
Plot hardcoded times it takes to achieve certain percentiles of optimality for the small network scenario versus number of intents.

Saves a single figure with 4 series (10%, 25%, 30%, 50%) to ./results/hardcoded_percentiles/time_until_percentiles.png
"""

import os

import matplotlib.pyplot as plt


def plot_hardcoded_times() -> None:
    """Plot the provided hardcoded data using the same style as plot_results.py."""

    # X-axis: number of intents
    num_intents = [20, 30, 40, 50, 60, 70]

    # Times are in seconds; convert to minutes for plotting
    time_until_10pct_sec = [15, 67, 4033, 29729, 90019, 223590]
    time_until_25pct_sec = [15, 46, 1636, 4910, 18090, 41368]
    time_until_30pct_sec = [15, 46, 1093, 4910, 15211, 35667]
    time_until_50pct_sec = [15, 30, 151, 778, 5678, 4514]

    time_until_10pct_min = [v / 60.0 for v in time_until_10pct_sec]
    time_until_25pct_min = [v / 60.0 for v in time_until_25pct_sec]
    time_until_30pct_min = [v / 60.0 for v in time_until_30pct_sec]
    time_until_50pct_min = [v / 60.0 for v in time_until_50pct_sec]

    # Styling similar to plot_results.py
    fontsize = 24
    linewidth = 3

    colors = {
        "10%": "mediumaquamarine",
        "25%": "violet",
        "30%": "tomato",
        "50%": "steelblue",
    }

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    ax.plot(num_intents, time_until_10pct_min, marker="o", ls="-", linewidth=linewidth, color=colors["10%"], label="10%")
    ax.plot(num_intents, time_until_25pct_min, marker="o", ls="-", linewidth=linewidth, color=colors["25%"], label="25%")
    # ax.plot(num_intents, time_until_30pct_min, marker="o", ls="-", linewidth=linewidth, color=colors["30%"], label="30%")
    ax.plot(num_intents, time_until_50pct_min, marker="o", ls="-", linewidth=linewidth, color=colors["50%"], label="50%")

    ax.set_xlabel("Number of flight intents", family="serif", fontsize=fontsize)
    ax.set_ylabel("Computational time (min)", family="serif", fontsize=fontsize, labelpad=17)
    ax.set_xticks(num_intents)
    ax.tick_params(axis="both", labelsize=fontsize)
    ax.legend(fontsize=fontsize)

    fig.tight_layout(pad=1.5)

    out_dir = "./results/hardcoded_percentiles"
    os.makedirs(out_dir, exist_ok=True)

    fig.savefig(os.path.join(out_dir, "time_until_percentiles_medium.pdf"), format="pdf", bbox_inches="tight", pad_inches=0.4)
    
    print("Saved plot to ./results/hardcoded_percentiles/time_until_percentiles_medium.png")


if __name__ == "__main__":
    plot_hardcoded_times()

# ================= END OF FILE =================


