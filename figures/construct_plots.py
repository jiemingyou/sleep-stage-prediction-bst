import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from preprocessing.feature_engineering import load_data

cmap = sns.color_palette("hls", 4)
ci = None


def plot1(df, x, y, ax, idx, label=None):
    """
    Default line plot
    """
    sns.lineplot(
        df,
        x=x,
        y=y,
        color=cmap[idx],
        errorbar=ci,
        lw=3,
        ax=ax[idx],
        label=label,
    )


def plot_data(df: pd.DataFrame, uid: str, save=False):
    """
    Create a plot of the data for the given user.

    Args:
        df (pd.DataFrame): Dataframe containing the data.
        uid (str): User id.
    """
    df_plot = df[df["id"] == uid]
    ci = None
    cmap = sns.color_palette("hls", 4)
    fontsize = 12

    fig, ax = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

    plot1(df_plot, "time", "count", ax, 0, "Activity count")
    plot1(df_plot, "time", "hr", ax, 1, "Normalized HR variance")
    plot1(df_plot, "time", "cosine", ax, 2, "Circadian phase")
    plot1(df_plot, "time", "psg", ax, 3, "Labeled sleep stage")

    # Configure axis
    ax[0].set_ylabel("Activity count", fontsize=fontsize)
    ax[1].set_ylabel("HR variance", fontsize=fontsize)
    ax[2].invert_yaxis()
    ax[2].set_ylabel("Relative phase", fontsize=fontsize)
    ax[3].set_ylabel("Sleep stage", fontsize=fontsize)
    ax[3].set_yticks([0, 1, 2, 3, 5])
    ax[3].set_yticklabels(["wake", "N1", "N2", "N3", "REM"])

    for a in ax:
        # Plot horizontal gridlines
        a.grid(axis="x", color="black", alpha=0.3)
        a.set_axisbelow(True)
        a.set_xlabel("Time since sleep")
        a.set_xlim(0, 8)

        # Move legend to top left and remove box
        handles, labels = a.get_legend_handles_labels()
        a.legend(
            handles=handles,
            labels=labels,
            loc="upper left",
            bbox_to_anchor=(0, 1.35),
            ncol=4,
            frameon=False,
        )
        for t in a.legend_.texts:
            t.set_fontsize(fontsize)

        # Increase the tick label size
        for t in a.get_xticklabels() + a.get_yticklabels():
            t.set_fontsize(fontsize)

        # Incrase x label size
        a.xaxis.label.set_size(fontsize)

    sns.despine()
    fig.subplots_adjust(hspace=0.5)
    fig.align_ylabels(ax)

    if save:
        plt.savefig(f"figures/data_{uid}.png", dpi=300, bbox_inches="tight")

    plt.show()


def hr_plot(
    df: pd.DataFrame,
    df_hr_raw: pd.DataFrame,
    uid: str,
    save=False,
):
    """
    Plot the heart rate data for the given user.
    """

    # HR Before and After transformation
    df_hr = df[df["id"] == uid].sort_values("time")["hr"]

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(df_hr_raw, color=cmap[0], lw=1, label="HR (bpm)")
    ax[0].set_xlim(0, 60 * 60 * 8)
    ax[0].set_xticks(range(0, 60 * 60 * 9, 60 * 60))
    ax[0].set_xticklabels([0, 1, 2, 3, 4, 5, 6, 7, 8])

    ax[1].plot(df_hr, color=cmap[0], lw=2, label="HR variance")
    ax[1].set_xlim(0, 2 * 60 * 8)
    ax[1].set_xticks(range(0, 2 * 60 * 9, 2 * 60))
    ax[1].set_xticklabels([0, 1, 2, 3, 4, 5, 6, 7, 8])

    # Bold titles
    ax[0].set_title("Raw heart rate measurements", fontweight="bold", fontsize=14)
    ax[1].set_title("HR variance after transformation", fontweight="bold", fontsize=14)

    ax[0].set_xlabel("Hours since sleep", fontsize=14)
    ax[1].set_xlabel("Hours since sleep", fontsize=14)

    ax[0].set_ylim(40, 120)

    ax[0].legend(loc="upper left", frameon=False, fontsize=14)
    ax[1].legend(loc="upper left", frameon=False, fontsize=14)
    fig.tight_layout()

    if save:
        plt.savefig(f"figures/hr_{uid}.png", dpi=300, bbox_inches="tight")

    plt.show()


def motion_plot(
    df: pd.DataFrame,
    df_motion_raw: pd.DataFrame,
    uid: str,
    save=False,
):
    """
    Plot the motion data for the given user.
    """
    # Motion Before and After transformation
    df_motion = df[df["id"] == uid].sort_values("time")["count"]

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(
        df_motion_raw["time"],
        df_motion_raw[["x", "y", "z"]].pow(2).sum(axis=1).pow(0.5),
        color=cmap[1],
        lw=1,
        label=r"Acceleration $||\vec{a} ||$",
    )

    # Second labels to hours on ax[0]
    ax[0].set_xlim(0, 60 * 60 * 8)
    ax[0].set_xticks(range(0, 60 * 60 * 9, 60 * 60))
    ax[0].set_xticklabels([0, 1, 2, 3, 4, 5, 6, 7, 8])

    ax[1].plot(df_motion, color=cmap[1], lw=2, label="Activity count")
    ax[1].set_xlim(0, 2 * 60 * 8)
    ax[1].set_xticks(range(0, 2 * 60 * 9, 2 * 60))
    ax[1].set_xticklabels([0, 1, 2, 3, 4, 5, 6, 7, 8])

    # Bold titles
    ax[0].set_title("Raw acceleration measurements", fontweight="bold", fontsize=14)
    ax[1].set_title("Activity count feature", fontweight="bold", fontsize=14)

    ax[0].set_xlabel("Hours since sleep", fontsize=14)
    ax[1].set_xlabel("Hours since sleep", fontsize=14)

    ax[0].legend(loc="upper left", frameon=False, fontsize=14)
    ax[1].legend(loc="upper left", frameon=False, fontsize=14)
    fig.tight_layout()

    if save:
        plt.savefig(f"figures/activity_{uid}.png", dpi=300, bbox_inches="tight")

    plt.show()


def circadian_plot(df: pd.DataFrame, uid: str, save=False):
    """
    Plot the circadian data for the given user.
    """
    df_proxy = df.query(f"id == '{uid}'").sort_values("time")
    # Plot clock proxy
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    ax.plot(
        df_proxy["time"],
        df_proxy["cosine"],
        color=cmap[2],
        lw=4,
        label="Relative circadian phase",
    )

    # Bold titles
    ax.set_title("Proxy clock", fontweight="bold", fontsize=14)
    ax.set_xlabel("Hours since sleep", fontsize=14)
    ax.legend(loc="upper right", frameon=False, fontsize=14)

    if save:
        plt.savefig(f"figures/circadian_{uid}.png", dpi=300, bbox_inches="tight")

    plt.show()


def psg_plot(df: pd.DataFrame, uid: str, save=False):
    """
    Plot the sleep stage data for the given user.
    """
    df_proxy = df.query(f"id == '{uid}'").sort_values("time")
    # Plot clock proxy
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    ax.plot(df_proxy["time"], df_proxy["psg"], color=cmap[3], lw=3)

    # Bold titles
    ax.set_title("Sleep stages", fontweight="bold", fontsize=14)
    ax.set_xlabel("Hours since sleep", fontsize=14)
    ax.legend(loc="upper right", frameon=False, fontsize=14)

    ax.set_yticks([0, 1, 2, 3, 5])
    ax.set_yticklabels(["wake", "N1", "N2", "N3", "REM"])

    if save:
        plt.savefig(f"figures/psg_{uid}.png", dpi=300, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    # Load the data
    df = load_data("data/")
    df_hr_raw = pd.read_csv("../data/heart_rate/8692923_heartrate.txt", names=["hr"])
    df_motion_raw = pd.read_csv(
        "../data/motion/8692923_acceleration.txt",
        names=["time", "x", "y", "z"],
        sep=" ",
    ).query("time >= 0")

    # Plot the data
    plot_data(df, "8692923", save=True)
    hr_plot(df, df_hr_raw, "8692923", save=True)
    motion_plot(df, df_motion_raw, "8692923", save=True)
    circadian_plot(df, "8692923", save=True)
    psg_plot(df, "8692923", save=True)
