import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# ==========================================================
# Read participant files
# ==========================================================

folder = Path(".")

files = sorted(folder.glob("*.csv"))
print(f"Found {len(files)} CSV files")

# Dictionary: list number -> list of participant dataframes
lists = {i: [] for i in range(1, 9)}

for file in files:

    match = re.search(r"list(\d+)", file.stem)
    if match is None:
        continue

    list_num = int(match.group(1))

    df = pd.read_csv(file)

    # Keep only rating trials
    df = df[df["rating"].notna()].copy()

    df["sentence_id"] = pd.to_numeric(df["sentence_id"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    df = df.dropna(subset=["sentence_id", "rating"])

    # Keep only needed columns
    df = df[["sentence_id", "rating"]]

    # Sort by sentence ID
    df = df.sort_values("sentence_id")

    participant = f"P{len(lists[list_num]) + 1}"

    df = df.set_index("sentence_id")
    df.columns = [participant]

    lists[list_num].append(df)

# ==========================================================
# Plot
# ==========================================================

# Rating colors: 1 -> 7
colors = [
    "#4E04A2",  # 1 - blue
    "#048DA9",  # 2 - light blue
    "#007558",  # 3 - green
    "#51C106",  # 4 - yellow-green
    "#FFEA00",  # 5 - yellow
    "#F09814",  # 6 - orange
    "#C00303",  # 7 - red
]

cmap = ListedColormap(colors)
norm = BoundaryNorm(np.arange(0.5, 8.5, 1), cmap.N)

fig, axes = plt.subplots(
    4,
    2,
    figsize=(18, 16),
    constrained_layout=True,
)

im = None

for list_num, ax in zip(range(1, 9), axes.flat):

    if len(lists[list_num]) == 0:
        ax.set_visible(False)
        continue

    heatmap = pd.concat(lists[list_num], axis=1).T

    # Make sentence IDs integers
    heatmap.columns = heatmap.columns.astype(int)

    print(f"List {list_num}: {heatmap.shape}")

    im = ax.imshow(
        heatmap,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        aspect="auto",
    )

    # White gridlines between cells
    ax.set_xticks(np.arange(-0.5, heatmap.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, heatmap.shape[0], 1), minor=True)

    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)

    ax.tick_params(which="minor", bottom=False, left=False)

    # Remove white strip at top
    ax.set_ylim(heatmap.shape[0] - 0.5, -0.5)

    # Bold titles
    ax.set_title(
        f"List {list_num}",
        fontsize=14,
        fontweight="bold",
    )

    # Only label axes on first plot
    if list_num == 1:
        ax.set_xlabel("Sentence ID", fontsize=10)
        ax.set_ylabel("Participant", fontsize=10)
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")

    # Participant labels
    ax.set_yticks(np.arange(len(heatmap.index)))
    ax.set_yticklabels(heatmap.index, fontsize=9)

    # Show ~12 evenly spaced ACTUAL sentence IDs
    sentence_ids = heatmap.columns.to_numpy(dtype=int)

    ax.set_xticks(np.arange(len(sentence_ids)))
    ax.set_xticklabels(
        sentence_ids,
        rotation=90,
        fontsize=5,   # decrease if labels overlap
    )

# Thin colorbar outside plots
cbar = fig.colorbar(
    im,
    ax=axes,
    location="right",
    fraction=0.02,
    pad=0.015,
    aspect=50,
    shrink=0.9,
)

cbar.set_ticks(range(1, 8))
cbar.set_ticklabels([str(i) for i in range(1, 8)])
cbar.set_label("Rating", fontsize=10)

plt.show()