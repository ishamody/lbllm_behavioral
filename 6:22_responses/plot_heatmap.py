import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path

folder = Path(".")

all_data = []

for file in folder.glob("*.csv"):
    df = pd.read_csv(file)

    df["sentence_id"] = df["sentence_id"].astype(str)
    df = df[~df["sentence_id"].str.contains("attention_check", na=False)]

    df["sentence_id"] = pd.to_numeric(df["sentence_id"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["sentence_id", "rating"])

    df["participant"] = file.stem

    all_data.append(df[["sentence_id", "rating", "participant"]])

data = pd.concat(all_data, ignore_index=True)

matrix = data.pivot_table(
    index="participant",
    columns="sentence_id",
    values="rating",
    aggfunc="mean"
)

matrix = matrix.reindex(sorted(matrix.columns), axis=1)

# convert to numpy for pcolormesh
Z = matrix.values

# discrete colormap (1–7)
colors = [
    "#440154", "#3b528b", "#21918c",
    "#5ec962", "#fde725", "#fdae61", "#d7191c"
]

cmap = mcolors.ListedColormap(colors)
bounds = [0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

fig = plt.figure(figsize=(22, 7))

ax = fig.add_axes([0.05, 0.12, 0.88, 0.80])

# --- MAIN HEATMAP WITH EDGES ---
mesh = ax.pcolormesh(
    Z,
    cmap=cmap,
    norm=norm,
    edgecolors="white",   #this creates borders
    linewidth=0.5
)

ax.set_xticks(np.arange(len(matrix.columns)) + 0.5)
ax.set_xticklabels(matrix.columns.astype(int), rotation=90, fontsize=6)

ax.set_yticks(np.arange(len(matrix.index)) + 0.5)
ax.set_yticklabels(matrix.index)

ax.set_xlabel("Sentence ID")
ax.set_ylabel("Participant")
ax.set_title("Ratings Heatmap (All Participants)")

# colorbar
cbar_ax = fig.add_axes([0.94, 0.12, 0.02, 0.80])
cbar = fig.colorbar(mesh, cax=cbar_ax, ticks=[1,2,3,4,5,6,7])
cbar.set_label("Rating")

plt.show()
