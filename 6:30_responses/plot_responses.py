import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.patches as mpatches

folder = Path(".")
output_folder = folder / "plots"
output_folder.mkdir(exist_ok=True)

# discrete 7-color map (viridis-like)
color_map = {
    1: "#440154",
    2: "#3b528b",
    3: "#21918c",
    4: "#5ec962",
    5: "#fde725",
    6: "#fdae61",
    7: "#d7191c"
}

# legend handles
legend_patches = [
    mpatches.Patch(color=color_map[i], label=str(i)) for i in range(1, 8)
]

for file in folder.glob("*.csv"):
    df = pd.read_csv(file)

    df["sentence_id"] = df["sentence_id"].astype(str)
    df = df[~df["sentence_id"].str.contains("attention_check", na=False)]

    df["sentence_id"] = pd.to_numeric(df["sentence_id"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["sentence_id", "rating"])

    df = df.groupby("sentence_id", as_index=False)["rating"].mean()

    df["sentence_id"] = df["sentence_id"].astype(int).astype(str)
    df = df.sort_values(by="sentence_id", key=lambda x: x.astype(int))

    df["rating_round"] = df["rating"].round().astype(int)
    colors = df["rating_round"].map(color_map)

    fig, ax = plt.subplots(figsize=(18, 5))

    ax.bar(df["sentence_id"], df["rating"], color=colors)

    ax.set_title(file.stem)
    ax.set_xlabel("Sentence ID")
    ax.set_ylabel("Rating")
    ax.set_ylim(0.5, 7.5)

    # REMOVE SIDE PADDING
    ax.margins(x=0)

    ax.tick_params(axis='x', labelrotation=90, labelsize=6)

    # LEGEND (discrete rating colors)
    ax.legend(handles=legend_patches, title="Rating", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()

    plt.savefig(output_folder / f"{file.stem}.png", dpi=300, bbox_inches="tight")
    plt.close()

print(f"Saved plots to: {output_folder}")