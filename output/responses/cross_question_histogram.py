import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

# Parent directory containing q1, q2, ..., q5
base_folder = Path(".")

questions = ["q1", "q2", "q3", "q4", "q5"]

fig, axes = plt.subplots(1, 5, figsize=(22, 4), sharex=True, sharey=True)

for ax, question in zip(axes, questions):

    folder = base_folder / question

    if not folder.exists():
        print(f"{folder} not found.")
        ax.set_visible(False)
        continue

    files = sorted(folder.glob("*.csv"))

    # store participant dfs by list
    lists = {i: [] for i in range(1, 9)}

    for file in files:

        m = re.search(r"list(\d+)", file.stem)
        if m is None:
            continue

        list_num = int(m.group(1))

        df = pd.read_csv(file)

        # keep only rating trials
        df = df[df["rating"].notna()].copy()

        df["sentence_id"] = pd.to_numeric(df["sentence_id"], errors="coerce")
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

        df = df.dropna(subset=["sentence_id", "rating"])

        participant = file.stem.split("_")[-1]

        df = (
            df[["sentence_id", "rating"]]
            .sort_values("sentence_id")
            .rename(columns={"rating": participant})
            .set_index("sentence_id")
        )

        lists[list_num].append(df)

    loo_rs = []

    # Compute leave-one-out correlations
    for list_num in range(1, 9):

        if len(lists[list_num]) < 2:
            continue

        ratings = pd.concat(lists[list_num], axis=1)

        for participant in ratings.columns:

            participant_ratings = ratings[participant]

            others_mean = ratings.drop(columns=participant).mean(axis=1)

            r = participant_ratings.corr(others_mean)

            loo_rs.append(r)

    print(f"{question}:")
    print(f"Participants: {len(loo_rs)}")
    print(f"Mean r = {np.mean(loo_rs):.3f}")
    print(f"Median r = {np.median(loo_rs):.3f}")
    print()

    ax.hist(
        loo_rs,
        bins=np.linspace(-1, 1, 21),
        edgecolor="black"
    )

    ax.set_title(question.upper())
    ax.set_xlim(-1, 1)
    ax.set_xlabel("LOO correlation (r)")

axes[0].set_ylabel("Participants")

plt.suptitle("Leave-One-Out Correlation Distributions")
plt.tight_layout()
plt.show()