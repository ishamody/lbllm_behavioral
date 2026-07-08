import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

# Folder containing participant CSVs
folder = Path(".")
files = sorted(folder.glob("*.csv"))

# Group participant dataframes by list
lists = {i: [] for i in range(1, 9)}
failed_attention = []

for file in files:

    m = re.search(r"list(\d+)", file.stem)
    if m is None:
        continue

    list_num = int(m.group(1))
    df = pd.read_csv(file)

    # Inspect attention checks
    attention = df[df["bucket"].astype(str).str.strip().str.lower() == "attention_check"].copy()

    if not attention.empty:
        attention["rating"] = pd.to_numeric(attention["rating"], errors="coerce")
        attention["correct_response"] = pd.to_numeric(attention["correct_response"], errors="coerce")
        print(attention[["sentence_id", "rating", "correct_response"]])
        passed = (attention["rating"] == attention["correct_response"]).all()
        print("Passed:", passed)
        if not passed:
            failed_attention.append(file.name)
            continue

    print(f"Rows before removing attention checks and top rows removed: ", len(df))

    # Keep only rating trials
    df = df[df["rating"].notna()].copy()
    df["sentence_id"] = pd.to_numeric(df["sentence_id"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["sentence_id", "rating"])

    print(f"Rows after removing attention checks and top rows removed: ", len(df))

    participant = file.stem.split("_")[-1]

    df = (
        df[["sentence_id", "rating"]]
        .sort_values("sentence_id")
        .rename(columns={"rating": participant})
        .set_index("sentence_id")
    )

    lists[list_num].append(df)

print("files that fail attention checks: ", failed_attention)

fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)

for list_num, ax in zip(range(1, 9), axes.flat):
    if len(lists[list_num]) == 0:
        ax.set_visible(False)
        continue
    # rows = sentences, cols = participants
    ratings = pd.concat(lists[list_num], axis=1)

    r_values = []

    for participant in ratings.columns:
        participant_ratings = ratings[participant]
        others_mean = ratings.drop(columns=participant).mean(axis=1)
        r = participant_ratings.corr(others_mean)
        r_values.append(r)

    ax.hist(r_values, bins=np.linspace(-1, 1, 11), edgecolor="black")

    ax.set_title(f"List {list_num}")
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Correlation (r)")
    ax.set_ylabel("Participants")

    print(f"List {list_num}:")
    print(np.round(r_values, 3))
    print(f"Mean r = {np.mean(r_values):.3f}\n")

plt.tight_layout()
plt.show()