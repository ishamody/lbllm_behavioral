import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

# Folder containing participant CSVs
folder = Path(".")

files = sorted(folder.glob("*.csv"))

# Store participants by list
lists = {i: [] for i in range(1, 9)}

# -----------------------------
# Read participant files
# -----------------------------
for file in files:
    m = re.search(r"list(\d+)", file.stem)
    if m is None:
        continue
    list_num = int(m.group(1))
    df = pd.read_csv(file)
    # Keep only rating trials
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

# ------------------------------------
# Compute split-half averages
# ------------------------------------

all_points = []
for list_num in range(1, 9):
    if len(lists[list_num]) == 0:
        continue
    ratings = pd.concat(lists[list_num], axis=1)
    # Ensure same sentence order
    ratings = ratings.sort_index()
    participants = list(ratings.columns)
    # Split into first half and second half
    half1 = participants[:len(participants)//2]
    half2 = participants[len(participants)//2:]
    mean1 = ratings[half1].mean(axis=1)
    mean2 = ratings[half2].mean(axis=1)
    temp = pd.DataFrame({
        "list": list_num,
        "sentence_id": ratings.index,
        "half1": mean1,
        "half2": mean2
    })
    all_points.append(temp)
# Combine all lists (≈800 rows)
all_points = pd.concat(all_points, ignore_index=True)
print(all_points.head())
print(f"\nTotal points: {len(all_points)}")

# ------------------------------------
# Correlation
# ------------------------------------
r = all_points["half1"].corr(all_points["half2"])
print(f"\nSplit-half correlation = {r:.3f}")
# ------------------------------------
# Regression line
# ------------------------------------
m, b = np.polyfit(all_points["half1"], all_points["half2"], 1)
x = np.linspace(1, 7, 100)
y = m*x + b

# ------------------------------------
# Plot
# ------------------------------------
plt.figure(figsize=(8,8))
plt.scatter(
    all_points["half1"],
    all_points["half2"],
    alpha=0.5,
    s=20
)
plt.plot(
    x,
    y,
    linewidth=3,
    label=f"Best fit (r = {r:.3f})"
)

# Identity line (perfect agreement)
plt.plot(
    [1,7],
    [1,7],
    "--",
    linewidth=1,
    alpha=0.7,
    label="Perfect agreement"
)

plt.xlabel("Average rating (Half 1)")
plt.ylabel("Average rating (Half 2)")
plt.title("Split-half reliability across all lists (Plausibility)")

plt.xlim(1,7)
plt.ylim(1,7)

plt.legend()
plt.tight_layout()
plt.show()