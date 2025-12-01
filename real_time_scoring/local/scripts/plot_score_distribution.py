"""
Generate score distribution plot for the technical document
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

# Set font
rcParams["font.family"] = "Avenir"

# Load trained model results
from pathlib import Path

base_dir = Path(__file__).parent.parent
model_output_dir = base_dir / "model_output"

with open(model_output_dir / "scorecard_metadata.json", "r") as f:
    metadata = json.load(f)

# Load threshold analysis
threshold_df = pd.read_csv(model_output_dir / "threshold_analysis.csv")

import sys
from pathlib import Path

# For visualization, we'll load the training data and recalculate scores
import joblib

sys.path.insert(0, str(Path(__file__).parent))
from local_train import load_data

df = load_data()
X = df.drop("Good_Bad", axis=1)
y = 1 - df["Good_Bad"]

pipeline = joblib.load(model_output_dir / "scorecard_pipeline.joblib")

# Calculate scores
X_transformed = pipeline.named_steps["woe"].transform(X)
logistic = pipeline.named_steps["logistic"]
log_odds = logistic.decision_function(X_transformed)

factor = metadata["factor"]
base_score = metadata["base_score"]
offset = metadata["offset"]
offset = base_score - factor * (-logistic.intercept_[0])
scores = offset - factor * log_odds

# Create visualization dataframe
dataframe_viz = pd.DataFrame({"score": scores, "is_bad": y.values})

# Verify the logic
print("\nVerifying score distribution:")
print(
    f"  Bad risks (is_bad=1): mean score = {dataframe_viz[dataframe_viz['is_bad'] == 1]['score'].mean():.1f}"
)
print(
    f"  Good risks (is_bad=0): mean score = {dataframe_viz[dataframe_viz['is_bad'] == 0]['score'].mean():.1f}"
)

# Determine the range of scores
score_min = dataframe_viz["score"].min()
score_max = dataframe_viz["score"].max()

# Define the number of bins and bin width
num_bins = 30
bin_width = (score_max - score_min) / num_bins

filter_goods = dataframe_viz["is_bad"] == 0
filter_bads = dataframe_viz["is_bad"] == 1

plt.figure(figsize=(10, 5), dpi=300, facecolor="none")
ax = plt.gca()
ax.set_facecolor("none")

# Plot GOOD risk first (BLUE) - lower scores on left
_ = plt.hist(
    dataframe_viz[filter_goods]["score"].sample(
        frac=1.0, replace=True, random_state=42
    ),
    label="Good risk",
    color="#52a1ec",
    edgecolor="#0b62af",
    alpha=0.6,
    bins=np.arange(score_min, score_max + bin_width, bin_width),
)
# Plot BAD risk second (RED) - higher scores on right
_ = plt.hist(
    dataframe_viz[filter_bads]["score"].sample(frac=1.0, replace=True, random_state=42),
    label="Bad risk",
    color="#fe595f",
    edgecolor="#cc212e",
    alpha=0.6,
    bins=np.arange(score_min, score_max + bin_width, bin_width),
)
plt.axvline(
    x=metadata["cutoff"],
    color="fuchsia",
    linewidth=2.5,
    linestyle="--",
    label="Cut-off",
)
plt.legend(fontsize=16, framealpha=0.95, loc="upper left")
plt.title("WOE Logistic Regression Scorecard", fontsize=18, fontweight="bold", pad=15)
plt.xlabel("Credit Score", fontsize=16)
plt.ylabel("Frequency", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(alpha=0.15, linestyle="--")
plt.tight_layout()

# Save with transparent background
plt.savefig(
    "../score_distribution.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="none",
    transparent=True,
)
print("✓ Score distribution plot saved to: ../score_distribution.png")
    transparent=True,
)
print("✓ Score distribution plot saved to: ../score_distribution.png")
    transparent=True,
)
print("✓ Score distribution plot saved to: ../score_distribution.png")
