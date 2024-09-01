import json

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

with open("benchmark_results.json", "r") as file:
    data = json.load(file)

data_filtered = {}
for model, versions in data.items():
    data_filtered[model] = {
        version: perf["average_fps_sequential"] for version, perf in versions.items()
    }

data = data_filtered
# Convert the data into a format suitable for seaborn
plot_data = []
for model, versions in data.items():
    for version, fps in versions.items():
        plot_data.append({"Model": model, "Version": version, "FPS": fps})

# Create a DataFrame for seaborn
df = pd.DataFrame(plot_data)

# Set the aesthetic style of the plots
sns.set_theme(style="whitegrid")
# Create the bar plot with seaborn
plt.figure(figsize=(12, 8))
ax = sns.barplot(x="Model", y="FPS", hue="Version", data=df, palette="viridis")
# Adding the value labels on top of the bars
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", label_type="edge", padding=2)

# Add labels and title
plt.xlabel("Models")
plt.ylabel("Average FPS (Sequential)")
plt.title("Comparison of Average FPS (Sequential) Across Models and Versions")
plt.xticks(rotation=45)
plt.legend(title="Versions")

plt.tight_layout()
plt.show()
plt.savefig("benchmark_results.png", dpi=300)

# same but with relative speedup compared to eager version
# Create a DataFrame for seaborn
df_relative = df.copy()
df_relative["FPS"] = df_relative.groupby("Model")["FPS"].transform(
    lambda x: x / x.min()
)

# Create the bar plot with seaborn
plt.figure(figsize=(12, 8))
ax = sns.barplot(x="Model", y="FPS", hue="Version", data=df_relative, palette="viridis")
# Adding the value labels on top of the bars
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", label_type="edge", padding=2)

# Add labels and title
plt.xlabel("Models")
plt.ylabel("Relative Speedup")
plt.title("Comparison of Relative Speedup Across Models and Versions")
plt.xticks(rotation=45)
plt.legend(title="Versions")

plt.tight_layout()
plt.show()
plt.savefig("benchmark_results_relative.png", dpi=300)
