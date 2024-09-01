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

# compare optim versions of rt_detr against non-optim versions
data_rt_detr = data_filtered["rt_detr"]
data_optim_rt_detr = data_filtered["optim_rt_detr"]
data_compare = {}
for version, fps in data_rt_detr.items():
    data_compare[version] = {
        "Transformers": fps,
        "Optim": data_optim_rt_detr[version],
    }


data = data_compare
# Convert the data into a format suitable for seaborn
plot_data = []
for version, libraries in data.items():
    for library, fps in libraries.items():
        plot_data.append({"Library": library, "Version": version, "FPS": fps})

# Create a DataFrame for seaborn
df = pd.DataFrame(plot_data)

# Set the aesthetic style of the plots
sns.set_theme(style="whitegrid")
# Create the bar plot with seaborn
plt.figure(figsize=(12, 8))
ax = sns.barplot(x="Version", y="FPS", hue="Library", data=df, palette="viridis")
# Adding the value labels on top of the bars
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", label_type="edge", padding=2)

# Add labels and title
plt.xlabel("Versions")
plt.ylabel("Average FPS (Sequential)")
plt.title(
    "Comparison of Average FPS (Sequential) Across Versions between RT-DETR and Optim RT-DETR"
)
plt.xticks(rotation=45)
plt.legend(title="Library")
plt.tight_layout()
plt.savefig("benchmark_results_rt_detr.png", dpi=300)

# same but with relative speedup compared to eager version
# Create a DataFrame for seaborn
df_relative = df.copy()
df_relative["FPS"] = df_relative.groupby("Version")["FPS"].transform(
    lambda x: x / x.min()
)

# Create the bar plot with seaborn
plt.figure(figsize=(12, 8))
ax = sns.barplot(
    x="Version", y="FPS", hue="Library", data=df_relative, palette="viridis"
)
# Adding the value labels on top of the bars
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", label_type="edge", padding=2)

# Add labels and title
plt.xlabel("Versions")
plt.ylabel("Relative Speedup")
plt.title(
    "Comparison of Relative Speedup Across Versions between RT-DETR and Optim RT-DETR"
)
plt.xticks(rotation=45)
plt.legend(title="Library")
plt.tight_layout()
plt.savefig("benchmark_results_relative_rt_detr.png", dpi=300)


# Same comparison but for deformable detr
data_deformable_detr = data_filtered["deformable_detr"]
data_optim_deformable_detr = data_filtered["optim_deformable_detr"]
data_compare = {}
for version, fps in data_deformable_detr.items():
    data_compare[version] = {
        "Transformers": fps,
        "Optim": data_optim_deformable_detr[version],
    }

data = data_compare
# Convert the data into a format suitable for seaborn
plot_data = []
for version, libraries in data.items():
    for library, fps in libraries.items():
        plot_data.append({"Library": library, "Version": version, "FPS": fps})

# Create a DataFrame for seaborn
df = pd.DataFrame(plot_data)

# Set the aesthetic style of the plots
sns.set_theme(style="whitegrid")
# Create the bar plot with seaborn
plt.figure(figsize=(12, 8))
ax = sns.barplot(x="Version", y="FPS", hue="Library", data=df, palette="viridis")
# Adding the value labels on top of the bars
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", label_type="edge", padding=2)

# Add labels and title
plt.xlabel("Versions")
plt.ylabel("Average FPS (Sequential)")
plt.title(
    "Comparison of Average FPS (Sequential) Across Versions between Deformable DETR and Optim Deformable DETR"
)

plt.xticks(rotation=45)
plt.legend(title="Library")
plt.tight_layout()
plt.savefig("benchmark_results_deformable_detr.png", dpi=300)

# same but with relative speedup compared to eager version
# Create a DataFrame for seaborn
df_relative = df.copy()
df_relative["FPS"] = df_relative.groupby("Version")["FPS"].transform(
    lambda x: x / x.min()
)

# Create the bar plot with seaborn
plt.figure(figsize=(12, 8))
ax = sns.barplot(
    x="Version", y="FPS", hue="Library", data=df_relative, palette="viridis"
)
# Adding the value labels on top of the bars
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", label_type="edge", padding=2)

# Add labels and title
plt.xlabel("Versions")
plt.ylabel("Relative Speedup")
plt.title(
    "Comparison of Relative Speedup Across Versions between Deformable DETR and Optim Deformable DETR"
)
plt.xticks(rotation=45)
plt.legend(title="Library")
plt.tight_layout()
plt.savefig("benchmark_results_relative_deformable_detr.png", dpi=300)
