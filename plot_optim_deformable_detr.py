import json

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

with open("benchmark_results_deformable_detr_3.json", "r") as file:
    data = json.load(file)

data_filtered = {}
for model, versions in data.items():
    data_filtered[model] = {
        version: perf["average_fps"]
        for version, perf in versions.items()
        if "best" not in version
    }

# compare optim versions of rt_detr against non-optim versions
data_rt_detr = data_filtered["deformable_detr"]
data_optim_rt_detr = data_filtered["optim_deformable_detr"]
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
        plot_data.append(
            {
                "Library": library,
                "Version": version.replace("_", " ").capitalize(),
                "FPS": fps,
            }
        )

# Create a DataFrame for seaborn
df = pd.DataFrame(plot_data)

# Set the aesthetic style of the plots
sns.set_theme(style="whitegrid")
# Create the bar plot with seaborn
plt.figure(figsize=(12, 8))
ax = sns.barplot(x="Version", y="FPS", hue="Library", data=df, palette="viridis")
# Adding the value labels on top of the bars
for container in ax.containers:
    ax.bar_label(container, fmt="%.1f", label_type="edge", padding=2)

# Add labels and title
plt.xlabel("Compiled models use the mode 'reduce-overhead'")
plt.ylabel("Average FPS")
plt.title(
    "Comparison of average FPS between Deformable DETR and Optim Deformable DETR on a NVIDIA A10 GPU\n using 'SenseTime/deformable-detr' checkpoint."
)
plt.xticks(rotation=45)
plt.legend(title="Library")
plt.tight_layout()
plt.savefig("benchmark_results_deformable_detr_3.png", dpi=300)

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
    ax.bar_label(container, fmt="%.1f", label_type="edge", padding=2)

# Add labels and title
plt.xlabel("Compiled models were use the mode 'reduce-overhead'")
plt.ylabel("Relative Speedup")
plt.title(
    "Comparison of relative speedup between Deformable DETR and Optim Deformable DETR on a NVIDIA A10 GPU\n using 'SenseTime/deformable-detr' checkpoint."
)
plt.xticks(rotation=45)
plt.legend(title="Library")
plt.tight_layout()
plt.savefig("benchmark_results_relative_deformable_detr_3.png", dpi=300)
