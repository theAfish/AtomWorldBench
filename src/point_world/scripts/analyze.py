import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ========== Argument Parsing ==========
parser = argparse.ArgumentParser(description="Analyze evaluation results.")
parser.add_argument("-m", "--model_name", type=str, required=True, help="Name of the model")
parser.add_argument("-a", "--action_name", type=str, required=True, help="Name of the action")
args = parser.parse_args()

# ========== Setup ==========
base_folder = "results/PointWorld"
model_name = args.model_name
action_name = args.action_name
results_folder = f"{base_folder}/{model_name}/{action_name}"

# Find the latest datetime subfolder under the model/action folder
subfolders = [d for d in os.listdir(results_folder) if os.path.isdir(os.path.join(results_folder, d))]
if subfolders:
    latest_datetime_subfolder = sorted(subfolders)[-1]
    results_folder = os.path.join(results_folder, latest_datetime_subfolder)

print(f"Analysing folder: {results_folder}")
results_file = os.path.join(results_folder, f"{action_name}_evaluation_results.csv")
wrongs_file = os.path.join(results_folder, f"{action_name}_evaluation_wrongs.csv")

# ========== Load Data ==========
df_results = pd.read_csv(results_file, sep=',')
try:
    df_errs = pd.read_csv(wrongs_file, sep=',')
except pd.errors.EmptyDataError:
    df_errs = pd.DataFrame()

# ========== Statistics ==========
tolerance = 0.5
# if max_error is larger than tolerance, consider it an error
df_results['max_error'] = df_results['max_error'].astype(float)
num_larger_than_tolerance = df_results[df_results['max_error'] > tolerance].shape[0]
df_results = df_results[df_results['max_error'] <= tolerance]

stats = df_results['max_error'].describe()
err_rate = len(df_errs) / (len(df_results) + len(df_errs))

# ========== Pretty Summary Text ==========
summary_lines = [
    f"{len(df_results)} processable output over {len(df_results) + len(df_errs) + num_larger_than_tolerance} results",
    f"{'Error rate':<10}: {err_rate:>10.2%}",
    f"{'Mean':<10}: {stats['mean']:>10.4f}",
    f"{'Std dev':<10}: {stats['std']:>10.4f}",
    f"{'Min':<10}: {stats['min']:>10.4f}",
    f"{'Max':<10}: {stats['max']:>10.4f}"
]
summary_text = "\n".join(summary_lines)

# ========== Style Settings ==========
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

# ========== Histogram ==========
plt.figure(figsize=(10, 5))
sns.histplot(df_results['max_error'], bins=80, kde=False, color="skyblue", edgecolor="black")
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.xlabel("max_dist", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.title(f"{model_name} - {action_name} - max_dist Histogram", fontsize=14, weight='bold')
plt.text(
    0.98, 0.98, summary_text,
    ha='right', va='top',
    transform=plt.gca().transAxes,
    fontsize=11,
    family='monospace',
    bbox=dict(facecolor='white', alpha=0.85, boxstyle='round,pad=0.4')
)

plt.tight_layout()
plt.savefig(f"{results_folder}/{model_name}-{action_name}-max_dist.png", dpi=300, bbox_inches='tight')
plt.show()

# # ========== Boxplot ==========
# plt.figure(figsize=(6, 3))
# sns.boxplot(x=df_results['max_error'], color="lightcoral")
# plt.title(f"{model_name} - {action_name} - max_dist Boxplot", fontsize=14)
# plt.xlabel("max_dist")
# plt.tight_layout()
# plt.show()
#
#
# sns.stripplot(x=df_results['max_error'], color='black', jitter=True, size=2)
# plt.show()