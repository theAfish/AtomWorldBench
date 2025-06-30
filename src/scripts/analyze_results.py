import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ========== Setup ==========
results_folder = "results"
model_name = "llama3_70b"
action_name = "insert_between_atoms_action"
folder = f"{results_folder}/{model_name}/{action_name}"
results_file = os.path.join(folder, f"{action_name}_evaluation_results.csv")
wrongs_file = os.path.join(folder, f"{action_name}_evaluation_wrongs.csv")

# ========== Load Data ==========
df_results = pd.read_csv(results_file, sep=',')
df_errs = pd.read_csv(wrongs_file, sep=',')

# ========== Statistics ==========
stats = df_results['max_diff'].describe()
err_rate = len(df_errs) / (len(df_results) + len(df_errs))

# ========== Pretty Summary Text ==========
summary_lines = [
    f"{len(df_results)} processable CIF over {len(df_results) + len(df_errs)} results",
    f"{'Error rate':<10}: {err_rate:.2%}",
    f"{'Mean':<10}: {stats['mean']:.4f}",
    f"{'Std dev':<10}: {stats['std']:.4f}",
    f"{'Min':<10}: {stats['min']:.4f}",
    f"{'Max':<10}: {stats['max']:.4f}"
]
summary_text = "\n".join(summary_lines)

# ========== Style Settings ==========
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

# ========== Histogram ==========
plt.figure(figsize=(10, 5))
sns.histplot(df_results['max_diff'], bins=80, kde=False, color="skyblue", edgecolor="black")
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
plt.savefig(f"{results_folder}/{model_name}/{model_name}-{action_name}-max_dist.png", dpi=300, bbox_inches='tight')
plt.show()

# ========== Boxplot ==========
# plt.figure(figsize=(6, 3))
# sns.boxplot(x=df_results['max_diff'], color="lightcoral")
# plt.title(f"{model_name} - {action_name} - max_dist Boxplot", fontsize=14)
# plt.xlabel("max_dist")
# plt.tight_layout()
# plt.show()


# sns.stripplot(x=df_results['max_diff'], color='black', jitter=True, size=2)
# plt.show()

# save some example cif 
index2save = 2
init_cif = df_results["input_cif"][index2save]
gen_cif = df_results["generated_cif"][index2save]
target_cif = df_results["target_cif"][index2save]

with open(f"{results_folder}/{model_name}/example_2.cif", "w") as f:
    f.write(gen_cif)

with open(f"{results_folder}/{model_name}/example_1.cif", "w") as f:
    f.write(target_cif)

with open(f"{results_folder}/{model_name}/example_0.cif", "w") as f:
    f.write(init_cif)