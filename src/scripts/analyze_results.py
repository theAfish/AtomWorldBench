import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os 


# Replace with your actual action name
folder = "results/deepseek_chat/add_atom_action"
action_name = "add_atom_action"
results_file = os.path.join(folder, f"{action_name}_evaluation_results.csv")
wrongs_file = os.path.join(folder, f"{action_name}_evaluation_wrongs.csv")


# Load data
df_results = pd.read_csv(results_file, sep=',')
df_wrongs = pd.read_csv(wrongs_file, sep=',')

# Basic statistics for max_diff
max_diff_stats = df_results['max_diff'].describe()
print("max_diff statistics:\n", max_diff_stats)

# Wrong rate
total = len(df_results) + len(df_wrongs)
wrong_rate = len(df_wrongs) / total if total > 0 else 0
print(f"Wrong rate: {wrong_rate:.2%} ({len(df_wrongs)}/{total})")

# Histogram for max_diff
plt.figure(figsize=(10, 4))
sns.histplot(df_results['max_diff'], bins=30, kde=True)
plt.title(f'{action_name} - max_diff Histogram')
plt.xlabel('max_diff')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Boxplot for max_diff
plt.figure(figsize=(6, 4))
sns.boxplot(x=df_results['max_diff'])
plt.title(f'{action_name} - max_diff Boxplot')
plt.xlabel('max_diff')
plt.tight_layout()
plt.show()