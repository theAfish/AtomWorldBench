import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Dict

def plot_metrics_distribution(results: List[Dict], output_folder: str):
    """
    Plots the distribution of RMSD and Max Diff metrics for correct results.
    
    Args:
        results: List of result dictionaries.
        output_folder: Folder to save the plots.
    """
    # Ensure output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    # Filter correct results
    valid_results = [r for r in results if r.get('correct', False)]
    
    if not valid_results:
        print("No valid results to plot.")
        return

    # Extract metrics
    data = []
    for res in valid_results:
        # Handle potential missing keys gracefully
        rmsd = res.get('rmsd')
        max_dist = res.get('max_dist')
            
        data.append({
            'RMSD': rmsd,
            'Max Dist': max_dist
        })
    
    df = pd.DataFrame(data)
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    # Prepare plots configuration
    plots_config = []
    if 'RMSD' in df.columns and not df['RMSD'].isnull().all():
        plots_config.append({
            'column': 'RMSD',
            'color': '#3498db',
            'title': 'RMSD Distribution',
            'xlabel': 'RMSD (Å)'
        })
        
    if 'Max Dist' in df.columns and not df['Max Dist'].isnull().all():
        plots_config.append({
            'column': 'Max Dist',
            'color': '#e74c3c',
            'title': 'Max Distance Distribution',
            'xlabel': 'Max Distance (Å)'
        })

    if plots_config:
        num_plots = len(plots_config)
        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots), dpi=300)
        
        # Ensure axes is iterable even if there's only one plot
        if num_plots == 1:
            axes = [axes]
            
        for ax, config in zip(axes, plots_config):
            sns.histplot(data=df, x=config['column'], kde=True, kde_kws={"bw_adjust": 0.02, "gridsize": 1000}, color=config['color'], bins=60, ax=ax)
            ax.set_title(config['title'])
            ax.set_xlabel(config['xlabel'])
            ax.set_ylabel('Count')
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'metrics_distribution.png'))
        plt.close()
