from plotting_functions import *
from chain_functions import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import math

def main():
    colors = {
    "538-Dem": 'blue',
    "538-GOP": 'red',
    "538-Pro": 'purple'
    }
    for state in ["PA", "MD", "MA", "TX", "NC"]:
        plan_mvs = pd.read_csv(f"outputs/{state}/{state}_mean_var_stats.csv", index_col=0)
        df = pd.read_csv(f"outputs/{state}/{state}_disprop_scores_100000.csv")
        variances, means = find_variances_and_means(df)

        fig, ax = plt.subplots(figsize=(10,10))
        plt.scatter(means, variances, color='gray', s=20)
        
        yellow_colors = ['darkorange', 'goldenrod', 'gold', 'yellow']
        for plan in plan_mvs.index:
            nice_name = plan_names[state][plan]
            if nice_name in colors:
                color = colors[nice_name]
            else:
                color = random.choice(yellow_colors)
                yellow_colors.remove(color)
            mean = plan_mvs.loc[plan]['disprop_mean']
            var = plan_mvs.loc[plan]['disprop_var']
            plt.scatter(mean, var, label=nice_name, color=color, s=100)

        ax.set_xlim(-0.35, 0.35)
        ax.set_ylim(-0.001, 0.1)
        plt.axvline(0, color='gray', alpha=0.5)
        _ = plt.legend()
        plt.savefig(f"outputs/{state}/plots/{state}_scatter.png")
    return

if __name__=="__main__":
    main()