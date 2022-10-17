from chain_functions import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import os

PALETTE = "bright"
colors = sns.color_palette(PALETTE) + sns.color_palette(PALETTE)
DPI_SIZE = 50

def make_df(path):
    """
    Returns two pandas DataFrames, one with the proportionality vector for each plan, and one
    with the signed binary violation vector for each plan.

    Parameters:
    ----------
    path: str
        Path to CSV file saved from chain run
    """
    df = pd.read_csv(path, index_col=0)
    elections = [e for e in df.columns if "bv" not in e]
    bvs = [f"{e}_bv" for e in elections]
    prop_df = df[elections]
    bin_df = df[bvs]
    return prop_df, bin_df

def find_variances(df):
    """
    Returns a list of the variance of the proportionality vector in each plan.

    Parameters:
    ----------
    df: pandas DataFrame
        DataFrame of the proportionality vector for each plan in an ensemble.
    """
    variances = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        variances.append(np.var(row))
    return variances

def find_means(df):
    """
    Returns a list of the mean of the proportionality vector in each plan.

    Parameters:
    ----------
    df: pandas DataFrame
        DataFrame of the proportionality vector for each plan in an ensemble.
    """
    means = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        means.append(np.mean(row))
    return means

def find_variances_and_means(df):
    variances = []
    means = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        variances.append(np.var(row))
        means.append(np.mean(row))
    return variances, means
    

def low_variance_plans(df, variances, percentile):
    """
    Returns the indices of the plans in the given DataFrame which have variance in the given percentile.

    Parameters:
    ----------
    df: pandas DataFrame
        DataFrame of the proportionality vector for each plan in an ensemble
    variances: list
        List of variances in the ensemble
    percentile: float
        Lowest `percentile`% of plans in the ensemble.
    """
    threshold = np.percentile(variances, percentile)
    idxs = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        if np.var(row) < threshold:
            idxs.append(i)
    return idxs, threshold

def get_enacted_stats(state):
    """
    Returns a dictionary with the variance and mean of the proportionality vector for each enacted plan
    on a given state.

    Parameters:
    ----------
    state: str
        Name of U.S. state, e.g. "NC"
    """
    df = pd.read_csv(f"outputs/{state}/{state}_enacted_plan_stats.tsv", sep="\t")
    enacted_plans = {}
    for i, row in df.iterrows():
        enacted_plans[row["plan"]] = [row["var"], row["mean"]]
    return enacted_plans

def plot_score_histogram(scores, ax, i, step, num_scores, title):
    bins = range(min(scores), max(scores)+2, 1)
    sns.histplot(scores,
                 bins=bins,
                 ax=ax[i],
                 )
    if i == 0:
        for j, patch in enumerate(ax[i].patches):
            if step != 1:
                if j%2 == 1:
                    patch.set_fc(colors[int((j-1)/2)])
                    continue
            patch.set_fc(colors[int(j/step)])
    else:
        mean = np.mean(scores)
        for patch in ax[i].patches:
            patch.set_fc(colors[i-1])
        ax[i].axvline(mean+0.5, color="black", lw=4, label=f"mean={mean:.2f}") # offset bc of x labeling
        ax[i].legend()
    ax[i].set_xlim(0, num_scores)
    ax[i].set_xticks(np.arange(0.5, num_scores + 0.5))
    ax[i].set_xticklabels(np.arange(0, num_scores))
    ax[i].set_title(title)
    ax[i].set_xlabel(f"Proportionality Score (out of {num_scores-1} elections)")
    return

def get_hist_data(by_score_dictionary):
    data = []
    for score, num in by_score_dictionary.items():
        for _ in range(num):
            data.append(int(score))
    return data

def load_pickles(state, chain_length):
    plans_by_training_score = pickle.load(open(f"outputs/{state}/plans_by_training_score_{chain_length}.p", "rb"))
    plans_by_testing_score = pickle.load(open(f"outputs/{state}/plans_by_testing_score_{chain_length}.p", "rb"))
    return plans_by_training_score, plans_by_testing_score

def plot_full_score_histogram(state, chain_length):
    plans_by_training_score, plans_by_testing_score = load_pickles(state, chain_length)
    training_elecs = states[state]["early"]
    testing_elecs = [e for e in states[state]["elections"].keys() if e not in training_elecs]

    num_training_scores = max([k for k in plans_by_training_score.keys() if plans_by_training_score[k] > 0]) + 1
    total_training_scores = len(training_elecs) + 1
    num_testing_scores = len(testing_elecs) + 1

    if num_training_scores >= 6:
        num_bins = int(np.ceil((num_training_scores)/2))
        step = 2
    else:
        num_bins = num_training_scores
        step = 1
    fig, ax = plt.subplots(num_bins + 1, 1, figsize=(12, 5 * (num_bins+1)))
    plt.subplots_adjust(hspace=0.5)

    # print(f"Total training scores: {total_training_scores}")
    # print(f"Num training scores: {num_training_scores}")
    # print(f"Bins: {num_bins}")
    # plt.suptitle(f"{state}: {chain_length} plans, binned by proportionality score")
    training_data = get_hist_data(plans_by_training_score)
    plot_score_histogram(training_data,
                         ax,
                         0,
                         step,
                         total_training_scores,
                         f"{state} Training Data\n{training_elecs}",
                         )
    for idx, training_bin in enumerate(range(0, num_testing_scores, step)):
        if training_bin not in plans_by_testing_score:
            continue
        if step == 1:
            testing_data = get_hist_data(plans_by_testing_score[training_bin])
            training_score_str = training_bin
        else:
            training_scores = [training_bin, training_bin+1]
            testing_data = get_hist_data(plans_by_testing_score[training_scores[0]])
            testing_data += get_hist_data(plans_by_testing_score[training_scores[1]])
            training_score_str = f"{training_scores[0]}-{training_scores[1]}"
        if testing_data:
            plot_score_histogram(testing_data,
                                 ax,
                                 idx+1,
                                 step,
                                 num_testing_scores,
                                 f"Training Score = {training_score_str}\nTesting Data: {testing_elecs}",
                                 )
    if not os.path.exists(f"outputs/{state}/plots"):
        os.makedirs(f"outputs/{state}/plots")
    plt.savefig(f"outputs/{state}/plots/{state}_early_later.png", dpi=DPI_SIZE, bbox_inches='tight')
    return

def make_histogram(state, output_string, percentile=5):
    """
    Saves a histogram of the distribution of variances and means of the proportionality vector
    over the ensemble of plans, given a state and ensemble run.

    Parameters:
    ----------
    state: str
        Name of U.S. state, e.g. "NC"
    output_string: str
        Suffix of run name we're interested in, e.g. "guided_proportionality_scores_100000"
    """
    data_path = f"outputs/{state}/{state}_{output_string}.csv"
    output_file = f"outputs/{state}/plots/{state}_{output_string}.png"
    prop_df, bin_df = make_df(data_path)

    fig, ax = plt.subplots(3, 1, figsize=(10, 8))
    plt.subplots_adjust(hspace=0.3)

    print("Calculating variances and means...")
    variances, means = find_variances_and_means(prop_df)
    print("Calculating low-variance means...")
    low_v_plans, threshold = low_variance_plans(prop_df, variances, percentile)
    low_v_means = find_means(prop_df.iloc[low_v_plans])
    print("Grabbing enacted stats...")
    enacted_plans = get_enacted_stats(state)

    data_len = int(np.floor(len(prop_df)/1000))
    low_v_len = len(low_v_means)

    # Plot ensemble data
    var_scale = int(np.floor(np.log10(variances[0])))
    mean_scale = int(np.floor(np.log10(abs(means[0]))))
    var_step = 10**(var_scale-1)
    mean_step = 10**(mean_scale-1)
    var_bins = np.arange(round(min(variances), -var_scale), round(max(variances) + 2*var_step, -var_scale), var_step)
    mean_bins = np.arange(round(min(means), -mean_scale), round(max(means) + 2*mean_step, -mean_scale), mean_step)
    sns.histplot(variances,
                bins=var_bins,
                ax=ax[0],
                )
    sns.histplot(means,
                bins=mean_bins,
                ax=ax[1],
                )
    sns.histplot(low_v_means,
                 bins=mean_bins,
                 ax=ax[2],
                )
                
    # Add low-variance shading
    ax[0].axvspan(max(0,ax[0].get_xlim()[0]), 
                  threshold, 
                  color="gray", 
                  alpha=0.2, 
                  label="low variance")
                  
    # Add enacted plans
    stat = [0,1,1] # second two plots record means
    for i, plan in enumerate(enacted_plans.keys()):
        for j in range(3):
            val = enacted_plans[plan][stat[j]]
            if j == 2 and enacted_plans[plan][stat[0]] > threshold:
                continue
            ax[j].axvline(val,
                          color=colors[i],
                          lw=2,
                          label=f"{plan}={val:0.3f}",
                          )
            ax[j].legend()
    # Annotate
    ax[0].set_title(f"{data_len}K {state} plans: Variance over {len(prop_df.columns)} elections")
    ax[1].set_title(f"{data_len}K {state} plans: Means over {len(prop_df.columns)} elections")
    ax[2].set_title(f"{low_v_len} {state} plans with var < {threshold:0.2f}: Means over {len(prop_df.columns)} elections")

    ax[2].set_xlim(ax[1].get_xlim())
    if not os.path.exists(f"outputs/{state}/plots"):
        os.makedirs(f"outputs/{state}/plots")
    plt.savefig(output_file, dpi=DPI_SIZE, bbox_inches='tight')
    return

def make_swinginess_chart(state, output_string, chain_length):
    outfile = f"outputs/{state}/plots/{state}_swinginess_{chain_length}.png"
    df = pd.read_csv(f"outputs/{state}/{state}_{output_string}_{chain_length}.csv", index_col=0)
    winnowed_df = pd.read_csv(f"outputs/{state}/{state}_{output_string}_winnowed_{chain_length}.csv", index_col=0)

    fig, ax = plt.subplots(2,3, figsize=(16, 8), sharex=False)
    plt.subplots_adjust(hspace=0.3)
    cols = list(df.columns)
    def binning(data):
        return np.arange(round(min(data),1), round(max(data) + 0.04,1), 0.02)
    
    for i in range(3):
        sns.histplot(df[cols[i]],
                     bins=binning(df[cols[i]]),
                     ax=ax[0][i])
        if len(winnowed_df.columns) > 1:
            sns.histplot(winnowed_df[cols[i]],
                         bins=binning(df[cols[i]]),
                         ax=ax[1][i])
    ax[0][0].set_xlabel("All plans\n% Districts that ever swing")
    ax[0][1].set_xlabel("All plans\n% Districts that ever swing, weighted")
    ax[0][2].set_xlabel("All plans\n% (District, E_i, E_j) that swing")
    ax[1][0].set_xlabel("Proportional plans\n% Districts that ever swing")
    ax[1][1].set_xlabel("Proportional plans\n% Districts that ever swing, weighted")
    ax[1][2].set_xlabel("Proportional plans\n% (District, E_i, E_j) that swing")


    plt.savefig(outfile, dpi=DPI_SIZE, bbox_inches='tight')
    return

def make_one_pie_chart(state, ax, chain_length, threshold):
    data_path = f"outputs/{state}/{state}_proportionality_scores_{chain_length}.csv"
    prop_df, bin_df = make_df(data_path)
    recent_elections = states[state]["past4"]
    recent_df = prop_df[recent_elections]
    plans_by_score = {s:[] for s in range(len(recent_elections)+1)}
    
    def is_vec_proportional_score(vec, elections, threshold):
        score = 0
        for v in vec:
            if abs(v) <= threshold:
                score += 1
        return score
    
    for i, row in tqdm(recent_df.iterrows(), total=len(recent_df)):
        score = is_vec_proportional_score(row, recent_elections, threshold)
        plans_by_score[score].append(1)
    
    sizes = [len(plans_by_score[s]) for s in plans_by_score.keys()]
    labels = [f"{i}/{len(plans_by_score)-1} passed" for i in plans_by_score.keys()]
    patches, texts, autotexts = ax.pie(sizes,
                                       autopct='%1.1f%%',
                                       startangle=90,
                                       )
    for patch, txt in zip(patches, autotexts):
        ang = (patch.theta2 + patch.theta1) / 2.
        x = patch.r * 1.05 * np.cos(ang*np.pi/180)
        y = patch.r * 1.05 * np.sin(ang*np.pi/180)
        if (patch.theta2 - patch.theta1) < 10.:
            txt.set_position((x, y))
    
    ax.axis('equal')
    ax.set_title(f"{state}")
    return patches, labels

def make_pie_charts(states, chain_length=100000, threshold=0.07):
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    for i in range(len(states)):
        x = int(np.floor(i/3))
        y = i % 3
        print(f"Making {states[i]} pie chart")
        patches, labels = make_one_pie_chart(states[i],
                                             ax[x][y],
                                             chain_length,
                                             threshold)
    plt.suptitle(f"Breakdown of {chain_length} districting plans on each state by proportionality score")
    # plt.legend(patches, labels)
    plt.savefig(f"outputs/pie_charts.png", dpi=DPI_SIZE, bbox_inches='tight')
    return


if __name__=="__main__":
    make_pie_charts(list(states.keys()))
    for state in states.keys():
        print(state)
        plot_full_score_histogram(state, 100000)
        make_histogram(state, "proportionality_scores_100000")
    # plot_full_score_histogram("TX", 100000)
    # plot_full_score_histogram("PA", 100000)
    # make_histogram("PA", "proportionality_scores_100000")
    # make_histogram("WI", "proportionality_scores_100")
    # make_histogram("NC", "proportionality_scores_100000")
    # make_histogram("NC", "guided_proportionality_scores_100000")
    # make_histogram("NC", "guided_proportionality_scores_no_SEN10_100000")
    # make_histogram("NC", "guided_proportionality_scores_no_SEN0810_100000")
    # make_histogram("NC", "guided_proportionality_scores_only_SEN0810_100000")