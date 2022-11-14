from chain_functions import *
from plotting_functions import make_df
import pandas as pd
import numpy as np

def main(score_type):
    for state in ["NC", "PA", "WI", "MA", "MD", "TX"]:
        print(f"Calculating {score_type} FTV score for {state}...")
        if state == "WI":
            print("Skipping WI because old vs. new json weirdness")
            continue
        output_path = f"outputs/{state}/{state}_{score_type}_FTV_score.csv"
        if os.path.exists(output_path):
            os.remove(output_path)
        else:
            if not os.path.exists(f"outputs/{state}/"):
                os.makedirs(f"outputs/{state}/")
        graph = initialize_graph(state)
        plans = states[state]["ENACTED_COL"]
        past4 = states[state]["past4"]
        df = pd.DataFrame(index=plans, columns=past4 + ["FTV", "Pass?"])

        # Prep R seats (from ensemble) dataframe
        stats = pd.read_csv(f"outputs/{state}/{state}_election_stats.csv", index_col=0)
        R_vshare = dict(stats.loc['vote_share'])
        k = int(stats.loc['seats'][0])
        disprop_df, _ = make_df(f"outputs/{state}/{state}_proportionality_scores_100000.csv")
        seats_df = disprop_df.copy()   
        for e in past4:
            seats_df[e] = disprop_df[e].apply(lambda x: round((x + R_vshare[e]) * k))
        mean_seats = dict(seats_df.mean())

        for idx, plan in enumerate(plans):
            partition = initialize_partition(graph, state, idx)
            t = max(0.07, 1/len(partition))
            passes = 0
            for election in past4:
                if score_type == "disprop":
                    score = proportionality_score(partition, election)
                elif score_type == "eg":
                    R_sshare = partition[election].wins("Rep") / len(partition)
                    score = R_sshare - (2 * R_vshare[election]) + 0.5
                elif score_type == "ens-discrep":
                    R_sshare = partition[election].wins("Rep") / len(partition)
                    score = R_sshare - (mean_seats[election] / len(election))
                else:
                    print("score_type needs to be either `disprop`, `eg`, or `ens-discrep`")
                    exit()
                df.loc[plan][election] = np.round(score, 4)
                if np.abs(score) < t:
                    passes += 1
            df.loc[plan]["FTV"] = passes
            df.loc[plan]["Pass?"] = "Y" if passes >= 3 else "N"
            print(f" {plan}: {passes}")
        df.to_csv(output_path)

if __name__=="__main__":
    main("disprop")
    main("eg")
    main("ens-discrep")