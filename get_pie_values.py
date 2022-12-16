from plotting_functions import *
from chain_functions import *
import pandas as pd

def get_pie_values(states_list, score_kind):
    for state in states_list:
        print(state)
        graph = initialize_graph(state)
        partition = initialize_partition(graph, state)
        threshold = max(0.07, 1/len(partition))
        if score_kind == "disprop":
            data_path = f"outputs/{state}/{state}_proportionality_scores_100000.csv"
            df, _ = make_df(data_path)
        elif score_kind == "eg":
            data_path = f"outputs/{state}/{state}_eg_scores_100000.csv"
            df = pd.read_csv(data_path)
        elif score_kind == "ens_discrep":
            data_path = f"outputs/{state}/{state}_ensemble_mean_scores_100000.csv"
            df = pd.read_csv(data_path)
        else:
            print("Error: `score` needs to be either `disprop`, `eg`, or `ens_discrep`")
            return
        past4 = states[state]["past4"]
        df = df[past4]
        plans_by_score = {s:[] for s in range(len(past4)+1)}

        def is_vec_proportional_score(vec, threshold):
            score = 0
            for v in vec:
                if abs(v) <= threshold:
                    score += 1
            return score

        for _, row in df.iterrows():
            score = is_vec_proportional_score(row, threshold)
            plans_by_score[score].append(1)

        sizes = [len(plans_by_score[s])/len(df) for s in plans_by_score]
        print("\\begin{tikzpicture}")
        print("\\pie{",end="")
        for idx, value in enumerate(sizes):
            print(f"{value*100:0.1f}/{idx},", end="")
        print("}\n\\end{tikzpicture}")
        

if __name__=="__main__":
    states_list = ["NC", "PA", "WI", "MA", "MD", "TX"]
    get_pie_values(states_list, "eg")