from chain_functions import *
import pandas as pd
import numpy as np

def main():
    for state in ["NC", "PA", "WI", "MA", "MD", "TX"]:
        print(f"Calculating FTV score for {state}...")
        output_path = f"outputs/{state}/{state}_FTV_score.csv"
        if os.path.exists(output_path):
            os.remove(output_path)
        else:
            if not os.path.exists(f"outputs/{state}/"):
                os.makedirs(f"outputs/{state}/")
        graph = initialize_graph(state)
        plans = states[state]["ENACTED_COL"]
        past4 = states[state]["past4"]
        df = pd.DataFrame(index=plans, columns=past4 + ["FTV", "Pass?"])
        for idx, plan in enumerate(plans):
            partition = initialize_partition(graph, state, idx)
            t = max(0.07, 1/len(partition))
            passes = 0
            for election in past4:
                disprop = proportionality_score(partition, election)
                df.loc[plan][election] = np.round(disprop, 4)
                if np.abs(disprop) < t:
                    passes += 1
            df.loc[plan]["FTV"] = passes
            df.loc[plan]["Pass?"] = "Y" if passes >= 3 else "N"
            print(f" {plan}: {passes}")
        df.to_csv(output_path)

if __name__=="__main__":
    main()