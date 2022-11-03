from plotting_functions import *
from chain_functions import *
from tqdm import tqdm
import pandas as pd

def main():
    states = ["NC", "WI", "TX", "PA", "MD", "MA"]
    for state in states:
        # EG
        df = pd.read_csv(f"outputs/{state}/{state}_eg_scores_100000.csv")
        variances, means = find_variances_and_means(df)
        csv = pd.DataFrame(list(zip(variances, means)), columns=["variances", "means"])
        csv.to_csv(f"outputs/{state}_eg_mean_variance_100000.csv", index=0)

        # Ensemble
        df = pd.read_csv(f"outputs/{state}/{state}_ensemble_mean_scores_100000.csv")
        variances, means = find_variances_and_means(df)
        csv = pd.DataFrame(list(zip(variances, means)), columns=["variances", "means"])
        csv.to_csv(f"outputs/{state}/{state}_ens_discrep_mean_variance_100000.csv", index=0)


if __name__=="__main__":
    main()