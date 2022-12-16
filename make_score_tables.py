from plotting_functions import make_df
from tqdm import tqdm
import pandas as pd

# proportional ideal: S = V => disprop = S - V
# EG ideal: S = 2V - 0.5 => EG score = S - 2V + 0.5
# Ensemble ideal: S = mean(S) => Ens score = S - mean(S)

def make_tables(state):
    df, _ = make_df(f"outputs/{state}/{state}_proportionality_scores_100000.csv")
    stats = pd.read_csv(f"outputs/{state}/{state}_election_stats.csv", index_col=0)
    seats_df = df.copy()
    eg_df = df.copy()
    ensemble_df = df.copy()

    k = int(stats.loc["seats"][0])
    seat_share = dict(stats.loc["seat_share"])
    vote_share = dict(stats.loc["vote_share"])
    elections = list(df.columns)
    elections_w_stats = list(vote_share)
    if len(elections) != len(elections_w_stats):
        print(f"{state}: We have {len(elections_w_stats)} elections with stats out of {len(elections)} elections")
        elections = list(set(elections).intersection(set(elections_w_stats)))
        print(f"Winnowing to {len(elections)} elections...")

    # We can recreate the number of R seats in each plan by adding the vote share to the disprop score
    # which was calculated as disprop = plan's seat share - election's vote share
    for election in elections:
        seats_df[election] = df[election].apply(lambda x: round((x + vote_share[election]) * k))

    mean_seats = dict(seats_df.mean())

    for election in elections:
        eg_df[election] = seats_df[election].apply(lambda x: (x/k) - (2 * vote_share[election]) + 0.5)
        ensemble_df[election] = seats_df[election].apply(lambda x: (x/k) - (mean_seats[election]/k))

    df.to_csv(f"outputs/{state}/{state}_disprop_scores_100000.csv", index=0)
    seats_df.to_csv(f"outputs/{state}/{state}_R_seats_100000.csv", index=0)
    eg_df.to_csv(f"outputs/{state}/{state}_eg_scores_100000.csv", index=0)
    ensemble_df.to_csv(f"outputs/{state}/{state}_ensemble_mean_scores_100000.csv", index=0)
    return

if __name__=="__main__":
    for state in tqdm(["NC", "PA", "WI", "TX", "MD", "MA"]):
        make_tables(state)
