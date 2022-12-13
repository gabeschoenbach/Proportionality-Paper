from chain_functions import *
from plotting_functions import make_df

def main():
    for state in ["TX", "WI"]:
        print(f"Saving plans stats on {state}...")
        output_path = f"outputs/{state}/{state}_mean_var_stats.csv"
        if os.path.exists(output_path):
            os.remove(output_path)
        else:
            if not os.path.exists(f"outputs/{state}/"):
                os.makedirs(f"outputs/{state}/")
        num_seeds = len(states[state]["ENACTED_COL"])
        graph = initialize_graph(state)
        with open(output_path, "w+") as f:
            f.write("plan,disprop_mean,disprop_var,eg_mean,eg_var,ens-discrep_mean,ens-discrep_var\n")
        f.close()

        # Prep R seats (from ensemble) dataframe
        stats = pd.read_csv(f"outputs/{state}/{state}_election_stats.csv", index_col=0)
        elections = states[state]["elections"].keys()
        R_vshare = dict(stats.loc['vote_share'])
        k = int(stats.loc['seats'][0])
        df, _ = make_df(f"outputs/{state}/{state}_proportionality_scores_100000.csv")
        seats_df = df.copy()   
        for e in elections:
            seats_df[e] = df[e].apply(lambda x: round((x + R_vshare[e]) * k))
        mean_seats = dict(seats_df.mean())


        for i in range(num_seeds):
            ENACTED_COL = states[state]["ENACTED_COL"][i]
            initial_partition = initialize_partition(graph, state, i)
            R_sshare = {e:initial_partition[e].wins("Rep")/len(initial_partition) for e in elections}
            disprop_v = proportionality_vector(initial_partition, elections)
            eg_v = [R_sshare[e] - (2 * R_vshare[e]) + 0.5 for e in elections]
            ens_v = [R_sshare[e] - (mean_seats[e]/k) for e in elections]

            with open(output_path, "a+") as f:
                f.write(f"{ENACTED_COL},{np.mean(disprop_v)},{np.var(disprop_v)},")
                f.write(f"{np.mean(eg_v)},{np.var(eg_v)},")
                f.write(f"{np.mean(ens_v)},{np.var(ens_v)}\n")
            f.close()
            print(f"  Wrote stats for {ENACTED_COL}")
    return

if __name__=="__main__":
    main()