from chain_functions import *

def main():
    for state in states:
        print(f"Saving enacted plans stats on {state}...")
        output_path = f"outputs/{state}/{state}_enacted_plan_stats.tsv"
        if os.path.exists(output_path):
            os.remove(output_path)
        else:
            if not os.path.exists(f"outputs/{state}/"):
                os.makedirs(f"outputs/{state}/")
        num_seeds = len(states[state]["ENACTED_COL"])
        graph = initialize_graph(state)
        with open(output_path, "w+") as f:
            f.write("plan\tvar\tmean\tproportionality_vector\n")
        f.close()
        for i in range(num_seeds):
            ENACTED_COL = states[state]["ENACTED_COL"][i]
            initial_partition = initialize_partition(graph, state, i)
            elections = states[state]["elections"].keys()
            v = proportionality_vector(initial_partition, elections)
            with open(output_path, "a+") as f:
                f.write(f"{ENACTED_COL}\t{np.var(v)}\t{np.mean(v)}\t{v}\n")
            f.close()
            print(f"  Wrote stats for {ENACTED_COL}")
    return

if __name__=="__main__":
    main()