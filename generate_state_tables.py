from chain_functions import *
import numpy as np

def main():
    for state in states.keys():
        print(f"Saving election stats for {state}...")
        graph = initialize_graph(state)
        plans = states[state]["ENACTED_COL"]
        for idx, plan in enumerate(plans):
            partition = initialize_partition(graph, state, idx)
            print(plan, len(partition))
            elections = list(states[state]["elections"].keys())
            output_path = f"outputs/{state}/{state}_{plan}_election_stats.csv"
            if os.path.exists(output_path):
                os.remove(output_path)
            else:
                if not os.path.exists(f"outputs/{state}/"):
                    os.makedirs(f"outputs/{state}/")
            
            with open(output_path, "w+") as f:
                f.write("stat,")
                for elec in elections:
                    if elec != elections[-1]:
                        f.write(f"{elec},")
                    else:
                        f.write(f"{elec}\n")
                f.write("vote_share,")
                for elec in elections:
                    vshare = partition[elec].percent('Rep')
                    if elec != elections[-1]:
                        f.write(f"{vshare:0.3f},")
                    else:
                        f.write(f"{vshare:0.3f}\n")
                f.write("propor_seats,")
                for elec in elections:
                    pshare = partition[elec].percent('Rep')
                    if elec != elections[-1]:
                        f.write(f"{pshare:0.3f},")
                    else:
                        f.write(f"{pshare:0.3f}\n")
                f.write("2propor_seats,")
                for elec in elections:
                    propor_share = partition[elec].percent('Rep')
                    double = propor_share + (propor_share - 0.5)
                    double_share = double * len(partition)
                    if elec != elections[-1]:
                        f.write(f"{double_share:0.3f},")
                    else:
                        f.write(f"{double_share:0.3f}\n")
                f.write("seats,")
                for elec in elections:
                    if elec != elections[-1]:
                        f.write(f"{len(partition)},")
                    else:
                        f.write(f"{len(partition)}\n")
                f.write("Rseats,")
                for elec in elections:
                    seats = partition[elec].wins('Rep')
                    if elec != elections[-1]:
                        f.write(f"{seats},")
                    else:
                        f.write(f"{seats}\n")
                f.write("Rseat_share,")
                for elec in elections:
                    sshare = partition[elec].wins('Rep')/len(partition)
                    if elec != elections[-1]:
                        f.write(f"{sshare:0.3f},")
                    else:
                        f.write(f"{sshare:0.3f}\n")
                f.write("disprop,")
                for elec in elections:
                    pscore = (partition[elec].wins('Rep')/len(partition)) - partition[elec].percent('Rep')
                    if elec != elections[-1]:
                        f.write(f"{pscore:0.3f},")
                    else:
                        f.write(f"{pscore:0.3f}\n")
            f.close()
    return

if __name__=="__main__":
    main()