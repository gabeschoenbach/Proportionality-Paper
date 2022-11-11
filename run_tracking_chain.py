from chain_functions import *
import click

# @click.command()
# @click.option('--state')
# @click.option('--num_steps', default=10)
# @click.option('--guided', default=False)
# @click.option('--elecs', default=[])
def main(state, 
         num_steps, 
         is_guided=False, 
         elecs=[], 
         output_string="proportionality_scores"):
    print(f"Running a chain on {state}")
    graph = initialize_graph(state)
    initial_partition = initialize_partition(graph, state, 0)
    k = len(initial_partition)
    threshold = max(0.07, 1/k)
    chain = initialize_chain(state,
                             initial_partition,
                             num_steps,
                             guided=is_guided,
                             untracked_elections=elecs)
    # run_chain_track_proportionality(state, chain, output_string)
    run_chain_track_training_testing(state, chain, threshold)

if __name__=="__main__":
    # main("WI",
    #      10000,
    #      elecs=[])
    # main("NC",
    #      100000,
    #      elecs=["GOV08", "SEN08", "SEN10", "GOV12", "PRES12"],
    #      )
    main("MD",
         100000,
         )
