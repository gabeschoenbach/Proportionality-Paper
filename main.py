from chain_functions import *
from plotting_functions import *
import click


@click.command()
@click.option('-state')
@click.option('-steps', default=100000)
def score_run(state, 
              steps):
    print(f"Running a score-run chain on {state}...")
    graph = initialize_graph(state)
    initial_partition = initialize_partition(graph, state, 0)
    chain = initialize_chain(state,
                             initial_partition,
                             steps)
    run_chain_track_training_testing(state, chain)
    print(f"Plotting {state} score-run results...")
    plot_full_score_histogram(state, steps)
    return

@click.command()
@click.option('-state')
@click.option('-steps', default=100000)
@click.option('-guided', default=0)
def histogram_run(state, 
         steps, 
         guided, 
         output_string="proportionality_scores",
         elecs=[]):
    isGuided = guided == 1
    if isGuided:
        prefix = "guided_"
    else:
        prefix = ""
    print(f"Running a {prefix}chain on {state}...")
    graph = initialize_graph(state)
    initial_partition = initialize_partition(graph, state, 0)
    chain = initialize_chain(state,
                             initial_partition,
                             steps,
                             guided=isGuided,
                             untracked_elections=elecs)
    run_chain_track_proportionality(state, chain, prefix+output_string, elecs)
    print(f"Plotting {state} chain results...")
    make_histogram(state, f"{prefix}{output_string}_{str(steps)}")
    return


@click.command()
@click.option('-state')
@click.option('-steps', default=100000)
@click.option('-threshold', default=0.07)
def swinginess_run(state, 
                   steps,
                   threshold):
    print(f"Running a swinginess run on {state}...")
    graph = initialize_graph(state)
    initial_partition = initialize_partition(graph, state, 0)
    chain = initialize_chain(state,
                             initial_partition,
                             steps)
    run_chain_track_swinginess(state, chain, threshold)
    print(f"Plotting {state} swinginess chain results...")
    make_swinginess_chart(state, "swinginess", len(chain))
    return

if __name__=="__main__":
    #score_run()
    histogram_run()
    #swinginess_run()
