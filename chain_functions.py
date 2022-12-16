from gerrychain import *
from gerrychain.updaters import Tally
from gerrychain.proposals import recom
from gerrychain import accept
from functools import partial
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import os
import json
import pickle
import math
from itertools import combinations

plan_names_538 = ["Proportional_DIST", "Dem_DIST", "GOP_DIST"]

states = {
    "NC": {
        "elections": {
            "GOV08":["EL08G_GV_D", "EL08G_GV_R"],
            "SEN08":["EL08G_USS_", "EL08G_US_1"],
            "SEN10":["EL10G_USS_", "EL10G_US_1"],
            "GOV12":["EL12G_GV_D", "EL12G_GV_R"],
            "PRES12":["EL12G_PR_D", "EL12G_PR_R"],
            "SEN14":["EL14G_US_1", "EL14G_USS_"],
            "PRES16":["EL16G_PR_D", "EL16G_PR_R"],
            "SEN16":["EL16G_US_1", "EL16G_USS_"],
            "GOV16":["EL16G_GV_D", "EL16G_GV_R"]
        },
        "past4": ["PRES16", "PRES12", "SEN16", "SEN14"],
        "early": ["GOV08", "SEN08", "SEN10", "GOV12", "PRES12"],
        "POP_COL": "TOTPOP",
        "ENACTED_COL": ["newplan", "oldplan", "judge"] + ["NC_" + col for col in plan_names_538],
    },
    "PA": {
        "elections": {
            "SEN16":["T16SEND", "T16SENR"],
            "PRES16":["T16PRESD", "T16PRESR"],
            "PRES12":["PRES12D", "PRES12R"],
            "SEN12":["USS12D", "USS12R"],
            "AG14":["ATG12D", "ATG12R"],
            "GOV14":["GOV14D", "GOV14R"],
            "GOV10":["GOV10D", "GOV10R"],
            "SEN10":["SEN10D", "SEN10R"],
            "AG16":["T16ATGD", "T16ATGR"],
        },
        "past4": ["PRES16", "PRES12", "SEN16", "SEN12"],
        "early": ["GOV10", "SEN10", "SEN12", "PRES12"],
        "POP_COL": "TOTPOP",
        "ENACTED_COL": ["CD_2011", "TS", "REMEDIAL", "GOV"] + ["PA_" + col for col in plan_names_538],
    },
    "WI": {
        # "elections": {
        #     "GOV18":["GOV18D", "GOV18R"],
        #     "SOS18":["SOS18D", "SOS18R"],
        #     "TRE18":["TRE18D", "TRE18R"],
        #     "SEN18":["SEN18D", "SEN18R"],
        #     "PRES16":["PRES16D", "PRES16R"],
        #     "SEN16":["SEN16D", "SEN16R"],
        #     "GOV14":["GOV14D", "GOV14R"],
        #     "SOS14":["SOS14D", "SOS14R"],
        #     "TRE14":["TRE14D", "TRE14R"],
        #     "AG14":["AG14D", "AG14R"],
        #     "GOV12":["GOV12D", "GOV12R"],
        #     "PRES12":["PRES12D", "PRES12R"],
        #     "SEN12":["SEN12D", "SEN12R"],
        # },
        "elections": {
            "GOV18":["GOVDEM18", "GOVREP18"],
            "SOS18":["SOSDEM18", "SOSREP18"],
            "TRE18":["TRSDEM18", "TRSREP18"],
            "SEN18":["USSDEM18", "USSREP18"],
            "PRES16":["PREDEM16", "PREREP16"],
            "SEN16":["USSDEM16", "USSREP16"],
            "GOV14":["GOVDEM14", "GOVREP14"],
            "SOS14":["SOSDEM14", "SOSREP14"],
            "TRE14":["TRSDEM14", "TRSREP14"],
            "AG14":["WAGDEM14", "WAGREP14"],
            "GOV12":["GOVDEM12", "GOVREP12"],
            "PRES12":["PREDEM12", "PREREP12"],
            "SEN12":["USSDEM12", "USSREP12"],
        },
        "past4": ["PRES16", "PRES12", "SEN18", "SEN16"],
        "early": ["GOV12", "PRES12", "SEN12", "GOV14", "SOS14", "TRE14", "AG14"],
        "POP_COL": "TOTPOP",
        "ENACTED_COL": ["WI_current_DIST"] + ["WI_" + col for col in plan_names_538],
    },
    "MA": {
        "elections": {
            "SEN12":["SEN12D", "SEN12R"],
            "PRES12":["PRES12D", "PRES12R"],
            "SEN13":["SEN13D", "SEN13R"],
            "SEN14":["SEN14D", "SEN14R"],
            "PRES16":["PRES16D", "PRES16R"],
            "GOV14":["GOV14D", "GOV14R"],
            "GOV18":["GOV18D", "GOV18R"],
            "SEN18":["SEN18D", "SEN18R"],
        },
        "past4": ["PRES16", "PRES12", "SEN18", "SEN14"],
        "early": ["SEN12", "PRES12", "SEN13", "SEN14", "GOV14"],
        "POP_COL": "TOTPOP",
        "ENACTED_COL": ["CD"] + ["MA_" + col for col in plan_names_538],
    },
    "MD": {
        "elections": {
            "PRES12":["PRES12D", "PRES12R"],
            "SEN12":["SEN12D", "SEN12R"],
            "GOV14":["GOV14D", "GOV14R"],
            "AG14":["AG14D", "AG14R"],
            "COMP14":["COMP14D", "COMP14R"],
            "PRES16":["PRES16D", "PRES16R"],
            "SEN16":["SEN16D", "SEN16R"],
            "SEN18":["SEN18D", "SEN18R"],
            "AG18":["AG18D", "AG18R"],
            "COMP18":["COMP18D", "COMP18R"],
        },
        "past4": ["PRES16", "PRES12", "SEN18", "SEN16"],
        "early": ["PRES12", "SEN12", "GOV14", "AG14", "COMP14"],
        "POP_COL": "TOTPOP",
        "ENACTED_COL": ["CD", "CNG02"] + ["MD_" + col for col in plan_names_538],
    },
    "TX": {
        "elections": {
            "PRES12":["ObamaD_12G", "RomneyR_12"],
            "SEN12":["SadlerD_12", "CruzR_12G_"],
            # "AGCOMM14":["HoganD_14G", "MillerR_14"],
            "AG14":["HoustonD_1", "PaxtonR_14"],
            # "COMP14":["CollierD_1", "HegarR_14G"],
            "GOV14":["DavisD_14G", "AbbottR_14"],
            # "LANDCOMM14":["CookD_14G_", "BushR_14G_"],
            # "LTGOV14":["Van De Put", "PatrickR_1"],
            "SEN14":["AlameelD_1", "CornynR_14"],
            "PRES16":["ClintonD_1", "TrumpR_16G"],
            # "AGCOMM18":["OlsonD_18G", "MillerR_18"],
            "AG18":["NelsonD_18", "PaxtonR_18"],
            # "COMP18":["ChevalierD", "HegarR_18G"],
            "GOV18":["ValdezD_18", "AbbottR_18"],
            # "LANDCOMM18":["SuazoD_18G", "BushR_18G_"],
            # "LTGOV18":["CollierD_3", "PatrickR_4"],
            "SEN18":["O'RourkeD_", "CruzR_18G_"],
            # "RRCOMM18":["McAllenD_1", "CraddickR_"],
            # "RRCOMM16":["Yarbroug_3", "ChristianR"],
            # "RRCOMM14":["BrownD_14G", "SittonR_14"],
            # "RRCOMM12":["HenryD_12G", "Craddick_1"],
        },
        "past4": ["PRES16", "PRES12", "SEN18", "SEN14"],
        # "early": ["RRCOMM12", "PRES12", "SEN12", "RRCOMM14", "SEN14", "LTGOV14", "LANDCOMM14", "GOV14", "COMP14", "AG14", "AGCOMM14"],
        "early": ["PRES12", "SEN12", "SEN14", "GOV14", "AG14"],
        "POP_COL": "TOTPOP_x",
        "ENACTED_COL": ["PLANC185_DIST", "CD"] + ["TX_" + col for col in plan_names_538],
    },
}

def initialize_graph(state):
    """
    Initialize a GerryChain Graph object for a given state.

    Parameters:
    ----------
    state: str
        The abbreviation for the U.S. state, e.g. "NC"
    """
    if os.path.exists(f"shapes/{state}.json"):
        graph = Graph.from_json(f"shapes/{state}.json")
    else:
        graph = Graph.from_file(f"shapes/{state}/{state}.shp")
    return graph

def initialize_partition(graph, state, i=0):
    """
    Return a GerryChain Partition that will track every statewide election specified in `states`.
    By default, the initial plan represented by this partition will be the first plan in "ENACTED_COL".

    Parameters:
    ----------
    graph: GerryChain Graph
        The Graph object returned by `initialize_graph()`
    state: str
        The abbreviation for the U.S. state, e.g. "NC"
    i: int
        Specifies the index in `states[state]["ENACTED_COL"] that corresponds to the initial plan
        default = 0
    """
    POP_COL = states[state]["POP_COL"]
    ENACTED_COL = states[state]["ENACTED_COL"][i]
    elections = states[state]["elections"]
    my_updaters = {
        "population":Tally(POP_COL, alias="population")
    }
    for elec in elections.keys():
        my_updaters[elec] = Election(elec,{"Dem":elections[elec][0], "Rep":elections[elec][1]})
    initial_partition = Partition(graph, assignment=ENACTED_COL, updaters=my_updaters)
    return initial_partition


def L1(vec):
    return sum(abs(v) for v in vec)

def make_proportionality_acceptance_function(election_list):
    """
    Function factory to return an acceptance function based on the `election_list`.

    Parameters:
    ----------
    election_list: str
        The set of elections upon which the `accept_more_proportional()` function will consider.
    """
    def accept_more_proportional(partition):
        """ 
        Acceptance function for probabilistically choosing partitions with a proportionality vector L1 score
        closer to 0, given the set of elections that is passed to the factory function, originally 
        set in `initialize_chain()`.

        Parameters:
        ----------
        partition: GerryChain Partition
            A districting plan on the state.
        """

        parent = partition.parent
        child_v = proportionality_vector(partition, election_list)
        parent_v = proportionality_vector(parent, election_list)
        child_score = L1(child_v)
        parent_score = L1(parent_v)
        probability = min([0.0055*math.exp(parent_score - child_score), 1])
        if random.random() < probability:
            return True
        else:
            return False
        
    return accept_more_proportional


def initialize_chain(state, 
                     initial_partition, 
                     num_steps, 
                     guided=False, 
                     untracked_elections=[], 
                     epsilon=0.01):
    """
    Returns a GerryChain MarkovChain object that will be used to generate an ensemble of district plans.
    If the acceptance function is `always_accept`, then this will be a neutral ensemble. Passing our custom
    acceptance function will give a guided ReCom ensemble.

    Parameters:
    ----------
    state: str
        The abbreviation for the U.S. state, e.g. "NC"
    initial_partition: GerryChain Partition
        The Partition object returned by `initialize_partition()`
    num_steps: int
        The number of plans generated in the ensemble
    acceptance: function :: Partition -> Boolean
        The acceptance function for the transition phase.
    untracked_elections: list of str
        The set of elections upon which the `accept_more_proportional()` function will NOT consider, if used.
    epsilon: float
        The maximum population deviation tolerated by plans in the ensemble
        default = 0.01
    """
    if state == "MD":
        epsilon = 0.02
    POP_COL = states[state]["POP_COL"]
    num_dists = len(initial_partition)
    ideal_pop = sum(initial_partition.population.values()) / len(initial_partition)
    proposal = partial(recom,
                       pop_col=POP_COL,
                       pop_target=ideal_pop,
                       epsilon=epsilon,
                       node_repeats=3)
    compactness_bound = constraints.UpperBound(
        lambda p: len(p["cut_edges"]), 2 * len(initial_partition["cut_edges"])
    )
    election_list = [elec for elec in states[state]["elections"] if elec not in untracked_elections]
    acceptance = make_proportionality_acceptance_function(election_list) if guided else accept.always_accept
    chain = MarkovChain(
        proposal=proposal,
        constraints=[
            constraints.within_percent_of_ideal_population(initial_partition, epsilon),
        ],
        accept=acceptance,
        initial_state=initial_partition,
        total_steps=num_steps
    )
    return chain

def proportionality_score(partition, election):
    """
    Calculate the proportionality score of a given districting plan, election pair.

    Parameters:
    ----------
    partition: GerryChain Partition
        The districting plan on the state
    election: str
        The name of the election to be analyzed, from `states[state]["elections"]`
    """
    num_dists = len(partition)
    vote_share = partition[election].percent("Rep")
    seat_share = partition[election].wins("Rep") / num_dists
    return seat_share - vote_share

def is_proportional(partition, election, threshold=0.07):
    """
    Return True if the proportionality score has smaller magnitude than `threshold`.

    Parameters:
    ----------
    partition: GerryChain Partition
        The districting plan on the state
    election: str
        The name of the election to be analyzed, from `states[state]["elections"]`
    threshold: float
        The maximum allowed magnitude of proportionality score to be considered proportional
        default = 0.07 (per S.1. draft) 
    """
    return abs(proportionality_score(partition, election)) <= threshold

def is_proportional_score(partition, elections, threshold=0.07):
    """
    Return the number of elections on which this partition is proportional, relative to threshold.

    Parameters:
    ----------
    partition: GerryChain Partition
        The districting plan on the state
    elections: list of str
        The names of the elections to be analyzed, from `states[state]["elections"]`
    threshold: float
        The maximum allowed magnitude of proportionality score to be considered proportional
        default = 0.07 (per S.1. draft)
    """
    score = 0
    for election in elections:
        if is_proportional(partition, election, threshold):
            score += 1
    return score

def proportionality_vector(partition, elections):
    """
    Calculate the proportionality vector of a given districting plan.

    Parameters:
    ----------
    partition: GerryChain Partition
        The districting plan on the state
    elections: list of str
        The names of the elections to be analyzed, from `states[state]["elections"]`
    """
    v = []
    for elec in elections:
        v.append(proportionality_score(partition, elec))
    return v

def swinginess(partition, elections):
    """
    Calculate the "swinginess" of a districting plan over our election list. This corresponds to
    the percent of districts that ever swing from one party to the other over the elections.

    Parameters:
    ----------
    partition: GerryChain Partition
        The districting plan on the state
    elections: list of str
        The names of the elections to be analyzed, from `states[state]["elections"]`
    """
    num_swing_dists = 0
    num_swing_dists_weighted = 0
    num_swing_swaps = 0
    num_pairs = 0
    for i in range(len(partition)):
        party_list = [round(partition[e].percents('Rep')[i]) for e in elections] # 0 for Dem, 1 for Rep
        

        num_swings = min(sum(party_list), len(party_list)-sum(party_list))
        if num_swings > 0:
            num_swing_dists += 1
        num_swing_dists_weighted += num_swings/np.floor(len(party_list)/2)

        pairs = list(combinations(party_list, 2))
        for pair in pairs:
            if len(set(pair)) == 2:
                num_swing_swaps += 1
            num_pairs += 1
    
    swing_dist_pct = num_swing_dists/len(partition)
    swing_dist_weighted_pct = num_swing_dists_weighted/len(partition)
    swing_swaps_pct = num_swing_swaps/num_pairs
    return swing_dist_pct, swing_dist_weighted_pct, swing_swaps_pct

def binary_violation(partition, election):
    """
    A binary violation occurs when a party wins > 50% of the seats with < 50% of the vote.
    Returns 1 if the given (partition, election) pair is a Republican-favoring binary violation, 
    -1 if it is a Democratic-favoring binary violation, and 0 otherwise.

    Parameters:
    ----------
    partition: GerryChain Partition
        The districting plan on the state
    election: str
        The name of the election to be analyzed, from `states[state]["elections"]`
    """
    num_dists = len(partition)
    vote_share = partition[election].percent("Rep")
    seat_share = partition[election].wins("Rep") / num_dists
    if seat_share > 0.5 and vote_share < 0.5:
        return 1
    elif seat_share < 0.5 and vote_share > 0.5:
        return -1
    else:
        return 0


def run_chain_track_proportionality(state, chain, output_string, elecs):
    """
    Saves a CSV in the `outputs` folder that tracks the proportionality score and
    binary violations for every districting plan in the ensemble.

    Parameters:
    ----------
    state: str
        The abbreviation for the U.S. state, e.g. "NC"
    chain: GerryChain MarkovChain object
        The object that specifies the Markov Chain that will generate the ensemble of plans.
    output_string: str
        The middle part (not the state or number of steps) of the CSV file storing results
    elecs: list of str
        The untracked elections in the acceptance function, if guided
    """
    output_path = f"outputs/{state}/"
    output_file = output_path + f"{state}_{output_string}_{len(chain)}.csv"
    if os.path.exists(output_file):
        print(f" `{output_file}` chain already run on {state}! Skipping {state}.")
        return
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    elections = states[state]["elections"].keys()
    tracked_elections = [elec for elec in elections if elec not in elecs]
    df = pd.DataFrame(columns=elections)
    L1_scores = []
    for part in tqdm(chain):
        v = proportionality_vector(part, tracked_elections)
        L1_scores.append(L1(v))
        partition_dict = {}
        for elec in elections:
            partition_dict[elec] = proportionality_score(part, elec)
            partition_dict[f"{elec}_bv"] = binary_violation(part, elec)
        partition_df = pd.DataFrame([pd.Series(partition_dict)])
        df = pd.concat([df, partition_df], ignore_index=True)
    df.to_csv(output_file)

    plt.title("L1 distance from 0")
    plt.scatter(range(len(L1_scores)), L1_scores)
    plt.xlabel("Chain step")
    plt.savefig(f"outputs/{state}/plots/{state}_{output_string}_{len(chain)}_traceplot.png")
    return


def run_chain_track_training_testing(state, chain, threshold):
    elections = states[state]["elections"]
    training_elections = states[state]["early"]
    testing_elections = [e for e in elections if e not in training_elections]

    plans_by_training_score = {i:0 for i in range(len(training_elections) + 1)}
    plans_by_testing_score = {
        i:{
            j:0 for j in range(len(testing_elections) + 1)
        } for i in range(len(training_elections) + 1)
    }

    for partition in tqdm(chain):
        training_score = is_proportional_score(partition, training_elections, threshold=threshold)
        testing_score = is_proportional_score(partition, testing_elections, threshold=threshold)
        plans_by_training_score[training_score] += 1
        plans_by_testing_score[training_score][testing_score] += 1

    pickle.dump(plans_by_training_score, open(f"outputs/{state}/plans_by_training_score_{len(chain)}.p", "wb"))
    pickle.dump(plans_by_testing_score, open(f"outputs/{state}/plans_by_testing_score_{len(chain)}.p", "wb"))
   
    return


def run_chain_track_swinginess(state, chain, threshold=0.07, output_string="swinginess"):
    elections = states[state]["elections"]
    output_path = f"outputs/{state}/"
    output_file = output_path + f"{state}_{output_string}_{len(chain)}.csv"
    winnowed_output_file = output_path + f"{state}_{output_string}_winnowed_{len(chain)}.csv"
    if os.path.exists(output_file):
        print(f" `{output_file}` chain already run on {state}! Skipping {state}.")
        return
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    df = pd.DataFrame()
    winnowed_df = pd.DataFrame()
    for part in tqdm(chain):
        part_df = pd.DataFrame([pd.Series(swinginess(part, elections))])
        df = pd.concat([df, part_df], ignore_index=True, axis=0)
        if np.mean(proportionality_vector(part, elections)) < threshold:
            winnowed_df = pd.concat([winnowed_df, part_df], ignore_index=True, axis=0)

    df = df.rename(columns={0:"swing_pct", 1:"swing_pct_w", 2:"pair_pct"})
    winnowed_df = winnowed_df.rename(columns={0:"swing_pct", 1:"swing_pct_w", 2:"pair_pct"})
    df.to_csv(output_file)
    winnowed_df.to_csv(winnowed_output_file)
    return
