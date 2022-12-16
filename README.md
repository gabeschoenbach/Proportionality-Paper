# Redistricting for Proportionality
This repo contains data, experiments, and outputs for `Redistricting for Proportionality`, exploring the Freedom to Vote Act's proposal of a standard to flag partisan gerrymandering. All outputs used in the paper can be found in the `outputs` folder, split into subfolders for each of the six states on which we focus.

## Replication
If desired, following the below steps will suffice to replicate the results found in the paper.

### Programming environment
You will need to use [conda](https://conda.io/projects/conda/en/latest/index.html) to set up the correct programming environment, by running
```sh
conda env create -f environment.yml
```

### Replication steps
`chain_functions.py` contains a dictionary `states` with metadata for each of the states. It also includes initialization functions for creating GerryChain Graphs and Partitions, as well as scoring functions and driver functions to kick off chains to track proportionality scores and training/testing FTV scores. To generate, for example, `NC_proportionality_scores_100000.csv`, one would run
```sh
python main.py -state NC
```
which calls the `run_chain_track_proportionality()` function and saves the proportionality scores as a CSV, which then can be post-processed to generate many of the figures we discuss in the paper. Running the `run_tracking_chain.py` is another way to kick off a chain that saves training/testing scores.

To generate a table with partisan statistics by election for every plan on file, run
```sh
python generate_state_tables.py
```
which will create, for example, `NC/NC_judge_election_stats.csv` in the `outputs/` folder — detailing statistics for the Judges' plan on North Carolina.

After doing this, you can run
```sh
python make_score_tables.py
```
which will use the `NC_proportionality_scores_100000.csv` to generate `NC_{score_type}_scores_100000.csv` for the score types `disprop`, `eg` (efficiency gap), `ensemble_mean` (difference between each plan's seat share and the mean seat share taken over the whole ensemble), and `R_seats`.

In order to make the scatterplots, you will first need to run
```sh
python save_plan_mean_var_stats.py
python make_variances_means.py
python make_scatters.py
```
which will generate outputs like `NC/NC_mean_var_stats.csv` which contains the mean and variance of scores across every election for each plan on file, for three scores: disproportionality, efficiency gap, and discrepancy from the ensemble mean. The second script saves means and variances for the latter two scores across every plan in the ensemble, e.g. `NC/NC_eg_mean_variance_100000.csv`. Finally, the `make_scatters.py` script uses those intermediate outputs to generate the scatterplots we use in the paper, saved as `NC/plots/NC_scatter.png`. Note that some utility functions for our plotting functionality are stored in `plotting_functions.py`

Running the `get_pie_values.py` script prints out the necessary values to plot the pie charts we use in the paper.

FYI, below is a mapping of plan column names (the way plans are represented in our data) to the prettier plan names we used in the paper (also found in `plotting_functions.py`):
```python
plan_names = {
    "NC": {"newplan":"Leg16",
            "oldplan":"Leg12",
            "judge":"Judges",
            "NC_Proportional_DIST": "538-Pro",
            "NC_Dem_DIST": "538-Dem",
            "NC_GOP_DIST": "538-GOP",
            },
    "MD": {"CD":"Leg11",
            "CNG02":"Leg02",
            "MD_Proportional_DIST": "538-Pro",
            "MD_Dem_DIST": "538-Dem",
            "MD_GOP_DIST": "538-GOP",
            },
    "MA": {"CD":"Leg11",
            "MA_Proportional_DIST": "538-Pro",
            "MA_Dem_DIST": "538-Dem",
            "MA_GOP_DIST": "538-GOP",
            },
    "TX": {"CD":"VRA",
           "PLANC185_DIST": "Leg11",
            "TX_Proportional_DIST": "538-Pro",
            "TX_Dem_DIST": "538-Dem",
            "TX_GOP_DIST": "538-GOP",
            },
    "WI": {"WI_current_DIST":"Leg11",
           "WI_Proportional_DIST": "538-Pro",
           "WI_Dem_DIST": "538-Dem",
           "WI_GOP_DIST": "538-GOP",
            # weirdness here...
            },
    "PA": {"CD_2011":"Leg11",
            "REMEDIAL":"Remedial",
            "GOV":"Gov",
            "TS":"Leg18",
            "PA_Proportional_DIST": "538-Pro",
            "PA_Dem_DIST": "538-Dem",
            "PA_GOP_DIST": "538-GOP",
            },
}
```
