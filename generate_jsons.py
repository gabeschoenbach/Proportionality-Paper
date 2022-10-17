from chain_functions import states
from gerrychain import Graph
import geopandas as gpd
import os

def main():
    for state in states:
        print(f"Generating {state} json...")
        if os.path.exists(f"shapes/{state}.json"):
            print(f"  {state}.json is already in the `shapes/` folder! Skipping {state}.")
            continue
        try:
            graph = Graph.from_file(f"shapes/{state}/{state}.shp")
            graph.to_json(f"shapes/{state}.json")
        except:
            print(f"  Error generating {state} json! Skipping {state}.")
            continue
    return

if __name__=="__main__":
    main()