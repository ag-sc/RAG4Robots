import argparse

from src.enums import ResourceType
from src.rag.manager import RagDBManager

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resources", type=str, choices=["recipes", "wikihow", "cutting_vids", "cskg_loc", "all"],
                        nargs="+", help="Choose for what resources the DBs should be created")
    args = parser.parse_args()

    if "recipes" in args.resources or "all" in args.resources:
        recipe_rag = RagDBManager("Recipes", ResourceType.RECIPES)
        print("Created the RAG vector database for the recipe corpus")
    if "wikihow" in args.resources or "all" in args.resources:
        wikihow_rag = RagDBManager("WikiHow", ResourceType.WIKIHOW)
        print("Created the RAG vector database for the WikiHow copus")
    if "cutting_vids" in args.resources or "all" in args.resources:
        cutting_rag = RagDBManager("CuttingVids", ResourceType.CUTTING_TUTORIALS)
        print("Created the RAG vector database for the cutting tutorials")
    if "cskg_loc" in args.resources or "all" in args.resources:
        cskg_loc_rag = RagDBManager("CSKGLocations", ResourceType.CSKG_LOC)
        print("Created the RAG vector database for the prototypical locations found in the CSKG")
