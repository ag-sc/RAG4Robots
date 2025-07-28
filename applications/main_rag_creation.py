from src.enums import ResourceType
from src.rag.manager import RagDBManager

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resources", type=str, choices=["recipes", "wikihow", "cutting_vids", "all"], nargs="+", help="Choose for what resources the DBs should be created")
    args = parser.parse_args()

    if "recipes" in args.tasks or "all" in args.tasks:
        recipe_rag = RagDBManager("Recipes", ResourceType.RECIPES)
        print("Created the RAG vector database for the recipe corpus")
    if "wikihow" in args.tasks or "all" in args.tasks:
        wikihow_rag = RagDBManager("WikiHow", ResourceType.WIKIHOW)
        print("Created the RAG vector database for the WikiHow copus")
    if "cutting_vids" in args.tasks or "all" in args.tasks:
        cutting_rag = RagDBManager("CuttingVids", ResourceType.CUTTING_TUTORIALS)
        print("Created the RAG vector database for the cutting tutorials")