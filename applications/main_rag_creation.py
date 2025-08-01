import argparse

from src.manager import RAGManager
from src.utils.enums import ResourceType

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resources", type=str, choices=["recipes", "wikihow", "cutting_vids", "cskg_loc", "all"],
                        nargs="+", help="Choose for what resources the DBs should be created")
    args = parser.parse_args()

    db_types = []
    if "recipes" in args.resources or "all" in args.resources:
        db_types.append(ResourceType.RECIPES)
    if "wikihow" in args.resources or "all" in args.resources:
        db_types.append(ResourceType.WIKIHOW)
    if "cutting_vids" in args.resources or "all" in args.resources:
        db_types.append(ResourceType.CUTTING_TUTORIALS)
    if "cskg_loc" in args.resources or "all" in args.resources:
        db_types.append(ResourceType.CSKG_LOC)
    man = RAGManager(db_types)
