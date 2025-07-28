from src.enums import ResourceType
from src.rag.manager import RagDBManager

if __name__ == "__main__":
    recipe_rag = RagDBManager("Recipes", ResourceType.RECIPES)
    wikihow_rag = RagDBManager("WikiHow", ResourceType.WIKIHOW)
    cutting_rag = RagDBManager("CuttingVids", ResourceType.CUTTING_TUTORIALS)
