from enum import Enum

file_ending = 'csv'

class ResourceType(Enum):
    RECIPES = ("Recipes", f'recipes1m+.{file_ending}', True)
    WIKIHOW = ("WikiHow", f'wikihow.{file_ending}', True)
    CUTTING_TUTORIALS = ("Cutting_Tutorials", f'cutting_tutorials.{file_ending}', True)

    def __init__(self, title: str, file: str, chunks: bool):
        self.type = title
        self.file_name = file
        self.needs_chunking = chunks
