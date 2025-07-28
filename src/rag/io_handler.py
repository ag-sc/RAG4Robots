import json
import os
import random
import tarfile
from typing import List

from tqdm import tqdm

from src.enums import ResourceType


def get_data_from_resource(resource: ResourceType) -> List[str]:
    if resource == ResourceType.RECIPES:
        return read_recipe_data()
    if resource == ResourceType.WIKIHOW:
        return read_wikihow_articles()
    if resource == ResourceType.CUTTING_TUTORIALS:
        return read_tutorial_videos()
    return []


def read_recipe_data(max_amount=-1) -> List[str]:
    folder = "data/recipes/"
    json_files = [pos_json for pos_json in os.listdir(folder) if pos_json.endswith('.json')]
    recipes = []
    # load all recipes
    for js in tqdm(json_files, 'Reading recipe files'):
        with open(os.path.join(folder, js)) as json_file:
            json_text = json.load(json_file)
            for rec in json_text:
                recipe = ""
                for ins in rec['instructions']:
                    recipe = f"{recipe}\n{ins['text']}"
                recipes.append(recipe.strip())
    # sample & return max_amount recipes
    if max_amount <= 0 or max_amount >= len(recipes):
        return recipes
    sampled_rec = random.sample(recipes, max_amount)
    return sampled_rec


def read_wikihow_articles(max_amount=-1) -> List[str]:
    archive = "data/wikihow/wikihow_corpus.tar.gz"
    articles = []
    # load all articles
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tqdm(tar.getmembers(), 'Reading WikiHow articles'):
            if member.isfile() and member.name.endswith(".json"):
                file = tar.extractfile(member)
                if file:
                    data = json.load(file)
                    article = data['title']
                    for idm, m in enumerate(data['methods']):
                        article = f"{article}\n{idm + 1}. {m['name']}"
                        for ids, s in enumerate(m['steps']):
                            article = f"{article}\n{idm + 1}.{ids + 1} {s['description']}"
                    articles.append(article)
    # sample & return max_amount recipes
    if max_amount <= 0 or max_amount >= len(articles):
        return articles
    sampled_rec = random.sample(articles, max_amount)
    return sampled_rec


def read_tutorial_videos() -> List[str]:
    folder = "data/cut_tutorials/"
    documents = []
    for filename in tqdm(os.listdir(folder), 'Reading tutorial video transcripts'):
        if filename.endswith(".txt"):
            with open(os.path.join(folder, filename), "r", encoding="utf-8") as file:
                documents.append(file.read())
    return documents
