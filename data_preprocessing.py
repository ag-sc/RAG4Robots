from data_reader import *
from data_vectorizer import convert_texts_to_vector, save_vector_db


def prepare_recipe_embeddings(max_amount=-1, chunk_size=500, chunk_overlap=50):
    recipes = read_recipe_data(max_amount)
    chunks = chunk_text_documents(recipes, chunk_size, chunk_overlap)
    vector_db = convert_texts_to_vector(chunks)
    if max_amount <= 0:
        file = f'all_recipes_cs{chunk_size}_co{chunk_overlap}'
    else:
        file = f'{max_amount}recipes_cs{chunk_size}_co{chunk_overlap}'
    save_vector_db(vector_db, file)


def prepare_wikihow_embeddings(max_amount=-1, chunk_size=500, chunk_overlap=50):
    articles = read_wikihow_articles(max_amount)
    chunks = chunk_text_documents(articles, chunk_size, chunk_overlap)
    vector_db = convert_texts_to_vector(chunks)
    if max_amount <= 0:
        file = f'all_wikihow_cs{chunk_size}_co{chunk_overlap}'
    else:
        file = f'{max_amount}wikihow_cs{chunk_size}_co{chunk_overlap}'
    save_vector_db(vector_db, file)


def prepare_tutorial_embeddings(chunk_size=500, chunk_overlap=50):
    tutorials = read_tutorial_videos()
    chunks = chunk_text_documents(tutorials, chunk_size, chunk_overlap)
    vector_db = convert_texts_to_vector(chunks)
    file = f'tutorials_cs{chunk_size}_co{chunk_overlap}'
    save_vector_db(vector_db, file)


if __name__ == '__main__':
    prepare_recipe_embeddings(5000)
    prepare_wikihow_embeddings(5000)
    prepare_tutorial_embeddings()
