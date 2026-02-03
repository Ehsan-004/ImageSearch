from chromadb.config import Settings
import chromadb


DB_STORE_PATH = "./database"
CATEGORIES_COLLECTION_NAME = "categories"
PRODUCTS_COLLECTION_NAME = "products"


def get_client(db_path=DB_STORE_PATH):
    client = chromadb.PersistentClient(
        path=db_path,
        settings = Settings(anonymized_telemetry=False)
    )
    return client


def get_pr_collection(client, collection_name=PRODUCTS_COLLECTION_NAME):
    products_collection = client.get_or_create_collection(
        name=collection_name
    )
    return products_collection


def get_cat_collection(client, collection_name=CATEGORIES_COLLECTION_NAME):
    categories_collection = client.get_or_create_collection(
        name=collection_name
    )
    return categories_collection

