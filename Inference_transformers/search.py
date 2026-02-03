import faiss
import torch

from Inference_transformers.chroma_config import get_cat_collection, get_client, get_pr_collection
from Inference_transformers.dataset import (
    extract_embeddings_img
)
from Inference_transformers.model import get_model, get_processor


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DB_STORE_PATH = "./database"
EMB_DIM = 768


# model
processor = get_processor()
model = get_model().to(DEVICE)


# chroma DB
client = get_client()
products_collection = get_pr_collection(client)
categories_collection = get_cat_collection(client)


data = products_collection.get(include=["embeddings"])
all_embeddings = data["embeddings"]
ids = data["ids"]


M = 48  # number of connections in graph
index = faiss.IndexHNSWFlat(EMB_DIM, M)
index.hnsw.efConstruction = 200  # accuracy while indexing
index.add(all_embeddings.astype('float32')) 

# در زمان جستجو
index.hnsw.efSearch = 64
# extract_embeddings_img(img, model, processor, device=DEVICE):


def search(img, k=10):
    query_embedding = extract_embeddings_img(img, model, processor, DEVICE)
    query_embedding = query_embedding.cpu().numpy().flatten().reshape(1,-1)

    # distances, indices = index.search(query_embedding, k)
    D, I = index.search(query_embedding, k)

    return D, I
