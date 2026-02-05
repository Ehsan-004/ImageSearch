import faiss
import numpy as np
import torch
from collections import defaultdict, Counter

from Inference.chroma_configs import get_cat_collection, get_client, get_pr_collection
from Inference.dataset import extract_embeddings_img
from Inference.model import get_model, get_transforms


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "./data/"
DB_STORE_PATH = "./database"
EMB_DIM = 1024

transforms = get_transforms()
model = get_model().to(DEVICE)

client = get_client()
products_collection = get_pr_collection(client)
categories_collection = get_cat_collection(client)


def load_faiss_index(path, emb_dim):
    print("loading faiss index ...")
    index = faiss.read_index(path)
    print("faiss index loaded.")
    return index

index = load_faiss_index(f"{DATA_PATH}/product_index.index", EMB_DIM)


def search(img, k=10):
    query_embedding = extract_embeddings_img(img, model, DEVICE, transforms)
    query_embedding = query_embedding.cpu().numpy().flatten().reshape(1,-1)
    D, I = index.search(query_embedding, k)
    return D, I
