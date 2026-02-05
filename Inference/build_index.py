
import faiss
import numpy as np
import torch
from tqdm import tqdm
from chroma_configs import get_cat_collection, get_client, get_pr_collection
from model import get_model, get_transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "./data/"
DB_STORE_PATH = "./database"
EMB_DIM = 1024
BATCH_SIZE = 2000
M = 300  # number of connections in graph

transforms = get_transforms()
model = get_model().to(DEVICE)

client = get_client()
products_collection = get_pr_collection(client)
categories_collection = get_cat_collection(client)

print("creating faiss index ...")
index = faiss.IndexHNSWFlat(EMB_DIM, M)
index.hnsw.efConstruction = 256  # accuracy while indexing


total_count = products_collection.count()
print(f"Total vectors: {total_count}")



offset = 0

with tqdm(total=total_count, desc="Indexing embeddings") as pbar:
    while offset < total_count:
        batch = products_collection.get(
            include=["embeddings"],
            limit=BATCH_SIZE,
            offset=offset
        )

        if not batch["embeddings"].any():
            break

        embeddings = np.array(batch["embeddings"], dtype="float32")

        index.add(embeddings)

        batch_size = len(embeddings)
        offset += batch_size
        pbar.update(batch_size)



index.hnsw.efSearch = 128  # accuracy while searching

print("faiss index created.")
faiss.write_index(index, f"{DATA_PATH}/product_index.index")
print("✅ faiss index saved")

