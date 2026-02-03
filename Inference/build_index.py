
import faiss
import numpy as np
import torch
from tqdm import tqdm
from chroma_configs import get_cat_collection, get_client, get_pr_collection
from model import get_model, get_transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "./data/"
DB_STORE_PATH = "./database"
EMB_DIM = 2048
BATCH_SIZE = 2000
M = 300  # number of connections in graph

transforms = get_transforms()
model = get_model().to(DEVICE)

client = get_client()
products_collection = get_pr_collection(client)
categories_collection = get_cat_collection(client)

print("creating faiss index ...")
index = faiss.IndexHNSWFlat(EMB_DIM, M)
index.hnsw.efConstruction = 300  # accuracy while indexing


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



index.hnsw.efSearch = 256  # accuracy while searching

print("faiss index created.")
faiss.write_index(index, f"{DATA_PATH}/product_index.index")
print("✅ faiss index saved")



# print("loading data ...")
# data = products_collection.get(include=["embeddings", "metadatas"])
# embeddings = np.array(data["embeddings"], dtype="float32")
# ids = data["ids"]
# print(f"Loaded {len(embeddings)} embeddings")
# # metadatas = data["metadatas"]


# print("creating faiss index ...")
# index = faiss.IndexHNSWFlat(EMB_DIM, M)  # create the index
# index.hnsw.efConstruction = 100  # accuracy while indexing
# index.add(embeddings)  # add vectors to the index
# index.hnsw.efSearch = 128  # accuracy while searching
# print("faiss index created.")
# # save index file
# faiss.write_index(index, f"{DATA_PATH}/product_index.index")
# print("✅ faiss index saved")






# import faiss
# import numpy as np
# from collections import defaultdict
# from tqdm import tqdm

# from chroma_configs import (
#     get_client,
#     get_pr_collection
# )
# from index_config import *

# # =========================
# # Load data from Chroma
# # =========================
# print("loading data ...")
# client = get_client()
# products = get_pr_collection(client)

# data = products.get(
#     include=["embeddings", "metadatas"],
# )

# embeddings = np.array(data["embeddings"], dtype="float32")
# metadatas = data["metadatas"]

# # =========================
# # Group by category
# # =========================
# print("gouping by category ...")
# category_to_items = defaultdict(list)

# for emb, meta in zip(embeddings, metadatas):
#     if meta is None:
#         print("non found")
#         continue

#     if "category" not in meta:
#         print("non for category found")
#         continue

#     category = meta["category"]
#     category_to_items[category].append(emb)

# print(f"Loaded {len(embeddings)} embeddings")
# print(f"Categories: {len(category_to_items)}")

# # =========================
# # Stage 1: Category Representatives
# # =========================
# rep_embeddings = []
# rep_categories = []

# for category, embs in category_to_items.items():
#     embs = np.array(embs)

#     if len(embs) <= REP_PER_CATEGORY:
#         reps = embs
#     else:
#         idx = np.random.choice(len(embs), REP_PER_CATEGORY, replace=False)
#         reps = embs[idx]

#     for r in reps:
#         rep_embeddings.append(r)
#         rep_categories.append(category)

# rep_embeddings = np.array(rep_embeddings, dtype="float32")

# faiss.normalize_L2(rep_embeddings)

# rep_index = faiss.IndexFlatIP(EMB_DIM)
# rep_index.add(rep_embeddings)

# faiss.write_index(rep_index, CATEGORY_REP_INDEX_PATH)

# np.save(f"{INDEX_DIR}/rep_categories.npy", np.array(rep_categories))

# print("✅ Category representative index saved")

# # =========================
# # Stage 2: Per-category HNSW indexes
# # =========================
# for category, embs in tqdm(category_to_items.items(), desc="Building category indexes"):
#     embs = np.array(embs, dtype="float32")
#     faiss.normalize_L2(embs)

#     index = faiss.IndexHNSWFlat(EMB_DIM, 80)
#     index.hnsw.efConstruction = 300
#     index.hnsw.efSearch = 128

#     for i in range(0, len(embs), BATCH_SIZE):
#         index.add(embs[i:i+BATCH_SIZE])

#     path = f"{PRODUCT_INDEX_DIR}/{category}.index"
#     faiss.write_index(index, path)

# print("✅ All product indexes saved")
