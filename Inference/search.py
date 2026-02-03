# import faiss
# import numpy as np
# from collections import Counter

# import torch
# from Inference.dataset import extract_embeddings_img
# from Inference.model import get_model, get_transforms
# from Inference.index_config import *

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = get_model().to(DEVICE)
# model.eval()
# transforms = get_transforms()

# # =========================
# # Load indexes
# # =========================
# rep_index = faiss.read_index(CATEGORY_REP_INDEX_PATH)
# rep_categories = np.load(f"{INDEX_DIR}/rep_categories.npy", allow_pickle=True)

# category_indexes = {}

# def load_category_index(category):
#     if category not in category_indexes:
#         path = f"{PRODUCT_INDEX_DIR}/{category}.index"
#         category_indexes[category] = faiss.read_index(path)
#     return category_indexes[category]

# # =========================
# # Search
# # =========================
# def search(img, k=10):
#     q = extract_embeddings_img(img, model, DEVICE, transforms)
#     q = q.cpu().numpy().reshape(1, -1).astype("float32")
#     faiss.normalize_L2(q)

#     # -------- Stage 1 --------
#     D, I = rep_index.search(q, 30)

#     votes = Counter()
#     for idx in I[0]:
#         votes[rep_categories[idx]] += 1

#     top_category, count = votes.most_common(1)[0]

#     if count >= int(0.7 * REP_PER_CATEGORY):
#         candidate_categories = [top_category]
#     else:
#         candidate_categories = [c for c, _ in votes.most_common(3)]

#     # -------- Stage 2 --------
#     results = []

#     for cat in candidate_categories:
#         index = load_category_index(cat)
#         D2, I2 = index.search(q, k)
#         for d, i in zip(D2[0], I2[0]):
#             results.append((cat, d, i))

#     results.sort(key=lambda x: x[1])
#     return results[:k]


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

# --- group by category ---
# category_to_embeddings = defaultdict(list)

# for emb, meta in zip(embeddings, metadatas):
#     cat = meta["category"]  # ← اگر اسمش فرق داره همینو عوض کن
#     category_to_embeddings[cat].append(emb)

# # --- select representatives ---
# rep_embeddings = []
# rep_categories = []

# for cat, embs in category_to_embeddings.items():
#     embs = np.array(embs)

#     if len(embs) <= REP_PER_CATEGORY:
#         selected = embs
#     else:
#         idx = np.random.choice(len(embs), REP_PER_CATEGORY, replace=False)
#         selected = embs[idx]

#     rep_embeddings.append(selected)
#     rep_categories.extend([cat] * len(selected))

# rep_embeddings = np.vstack(rep_embeddings).astype("float32")

# # cosine similarity
# faiss.normalize_L2(rep_embeddings)






# data = products_collection.get(include=["embeddings"])
# all_embeddings = data["embeddings"]
# ids = data["ids"]

# print("from saerch")
# print(ids[2:20])

# M = 50  # تعداد اتصالات در گراف HNSW (بیشتر = دقت بالاتر، ساخت کندتر)
# index = faiss.IndexHNSWFlat(EMB_DIM, M)
# index.hnsw.efConstruction = 200 # دقت در زمان ساخت ایندکس
# index.add(all_embeddings.astype('float32')) 

# # در زمان جستجو
# index.hnsw.efSearch = 128


# def search(img, k=10):
#     query_embedding = extract_embeddings_img(img, model, DEVICE, transforms)
#     query_embedding = query_embedding.cpu().numpy().flatten().reshape(1,-1)
#     D, I = index.search(query_embedding, k)

#     return D, I




# index_stage1 = faiss.IndexFlatIP(EMB_DIM)
# faiss.normalize_L2(rep_embeddings)
# index_stage1.add(rep_embeddings)




