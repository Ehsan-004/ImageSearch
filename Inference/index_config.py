import os

EMB_DIM = 2048
REP_PER_CATEGORY = 10

INDEX_DIR = "database/faiss_indexes"
os.makedirs(INDEX_DIR, exist_ok=True)

CATEGORY_REP_INDEX_PATH = f"{INDEX_DIR}/category_reps.index"
PRODUCT_INDEX_DIR = f"{INDEX_DIR}/per_category"

os.makedirs(PRODUCT_INDEX_DIR, exist_ok=True)

BATCH_SIZE = 4096
