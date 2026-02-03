from pathlib import Path

import torch
from torch.utils.data import Dataset

from PIL import Image
import numpy as np
import pandas as pd
import tqdm
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq




DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DB_STORE_PATH = "./database"
CATEGORIES_COLLECTION_NAME = "categories"
PRODUCTS_COLLECTION_NAME = "products"


def read_path_labels_from_parquet(parquet_path: str):
    df = pl.scan_parquet(
        parquet_path,
        use_statistics=True,
        rechunk=True,
        low_memory=False,
    ).select(
        pl.col("path"),
        pl.col("category")
        ).collect(
            streaming=True
            )
    
    paths, label = df['path'], df['category']
    
    del df
    return paths, label


def preprocess_dataset(root_path, ignore=None, save_dir=""):
    root_path = Path(root_path)
    ignore = ignore or []

    files, labels = [], []

    folders = [f for f in root_path.iterdir() if f.is_dir() and f.name not in ignore]
    folder_to_label = {folder.name: idx for idx, folder in enumerate(sorted(folders))}

    for folder in folders:
        for file_path in folder.rglob("*"):
            if file_path.is_file():
                files.append(str(file_path.resolve()))
                labels.append(int(folder_to_label[folder.name]))

    if not files:
        print("directory is empty!")
        return pd.DataFrame(columns=["path", "category"])

    df = pd.DataFrame({"path": files, "category": labels})

    p = Path(save_dir).joinpath("data.parquet")
    p2 = Path(save_dir).joinpath("mapping.csv")
    
    df = pa.Table.from_pandas(df)
    pq.write_table(df, p)
    
    df_map = pd.DataFrame(list(folder_to_label.items()), columns=['category', 'label'])
    df_map.to_csv(p2, index=False)
    
    print(f"Datasets saved to {p}")
    return df


class ImageDataset(Dataset):
    def __init__(self, parquet_path):
        self.image_paths, self.categories = read_path_labels_from_parquet(parquet_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        with Image.open(path).convert("RGB") as img:
            return img,{
                        "path": path,
                        "category": self.categories[idx]
                    }


def collate_fn(batch):
    # batch = [(PIL.Image, metadata), ...]
    images = [item[0] for item in batch]
    metadatas = [item[1] for item in batch]
    return images, metadatas


def extract_batch_embeddings(loader, model, processor, device=DEVICE):
    model.to(device)
    model.eval()
    
    all_embs = []
    all_meta = []
    
    with torch.no_grad():
        for images, metadatas in tqdm.tqdm(loader):

            inputs = processor(images=images, return_tensors="pt").to(device)
            outputs = model(**inputs)
            embs = outputs.last_hidden_state[:, 0, :]
            embs = torch.nn.functional.normalize(embs, dim=1)

            # a list of PIL.Images
            # inputs = processor(images=images, return_tensors="pt").to(device)
            # outputs = model.get_image_features(**inputs)
            # embs = outputs.pooler_output
            # embs = torch.nn.functional.normalize(embs, dim=1)
            
            all_embs.append(embs.cpu().numpy())
            all_meta.extend(metadatas)
            
    return np.vstack(all_embs).astype('float32'), all_meta


# -------------------------
# Single image from PIL.Image
# -------------------------
def extract_embeddings_img(img, model, processor, device=DEVICE):
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        inputs = processor(images=img, return_tensors="pt").to(device)
        outputs = model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :]
        emb = torch.nn.functional.normalize(emb, dim=1)


        # inputs = processor(images=img, return_tensors="pt").to(device)
        # outputs = model.get_image_features(**inputs)
        # emb = outputs.pooler_output
        # emb = torch.nn.functional.normalize(emb, dim=1)
        return emb

