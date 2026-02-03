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
        pl.col("label")
        ).collect(
            streaming=True
            )
    
    paths, label = df['path'], df['label']
    
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
        return pd.DataFrame(columns=["path", "label"])

    df = pd.DataFrame({"path": files, "label": labels})

    p = Path(save_dir).joinpath("data.parquet")
    p2 = Path(save_dir).joinpath("mapping.csv")
    
    df = pa.Table.from_pandas(df)
    pq.write_table(df, p)
    
    df_map = pd.DataFrame(list(folder_to_label.items()), columns=['category', 'label'])
    df_map.to_csv(p2, index=False)
    
    print(f"Datasets saved to {p}")
    return df




class ImageDataset(Dataset):
    def __init__(self, parquet_path, transform):
        self.image_paths, self.categories = read_path_labels_from_parquet(parquet_path)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        # img = read_image(path).float() / 255.0  # Normalize to [0, 1]
        with Image.open(path).convert("RGB") as img:
            img = self.transform(img)
            return img,{
                        "path": path,
                        "label": self.categories[idx]
                    }


def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    metadatas = [item[1] for item in batch]
    return images, metadatas


def extract_batch_embeddings(loader, model, device):
    model.eval()
    all_embs = []   
    all_meta = []
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(): # سرعت بالاتر با دقت نیم‌شناور
            for images, metadatas in tqdm.tqdm(loader):
                images = images.to(device, non_blocking=True)
                embs = model(images)
                embs = torch.nn.functional.normalize(embs, dim=1)
                all_embs.append(embs.cpu().numpy())
                all_meta.extend(metadatas)
                
    return np.vstack(all_embs).astype('float32'), all_meta



def extract_batch_embeddings_collection(
    loader,
    model,
    device,
    collection,
    start_id=0,
    chroma_batch_size=5000,
    output_metadata_path="data/product_metadata.parquet"
):
    model.eval()
    current_id = start_id

    emb_buffer = []
    meta_buffer = []
    id_buffer = []
    all_processed_metadata = [] 
    buffer_count = 0

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            for images, metadatas in tqdm.tqdm(loader, mininterval=1.0):
                images = images.to(device, non_blocking=True)
                embs = model(images)
                embs = torch.nn.functional.normalize(embs, dim=1)

                embs = embs.detach().cpu().numpy().astype("float32", copy=False)
                batch_size = embs.shape[0]

                ids = [str(current_id + i) for i in range(batch_size)]
                current_id += batch_size

                emb_buffer.append(embs)
                meta_buffer.extend(metadatas)
                id_buffer.extend(ids)
                buffer_count += batch_size

                if buffer_count >= chroma_batch_size:
                    all_embs = np.vstack(emb_buffer)
                    collection.add(
                        embeddings=all_embs,
                        metadatas=meta_buffer,
                        ids=id_buffer,
                    )
                    # reset buffers
                    emb_buffer = []
                    meta_buffer = []
                    id_buffer = []
                    buffer_count = 0

    if buffer_count > 0:
        all_embs = np.vstack(emb_buffer)
        collection.add(
            embeddings=all_embs,
            metadatas=meta_buffer,
            ids=id_buffer,
        )

    
    # --- بخش حیاتی: ذخیره فایل Mapping نهایی ---
    print(f"💾 Saving sync metadata to {output_metadata_path}...")
    # تبدیل لیست دیکشنری‌ها به دیتافریم و ذخیره
    df_sync = pd.DataFrame(all_processed_metadata)
    # اضافه کردن ستون faiss_id برای اطمینان (اختیاری ولی سینیوری)
    # df_sync['faiss_id'] = range(len(df_sync))
    
    df_sync.to_parquet(output_metadata_path, index=False)
    print(f"✅ Sync complete. Total processed: {len(df_sync)}")
    
    return current_id





def extract_embeddings(model, path, device, transform):
    model.to(device)
    model.eval()
    
    with Image.open(path).convert("RGB") as img:
        with torch.no_grad():
            img = transform(img).unsqueeze(0).to(device)
            emb = model(img)
            emb = torch.nn.functional.normalize(emb, dim=1)
            
            return emb

def extract_embeddings_img(img, model, device, transform):
    model.to(device)
    model.eval()

    with torch.no_grad():
        img = transform(img).unsqueeze(0).to(device)
        emb = model(img)
        emb = torch.nn.functional.normalize(emb, dim=1)
        
        return emb
    
