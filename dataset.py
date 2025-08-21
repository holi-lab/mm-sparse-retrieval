#dataset.py
from torch.utils.data import Dataset, get_worker_info
import torch
from PIL import Image
import h5py

import lmdb
import io
import os
import torch.nn.functional as F
import numpy as np
from functools import lru_cache
import json
from torch.nn.utils.rnn import pad_sequence
from itertools import chain
from transformers.tokenization_utils_base import BatchEncoding
from collections import OrderedDict
import os, io, json, lmdb, numpy as np, torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image

class TrainDataset(Dataset):
    def __init__(
        self,
        texts_dict,
        txtid2row,
        imgid2row,
        text_embs,
        img_embs,
        text_image_pairs,
        img_paths,
        image_root,
        load_dense_model,
        use_all_tokens,
        train_from_proj,
    ):
        super().__init__()
        self.texts_dict = texts_dict
        self.txtid2row = txtid2row
        self.imgid2row = imgid2row
        self.text_embs = text_embs
        self.img_embs = img_embs
        self.pairs = text_image_pairs
        self.img_paths = img_paths
        self.img_root = image_root
        self.load_dense_model = load_dense_model
        self.use_all_tokens = use_all_tokens
        self.train_from_proj = train_from_proj


        # Store only LMDB path, environment will be opened in open_handlers()
        self.txt_lmdb_path = (
            os.path.join(self.img_root, "train_text_before_proj_emb.lmdb")
            if self.train_from_proj
            else os.path.join(self.img_root, "train_text_emb.lmdb")
        )
        self.img_lmdb_path = (
            os.path.join(self.img_root, "train_image_before_proj_emb.lmdb")
            if self.train_from_proj
            else os.path.join(self.img_root, "train_image_emb.lmdb")
        )

        # env/txn are still None
        self.env_txt = None
        self.txn_txt = None
        self.env_img = None
        self.txn_img = None

    def open_handlers(self):
        """Worker init 시 한 번만 호출: LMDB env + txn 을 연다."""
        if self.env_txt is None:
            self.env_txt = lmdb.open(
                self.txt_lmdb_path,
                readonly=True, lock=False, readahead=False, meminit=False, subdir=False
            )
            self.txn_txt = self.env_txt.begin(write=False, buffers=True)

        if self.env_img is None:
            self.env_img = lmdb.open(
                self.img_lmdb_path,
                readonly=True, lock=False, readahead=False, meminit=False, subdir=False
            )
            self.txn_img = self.env_img.begin(write=False, buffers=True)

    def __len__(self):
        return len(self.pairs)

    def __del__(self):
        try:
            if self.env_txt is not None: self.env_txt.close()
            if self.env_img is not None: self.env_img.close()
        except:
            pass

    def _get_text_emb(self, text_id):
        data = self.txn_txt.get(str(text_id).encode("ascii"))
        arr = np.load(io.BytesIO(data), allow_pickle=False)
        return torch.from_numpy(arr)

    def _get_img_emb(self, image_id):
        data = self.txn_img.get(str(image_id).encode("ascii"))
        arr = np.load(io.BytesIO(data), allow_pickle=False)
        return torch.from_numpy(arr)

    def __getitem__(self, idx):
        
        if self.use_all_tokens and (self.txn_txt is None or self.txn_img is None):
                self.open_handlers()
        # Assume open_handlers() is already called in worker_init_fn
        text_id, image_id = self.pairs[idx]
        text  = self.texts_dict[text_id]
        img_p = self.img_paths[image_id]
        text_emb, img_emb = None, None
        if self.load_dense_model:
            with h5py.File(os.path.join(self.img_root, "train.h5"), 'r', swmr = True) as f:
                img = f[img_p][()]
            img = Image.fromarray(img).convert("RGB")
        else:
            img = None
        if self.use_all_tokens:
            text_emb = self._get_text_emb(text_id)
            img_emb  = self._get_img_emb(image_id)
        else:
            text_emb = torch.from_numpy(self.text_embs[self.txtid2row[text_id]])
            img_emb  = torch.from_numpy(self.img_embs[self.imgid2row[image_id]])


        return {
            "type": "train",
            "text": text,
            "text_emb": text_emb,
            "img_emb": img_emb,
            "img": img,
        }
class TripleDataset(Dataset):
    def __init__(self, texts_dict, image_root, train_triplets):
        super().__init__()
        self.texts_dict = texts_dict
        # self.txtid2row = txtid2row
        # self.imgid2row = imgid2row
        # self.text_embs = text_embs
        # self.img_embs = img_embs
        # self.text_image_pairs = text_image_pairs
        # self.img_paths = img_paths
        self.img_root =image_root
        # self.load_dense_model = load_dense_model
        # self.use_all_tokens = use_all_tokens
        self.train_triplets = train_triplets
    def __len__(self):
        return len(self.train_triplets)

    def __getitem__(self, index):
        seed_id, pos_id, neg_id = self.train_triplets[index]
        seed_text = self.texts_dict[seed_id]
        pos_text = self.texts_dict[pos_id]
        neg_text = self.texts_dict[neg_id]
        with h5py.File(os.path.join(self.img_root, "train_text_emb.h5"), 'r', swmr = True) as f:
            seed_emb = f[seed_id][()]
            pos_emb = f[pos_id][()]
            neg_emb = f[neg_id][()]

        return {"type": "triple_train", "seed_text": seed_text, "seed_emb": torch.tensor(seed_emb), \
                "pos_text": pos_text, "pos_emb": torch.tensor(pos_emb), \
                "neg_text": neg_text, "neg_emb": torch.tensor(neg_emb)}     


class VectorCollator:
    def __init__(self, tokenizer_cls, tokenizer_name, use_all_tokens) -> None:
        self.tokenizer_cls = tokenizer_cls
        self.tokenizer_name = tokenizer_name
        self.tokenizer = None
        self.use_all_tokens = use_all_tokens
    def __call__(self, batch):
        if self.tokenizer is None:
            self.tokenizer = self.tokenizer_cls.from_pretrained(self.tokenizer_name)
        if batch[0]["type"] == "train":
            if self.use_all_tokens:
                batch_texts = [item["text"] for item in batch]
                batch_txt_embs = [item["text_emb"] for item in batch]
                batch_img_embs = [item["img_emb"] for item in batch]
                batch_imgs = [item["img"] for item in batch]

                # Pre-calculate maximum length

                if not all(emb is None for emb in batch_txt_embs):
                    max_len = max(emb.shape[0] for emb in batch_txt_embs)
                    batch_txt_embs = torch.stack([
                        F.pad(emb, (0, 0, 0, max_len - emb.shape[0])) if emb.shape[0] < max_len else emb
                        for emb in batch_txt_embs
                    ])
                if not all(emb is None for emb in batch_img_embs):
                    batch_img_embs = torch.stack(batch_img_embs, dim=0)     # (B, D)

                batch_texts = self.tokenizer(
                    batch_texts, truncation=True, padding=True, return_tensors="pt")
            else:
                batch_texts = []
                batch_txt_embs = []
                batch_img_embs = []
                batch_imgs = []
                for item in batch:
                    batch_texts.append(item["text"])
                    batch_txt_embs.append(item["text_emb"])
                    batch_img_embs.append(item["img_emb"])
                    batch_imgs.append(item["img"])
                batch_txt_embs = torch.stack(batch_txt_embs, dim=0)
                batch_img_embs = torch.stack(batch_img_embs, dim=0)
                batch_texts = self.tokenizer(
                    batch_texts, truncation=True, padding=True, return_tensors="pt")
            # return batch_texts, batch_txt_embs, batch_img_embs
            return batch_texts, batch_txt_embs, batch_img_embs, batch_imgs 

        elif batch[0]["type"] == "triple_train":
            seed_batch_texts = []
            seed_batch_txt_embs = []
            pos_batch_texts = []
            pos_batch_txt_embs = []
            neg_batch_texts = []
            neg_batch_txt_embs = []

            for item in batch:
                seed_batch_texts.append(item["seed_text"])
                seed_batch_txt_embs.append(item["seed_emb"])  # [L, D]
                pos_batch_texts.append(item["pos_text"])
                pos_batch_txt_embs.append(item["pos_emb"])  # [L, D]
                neg_batch_texts.append(item["neg_text"])
                neg_batch_txt_embs.append(item["neg_emb"])  # [L, D]

            # Pad text embeddings
            seed_max_len = max(emb.shape[0] for emb in seed_batch_txt_embs)
            pos_max_len = max(emb.shape[0] for emb in pos_batch_txt_embs)
            neg_max_len = max(emb.shape[0] for emb in neg_batch_txt_embs)
            max_len = max(seed_max_len, pos_max_len, neg_max_len)
            dim = seed_batch_txt_embs[0].shape[1]
            seed_padded_embs = []
            pos_padded_embs = []
            neg_padded_embs = []
            # attention_masks = []

            for emb in seed_batch_txt_embs:
                pad_len = max_len - emb.shape[0]
                if pad_len > 0:
                    padded = F.pad(emb, (0, 0, 0, pad_len))  # (L, D) → (max_len, D)
                else:
                    padded = emb
                seed_padded_embs.append(padded)
            for emb in pos_batch_txt_embs:
                pad_len = max_len - emb.shape[0]
                if pad_len > 0:
                    padded = F.pad(emb, (0, 0, 0, pad_len))  # (L, D) → (max_len, D)
                else:
                    padded = emb
                pos_padded_embs.append(padded)
            for emb in neg_batch_txt_embs:
                pad_len = max_len - emb.shape[0]
                if pad_len > 0:
                    padded = F.pad(emb, (0, 0, 0, pad_len))  # (L, D) → (max_len, D)
                else:
                    padded = emb
                neg_padded_embs.append(padded)


            seed_batch_txt_embs = torch.stack(seed_padded_embs, dim=0)        # (B, max_len, D)
            seed_batch_texts = self.tokenizer(
                seed_batch_texts, truncation=True, padding='max_length', max_length=max_len, return_tensors="pt")
            pos_batch_txt_embs = torch.stack(pos_padded_embs, dim=0)        # (B, max_len, D)
            pos_batch_texts = self.tokenizer(
                pos_batch_texts, truncation=True, padding='max_length', return_tensors="pt", max_length=max_len)
            neg_batch_txt_embs = torch.stack(neg_padded_embs, dim=0)        # (B, max_len, D)
            neg_batch_texts = self.tokenizer(
                neg_batch_texts, truncation=True, padding='max_length', return_tensors="pt", max_length=max_len)
        
            return seed_batch_texts, seed_batch_txt_embs, pos_batch_texts, pos_batch_txt_embs, neg_batch_texts, neg_batch_txt_embs 
        elif batch[0]["type"] == "text":
            if self.use_all_tokens:
                batch_text_ids = []
                batch_texts = []
                batch_txt_embs = []

                for item in batch:
                    batch_text_ids.append(item["text_id"])
                    batch_texts.append(item["text"])
                    batch_txt_embs.append(item["text_emb"])  # shape: [L, D]

                # Calculate maximum length
                max_len = max(emb.shape[0] for emb in batch_txt_embs)
                dim = batch_txt_embs[0].shape[1]

                # Store padded tensors
                padded_embs = []
                # attention_masks = []

                for emb in batch_txt_embs:
                    pad_len = max_len - emb.shape[0]
                    if pad_len > 0:
                        padded = F.pad(emb, (0, 0, 0, pad_len))  # pad shape=(L, D) to (max_len, D)
                    else:
                        padded = emb
                    padded_embs.append(padded)
                    # attention_masks.append(torch.cat([torch.ones(emb.shape[0]), torch.zeros(pad_len)]))

                batch_txt_embs = torch.stack(padded_embs, dim=0)  # (B, max_len, D)
                # attention_masks = torch.stack(attention_masks, dim=0)  # (B, max_len)

                batch_texts = self.tokenizer(
                    batch_texts, truncation=True, padding=True, return_tensors="pt")

            else:
                batch_text_ids = []
                batch_texts = []
                batch_txt_embs = []
                for item in batch:
                    batch_text_ids.append(item["text_id"])
                    batch_texts.append(item["text"])
                    batch_txt_embs.append(item["text_emb"])
                batch_txt_embs = torch.stack(batch_txt_embs, dim=0)
                batch_texts = self.tokenizer(
                    batch_texts, truncation=True, padding=True, return_tensors="pt")
            return batch_text_ids, batch_texts, batch_txt_embs
        elif batch[0]["type"] == "image":
            batch_img_ids = []
            batch_img_embs = []
            batch_imgs = []
            for item in batch:
                batch_img_ids.append(item["img_id"])
                if item["img"] is not None:
                    batch_imgs.append(item["img"])
                else:
                    batch_imgs.append(None)
                if item["img_emb"] is not None:
                    batch_img_embs.append(item["img_emb"])
                else:
                    batch_img_embs.append(None)
            if item["img_emb"] is not None:
                batch_img_embs = torch.stack(batch_img_embs, dim=0)
            return batch_img_ids, batch_img_embs, batch_imgs
            # return batch_img_ids, batch_img_embs


class TextCollection(Dataset):
    def __init__(self, ids, texts, txtid2row, txt_embs, mode=None, use_all_tokens=False, img_root=None, train_from_proj = False, load_dense_model = False):
        super().__init__()
        self.ids = ids
        self.texts = texts
        self.txtid2row = txtid2row
        self.txt_embs = txt_embs
        self.mode = mode
        self.use_all_tokens = use_all_tokens
        self.img_root = img_root
        self.train_from_proj = train_from_proj
        self.load_dense_model = load_dense_model
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        item_id = self.ids[index]
        item_text = self.texts[index]
        if self.use_all_tokens:
            if self.train_from_proj:
                with h5py.File(os.path.join(self.img_root, f"{self.mode}_text_before_proj_emb.h5"), 'r', swmr = True) as f:
                    item_emb = f[item_id][()]
            elif self.load_dense_model:
                item_emb = self.txt_embs[self.txtid2row[item_id]]
            else:
                with h5py.File(os.path.join(self.img_root, f"{self.mode}_text_emb.h5"), 'r', swmr = True) as f:
                    item_emb = f[item_id][()]
        else:
            item_emb = self.txt_embs[self.txtid2row[item_id]]
        return {"type": "text", "text_id": item_id, "text": item_text, "text_emb": torch.tensor(item_emb)}


class ImageCollection(Dataset):
    def __init__(self, ids, imgid2row, img_embs, img_paths, image_root, mode=None, load_dense_model=False, use_all_tokens=False, train_from_proj = False, dataset = "coco"):
        super().__init__()
        self.ids = ids
        self.imgid2row = imgid2row
        self.img_embs = img_embs
        self.img_paths = img_paths
        self.img_root = image_root
        self.mode = mode
        self.load_dense_model = load_dense_model
        self.use_all_tokens = use_all_tokens
        self.train_from_proj = train_from_proj
        self.dataset = dataset
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        item_id = self.ids[index]
        img_path = self.img_paths[item_id]
        item_emb = None
        if self.load_dense_model:
            if self.dataset == "flickr30k":
                with h5py.File(os.path.join(self.img_root, f"image.h5"), 'r', swmr = True) as f:
                    item = f[img_path][()]
            elif self.dataset == "coco":
                with h5py.File(os.path.join(self.img_root, f"{self.mode}.h5"), 'r', swmr = True) as f:
                    item = f[img_path][()]
            
            item = Image.fromarray(item).convert("RGB")
        else:
            item = None
        if self.use_all_tokens:
            if self.train_from_proj:
                with h5py.File(os.path.join(self.img_root, f"{self.mode}_image_before_proj_emb.h5"), 'r', swmr = True) as f:
                    item_emb = f[item_id][()]
            elif self.load_dense_model:
                item_emb = self.img_embs[self.imgid2row[item_id]]
            else:
                with h5py.File(os.path.join(self.img_root, f"{self.mode}_image_emb.h5"), 'r', swmr = True) as f:
                    item_emb = f[item_id][()]
        else:
            item_emb = self.img_embs[self.imgid2row[item_id]]
        img_path = self.img_paths[self.ids[index]]

            
        if item_emb is not None: item_emb = torch.tensor(item_emb)
        return {"type": "image", "img_id": item_id, "img_emb": item_emb, "img": item}
        # return {"type": "image", "img_id": item_id, "img_emb": torch.tensor(item_emb)}

class MLMDataset(Dataset):
    def __init__(self, ids, texts, tokenizer):
        super().__init__()
        self.ids = ids
        self.texts = texts
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        item_id = self.ids[index]
        item_text = self.texts[index]
        encoded_text = self.tokenizer(
            item_text,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        return {
                "input_ids": encoded_text["input_ids"].squeeze(0),
                "attention_mask": encoded_text["attention_mask"].squeeze(0),
            }