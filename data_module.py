# data_module.py
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch
import json
from huggingface_hub import hf_hub_download
from datasets import load_dataset
from dataset import TrainDataset, TextCollection, ImageCollection, VectorCollator
from transformers import AutoTokenizer
from config import Config
from typing import Tuple, Dict, Any
from collections import defaultdict
import numpy as np
import random
from torch.utils.data import get_worker_info
class RetrievalDataModule(pl.LightningDataModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.vector_collator = VectorCollator(AutoTokenizer, "distilbert-base-uncased", config.use_all_tokens)

    def setup(self, stage: str):
        if self.config.finetune_from_proj:
            dense_embs = load_dataset( "parquet", data_files={  "img_emb": f'./.cache/{self.config.dataset}/img_embs_before_proj_{self.config.backbone_model}.parquet', "text_emb": f'./.cache/{self.config.dataset}/text_embs_before_proj_{self.config.backbone_model}.parquet'},   keep_in_memory=True).with_format("numpy")
        else:
            dense_embs = load_dataset(self.config.data, data_files={"img_emb": "img_embs.parquet", "text_emb": "text_embs.parquet"}, keep_in_memory=True).with_format("numpy")
        text_ids = dense_embs['text_emb']["id"]
        text_embs = dense_embs['text_emb']['emb']
        img_ids = dense_embs['img_emb']["id"]
        img_embs = dense_embs['img_emb']['emb']
        txtid2row = dict(zip(text_ids, range(len(text_ids))))
        imgid2row = dict(zip(img_ids, range(len(img_ids))))

        meta_data = json.load(open(hf_hub_download(repo_id=self.config.data, repo_type="dataset", filename="dataset_meta.json")))
        
        train_image_ids, train_captions, train_caption_ids, train_pairs, train_image_paths = [], [], [], [], {}
        val_image_ids, val_captions, val_caption_ids, val_qrels, val_image_paths = [], [], [], defaultdict(dict), {}
        test_image_ids, test_captions, test_caption_ids, test_qrels, test_image_paths = [], [], [], defaultdict(dict), {}

        for image in meta_data['images']:
            image_id = str(image["imgid"])
            caption_texts = [sent["raw"] for sent in image["sentences"]]
            caption_ids = [str(sent["sentid"]) for sent in image["sentences"]]
            image_path = image["filename"]
            if image["split"] == "train":
                train_image_ids.append(image_id)
                train_captions.extend(caption_texts)
                train_caption_ids.extend(caption_ids)
                train_pairs.extend([(sent_id, image_id) for sent_id in caption_ids])
                train_image_paths[image_id] = image_path
            elif image['split'] == "val":
                val_image_ids.append(image_id)
                val_captions.extend(caption_texts)
                val_caption_ids.extend(caption_ids)
                for sent_id in caption_ids:
                    val_qrels[sent_id][image_id] = 1
                val_image_paths[image_id] = image_path
            elif image['split'] == 'test':
                test_image_ids.append(image_id)
                test_captions.extend(caption_texts)
                test_caption_ids.extend(caption_ids)    
                for sent_id in caption_ids:
                    test_qrels[sent_id][image_id] = 1
                test_image_paths[image_id] = image_path

        if self.config.debug:
            num_debug = 100
            train_image_ids = train_image_ids[:num_debug]
            train_caption_ids = train_caption_ids[:num_debug]
            train_captions = train_captions[:num_debug]
            train_pairs = train_pairs[:num_debug]
            train_image_paths = dict(list(train_image_paths.items())[:num_debug])
            val_image_ids = val_image_ids[:num_debug]
            val_caption_ids = val_caption_ids[:num_debug]
            val_captions = val_captions[:num_debug]
            val_image_paths = dict(list(val_image_paths.items())[:num_debug])
            val_qrels = defaultdict(dict, {k: val_qrels[k] for k in val_caption_ids})
            # test_image_ids = test_image_ids[:num_debug]
            # test_caption_ids = test_caption_ids[:num_debug]
            # test_captions = test_captions[:num_debug]
            # test_image_paths = dict(list(test_image_paths.items())[:num_debug])
            # test_qrels = defaultdict(dict, {k: test_qrels[k] for k in test_caption_ids})

        self.train_dataset = TrainDataset(
            dict(zip(train_caption_ids, train_captions)), txtid2row, imgid2row, text_embs, img_embs,
            train_pairs, train_image_paths, self.config.image_root, self.config.load_dense_model, self.config.use_all_tokens, self.config.finetune_from_proj
        )
        self.val_dataset = (
            TextCollection(val_caption_ids, val_captions, txtid2row, text_embs, "val", self.config.use_all_tokens, self.config.image_root, self.config.finetune_from_proj, self.config.load_dense_model),
            ImageCollection(val_image_ids, imgid2row, img_embs, val_image_paths, self.config.image_root, "val", self.config.load_dense_model, self.config.use_all_tokens, self.config.finetune_from_proj, self.config.dataset), 
            val_qrels
        )
        self.test_dataset = (
            TextCollection(test_caption_ids, test_captions, txtid2row, text_embs, "test", self.config.use_all_tokens, self.config.image_root, self.config.finetune_from_proj, self.config.load_dense_model),
            ImageCollection(test_image_ids, imgid2row, img_embs, test_image_paths, self.config.image_root, "test", self.config.load_dense_model, self.config.use_all_tokens, self.config.finetune_from_proj, self.config.dataset),
            test_qrels
        )
    def seed_worker(self, worker_id):
        worker = get_worker_info()
        dataset = worker.dataset       # TrainDataset instance
        if self.config.use_all_tokens:
            dataset.open_handlers() 
        # torch.initial_seed() itself gives different seeds for each process.
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    def train_dataloader(self):
        if not hasattr(self, "_train_loader"):

            self._train_loader =  DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.config.train_batch_size,
            num_workers=self.config.num_workers,
            collate_fn=self.vector_collator,
            persistent_workers=(self.config.num_workers > 0),
            pin_memory=True,
            worker_init_fn=self.seed_worker,   
            multiprocessing_context=mp.get_context('fork') if self.config.num_workers > 0 else None,
            prefetch_factor=4 if self.config.num_workers > 0 else None        # ← picklable function
            )
        return self._train_loader
    def val_dataloader(self):
        return DataLoader(self.val_dataset[0], batch_size=self.config.txt_eval_batch_size, shuffle=False, num_workers=self.config.num_workers, collate_fn=self.vector_collator)

    def test_dataloader(self):
        return DataLoader(self.test_dataset[0], batch_size=self.config.txt_eval_batch_size, shuffle=False, num_workers=self.config.num_workers, collate_fn=self.vector_collator)
