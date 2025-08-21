# lightning_model.py
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
import wandb
import ir_measures
from ir_measures import R, MRR
from pathlib import Path
from model import RetrievalModel, BLIPRetrievalModel, RetrievalProjModel
from loss import BICELoss, BaselineLoss
from regularizer import L1
from utils import write_trec_file
from dataset import VectorCollator
from typing import Dict, Any, Tuple, Callable
from config import Config
from collections import defaultdict, OrderedDict
import json
from tqdm import tqdm
from collections import defaultdict, OrderedDict

class RetrievalLightningModule(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config.__dict__)
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_all_tokens = config.use_all_tokens
        # Initialize model
        if config.finetune_from_proj:
            self.model = RetrievalProjModel(device=self.device_type, args=config)
        elif config.load_dense_model:
            self.model = BLIPRetrievalModel(device=self.device_type, args=config)
        else:
            self.model = RetrievalModel(args=config)
        self.temp = nn.Parameter(config.temp*torch.ones([]))   
        self.w1 = config.w1
        self.w2 = config.w2
        self.lam1 = config.lam1
        self.lam2 = config.lam2
        if self.config.baseline:
            self.loss_fn = BaselineLoss(temp=self.temp, q_reg=config.q_reg, d_reg=config.d_reg, T=None)
        else:
            self.loss_fn = BICELoss(temp=self.temp, q_reg=config.q_reg, d_reg=config.d_reg, T=None,
            w1 = self.w1, w2 = self.w2, 
            lam1 = self.lam1, lam2 = self.lam2,
            self_supervised = config.self_supervised, pooling = config.pooling)  # T will be updated in configure_optimizers
        self._microbatch_counter = 0        
        # Metrics
        self.highest_recall_1 = -1
        self.early_stop_counter = 0
        self.best_epoch = -1
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.vector_collator = VectorCollator(AutoTokenizer, "distilbert-base-uncased", self.use_all_tokens,)
        if self.config.disable_mask: 
            self.mask_ratio = torch.tensor(0.0)
        else:
            self.mask_ratio = torch.tensor(1.0)  
            self.step = self.mask_ratio/(self.config.epochs * 0.95)
    def forward(self, dense_texts, input_ids, attention_mask, dense_imgs=None, imgs=None, bow=False, mask_ratio=0.0):
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)
        mask_decision = torch.bernoulli(torch.tensor(mask_ratio)).item()
        
        if self.config.load_dense_model:
            dense_vec_texts, single_sparse_texts, _, _ = self.model.forward_text(input_ids, attention_mask, bow=bow, mask_ratio=mask_ratio, all_tokens = self.config.use_all_tokens)
            dense_vec_imgs, single_sparse_imgs, _, _ = self.model.forward_image(imgs, all_tokens = self.config.use_all_tokens) if imgs is not None else None
        elif self.config.finetune_from_proj:
            dense_vec_texts, single_sparse_texts, _, _ = self.model.forward_text(dense_texts, attention_mask, input_ids = input_ids, bow=bow, mask_ratio=mask_decision if mask_decision else 0, all_tokens = self.config.use_all_tokens)
            dense_vec_imgs, single_sparse_imgs, _, _ = self.model.forward_image(dense_imgs, all_tokens = self.config.use_all_tokens) if imgs is not None else None
        else:
            assert self.config.single
            dense_vec_texts, single_sparse_texts, _, _ = self.model.forward_text(dense_texts, input_ids, attention_mask, bow=bow, mask_ratio=mask_decision if mask_decision else 0)
            dense_vec_imgs, single_sparse_imgs, _, _ = self.model.forward_image(dense_imgs) if dense_imgs is not None else None
        return dense_vec_texts, dense_vec_imgs, single_sparse_texts, single_sparse_imgs, None, None, None, None

    def training_step(self, batch, batch_idx):
        batch_tokenized_texts, dense_texts, dense_imgs, imgs = batch
        batch_tokenized_texts = batch_tokenized_texts.to(self.device)
        if isinstance(dense_texts, torch.Tensor): dense_texts = dense_texts.to(self.device)
        if isinstance(dense_imgs, torch.Tensor): dense_imgs = dense_imgs.to(self.device)
        with torch.amp.autocast('cuda'):
            dense_vec_texts, dense_vec_imgs, single_sparse_texts, single_sparse_imgs, _, _, _, _ = self.forward(
                dense_texts, batch_tokenized_texts["input_ids"], batch_tokenized_texts["attention_mask"], 
                dense_imgs, imgs, bow=self.config.bow, mask_ratio=self.mask_ratio
            )
            rel_loss, reg, text_reg, img_reg, sup_loss, distill_loss, single_text_reg, single_imgs_reg = self.loss_fn(dense_vec_texts, dense_vec_imgs, single_sparse_texts, single_sparse_imgs, None, None, None, None, self.config.bow, self.use_all_tokens)
            if reg is not None:
                loss = (rel_loss + reg)
            else:
                loss = rel_loss
        optimizer = self.trainer.optimizers[0] 
        current_lr = optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else optimizer.param_groups[0]['lr']
        self.log_dict({
            'train/rel_loss': rel_loss,
            'train/temp': self.temp,
            'train/loss': loss,
            'train/sup_loss': sup_loss,
            'train/distill_loss': distill_loss,
            'train/regularizer_t': self.loss_fn.q_regularizer.t,
            'train/regularizer_weight_t': self.loss_fn.q_regularizer.weight_t,
            'train/lr': current_lr,


        }, on_step=True, prog_bar=True, logger=True)
        if reg is not None:
            self.log_dict({
                'train/reg': reg,
                'train/text_reg': text_reg,
                'train/img_reg': img_reg,
                'train/single_text_reg': single_text_reg, 
                'train/single_imgs_reg': single_imgs_reg
            }, on_step=True, prog_bar=True, logger=True)

        return loss
    def validation_step(self, batch, batch_idx):
        # Save batch for later evaluation
        return

    def on_validation_epoch_end(self):
        recall_all = self.evaluate(self.trainer.datamodule.val_dataset)
        logs = {}
        for mode, vals in recall_all.items():
            # Example: mode='dense', vals={'R@1':0.97, ...}
            logs.update({
                f'val/{mode}_recall1':  vals['R@1'],
                f'val/{mode}_recall5':  vals['R@5'],
                f'val/{mode}_recall10': vals['R@10'],
                f'val/{mode}_mrr10':    vals['MRR@10'],
            })
        # Now log everything at once
        if len(self.config.pooling) == 1:
            if logs[self.config.best_monitor] > self.highest_recall_1:
                self.highest_recall_1 = logs[self.config.best_monitor]
                self.early_stop_counter = 0
                self.best_epoch = self.current_epoch
        elif len(self.config.pooling) > 1:
            if logs[self.config.best_monitor] > self.highest_recall_1:
                self.highest_recall_1 = logs[self.config.best_monitor]
                self.early_stop_counter = 0
                self.best_epoch = self.current_epoch
        else:
            self.early_stop_counter += 1
        logs.update({'highest_recall_1': self.highest_recall_1})
        self.log_dict(logs, on_epoch = True, prog_bar = True, logger = True)

    # ToDo: Pre-define array length instead of using append
    def evaluate(self,
                    dataset,
                    return_run_file = False, 
                    mask_ratio=0.0):
        """
        Returns a dict of (R@1, R@5, R@10, MRR@10) for five modes.
        """
        text_collection, image_collection, qrels = dataset
        flops = 0 # Only implemented for single for now
        self.model.eval()
                
        def truncate_topk(x: torch.Tensor, k: int = 1000):
            """
            Keeps only the top k activation values for each row vector and fills the rest with zeros.
            Args:
                x (Tensor): Matrix of shape [N, V]
                k (int): Number of top elements to keep
            Returns:
                Tensor: Sparse matrix of the same size
            """
            # Get values and indices for each row ([N, k])
            topk_vals, topk_idx = x.topk(k, dim=1)
            # Create zero matrix of same size as original
            sparse = torch.zeros_like(x)
            # Restore only top k values using scatter
            sparse.scatter_(1, topk_idx, topk_vals)
            return sparse


        if self.config.self_supervised:
            # 1) Extract embeddings (same as above example)
            text_dense_list, text_single_list, text_ids = [], [], []
            img_dense_list, img_single_list, image_ids = [], [], []
            score_dense_list, score_single_list = [], []
            # Image embeddings
            img_loader = DataLoader(
                image_collection,
                batch_size=self.config.img_eval_batch_size,
                shuffle=False, num_workers=8,
                collate_fn=self.vector_collator
            )
            for batch in tqdm(img_loader, desc='encode images'):
                ids = batch[0]    
                if self.config.load_dense_model:
                    img_feats = batch[2]
                else:
                    img_feats = batch[1]
                image_ids.extend(ids)
                with torch.no_grad(), torch.cuda.amp.autocast(self.config.use_amp):
                    id_, is_, _, _ = self.model.forward_image(
                        img_feats.to(self.device),
                        all_tokens = self.config.use_all_tokens,
                    )
                    if id_ is not None:
                        img_dense_list.append(id_)
                    if is_ is not None:
                        if self.config.truncate_topk > 0:
                            img_single_list.append(truncate_topk(is_, self.config.truncate_topk))
                        else:
                            img_single_list.append(is_)
            if len(img_dense_list) > 0: I_dense  = torch.cat(img_dense_list, dim=0)
            if len(img_single_list) > 0 : I_single = torch.cat(img_single_list, dim=0)

            # Text embeddings
            txt_loader = DataLoader(
                text_collection,
                batch_size=self.config.txt_eval_batch_size,
                shuffle=False, num_workers=8,
                collate_fn=self.vector_collator
            )
            for batch in tqdm(txt_loader, desc='encode text'):
                ids = batch[0]
                text_ids.extend(ids)
                if self.config.load_dense_model:
                    txt_inputs = batch[1]
                else:
                    input_ids = batch[1]["input_ids"]
                    attn_mask = batch[1]["attention_mask"]
                    txt_inputs = batch[2]
                if self.config.load_dense_model:
                    with torch.no_grad(), torch.amp.autocast('cuda'):
                        td, ts, _, _ = self.model.forward_text(
                            txt_inputs["input_ids"].to(self.device),
                            txt_inputs["attention_mask"].to(self.device)
                        )

                        if td is not None: 
                            score_dense_list.append((td @ I_dense.t()).cpu())
                        if ts is not None: 
                            if self.config.truncate_topk > 0:
                                score_single_list.append((truncate_topk(ts, self.config.truncate_topk)@I_single.t()).cpu())
                            else:
                                score_single_list.append((ts@I_single.t()).cpu())
                            flops += torch.sum((ts != 0).float().sum(dim = 0)*
                                               (I_single!=0).float().sum(dim = 0))
                else:
                    with torch.no_grad(), torch.amp.autocast('cuda'):
                        td, ts, _, _ = self.model.forward_text(
                            dense_vec = txt_inputs.to(self.device),
                            attention_mask = attn_mask.to(self.device),
                            input_ids = input_ids.to(self.device),
                            bow = self.config.bow,
                            all_tokens = self.config.use_all_tokens,
                        )
                        if td is not None: 
                            score_dense_list.append((td @ I_dense.t()).cpu())
                            # text_dense_list.append(td)
                        if ts is not None:
                            score =  ts@I_single.t()
                            score_single_list.append(score.cpu())
                            flops +=  torch.sum(score.sum())
                            
                            # text_single_list.append(ts)

            # 2) Basic score matrix
            score_dict = {}
            if len(score_dense_list) > 0:
                score_dict['dense'] = torch.cat(score_dense_list)
            if len(score_single_list) > 0:
                score_dict['single'] = torch.cat(score_single_list)
            # 3) Final scores by combination
            weight_map = {
                'dense': self.config.w1,
                'single': self.config.w2
            }

            # 1) Store single scores
            if 'single' in score_dict:
                avg_flops = flops / (score_dict['single'].shape[0] * score_dict['single'].shape[1] )
                self.log_dict({'flops': avg_flops}, on_epoch = True, prog_bar = True, logger = True)
            all_scores = {k: score_dict[k] for k in self.config.pooling if k in score_dict}

            # 2) 2-combination: dense + other (both dense and other pooling must exist)
            if 'dense' in self.config.pooling and 'dense' in score_dict:
                for p in self.config.pooling:
                    if p == 'dense' or p not in score_dict:
                        continue
                    key = f'dense_{p}'
                    all_scores[key] = weight_map['dense'] * score_dict['dense'] + weight_map[p] * score_dict[p]

            # 3) If 3 or more, calculate all combinations
            if len(self.config.pooling) >= 3:
                all_score = 0
                valid_keys = [p for p in self.config.pooling if p in score_dict]
                for p in valid_keys:
                    all_score += weight_map[p] * score_dict[p]
                all_scores['all'] = all_score

        else:
            # 1) Extract embeddings (same as above example)
            text_list, text_ids = [], []
            img_list, image_ids = [], []
            score_list = []
            # Image embeddings
            img_loader = DataLoader(
                image_collection,
                batch_size=self.config.img_eval_batch_size,
                shuffle=False, num_workers=8,
                collate_fn=self.vector_collator
            )
            for batch in tqdm(img_loader, desc='encode images'):
                ids = batch[0]    
                if self.config.load_dense_model:
                    img_feats = batch[2]
                else:
                    img_feats = batch[1]
                image_ids.extend(ids)
                with torch.no_grad(), torch.cuda.amp.autocast(self.config.use_amp):
                    id_, is_, im_, ic_ = self.model.forward_image(
                        img_feats.to(self.device),
                        all_tokens = self.config.use_all_tokens,
                    )
                    if self.config.single:
                        img_list.append(is_)


            I  = torch.cat(img_list, dim=0)

            # Text embeddings
            txt_loader = DataLoader(
                text_collection,
                batch_size=self.config.txt_eval_batch_size,
                shuffle=False, num_workers=8,
                collate_fn=self.vector_collator
            )
            # ids, txt_inputs, _
            for batch in tqdm(txt_loader, desc='encode text'):
                ids = batch[0]
                text_ids.extend(ids)
                if self.config.load_dense_model:
                    txt_inputs = batch[1]
                else:
                    input_ids = batch[1]["input_ids"]
                    attn_mask = batch[1]["attention_mask"]
                    txt_inputs = batch[2]
                if self.config.load_dense_model:
                    with torch.no_grad(), torch.amp.autocast('cuda'):
                        td, ts, _, _ = self.model.forward_text(
                            txt_inputs["input_ids"].to(self.device),
                            txt_inputs["attention_mask"].to(self.device)
                        )
                        if self.config.single:
                            text_list.append(ts)
                            score_list.append(ts @ I.t())
                else:
                    with torch.no_grad(), torch.amp.autocast('cuda'):
                        td, ts, _, _ = self.model.forward_text(
                            dense_vec=txt_inputs.to(self.device),
                            attention_mask=attn_mask.to(self.device),
                            input_ids = input_ids.to(self.device),
                            bow = self.config.bow,
                            all_tokens = self.config.use_all_tokens,
                        )
                        if self.config.single:
                            text_list.append(ts)
                            score_list.append(ts @ I.t())
            # 2) Basic score matrix
            S  = torch.cat(score_list)
            # 3) Final scores by combination
            w1, w2 = self.config.w1, self.config.w2
            all_scores = {
                f'{self.config.pooling[0]}': S,
            }

                    # 4) Calculate run + metrics for all modes
        all_runs = {}
        all_metrics = {}
        for mode, S in all_scores.items():
            # build run dict
            run = defaultdict(OrderedDict)
            for qi, qid in enumerate(text_ids):
                sorted_idxs = S[qi].argsort(dim=0, descending=True)
                for gj in sorted_idxs[:10]:
                    run[qid][image_ids[gj]] = float(S[qi, gj].cpu())
            all_runs[mode] = run
            # compute metrics
            if qrels:
                m = ir_measures.calc_aggregate(
                    [R@1, R@5, R@10, MRR@10], qrels, run)
                all_metrics[mode] = {
                    'R@1':  m[R@1],
                    'R@5':  m[R@5],
                    'R@10': m[R@10],
                    'MRR@10': m[MRR@10],
                }
            else:
                all_metrics[mode] = {k: None for k in ['R@1','R@5','R@10','MRR@10']}            
        self.model.train()

        if return_run_file:
            # Return runs and metrics together
            return all_runs, all_metrics
        else:
            # Return only metrics
            return all_metrics

    def test_step(self, batch, batch_idx):
        return
    def on_test_start(self):
        # Now PL has finished model-device setup
        test_sparse_run, recall_all = self.evaluate(
            self.trainer.datamodule.test_dataset,
            return_run_file=True
        )
        logs = {}
        for mode, vals in recall_all.items():
            # Example: mode='dense', vals={'R@1':0.97, ...}
            logs.update({
                f'test/{mode}_recall1':  vals['R@1'],
                f'test/{mode}_recall5':  vals['R@5'],
                f'test/{mode}_recall10': vals['R@10'],
                f'test/{mode}_mrr10':    vals['MRR@10'],
            })
        # Now log everything at once
        self.log_dict(logs, on_epoch=True, prog_bar=True, logger = True, sync_dist = True)

        # Save test results
        model_dir = Path(self.config.output_dir or f"output/{self.trainer.logger.experiment.id}")
        model_dir.mkdir(exist_ok=True, parents=True)
        run_file_path = model_dir / "test_run_file.trec"
        write_trec_file(test_sparse_run, str(run_file_path))

    def configure_optimizers(self):
        if self.config.baseline:
            if self.config.finetune_from_proj:
                optimizer = torch.optim.AdamW([
                {"params": list(self.model.single_vocab_layer_norm.parameters()) +
                list(self.model.single_vocab_projector.parameters()), "lr": 5e-5, "betas": (0.9, 0.999), "weight_decay": 0.0},
                {"params": list(self.model.single_proj.parameters()) + list(self.model.text_proj.parameters()) + list(self.model.vision_proj.parameters()) + [self.temp], "lr": 1e-3}
                ], eps=1e-8, betas=(0.9, 0.999), weight_decay=0.05)
            else:
                optimizer = torch.optim.AdamW([
                {"params": list(self.model.vocab_layer_norm.parameters()) +
                list(self.model.vocab_projector.parameters()), "lr": 5e-5, "betas": (0.9, 0.999), "weight_decay": 0.0},
                {"params": list(self.model.proj.parameters()) + [self.temp], "lr": 1e-3}
                ], eps=1e-8, betas=(0.9, 0.999), weight_decay=0.05)
        else:
            if self.config.load_dense_model:
                # ToDo: Different from BLIP here?
                projection_params = (
                list(self.model.single_proj.parameters()) +
                list(self.model.max_proj.parameters()) +
                list(self.model.colbert_proj.parameters()) +
                list(self.model.single_vocab_projector.parameters()) +
                list(self.model.max_vocab_projector.parameters()) +
                list(self.model.single_vocab_layer_norm.parameters()) +
                list(self.model.max_vocab_layer_norm.parameters())
                )
                blip_params = list(filter(lambda p: p.requires_grad, self.model.blip_model.parameters()))
                optimizer = torch.optim.AdamW([
                {
                    "params": projection_params,
                    "lr": self.config.lr,  # Learning rate for projection layer
                    # "betas": (0.9, 0.999),
                    "weight_decay": 0.05
                },
                {
                    "params": blip_params,
                    "lr": self.config.blip_lr,  # Learning rate for BLIP model 
                    # "betas": (0.9, 0.999),
                    "weight_decay": 0.05
                },
                {
                    "params": [self.temp],  # Learning rate for temp parameter
                    "lr": self.config.lr,
                    # "betas": (0.9, 0.999),
                    "weight_decay": 0.05
                }
            ], eps=1e-8)
            elif self.config.finetune_from_proj:
                optimizer = torch.optim.AdamW([
                    {"params": list(self.model.parameters()) + [self.temp], "lr": self.config.lr}
                ], eps=1e-8, betas=(0.9, 0.999), weight_decay=0.05)
            else:
                optimizer = torch.optim.AdamW([
                    {"params": list(self.model.vocab_layer_norm.parameters()) + list(self.model.vocab_projector.parameters()), "lr": 5e-5, "betas": (0.9, 0.999), "weight_decay": 0.0},
                    {"params": list(self.model.proj.parameters()) + [self.temp], "lr": self.config.lr}
                ], eps=1e-8, betas=(0.9, 0.999), weight_decay=0.05)

        # Compute num_training_steps using trainer.datamodule
        if self.trainer.datamodule is not None:
            num_training_steps = self.trainer.estimated_stepping_batches
            num_warm_up = int(num_training_steps * 0.2)
        else:
            raise ValueError("DataModule not initialized. Cannot compute num_training_steps.")

        # Update loss_fn T based on num_warm_up
        self.loss_fn.q_regularizer.T = int(0.2* self.trainer.estimated_stepping_batches*self.trainer.accumulate_grad_batches)
        self.loss_fn.d_regularizer.T = int(0.2* self.trainer.estimated_stepping_batches*self.trainer.accumulate_grad_batches)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warm_up, num_training_steps=num_training_steps)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def on_save_checkpoint(self, checkpoint):
        checkpoint['q_reg_t'] = self.loss_fn.q_regularizer.t
        checkpoint['q_reg_weight_t'] = self.loss_fn.q_regularizer.weight_t
        checkpoint['d_reg_t'] = self.loss_fn.d_regularizer.t
        checkpoint['d_reg_weight_t'] = self.loss_fn.d_regularizer.weight_t

    def on_load_checkpoint(self, checkpoint):
        if self.config.mlm_trained_path is None:
            self.loss_fn.q_regularizer.t = checkpoint['q_reg_t']
            self.loss_fn.q_regularizer.weight_t = checkpoint['q_reg_weight_t']
            self.loss_fn.d_regularizer.t = checkpoint['d_reg_t']
            self.loss_fn.d_regularizer.weight_t = checkpoint['d_reg_weight_t']
    def on_after_backward(self) -> None:
        # For each backward call (=micro-batch)
        self._microbatch_counter += 1
        # Only execute at the last micro-batch of accumulate_grad_batches group
        # if self._microbatch_counter % self.trainer.accumulate_grad_batches == 0:
            # self.loss_fn.q_regularizer.step()
            # self.loss_fn.d_regularizer.step()
    

    # def setup(self, stage: str):
    #     # if stage == "fit" and self.config.wandb_id and (ckpt_path := Path(self.config.output_dir) / "checkpoint_last.ckpt").exists():
    #     #     ckpt = torch.load(ckpt_path, map_location=self.device_type)
    #     #     self.model.load_state_dict(ckpt["model_state_dict"])
    #     #     self.optimizers().load_state_dict(ckpt["optimizer_state_dict"])
    #     #     # Note: Scheduler state will be loaded in configure_optimizers
    #     #     self.highest_recall_1 = ckpt["best_recall1"]
    #     #     self.early_stop_counter = ckpt["early_stop_counter"]
    #     #     self.loss_fn.q_regularizerസ

    #     #     self.loss_fn.q_regularizer.weight_t = ckpt["q_reg_weight_t"]
    #     #     self.loss_fn.q_regularizer.t = ckpt["q_reg_t"]
    #     #     self.loss_fn.d_regularizer.weight_t = ckpt["d_reg_weight_t"]
    #     #     self.loss_fn.d_regularizer.t = ckpt["d_reg_t"]
    #     #     self.trainer.fit_loop.current_epoch = ckpt["epoch"] + 1

    #     # if stage == "test":
    #     #     self.test_dense_run, *_ = self.evaluate(self.trainer.datamodule.test_dataset, return_run_file=False)
