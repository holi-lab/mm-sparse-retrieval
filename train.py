# TODO
# 1. Enable temp backprop

# train_pl.py
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from config import Config
from lightning_model import RetrievalLightningModule
from data_module import RetrievalDataModule
import torch
import wandb
from pytorch_lightning.strategies import DDPStrategy
from datetime import datetime
import os
import re
import gc
import json
import copy
import argparse
import torch.distributed as dist
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from types import SimpleNamespace
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)

def parse_args():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config", type=str, default="config.json")
    # return parser.parse_args()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args, unknown = parser.parse_known_args()

    # unknown args like --key=value -> convert to dict
    extra_args = {}
    for arg in unknown:
        if arg.startswith("--"):
            key_value = arg[2:].split("=", 1)
            if len(key_value) == 2:
                key, value = key_value
                try:
                    # Try to parse into int, float or bool
                    value = eval(value)
                except:
                    pass
                extra_args[key] = value
    return args, extra_args
def load_config(path):
    with open(path, "r") as f:
        return json.load(f, object_hook=lambda d: SimpleNamespace(**d))
def load_config(path="config.json"):
    with open(path, "r") as f:
        return json.load(f, object_hook=lambda d: SimpleNamespace(**d))
def sync_across_ranks(obj, src=0):
    """
    Broadcast any Python object (string, int, etc.) from rank 0 to all other ranks
    """
    if dist.is_available():
        container = [obj if dist.get_rank() == src else None]
        dist.broadcast_object_list(container, src)
        return container[0]
    return obj
def config_warning(config):
    if config.finetune_from_proj:
        assert config.finetune_encoder is False and config.load_dense_model is False
    # assert (config.wandb_id is None) == (config.ckpt_path is None)

    if not config.self_supervised:
        assert(sum((config.dense, config.single)) == 1)
    assert config.backbone_model == "blip" or config.backbone_model == "albef"
    active_configs = {
        "dense": config.dense,
        "single": config.single
    }
    config.pooling = [name for name, is_enabled in active_configs.items() if is_enabled]
    config.data = f"lsr42/{config.dataset}-{config.backbone_model}-dense"
    config.image_root = os.path.join(config.image_root, config.dataset)
    if not hasattr(config, "baseline"):
        config.baseline = False
    if not hasattr(config, "truncate_topk"):
        config.truncate_topk = -1
    if not hasattr(config, "best_monitor"):
        config.best_monitor = 'val/single_recall1'
    return config
def main():
    args, extra_args = parse_args()

    # Load base config from JSON if specified
    config = load_config(args.config) if args.config else SimpleNamespace()

    # Override with command-line arguments
    for k, v in extra_args.items():
        setattr(config, k, v)
    config = config_warning(config)
    pl.seed_everything(42, workers=True)
    now = os.environ.get("SYNCED_NOW")
    if now is None:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.environ["SYNCED_NOW"] = now  # Share environment variable to ensure all ranks are synchronized
    if config.wandb_id is None:
        wandb_id = f"{now}"  # or use UUID etc.
        config.wandb_id = wandb_id
    else:
        wandb_id = config.wandb_id
    # Initialize WandB
    if  config.dataset == "mscoco":
        project_name = "splade-clip-lightning" + ("-debug" if config.debug else "")  
    else:
        project_name = "splade-clip-lightning-flickr30k" + ("-debug" if config.debug else "")  
        
    wandb_logger = WandbLogger(
        project=project_name,
        config=config.__dict__,
        id=wandb_id,
        resume="allow"
    )
    # Initialize model and data
    model = RetrievalLightningModule(config)
    data_module = RetrievalDataModule(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    # Callbacks

    ckpt_path = None
    if config.mlm_trained_path is not None and config.ckpt_path is not None:
        raise ValueError("config.mlm_trained_path and config.ckpt_path is not supposed to be not None at the same time")
    elif config.mlm_trained_path is not None:
        ckpt_path = config.mlm_trained_path
    elif config.ckpt_path is not None:
        config.output_dir = os.path.dirname(config.ckpt_path)
        ckpt_path = config.ckpt_path
    else:
        config.output_dir = os.path.join(config.output_dir,f"{wandb_id}")
    best_checkpoint_callback = ModelCheckpoint(
        dirpath=config.output_dir,
        filename="checkpoint_best",
        monitor=config.best_monitor,
        mode="max",
        save_top_k=1
    )
    

    last_checkpoint_callback = ModelCheckpoint(
    dirpath=config.output_dir,
    filename="checkpoint_last",
    save_last=True  # This alone will save "last.ckpt"
    )
    early_stop_callback = EarlyStopping(
        monitor=f"val/{config.pooling[0]}_recall1" if len(config.pooling) == 1 else f"val/{config.pooling[1]}_recall1",
        patience=config.patience,
        mode="max"
    )
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=torch.cuda.device_count(),
        strategy= DDPStrategy(find_unused_parameters=True),
        logger=wandb_logger,
        callbacks=[best_checkpoint_callback, last_checkpoint_callback, early_stop_callback],
        gradient_clip_val=1.0,
        enable_progress_bar=True,
        precision="16-mixed" if config.use_amp else 32,
        accumulate_grad_batches = config.accumulation_steps,
        log_every_n_steps = 1,
        num_sanity_val_steps=0, 
    )
    # Train and test

    if ckpt_path is None:
        trainer.fit(model, data_module)
    else:
        if config.mlm_trained_path is None:
            try:
                trainer.fit(model, data_module, ckpt_path = ckpt_path)
            except: # load pretrained model (https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth)
                ckpt = torch.load(ckpt_path)
                state_dict = ckpt["model"]
                keys_to_remove = [k for k in state_dict if "pos_embed" in k]
                for k in keys_to_remove:
                    print(f"Removing incompatible key: {k} with shape {state_dict[k].shape}")
                    del state_dict[k]
                model.model.blip_model.load_state_dict(state_dict, strict=False)
                trainer.fit(model, data_module)            
        else:
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            trainer.fit(model, data_module)
    trainer.test(model, data_module, ckpt_path=os.path.join(config.output_dir,"checkpoint_best.ckpt"))

if __name__ == "__main__":
    main()