# config.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    data: str = "lsr42/mscoco-blip-dense"
    train_batch_size: int = 16 # Actual batch size. effective bsz is (train_batch_size)*(accumulation_steps)*(# of gpus)
    accumulation_steps: int = 8
    eval_batch_size: int = 16
    temp: float = 0.07
    use_amp: bool = True
    epochs: int = 6
    q_reg: float = 0.00001
    d_reg: float = 0.00001
    bow: bool = False
    blip_lr: float = 1e-7
    lr: float = 5e-4
    load_dense_model: bool = True
    annotation_file: str = '../splade-clip/.cache/coco/annotations/coco_karpathy_train.json'
    val_annotation_file: str = '../splade-clip/.cache/coco/annotations/coco_karpathy_val.json'
    test_annotation_file: str = '../splade-clip/.cache/coco/annotations/coco_karpathy_test.json'
    image_root: str = '../splade-clip/.cache/cache/coco/images'
    use_dense: bool = False
    debug: bool = False
    patience: int = 20
    disable_mask: bool = True
    weight_tying: bool = True
    pooling: str = 'single'  # choices: ['single', 'sum']
    output_dir: Optional[str] = "./output"
    wandb_id: Optional[str] = None
    ckpt_path: Optional[str] = None
    stage_two: bool = True
    mlm_trained_path: Optional[str] = '/shared/s3/lab07/jongsong/splade-clip/output/mlm_trained_checkpoint_converted.pth'
    finetune_encoder: bool = True
    w1: float = 1.0
    w2: float = 0.3
    lam1: float = 1.0
    lam2: float = 0.1