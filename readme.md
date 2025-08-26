# Sparse and Dense Retrievers Learn Better Together

This repository contains the official implementation for the paper **"Sparse and Dense Retrievers Learn Better Together: Joint Sparse-Dense Optimization for Text-Image Retrieval"** accepted to CIKM 2025 Short Track.

**ArXiv:** https://arxiv.org/abs/2508.16707

**Authors:** Jonghyun Song, Youngjune Lee, Gyu-Hwung Cho, Ilhyeon Song, Saehun Kim, Yohan Jo



## 📋 Abstract

Vision-Language Pretrained (VLP) models have achieved impressive performance on multimodal tasks, including text-image retrieval, based on dense representations. Meanwhile, Learned Sparse Retrieval (LSR) has gained traction in text-only settings due to its interpretability and efficiency with fast term-based lookup via inverted indexes. Inspired by these advantages, recent work has extended LSR to the multimodal domain. However, these methods often rely on computationally expensive contrastive pre-training, or distillation from a frozen dense model, which limits the potential for mutual enhancement. To address these limitations, we propose a simple yet effective framework that enables bi-directional learning between dense and sparse representations through Self-Knowledge Distillation. This bi-directional learning is achieved using an integrated similarity score—a weighted sum of dense and sparse similarities—which serves as a shared teacher signal for both representations. To ensure efficiency, we fine-tune the final layer of the dense encoder and the sparse projection head, enabling easy adaptation of any existing VLP model. Experiments on MSCOCO and Flickr30k demonstrate that our sparse retriever not only outperforms existing sparse baselines, but also achieves performance comparable to—or even surpassing—its dense counterparts, while retaining the benefits of sparse models.

![cikm 3](https://github.com/user-attachments/assets/fe7c5ee4-f820-417c-9fcb-0e124a4b7629)


## 🚀 Quick Start

### Environment Setup

This experiment runs under Python 3.9 and CUDA 12.4. To set up the conda environment:

```bash
sh conda.sh
```

### Data Preparation

**Required Data:** We only need embeddings right before being processed by the final layer.

**Download Links:**
- [Pre-computed embeddings](https://drive.google.com/file/d/1HyBkfAmfhNvCFgwPIimCkZNrxO__1AnH/view?usp=sharing)

**Original Dataset Sources** (not necessary for reproduction):
- **MS COCO:**
  - [Train 2014](http://images.cocodataset.org/zips/train2014.zip)
  - [Val 2014](http://images.cocodataset.org/zips/val2014.zip)
  - [Test 2014](http://images.cocodataset.org/zips/test2014.zip)
- **Flickr30k:** [Dataset](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset)

**Directory Structure:**
After downloading, organize your data as follows:
```
.cache/
├── mscoco/
│   ├── text_embs_before_proj_blip.parquet
│   ├── img_embs_before_proj_blip.parquet
│   ├── text_embs_before_proj_albef.parquet
│   └── img_embs_before_proj_albef.parquet
└── flickr30k/
    ├── text_embs_before_proj_blip.parquet
    ├── img_embs_before_proj_blip.parquet
    ├── text_embs_before_proj_albef.parquet
    └── img_embs_before_proj_albef.parquet
```

## 🏋️ Training

To train the model, use one of the following commands:

```bash
# For ALBEF on MS COCO
python train.py --config training_config/albef-coco.json

# For ALBEF on Flickr30k
python train.py --config training_config/albef-flickr.json

# For BLIP on MS COCO
python train.py --config training_config/blip-coco.json

# For BLIP on Flickr30k
python train.py --config training_config/blip-flickr.json
```

## 📁 Pre-trained Models

Download pre-trained checkpoints from [here](https://drive.google.com/drive/folders/1cDKGeZfgzofDroFarFjZ3UzmshtoetBN?usp=sharing).


## 📖 Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{song2025sparse,
  title={Sparse and Dense Retrievers Learn Better Together: Joint Sparse-Dense Optimization for Text-Image Retrieval},
  author={Jonghyun Song, Youngjune Lee, Gyu-Hwung Cho, Ilhyeon Song, Saehun Kim, and Yohan Jo},
  booktitle={Proceedings of the 34th ACM International Conference on Information and Knowledge Management (CIKM '25)},
  year={2025},
  pages={5},
  publisher={ACM},
  doi={10.1145/3746252.3760959}
}
```


