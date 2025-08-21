from torch import nn
from transformers import AutoModelForMaskedLM, AutoModel, AutoConfig, PretrainedConfig, PreTrainedModel
import torch
from lavis.models import load_model_and_preprocess
from transformers import BatchEncoding
import torch.nn.functional as F
import numpy as np
from torch import device
from PIL import Image
from torch.amp import autocast
import copy
class RetrievalConfig(PretrainedConfig):
    model_type = "retrieval"

    def __init__(self, mlm_head="distilbert-base-uncased", dense_size=256, **kwargs):
        self.mlm_head = mlm_head
        self.dense_size = dense_size
        super().__init__(**kwargs)

class RetrievalProjModel(PreTrainedModel):
    config_class = RetrievalConfig
    def __init__(self, device = "cuda" , args = None, config: RetrievalConfig = RetrievalConfig()):
        super().__init__(config)
        self.args = args
        self.device_type = device
        if self.args.dataset == "flickr30k":
            dataset_type = "flickr"
        elif self.args.dataset == "mscoco":
            dataset_type = "coco"
        self.blip_model, _, _ = load_model_and_preprocess(
                name=f"{self.args.backbone_model}_retrieval", model_type=dataset_type, is_eval=True, device="cpu"
                )
        self.text_proj = copy.deepcopy(self.blip_model.text_proj).to( device = self.device_type)
        self.vision_proj = copy.deepcopy(self.blip_model.vision_proj).to( device = self.device_type)
        if args.weight_tying:
            text_encoder = self.blip_model.text_encoder
            self.single_proj = nn.Linear(config.dense_size, text_encoder.config.hidden_size)
            word_embeddings = self.blip_model.text_encoder.get_input_embeddings()
            self.single_vocab_projector = nn.Linear(
            word_embeddings.embedding_dim, word_embeddings.num_embeddings, bias=False
                )
            self.single_vocab_projector.weight = nn.Parameter(word_embeddings.weight.detach().clone())
            self.single_vocab_layer_norm = nn.LayerNorm(text_encoder.config.hidden_size)  # Define new if not exists
            self.vocab_size = text_encoder.config.vocab_size
        elif self.args.baseline:
            assert self.args.pooling == ["single"] and self.args.bow == False
            proj_model = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")
            self.single_proj = nn.Linear(256, proj_model.config.hidden_size)
            self.single_vocab_layer_norm = proj_model.vocab_layer_norm
            self.single_vocab_projector = proj_model.vocab_projector
        else:
            model = AutoModelForMaskedLM.from_pretrained(config.mlm_head)
            self.single_proj = nn.Linear(config.dense_size, model.config.hidden_size)
            self.single_vocab_layer_norm = copy.deepcopy(model.vocab_layer_norm)
            self.single_vocab_projector = copy.deepcopy(model.vocab_projector)
            self.vocab_size = model.config.vocab_size
        del self.blip_model
        torch.cuda.empty_cache()
    def forward_text(self, dense_vec, attention_mask, input_ids = None, bow = False, dense = False, mask_ratio = 0, only_dense = False, all_tokens = True):
        model_dev = self.device_type
        attention_mask = attention_mask.to(model_dev, non_blocking=True)
        if all_tokens:
            dense_vec = F.normalize(self.text_proj(dense_vec), dim = -1)
            dense_vec = dense_vec* attention_mask.unsqueeze(-1) # (B, L, 256)
        else:    
            dense_vec = F.normalize(self.text_proj(dense_vec), dim=-1) # (B, 256)
        if only_dense:
            return dense_vec
        
        batch_size = len(dense_vec)
        
        if bow:
            single_term_importances = torch.ones(dense_vec.size(0), self.vocab_size).to(dense_vec.device)
            mask = torch.zeros_like(single_term_importances).float()
            weights = torch.ones_like(input_ids).float()
            weights = weights * attention_mask.float()
            mask[torch.arange(batch_size).unsqueeze(-1).int(),
                input_ids.int()] = weights
            single_term_importances = single_term_importances * mask  
        else:
            single_term_importances = None
            if 'single' in self.args.pooling:
                if all_tokens:
                    single_proj_feat = self.single_vocab_layer_norm(self.single_proj(dense_vec.select(dim = -2, index = 0)))
                else:
                    single_proj_feat = self.single_vocab_layer_norm(self.single_proj(dense_vec))
                single_term_importances = torch.log1p(
                    torch.relu(self.single_vocab_projector(single_proj_feat)))
        if mask_ratio == 1:
            mask = torch.zeros_like(single_term_importances).float()
            weights = torch.ones_like(input_ids).float()
            weights = weights * attention_mask.float()
            mask[torch.arange(batch_size).unsqueeze(-1).int(),
                input_ids.int()] = weights
            if 'single' in self.args.pooling:
                single_term_importances = single_term_importances * mask  
        if all_tokens:
            return dense_vec.select(dim = -2, index = 0), single_term_importances, None, None
        else:
            return dense_vec, single_term_importances, None, None


    def forward_image(self, dense_vec, only_dense = False, all_tokens = True):
        # Preprocess inputs using processor
        # img_tensors = self.vis_processors(image).to(self.device)
        with autocast("cuda", dtype=torch.float16):
            dense_vec = dense_vec.to(device=self.device_type)
            dense_vec = torch.nn.functional.normalize(self.vision_proj(dense_vec), dim=-1)
            
            if only_dense:
                return dense_vec
            
            single_term_importances = None
            if 'single' in self.args.pooling:
                if all_tokens:
                    single_proj_feat = self.single_vocab_layer_norm(self.single_proj(dense_vec.select(dim = -2, index = 0)))
                else:
                    single_proj_feat = self.single_vocab_layer_norm(self.single_proj(dense_vec))
                    
                single_term_importances = torch.log1p(
                    torch.relu(self.single_vocab_projector(single_proj_feat)))
                del single_proj_feat
            
            if all_tokens:
                return dense_vec.select(dim = -2, index = 0), single_term_importances, None, None
            else:
                return dense_vec, single_term_importances, None, None

class RetrievalModel(PreTrainedModel):
    config_class = RetrievalConfig

    def __init__(self, args = None, config: RetrievalConfig = RetrievalConfig()):
        super().__init__(config)
        self.args = args
        if args.weight_tying:
            self.blip_model, _, _ = load_model_and_preprocess(
        name="blip_retrieval", model_type="coco", is_eval=True, device="cpu"
        )
        
            text_encoder = self.blip_model.text_encoder
            self.proj = nn.Linear(config.dense_size, text_encoder.config.hidden_size).to(self.device)
            word_embeddings = self.blip_model.text_encoder.get_input_embeddings()

            self.vocab_projector = nn.Linear(
            word_embeddings.embedding_dim, word_embeddings.num_embeddings, bias=False
                )
            self.vocab_projector.weight = nn.Parameter(word_embeddings.weight.detach().clone())
            self.vocab_layer_norm = nn.LayerNorm(text_encoder.config.hidden_size)  # Define new if not exists
            self.vocab_size = text_encoder.config.vocab_size
            del self.blip_model
            torch.cuda.empty_cache()
        elif self.args.baseline:
            assert self.args.pooling == ["single"] and self.args.bow == False
            proj_model = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")
            self.proj = nn.Linear(256, proj_model.config.hidden_size)
            self.vocab_layer_norm = proj_model.vocab_layer_norm
            self.vocab_projector = proj_model.vocab_projector
            self.vocab_size = proj_model.config.vocab_size
        else:
            model = AutoModelForMaskedLM.from_pretrained(config.mlm_head)
            self.proj = nn.Linear(config.dense_size, model.config.hidden_size)
            self.vocab_layer_norm = model.vocab_layer_norm
            self.vocab_projector = model.vocab_projector
            self.vocab_size = model.config.vocab_size

    def forward_text(self, dense_vec, input_ids=None, attention_mask=None, bow = False, dense = False, mask_ratio = 0, all_tokens = False):

        dense_vec_processed = self.proj(dense_vec)
        dense_vec_processed = self.vocab_layer_norm(dense_vec_processed)
        batch_size = len(dense_vec_processed)
        if bow:
            term_importances = torch.ones(dense_vec_processed.size(0), self.vocab_size).to(dense_vec_processed.device)
        else:
            if attention_mask is None:
                attention_mask = torch.ones(dense_vec_processed.size(0), dense_vec_processed.size(1)).float().to(dense_vec_processed.device)

            if self.args.single:
                term_importances = torch.log1p(
                torch.relu(self.vocab_projector(dense_vec_processed)))
        if mask_ratio == 1:
            mask = torch.zeros_like(term_importances).float()
            weights = torch.ones_like(input_ids).float()
            weights = weights * attention_mask.float()
            mask[torch.arange(batch_size).unsqueeze(-1).int(),
                input_ids.int()] = weights
            if self.args.single:
                term_importances = term_importances * mask
        single_term_importances = None
        if self.args.single:
            single_term_importances = term_importances
        return dense_vec, single_term_importances, None, None


    def forward_image(self, dense_vec, input_ids=None, attention_mask=None, bow = False, dense = False, mask_ratio = 0, all_tokens = False):
        dense_vec_processed = self.proj(dense_vec)
        dense_vec_processed = self.vocab_layer_norm(dense_vec_processed)
        batch_size = len(dense_vec_processed)
        if bow:
            term_importances = torch.ones(dense_vec_processed.size(0), self.vocab_size).to(dense_vec_processed.device)
        else:
            if attention_mask is None:
                attention_mask = torch.ones(dense_vec_processed.size(0), dense_vec_processed.size(1)).float().to(dense_vec_processed.device)

            if self.args.single:
                term_importances = torch.log1p(
                torch.relu(self.vocab_projector(dense_vec_processed)))
        if mask_ratio == 1:
            mask = torch.zeros_like(term_importances).float()
            weights = torch.ones_like(input_ids).float()
            weights = weights * attention_mask.float()
            mask[torch.arange(batch_size).unsqueeze(-1).int(),
                input_ids.int()] = weights
            if self.args.single:
                term_importances = term_importances * mask
        single_term_importances = None
        if self.args.single:
            single_term_importances = term_importances
        return dense_vec, single_term_importances, None, None

# Set vit_grad_ckpt to False in blip_retrieval_coco.yaml config

class BLIPRetrievalModel(nn.Module):
    def __init__(self, device="cuda", args = None, config: RetrievalConfig = RetrievalConfig(), mlm = False):
        super().__init__()
        self.args = args
        self.device = device
        # Load BLIP processor and model from Hugging Face
        if self.args.backbone_model == "blip":
            self.blip_model, self.vis_processors, self.txt_processors = load_model_and_preprocess(
            name="blip_retrieval", model_type="coco", is_eval=True, device="cuda"
            )

        elif self.args.backbone_model == "albef":
            self.blip_model, self.vis_processors, self.txt_processors = load_model_and_preprocess(
            name="albef_retrieval", model_type="coco", is_eval=True, device="cuda"
            )
        if not args.finetune_encoder:
            for param in self.blip_model.parameters():
                param.requires_grad = False
            self.blip_model.vision_proj.weight.requires_grad = True
            self.blip_model.vision_proj.bias.requires_grad = True
            self.blip_model.text_proj.weight.requires_grad = True
            self.blip_model.text_proj.bias.requires_grad = True
        if hasattr(self.blip_model, 'text_proj_m'):
            del self.blip_model.text_proj_m
        if hasattr(self.blip_model, 'vision_proj_m'):
            del self.blip_model.vision_proj_m
        # Delete other momentum-related modules if needed (e.g., entire momentum network)
        if hasattr(self.blip_model, 'text_encoder_m'):
            del self.blip_model.text_encoder_m
        if hasattr(self.blip_model, 'visual_encoder_m'):
            del self.blip_model.visual_encoder_m
        if args.weight_tying:
            text_encoder = self.blip_model.text_encoder
            self.single_proj = nn.Linear(config.dense_size, text_encoder.config.hidden_size)
            word_embeddings = self.blip_model.text_encoder.get_input_embeddings()
            self.single_vocab_projector = nn.Linear(
            word_embeddings.embedding_dim, word_embeddings.num_embeddings, bias=False
                )
            self.single_vocab_projector.weight = nn.Parameter(word_embeddings.weight.detach().clone())
            self.single_vocab_layer_norm = nn.LayerNorm(text_encoder.config.hidden_size)  # Define new if not exists
            self.vocab_size = text_encoder.config.vocab_size
        
        else:
            model = AutoModelForMaskedLM.from_pretrained(config.mlm_head)
            self.single_proj = nn.Linear(config.dense_size, model.config.hidden_size)
            self.single_vocab_layer_norm = copy.deepcopy(model.vocab_layer_norm)
            self.single_vocab_projector = copy.deepcopy(model.vocab_projector)
            self.vocab_size = model.config.vocab_size
        self.mlm = mlm
        torch.cuda.empty_cache()

    def forward_text(self, input_ids=None, attention_mask=None, bow = False, mask_ratio = 0, only_dense = False, all_tokens = True):
        # Preprocess inputs using processor
        model_dev = next(self.parameters()).device
        input_ids      = input_ids.to(model_dev,      non_blocking=True)
        attention_mask = attention_mask.to(model_dev, non_blocking=True)
        tokenized_text = BatchEncoding({
            "input_ids": input_ids,
            "attention_mask": attention_mask
        })
        text_output = self.blip_model.text_encoder.forward_text(tokenized_text= tokenized_text) 
        text_embeds = text_output.last_hidden_state
        dense_vec = F.normalize(self.blip_model.text_proj(text_embeds), dim = -1)
        dense_vec = dense_vec* tokenized_text['attention_mask'].unsqueeze(-1) # (B, L, 256)
        if only_dense:
            return dense_vec
        
        batch_size = len(dense_vec)
        if bow:
            term_importances = torch.ones(dense_vec.size(0), self.vocab_size).to(dense_vec.device)
        elif self.mlm:
            single_proj_feat = self.single_vocab_layer_norm(self.single_proj(dense_vec.select(dim = -2, index = 0)))
            term_importances = self.vocab_projector(single_proj_feat)
            return term_importances

        else:
            single_term_importances = None
            if 'single' in self.args.pooling:
                single_proj_feat = self.single_vocab_layer_norm(self.single_proj(dense_vec.select(dim = -2, index = 0)))
                single_term_importances = torch.log1p(
                    torch.relu(self.single_vocab_projector(single_proj_feat)))

        if mask_ratio == 1:
            mask = torch.zeros_like(single_term_importances).float()
            weights = torch.ones_like(input_ids).float()
            weights = weights * attention_mask.float()
            mask[torch.arange(batch_size).unsqueeze(-1).int(),
                input_ids.int()] = weights
            single_term_importances = single_term_importances * mask  
        if self.args.self_supervised:
            return dense_vec.select(dim = -2, index = 0), single_term_importances, None, None
        else:
            if self.args.dense:
                return dense_vec.select(dim = -2, index = 0), single_term_importances, None, None
            else:
                return None, single_term_importances, None, None


    def forward_image(self, image, only_dense = False, all_tokens = True):
        # Preprocess inputs using processor
        # img_tensors = self.vis_processors(image).to(self.device)
        img_tensors = torch.stack([self.vis_processors["eval"](img) for img in image]).to(self.device)
        # img = self.vis_processors["eval"](img_t/ensors).unsqueeze(0).to(image.device)
        image_embeds = self.blip_model.visual_encoder.forward_features(img_tensors)
        dense_vec = torch.nn.functional.normalize(self.blip_model.vision_proj(image_embeds), dim=-1)
        if only_dense:
            return dense_vec
        single_term_importances = None
    
        if 'single' in self.args.pooling:
            single_proj_feat = self.single_vocab_layer_norm(self.single_proj(dense_vec.select(dim = -2, index = 0)))
            single_term_importances = torch.log1p(
                torch.relu(self.single_vocab_projector(single_proj_feat)))
        
        # img_sparse = torch.log1p(torch.relu(self.vocab_projector(img_proj_feat)))
        if self.args.self_supervised:
            return dense_vec[:,0,:], single_term_importances, None, None
        else:
            if self.args.dense:
                return dense_vec.select(dim = -2, index = 0), single_term_importances, None, None
            else:
                return None, single_term_importances, None, None

AutoConfig.register("retrieval", RetrievalConfig)
AutoModel.register(RetrievalConfig, RetrievalModel)
