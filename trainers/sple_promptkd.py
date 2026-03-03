
import copy
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .imagenet_templates import IMAGENET_TEMPLATES
from tqdm import tqdm
import math

from clip.model import VisionTransformer, convert_weights
from collections import OrderedDict
import os
from PIL import Image
from dassl.data.transforms.transforms import build_transform
from dassl.engine import build_trainer
import json
import random
from dassl.utils import read_image

import sys
from dassl.config import get_cfg_default
import re

from kmeans_pytorch import kmeans

from .gsl import gradient_scale_layer
import random
_tokenizer = _Tokenizer()

dataset_name_mapping = {
    "Caltech101": "caltech",
    "DescribableTextures": "dtd",
    "EuroSAT": "eurosat",
    "FGVCAircraft": "fgvc",
    "Food101": "food101",
    "ImageNet": "imagenet",
    "ImageNetA": "imagenet_a",
    "ImageNetR": "imagenet_r",
    "ImageNetSketch": "imagenet_sketch",
    "ImageNetV2": "imagenetv2",
    "OxfordFlowers": "oxford_flowers",
    "OxfordPets": "oxford_pets",
    "StanfordCars": "stanford_cars",
    "SUN397": "sun397",
    "UCF101": "ucf101",
}

 
def transform_image(cfg, img0, transform):

    def _transform_image(tfm, img0):
        img_list = []
        for k in range(1):
            img_list.append(tfm(img0))
        img = img_list
        if len(img) == 1:
            img = img[0]

        return img

    output = {}
    # introduce tfm function to transform images
    if isinstance(transform, (list, tuple)):
        for i, tfm in enumerate(transform):
            img = _transform_image(tfm, img0)
            keyname = "img"
            if (i + 1) > 1:
                keyname += str(i + 1)
            output[keyname] = img
    else:
        img = _transform_image(transform, img0)
        output["img"] = img  # [3, 224, 224]

    return output


 
def split_img_abs_path(abs_path, ref_path):
    split_sum = ref_path.count("/")  # count the number of "/" using path name as reference
    if "\\" in abs_path:
        split_result = abs_path.rsplit("\\", 1)  # Split based on the last "\\"
        path_prefix = split_result[0]
        path_suffix = split_result[1]
    elif "r'\'" in abs_path:
        split_result = abs_path.rsplit("r'\'", 1)  # Split based on the last "\"
        path_prefix = split_result[0]
        path_suffix = split_result[1]
    else:
        split_result = abs_path.rsplit("/", split_sum + 1)  # Split based on the n+1th "/" from the end
        path_prefix = split_result[0]
        path_suffix = split_result[1]
        if len(split_result) > 1:
            for split_id in range(2, len(split_result)):
                path_suffix = path_suffix + "/" + split_result[split_id]

    return path_prefix, path_suffix


 
def reformat_imagenet_path(path_str):
    # Windows or Linux path
    return re.sub(r'([\\/])train\1n\d{8}', '', path_str, count=1)


def load_backbone_prompt_vector(cfg):
    

    # Load model-best.pth.tar fine-tuned by PromptKD
    upper_path = cfg.SPLE.BACK_CKPT_PATH
    ckpt_epoch = cfg.SPLE.BACK_CKPT_EPOCH

    if osp.exists(upper_path + "/VLPromptLearner/model-best.pth.tar"):
        model_path = upper_path + "/VLPromptLearner/model-best.pth.tar"  # path of PromptKD
    else:
        model_path = upper_path + "/VLPromptLearner/model.pth.tar-" + str(ckpt_epoch)  # path of PromptSRC

 
    if cfg.TRAINER.MODAL == "base2novel":
        src_model_path = './teacher_model/' + str(cfg.DATASET.NAME) + '/VLPromptLearner/model-best.pth.tar'
    elif cfg.TRAINER.MODAL == "cross":
        src_model_path = './teacher_model/ImageNet-xd/VLPromptLearner_large/model.pth.tar-20'

    # Initialize the pre-trained weights loading of [PromptKD]
    kd_txt_prompts_depth = cfg.TRAINER.PROMPTKD.PROMPT_DEPTH_TEXT  # [PromptKD]: layer depth of text prompts
    kd_vis_prompts_depth = cfg.TRAINER.PROMPTKD.PROMPT_DEPTH_VISION  # [PromptKD]: layer depth of visual prompts
    #kd_prompt_learner = torch.load(model_path, map_location="cuda")["state_dict"]  # Pretrained Weight of PromptKD
    kd_prompt_learner = torch.load(model_path, map_location="cuda" )["state_dict"]
    kd_vis_prompts_list = []
    kd_layer_params = [[], [], [], []]  # Store all pre-trained parameters of 'VPT_image_trans' module

    # Initialize the pre-trained weights loading of [PromptSRC]
    src_txt_prompts_depth = cfg.TRAINER.PROMPTKD.PROMPT_DEPTH_TEXT
    #src_prompt_learner = torch.load(src_model_path, map_location="cuda")["state_dict"]  # Pretrained Weight of PromptSRC
    src_prompt_learner = torch.load(src_model_path, map_location="cuda" )["state_dict"] 
    src_txt_prompts_list = []

    # [PromptSRC]: Extract text prompt from Layer 0 (prompt_learner.ctx)
    src_ctx_txt = src_prompt_learner["prompt_learner.ctx"]  # ViT-L/14: [4, 768]
    # [PromptSRC]: Extract the list of text prompts from 'text_encoder.transformer.resblocks.X.VPT_shallow'
    for layer_id in range(1, src_txt_prompts_depth):
        src_txt_prompts_list.append(src_prompt_learner["text_encoder.transformer.resblocks."
                                                       + str(layer_id)
                                                       + ".VPT_shallow"])  # ViT-L/14: [4, 768]

    # [PromptKD]: Extract visual prompt from Layer 0: [image_encoder.VPT]
    kd_ctx_vpt = kd_prompt_learner["image_encoder.VPT"]  # ViT-B/16 VPT prompt: [4, 768]
    # [PromptKD]: Extract the list of visual prompts from 'image_encoder.transformer.resblocks.X.VPT_shallow':
    for layer_id in range(1, kd_vis_prompts_depth):
        kd_vis_prompts_list.append(kd_prompt_learner["image_encoder.transformer.resblocks."
                                                     + str(layer_id)
                                                     + ".VPT_shallow"])  # ViT-B/16 VPT prompt: [4, 768]

    # [PromptKD]: Extract all params from VPT_image_trans   
    for i in range(0, len(kd_layer_params)):
        '''
        Format: nested list
        [
         [1.0.weight, 1.0.bias], 
         [1.1.weight, 1.1.bias],
         [1.1.running_mean, 1.1.running_var, 1.1.num_batches_tracked], 
         [1.3.weight, 1.3.bias]
        ]
        '''
        if i != 2:
            kd_weight = kd_prompt_learner["VPT_image_trans.conv1." + str(i) + ".weight"]
            kd_bias = kd_prompt_learner["VPT_image_trans.conv1." + str(i) + ".bias"]
            kd_layer_params[i] = [kd_weight, kd_bias]
        else:
            kd_running_mean = kd_prompt_learner["VPT_image_trans.conv1.1.running_mean"]
            kd_running_var = kd_prompt_learner["VPT_image_trans.conv1.1.running_var"]
            kd_num_batches_tracked = kd_prompt_learner["VPT_image_trans.conv1.1.num_batches_tracked"]
            kd_layer_params[i] = [kd_running_mean, kd_running_var, kd_num_batches_tracked]

    return src_ctx_txt, src_txt_prompts_list, kd_ctx_vpt, kd_vis_prompts_list, kd_layer_params, kd_prompt_learner

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
       
        x = F.layer_norm(x.to(self.weight.dtype), self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    
 
    
class MoE(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.num_experts = 3
        self.top_k = 2
        self.dropout = 0.3  

        # Multi-head attention
        self.attn = nn.MultiheadAttention(d_model, n_head, dtype=torch.float16)
        self.ln_1 = LayerNorm(d_model)

        # Experts
        self.experts = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, d_model * 4)),
                ("gelu", QuickGELU()),
                ("dropout", nn.Dropout(p=self.dropout)),
                ("c_proj", nn.Linear(d_model * 4, d_model))    
            ])).half() for _ in range(self.num_experts)
        ])

        self.ln_2 = LayerNorm(d_model)

        # Gating network
        self.gate = nn.Linear(d_model, self.num_experts, bias=False).half() 

        # 用来记录 router_logits 的 token 数 (B*L)，初始化为空
        self.text = None   

    def build_attention_mask(self, seq_len: int, device, dtype):
        """根据序列长度生成 causal mask"""
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=1)
        return mask
    
    def attention(self, x: torch.Tensor):
        # 注意：这里一定要用 x.size(0)，而不是 self.text
        seq_len = x.size(0)
        attn_mask = self.build_attention_mask(seq_len, x.device, x.dtype)
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor):
        # Self-attention
        x = x + self.attention(self.ln_1(x))
        hidden_states = self.ln_2(x)

        # 如果输入是 (B, D)，补一个 L=1 维度
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(1)  # (B, 1, D)

        # Gating logits
        router_logits = self.gate(hidden_states)  # (B, L, num_experts)
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        # 展开 batch 和序列
        hidden_states = hidden_states.view(-1, hidden_dim)  # (B*L, D)
        router_logits = self.gate(hidden_states)            # (B*L, num_experts)

 
        self.text = router_logits.shape[0]   # = B*L

        # 计算 routing 权重
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        # 初始化最终输出
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), 
            dtype=hidden_states.dtype, 
            device=hidden_states.device
        )

        # One-hot mask
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

  
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.numel() == 0:   # 没有 token 分配给这个 expert
                continue

            current_state = hidden_states[top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        # reshape 回 (B, L, D)
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

      
        gate_score = routing_weights.max(dim=-1).values.view(batch_size, sequence_length, 1)
        out = x * gate_score

        return out , router_logits



class Feature_Trans_Module_two_layer(nn.Module):
    def __init__(self, input_dim=100, out_dim=256):
        super(Feature_Trans_Module_two_layer, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, out_dim, 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, 1)
        )
    def forward(self, input_feat):
        
        final_feat = self.conv1(input_feat.unsqueeze(-1).unsqueeze(-1))
        
        return final_feat.squeeze(-1).squeeze(-1)


def load_clip_to_cpu_teacher(cfg, zero_shot_model=False):
    backbone_name = cfg.TRAINER.PROMPTKD.TEACHER_NAME
    # url = clip._MODELS[backbone_name]
    
    if backbone_name == "ViT-B/16":
        model_path = './clip/ViT-B-16.pt'
    elif backbone_name == "ViT-L/14":
        model_path = './clip/ViT-L-14.pt'
    elif backbone_name == "ViT-B/32":
        model_path = './clip/ViT-B-32.pt'
    else:
        print('enter the wrong teacher name.')
    
    print(f"CLIP Teacher name is {backbone_name}")

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    # We default use PromptSRC to pretrain our teacher model   
    design_details = {"trainer": 'IVLP',
                        "vision_depth": 9,
                        "language_depth": 9,
                        "vision_ctx": 4,
                        "language_ctx": 4}
    
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


def load_clip_to_cpu(cfg, zero_shot_model=False):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    # url = clip._MODELS[backbone_name]
    model_path = './clip/ViT-B-16.pt'
    
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {"trainer": 'IVLP',
                      "vision_depth": cfg.TRAINER.PROMPTKD.PROMPT_DEPTH_VISION,
                      "language_depth": cfg.TRAINER.PROMPTKD.PROMPT_DEPTH_TEXT,
                      "vision_ctx": cfg.TRAINER.PROMPTKD.N_CTX_VISION,
                      "language_ctx": cfg.TRAINER.PROMPTKD.N_CTX_TEXT}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


def load_clip_to_teacher_cpu(cfg, zero_shot_model=False):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    # url = clip._MODELS[backbone_name]
    model_path = './clip/ViT-L-14.pt'
    
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {"trainer": 'IVLP',
                      "vision_depth": cfg.TRAINER.PROMPTKD.PROMPT_DEPTH_VISION,
                      "language_depth": cfg.TRAINER.PROMPTKD.PROMPT_DEPTH_TEXT,
                      "vision_ctx": cfg.TRAINER.PROMPTKD.N_CTX_VISION,
                      "language_ctx": cfg.TRAINER.PROMPTKD.N_CTX_TEXT}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model

 
    
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):

        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class DWConvNormAct(nn.Module):
    def __init__(self, d_model, k_size, dim):
        super().__init__()
        
        self.dim = dim
        if dim == 2:
            self.conv = nn.Conv2d(
                d_model, d_model, k_size, padding=k_size//2, 
                groups=d_model, bias=False
            )
        elif dim == 1:
            self.conv = nn.Conv1d(
                d_model, d_model, k_size, padding=k_size//2, 
                groups=d_model, bias=False
            )
        

    def forward(self, x):
        if self.dim == 1:
            x = x.permute(1,2,0)
            x = self.conv(x).permute(2,0,1)
            return x
        
        elif self.dim == 2:
            n_token, b_size, d_model = x.shape
            p_size = int(math.sqrt(n_token))
            x = x.permute(1,2,0)
            x = self.conv(x.reshape(b_size, d_model, p_size, p_size))
            x = x.reshape(b_size, d_model, p_size*p_size).permute(2,0,1)
            return x
        
class AdapterLearner(nn.Module):
    def __init__(self, cfg,  clip_model):
        super().__init__()

        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # build multi-modal adapter
        self.text_adapter_parser = lambda x : self.return_text_adapter(x)
        self.text_adapted_func = lambda x, y, z: self.return_text_adapted_x(x, y, z)
        self.start=1
        self.end=12
        self.scale=0.01
        self.text_adapter = self._build_adapter(
            clip_model.ln_final.weight.shape[0], 
            len(clip_model.transformer.resblocks), 
            self.start,
            self.end,
            dtype=clip_model.dtype
        )
        self.visual_adapter_parser = lambda x : self.return_visual_adapter(x)
        self.visual_adapted_func = lambda x, y, z: self.return_visual_adapted_x(x, y, z)
        self.visual_adapter = self._build_adapter(
            clip_model.visual.ln_post.weight.shape[0],
            len(clip_model.visual.transformer.resblocks), 
           self.start,
           self.end,
            is_visual=True,
            dtype=clip_model.dtype
        )
        self.adapter_scale = float( )
        self.adapter_scale_factor = float(self.scale)
        self.slow_fast_ratio =  0.5

    def _build_adapter(self, d_model, n_layers, l_start, l_end, is_visual=False, dtype=torch.float32):
        adapter = [None] * (n_layers + 1)
        channel = d_model * 4
        for i in range(l_start, l_end+1):
            if is_visual:
                adapter[i] = nn.Sequential(OrderedDict([
                    ("att_conv", DWConvNormAct(d_model, 3, 2)),
                    ("mlp_conv", DWConvNormAct(channel, 3, 2))
                ]))
            else:
                adapter[i] = nn.Sequential(OrderedDict([
                    ("att_conv", DWConvNormAct(d_model, 3, 1)),
                    ("mlp_conv", DWConvNormAct(channel, 3, 1))
                ]))

        adapter = nn.ModuleList([a for a in adapter])
        if dtype == torch.float16:
            for m in adapter.modules():
                m.half()

        return adapter

    def return_text_adapter(self, index):
        if torch.rand(1) > self.slow_fast_ratio and self.training:
            adapter_scale = self.adapter_scale * self.adapter_scale_factor
        else:
            adapter_scale = self.adapter_scale
        return self.text_adapter[index], adapter_scale, self.text_adapted_func
    
    def return_text_adapted_x(self, x, adapter, scale):
        y = gradient_scale_layer(x, scale)
        y = adapter(y)
        y = gradient_scale_layer(y*scale, 1.0/scale)
        x = x + y
        return x

    def return_visual_adapter(self, index):
        if torch.rand(1) > self.slow_fast_ratio and self.training:
            adapter_scale = self.adapter_scale * self.adapter_scale_factor
        else:
            adapter_scale = self.adapter_scale
        return self.visual_adapter[index], adapter_scale, self.visual_adapted_func
    
    def return_visual_adapted_x(self, x, adapter, scale):
        n_token = x.shape[0]
        # LNC -> NCL
        cls_token, x = torch.split(x, [1, n_token-1], dim=0)
        y = gradient_scale_layer(x, scale)
        y = adapter(y)
        y = gradient_scale_layer(y*scale, 1.0/scale)
        x = x + y
        x = torch.cat([cls_token, x], dim=0)
        return x

    def update_adapter_scale(self, scale_factor):
        self.adapter_scale = self.adapter_scale * scale_factor

    def forward(self):
        return self.text_adapter_parser, self.visual_adapter_parser


 
class VLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, is_teacher):
        super().__init__()
        n_cls = len(classnames)
        # Make sure Language depth >= 1
        assert cfg.TRAINER.PROMPTKD.PROMPT_DEPTH_TEXT >= 1, "In Independent VL prompting, Language prompt depth should be >=1" \
                                                            "\nPlease use VPT trainer if you want to learn only vision " \
                                                            "branch"
        n_ctx = cfg.TRAINER.PROMPTKD.N_CTX_TEXT
        ctx_init = cfg.TRAINER.PROMPTKD.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        self.trainer_name = cfg.TRAINER.NAME
        self.train_modal = cfg.TRAINER.MODAL

        if ctx_init and n_ctx <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization   
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print(f"[PromptKD-Teacher] Independent V-L design")
        print(f'[PromptKD-Teacher] Initial text context: "{prompt_prefix}"')
        print(f"[PromptKD-Teacher] Number of context words (tokens) for Language prompting: {n_ctx}")
        print(f"[PromptKD-Teacher] Number of context words (tokens) for Vision prompting: {cfg.TRAINER.PROMPTKD.N_CTX_VISION}")
        self.ctx_x = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)

        print(f'classnames size is {len(classnames)}')

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        # self.name_lens = name_lens

        clip_model_temp = load_clip_to_teacher_cpu(cfg, True).float().cuda()
        clip_model_temp_image = load_clip_to_teacher_cpu(cfg, True)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            self.ZS_image_encoder = clip_model_temp_image.visual
        
        with open(f"./descriptions/{dataset_name_mapping[cfg.DATASET.NAME]}_prompt_distinct.json") as f:
        # with open(f"./descriptions/{dataset_name_mapping[cfg.DATASET.NAME]}_prompt.json") as f:
            gpt3_prompt = json.load(f)
            
        with torch.no_grad():
            clip_weights = []
            for classname in classnames:
                # Tokenize the prompts
                classname = classname.replace("_", " ")
                texts = []
                for t in gpt3_prompt[classname]:
                    texts.append(t)
                texts = clip.tokenize(texts)
                if torch.cuda.is_available():
                    texts = texts.cuda()
                class_embeddings = clip_model_temp.encode_text(texts)
                class_embeddings = class_embeddings.mean(dim=0, keepdim=True)
                clip_weights.append(class_embeddings)
        clip_weights = torch.cat(clip_weights, dim=0)
        clip_weights = clip_weights / clip_weights.norm(dim=-1, keepdim=True)
        self.fixed_embeddings = clip_weights

        with torch.no_grad():
            _, cluster_centers = kmeans(X=clip_weights, num_clusters=5, distance='cosine', device=clip_weights.device)
            cluster_centers = cluster_centers.half().to(clip_weights.device)
            cluster_centers = cluster_centers / cluster_centers.norm(dim=-1, keepdim=True)
            self.topk = nn.Embedding.from_pretrained(cluster_centers).weight

        if self.train_modal == "base2novel":
            self.register_buffer("token_prefix", embedding[:math.ceil(self.n_cls / 2), :1, :])  # SOS
            self.register_buffer("token_suffix", embedding[:math.ceil(self.n_cls / 2), 1 + n_ctx:, :])  # CLS, EOS

            self.register_buffer("token_prefix2", embedding[math.ceil(self.n_cls / 2):, :1, :])  # SOS
            self.register_buffer("token_suffix2", embedding[math.ceil(self.n_cls / 2):, 1 + n_ctx:, :])  # CLS, EOS

        elif self.train_modal == "cross":
            self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

            self.register_buffer("token_prefix2", embedding[:, :1, :])  # SOS
            self.register_buffer("token_suffix2", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

    def construct_prompts(self, ctx_x, prefix, suffix, label=None):
 

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx_x,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx_x = self.ctx_x
        if ctx_x.dim() == 2:
            ctx_x = ctx_x.unsqueeze(0).expand(self.n_cls, -1, -1)
 

        prefix = self.token_prefix
    

        suffix = self.token_suffix
      

        if self.trainer_name == "PromptKD" or "PromptKD" in self.trainer_name and self.train_modal == "base2novel":
            
            prefix = torch.cat([prefix, self.token_prefix2], dim=0)
            suffix = torch.cat([suffix, self.token_suffix2], dim=0)

        prompts = self.construct_prompts(ctx_x, prefix, suffix)

        return prompts


 
class VLPromptLearner_SPLE(nn.Module):
    def __init__(self, cfg, classnames, clip_model, is_teacher):
        super().__init__()
        n_cls = len(classnames)
 
        assert cfg.TRAINER.PROMPTKD.PROMPT_DEPTH_TEXT >= 1, "In Independent VL prompting, Language prompt depth should be >=1" \
                                                        "\nPlease use VPT trainer if you want to learn only vision " \
                                                        "branch"
        n_ctx = cfg.TRAINER.PROMPTKD.N_CTX_TEXT
        ctx_init = cfg.TRAINER.PROMPTKD.CTX_INIT
        sple_init = cfg.SPLE.SPLE_TRAINER.SPLE_INIT   

   
        self.base2new = cfg.DATASET.SUBSAMPLE_CLASSES
        self.sple_stack_weight = cfg.SPLE.STACK.WEIGHT   
        self.sple_stack_weight_for_new = cfg.SPLE.STACK.WEIGHT_FOR_NEW   

     
        clip_model_teacher = load_clip_to_cpu_teacher(cfg)

        self.sple_stack_mode = cfg.SPLE.STACK.MODE  # [DISCARD] DO NOT CHANGE
        self.sple_stack_depth = cfg.SPLE.STACK.LOOP_DEPTH  # [DISCARD] DO NOT CHANGE
        self.cfg = cfg
        self.text_encoder = TextEncoder(clip_model_teacher)

        dtype = clip_model_teacher.dtype
        ctx_dim = clip_model_teacher.ln_final.weight.shape[0]
        print("***** ctx_dim for text prompt: ", ctx_dim)
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        
        self.trainer_name = cfg.TRAINER.NAME
        self.train_modal = cfg.TRAINER.MODAL

        if ctx_init and n_ctx <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
            
                embedding = clip_model_teacher.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)

        clip_model_temp = load_clip_to_teacher_cpu(cfg, True).float().cuda()
        clip_model_temp_image = load_clip_to_teacher_cpu(cfg, True)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            self.ZS_image_encoder = clip_model_temp_image.visual
        
        with open(f"./descriptions/{dataset_name_mapping[cfg.DATASET.NAME]}_prompt_common.json") as f:
        # with open(f"./descriptions/{dataset_name_mapping[cfg.DATASET.NAME]}_prompt.json") as f:
            gpt3_prompt = json.load(f)
            
        with torch.no_grad():
            clip_weights = []
            for classname in classnames:
                # Tokenize the prompts
                classname = classname.replace("_", " ")
                texts = []
                for t in gpt3_prompt[classname]:
                    texts.append(t)
                texts = clip.tokenize(texts)
                if torch.cuda.is_available():
                    texts = texts.cuda()
                class_embeddings = clip_model_temp.encode_text(texts)
                class_embeddings = class_embeddings.mean(dim=0, keepdim=True)
                clip_weights.append(class_embeddings)
        clip_weights = torch.cat(clip_weights, dim=0)
        clip_weights = clip_weights / clip_weights.norm(dim=-1, keepdim=True)
        self.fixed_embeddings = clip_weights

        with torch.no_grad():
            _, cluster_centers = kmeans(X=clip_weights, num_clusters=5, distance='cosine', device=clip_weights.device)
            cluster_centers = cluster_centers.half().to(clip_weights.device)
            cluster_centers = cluster_centers / cluster_centers.norm(dim=-1, keepdim=True)
            self.topk = nn.Embedding.from_pretrained(cluster_centers).weight


        if "converse" in self.sple_stack_mode or "simple" in self.sple_stack_mode:
            # Load 'ctx_txt / txt_prompts_list / ctx_vpt / vis_prompts_list / kd_layer_params' fine-tuned by PromptKD
            self.pre_ctx_txt, self.pre_txt_prompts_list, self.pre_ctx_vpt, self.pre_vis_prompts_list, self.pre_kd_layer_params, _ = load_backbone_prompt_vector(cfg)

            # Print readme   
            if "converse" in self.sple_stack_mode:
                print("Construct ctx prompt in --converse-- mode: use parallel prompt for prompt tuning, ",
                      "and save & use mixed prompt for inference.")
            else:
                print("Construct ctx prompt by mixed-ctx using --simple-- mode")
            print(" Stack method:", self.sple_stack_mode, "; Stack weight:", self.sple_stack_weight)
            print("Params for Weighting-Decoupling: prompt_learner.ctx")

            if sple_init:
                
                mixed_ctx_txt = self.pre_ctx_txt.type(dtype)  # ViT-L/14: [4, 768]
                print("***** mixed_ctx_txt", mixed_ctx_txt.size())
                mixed_ctx_img = self.pre_ctx_vpt  # ViT-B/16 VPT: [4, 768]
                mixed_shallow_txt = [self.pre_txt_prompts_list[i] for i in
                                     range(0, cfg.TRAINER.PROMPTKD.PROMPT_DEPTH_TEXT - 1)]  # ViT-L/14: [4, 768]
                mixed_shallow_img = [self.pre_vis_prompts_list[i] for i in
                                     range(0, cfg.TRAINER.PROMPTKD.PROMPT_DEPTH_VISION - 1)]  # ViT-B/16 VPT: [4, 768]
            else:
             
                mixed_ctx_txt = (self.sple_stack_weight * ctx_vectors + (
                        1 - self.sple_stack_weight) * self.pre_ctx_txt).type(dtype)

                mixed_ctx_img = self.sple_stack_weight * clip_model.visual.VPT + (
                        1 - self.sple_stack_weight) * self.pre_ctx_vpt

                mixed_shallow_txt = []
                mixed_shallow_img = []
              
                for i in range(0, cfg.TRAINER.PROMPTKD.PROMPT_DEPTH_TEXT - 1):
                    mixed_shallow_txt.append(self.sple_stack_weight * clip_model.transformer.ctx_list[i + 1]
                                             + (1 - self.sple_stack_weight) * self.pre_txt_prompts_list[i])
           
                for i in range(0, cfg.TRAINER.PROMPTKD.PROMPT_DEPTH_VISION - 1):
                    mixed_shallow_img.append(self.sple_stack_weight * clip_model.visual.transformer.ctx_list[i + 1]
                                             + (1 - self.sple_stack_weight) * self.pre_vis_prompts_list[i])

 
            self.ctx = nn.Parameter(mixed_ctx_txt)
            clip_model.visual.VPT = nn.Parameter(mixed_ctx_img)
            for i in range(0, cfg.TRAINER.PROMPTKD.PROMPT_DEPTH_TEXT - 1):
                clip_model.transformer.ctx_list[i + 1] = nn.Parameter(mixed_shallow_txt[i])
            for i in range(0, cfg.TRAINER.PROMPTKD.PROMPT_DEPTH_TEXT - 1):
                clip_model.visual.transformer.ctx_list[i + 1] = nn.Parameter(mixed_shallow_img[i])

   
        else:
            print(" No stack mode applied, following original PromptKD to init.")
            self.ctx = nn.Parameter(ctx_vectors)

        print(f" Independent V-L design")
        print(f'  Initial text context: "{prompt_prefix}"')
        print(f" Number of context words (tokens) for Language prompting: {n_ctx}")
        print(f"  Number of context words (tokens) for Vision prompting: {cfg.TRAINER.PROMPTKD.N_CTX_VISION}")

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        
        print(f'classnames size is {len(classnames)}')

        with torch.no_grad():
            # [DPC_PromptKD_TO] Load ViT-L/14 teacher model to build text prompts
            embedding = clip_model_teacher.token_embedding(tokenized_prompts).type(dtype)
        
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        # self.name_lens = name_lens

        if self.train_modal == "base2novel":
            self.register_buffer("token_prefix", embedding[:math.ceil(self.n_cls / 2), :1, :])  # SOS
            self.register_buffer("token_suffix", embedding[:math.ceil(self.n_cls / 2), 1 + n_ctx:, :])  # CLS, EOS

            self.register_buffer("token_prefix2", embedding[math.ceil(self.n_cls / 2):, :1, :])  # SOS
            self.register_buffer("token_suffix2", embedding[math.ceil(self.n_cls / 2):, 1 + n_ctx:, :])  # CLS, EOS
            
        elif self.train_modal == "cross":
            self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
            
            self.register_buffer("token_prefix2", embedding[:, :1, :])  # SOS
            self.register_buffer("token_suffix2", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

    def construct_prompts(self, ctx, prefix, suffix, label=None):
 

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
 
        if "converse" in self.sple_stack_mode:
            ctx = (self.ctx - (1 - self.sple_stack_weight) * self.pre_ctx_txt) * (1 / self.sple_stack_weight)
 
        else:
            ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)  # ViT-L/14: [50, 4, 768]  
 

        prefix = self.token_prefix
        # print(f'prefix size is {prefix.size()}')
        
        suffix = self.token_suffix
        # print(f'suffix size is {suffix.size()}')

        if self.trainer_name == "PromptKD" or "PromptKD" in self.trainer_name and self.train_modal == "base2novel":
            # print(f'n_cls is {self.n_cls}')
            prefix = torch.cat([prefix, self.token_prefix2], dim=0)
            suffix = torch.cat([suffix, self.token_suffix2], dim=0)

        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts   


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.n_cls = len(classnames)
        
        self.VPT_image_trans = Feature_Trans_Module_two_layer(512, 768)
       
        self.cfg = cfg
        
        self.VPT_image_trans = self.VPT_image_trans.cuda()
        convert_weights(self.VPT_image_trans)

    def forward(self, image, label=None):
        logit_scale = self.logit_scale.exp()
        
        image_features = self.image_encoder(image.type(self.dtype))  # torch.Size([bs, 512])
        image_features = self.VPT_image_trans(image_features)  # torch.Size([bs, 768])
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features, logit_scale


class CustomCLIP_teacher(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model, True)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.topk = self.prompt_learner.topk   
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model).cuda()
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
 
        self.n_cls = len(classnames)

        self.VPT_topk_t = nn.Linear(clip_model.visual.output_dim, clip_model.visual.output_dim).half()
   
        
        self.adapter=AdapterLearner(cfg, clip_model)
        self.alpha=0.5
    
    def encode_image(self, feature, adapter_parser=None):
       if adapter_parser is not None:
               feature=adapter_parser(feature)
       return feature    

   
    def forward(self, image=None, label=None, sp=False):
        text_adapter_parser, visual_adapter_parser = self.adapter()
        
        self.alpha=0.5
        prompts = self.prompt_learner()
        if self.prompt_learner.training:
          
            if torch.rand(1).item()>self.alpha:     #0.5
                 self.alpha=random.uniform(1,2)*self.alpha 
                 print("self.ahpha_tea",self.alpha)    #0.5~1
            else:
                 self.alpha=self.alpha
 
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts.cuda(), tokenized_prompts.cuda())
        device = text_features.device
        dtype = text_features.dtype
        self.topk=self.topk.to(device=device,dtype=dtype)
        proj = nn.Linear(self.topk.size(0), text_features.size(0)).to(device=device,dtype=dtype)
        topk=self.topk.transpose(0,1)
        topk=proj(topk).transpose(0,1)
        text_features = text_features + topk  
        if sp is False:
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        else:
            text_features_sp = text_features[:math.ceil(self.n_cls / 2), :]  # split BASE feat
            text_features = text_features_sp / text_features_sp.norm(dim=-1, keepdim=True)  # norm only on BASE
        
        logit_scale = self.logit_scale.exp()
        
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features_p = self.encode_image([image_features, visual_adapter_parser]) 
 
        image_features_p = F.normalize(image_features_p[0], dim=-1)
 

        logits = logit_scale * image_features @ text_features.t()
        
        return image_features, text_features, logits


 
class CustomSPLECLIP(nn.Module):
 
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.n_cls = len(classnames)
        self.VPT_image_trans = Feature_Trans_Module_two_layer(512, 768)
        self.cfg = cfg
        self.sple_stack_mode = cfg.SPLE.STACK.MODE  # [DISCARD] DO NOT CHANGE
        self.adapter=AdapterLearner(cfg, clip_model)
        self.alpha=0.5
 
        if "converse" in self.sple_stack_mode or "simple" in self.sple_stack_mode:
            print("[DPC_PromptKD] Load weights of VPT_image_trans from pre-tuned PromptKD backbone")
            _, _, _, _, pre_kd_layer_params, _ = load_backbone_prompt_vector(cfg)
            for i in range(0, len(pre_kd_layer_params)):
                if i != 2:
                    self.VPT_image_trans.conv1[i].weight = nn.Parameter(pre_kd_layer_params[i][0])
                    self.VPT_image_trans.conv1[i].bias = nn.Parameter(pre_kd_layer_params[i][1])
                else:
                    self.VPT_image_trans.conv1[1].running_mean = pre_kd_layer_params[i][0]
                    self.VPT_image_trans.conv1[1].running_var = pre_kd_layer_params[i][1]
                    self.VPT_image_trans.conv1[1].num_batches_tracked = pre_kd_layer_params[i][2]

        self.VPT_image_trans = self.VPT_image_trans.cuda()
        convert_weights(self.VPT_image_trans)
 
        self.prompt_learner = VLPromptLearner_SPLE(cfg, classnames, clip_model, False)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

   
        self.text_encoder = self.prompt_learner.text_encoder
        self.topk = self.prompt_learner.topk

        self.clip_model = clip_model
        self.sple_stack_mode = cfg.SPLE.STACK.MODE  # [DISCARD] DO NOT CHANGE
        self.sple_stack_weight = cfg.SPLE.STACK.WEIGHT   
        
 
 

    def encode_image(self, feature, adapter_parser=None):
       if adapter_parser is not None:
               feature=adapter_parser(feature)
       return feature    
 


    def forward(self, image, label=None):
        logit_scale = self.logit_scale.exp()
 
        text_adapter_parser, visual_adapter_parser = self.adapter()
        #alpha1=0
        image_features = self.image_encoder(image.type(self.dtype))  # torch.Size([bs*TopK, 512])
       
        image_features_kd = self.VPT_image_trans(image_features)  # torch.Size([bs*TopK, 768])
        image_features_kd = image_features_kd / image_features_kd.norm(dim=-1, keepdim=True)
 
        image_features_sp = image_features_kd  # ViT-B/16 + VPT_image_trans: torch.Size([bs*TopK, 768])
        
        self.alpha=0.4
        if self.prompt_learner.training:
      
            if torch.rand(1).item()>self.alpha:     #0.5
                 self.alpha=random.uniform(1,2)*self.alpha 
             
            else:
                 self.alpha= self.alpha
                 
 
            image_features_s=self.encode_image( [image_features_sp, visual_adapter_parser])
            image_features_s = F.normalize(image_features_s[0], dim=-1)
            image_features_s=image_features_s.float()
            image_features_sp=(1-self.alpha)*image_features_s+self.alpha*image_features_sp  
            tokenized_prompts = self.tokenized_prompts
            prompts = self.prompt_learner()  # ViT-L/14: [n_cls, 77, 768]
            text_features_sp = self.text_encoder(prompts.cuda(), tokenized_prompts.cuda())  # ViT-L/14: [n_cls, 768]
 
            text_features_sp = text_features_sp[label.tolist()]     #[27,768]
            text_features_sp = text_features_sp / text_features_sp.norm(dim=-1, keepdim=True)  # ViT-L/14   
 
            device = text_features_sp.device
            dtype = text_features_sp.dtype
            self.topk=self.topk.to(device=device,dtype=dtype)
            proj = nn.Linear(self.topk.size(0), text_features_sp.size(0)).to(device=device,dtype=dtype)
            topk=self.topk.transpose(0,1)
            topk=proj(topk).transpose(0,1)
            text_features_sp = text_features_sp + topk 
            image_features_sp = image_features_sp.float()
            text_features_sp = text_features_sp.float()
            image_features_s=image_features_s.float()
            logits_per_img = logit_scale * image_features_sp @ text_features_sp.t()  # torch.Size(bs*TopK, TopK*bs)
            logits_per_text = logits_per_img.t()  # torch.Size(TopK*bs, bs*TopK)   
           # print(f'logits_per_img size is {logits_per_img.size()}')
            # Label the mini-batch: [0,1,2,...,bs*TopK]
            label_ids = torch.arange(label.size(0), device=logits_per_img.device).long()
 
            sple_loss = (F.cross_entropy(logits_per_img, label_ids) +
                         F.cross_entropy(logits_per_text, label_ids)
                         ) / 2
            return image_features_kd, logit_scale, sple_loss

        else:
      
            tokenized_prompts = self.tokenized_prompts
            prompts = self.prompt_learner()
            text_features_sp = self.text_encoder(prompts.cuda(), tokenized_prompts.cuda())
            text_features_sp = text_features_sp / text_features_sp.norm(dim=-1, keepdim=True)

            
            return text_features_sp, image_features_sp, logit_scale


@TRAINER_REGISTRY.register()
class StackSPLE_PromptKD(TrainerX):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.transform_img = build_transform(cfg, is_train=False)   
        self.base2new = cfg.DATASET.SUBSAMPLE_CLASSES  #  
        self.sple_stack_weight = cfg.SPLE.STACK.WEIGHT   
        self.sple_stack_weight_for_new = cfg.SPLE.STACK.WEIGHT_FOR_NEW  #  

   
        cfg.defrost()   
        cfg.SPLE.KD_INFER = "PromptKDInfer"  # Inference head change to sample base+new to generate 'tea_text_feature'
        self.inference_trainer = self.build_inference_backbone(cfg)
        cfg.SPLE.KD_INFER = ""  # After generating the inference head, restore the original settings
        cfg.freeze()  # re-freeze the configs  

    def check_cfg(self, cfg):
        assert cfg.TRAINER.PROMPTKD.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        
        classnames = self.dm.dataset.classnames
        self.n_cls = len(classnames)
        
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        clip_model_teacher = load_clip_to_cpu_teacher(cfg)

        if cfg.TRAINER.PROMPTKD.PREC == "fp32" or cfg.TRAINER.PROMPTKD.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomSPLECLIP(cfg, classnames, clip_model)

        self.model_teacher = CustomCLIP_teacher(cfg, classnames, clip_model_teacher)
        
        if cfg.TRAINER.MODAL == "base2novel":
            model_path = './teacher_model/'+str(cfg.DATASET.NAME)+'/VLPromptLearner/model-best.pth.tar'
        elif cfg.TRAINER.MODAL == "cross":
            model_path = './teacher_model/ImageNet-xd/VLPromptLearner_large/model.pth.tar-20'
            
        self.train_modal = cfg.TRAINER.MODAL
        
        checkpoint = load_checkpoint(model_path)
        state_dict = checkpoint["state_dict"]
        
        if "prompt_learner.token_prefix" in state_dict:
            del state_dict["prompt_learner.token_prefix"]
        if "prompt_learner.token_prefix2" in state_dict:
            del state_dict["prompt_learner.token_prefix2"]

        if "prompt_learner.token_suffix" in state_dict:
            del state_dict["prompt_learner.token_suffix"]
        if "prompt_learner.token_suffix2" in state_dict:
            del state_dict["prompt_learner.token_suffix2"]

        self.model_teacher.load_state_dict(state_dict, strict=False)
        self.model_teacher.to(self.device)
        self.model_teacher.eval()
        
        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
            else:
                if "ZS_image_encoder" in name:
                    param.requires_grad_(False)

 
            if "text_encoder" in name and "VPT" not in name:
                param.requires_grad_(False)
            if "clip_model" in name:
                param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        print(f"Parameters count: {len(enabled)}")
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer

        self.trainable_list = nn.ModuleList([])
        self.trainable_list.append(self.model)

        self.optim = build_optimizer(self.trainable_list, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("VLPromptLearner", self.model, self.optim, self.sched)
        
        # Cosine scheduler
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.step_counter = 1
        N = cfg.OPTIM.MAX_EPOCH
        
        self.scaler = GradScaler() if cfg.TRAINER.PROMPTKD.PREC == "amp" else None
 
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

        self.temperature = cfg.TRAINER.PROMPTKD.TEMPERATURE
    
    def forward_backward(self, batch):
 
        image, label = self.parse_sple_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler
        
        prec = self.cfg.TRAINER.PROMPTKD.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            
            _, _, sple_loss = model(image, label)
            loss = sple_loss

            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
            
        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

 
    def parse_sple_batch_train(self, batch, gate=False):
 
        input = batch["img"]
        label = batch["label"]
        img_path = batch["impath"]  # torch.Size([bs])
     
        cfg = self.cfg
        topk_sum = cfg.SPLE.INFER_TOPK  # The number (Top-K) of sampled hard negatives
 
        pic_lib = cfg.SPLE.PIC_LIB   
        trainer_name = cfg.TRAINER.NAME   

        if trainer_name == "PromptKD":
            gate = True
        if gate is not True:
            class_label = self.dm.dataset.classnames  # Contain ALL classes
        else:
            class_label = self.dm.dataset.classnames[:math.ceil(self.n_cls / 2)]  # Only contain BASE classes

        with torch.no_grad():
 
            input_sple = torch.empty(0, 3, 224, 224)
            label_sple = torch.empty(0)
            objects_in_batch = label.tolist()

 
 
            tea_image_feats, tea_text_feats, _ = self.model_teacher(input.type(self.model_teacher.dtype).to(self.device),
                                                                    sp=gate)
            batch_similarity = (100.0 * tea_image_feats @ tea_text_feats.T).softmax(dim=-1)

            # For each input image and label in the batch, perform Top-K reasoning
            for sample_id in range(0, input.size(0)):
                values, indices = batch_similarity[sample_id].topk(topk_sum)

                # Object Filtering: sample Top K-1 negative samples other than ground-truth as hard negative objects
                hn_labels_before_selection = []
                hn_labels = []
                for value, index in zip(values, indices):
                    if index != label[sample_id] and len(hn_labels) < topk_sum - 1:
                        hn_labels_before_selection.append(index)

                # non-repeat filtering
                for item in hn_labels_before_selection:
                    if item not in objects_in_batch:
                        hn_labels.append(item)
                        objects_in_batch.append(item)

                # If 'hn_labels' is empty, then randomly select 2 base-class objects outside from 'objects_in_batch'
                if len(hn_labels) < 2:
                    for step in range(0, len(class_label) - 1):
                        neg_label = random.randint(1, len(class_label) - 1)
                        if neg_label not in objects_in_batch and len(hn_labels) < 2:
                            hn_labels.append(neg_label)
                            objects_in_batch.append(neg_label)
                        elif len(hn_labels) < 2:
                            continue
                        else:
                            break

 
                hn_pic_paths = []
                with open(pic_lib) as f:
                    pics_for_selection = json.load(f)
                    '''
                    The format of 'pics_for_selection' dict is like:
                    {
                        'train': [{'face': [0, ['1.jpg', '2.jpg']], 'leopard': [1, ['3.jpg', '4.jpg']], ... }],
                        'val': [{'face': [0, ['5.jpg', '6.jpg']], 'leopard': [1, ['7.jpg', '8.jpg']], ... }],
                        'train_obj_list': ['face', 'leopard', ...],
                        'val_obj_list': ['face', 'leopard', ...]
                    }
                    - This dict can be found in './DATA/SPLE_database' folder.
                    - The list length of the 'train' and 'val' values is always 1.
                    - DO NOT load val when fine-tuning to avoid data leakage.
                    '''
                
                    for obj_id in hn_labels:
 

                        hn_obj_name = class_label[obj_id]  # Get classname
                        pic_list = pics_for_selection["train"][0].get(hn_obj_name) 
 

                        random_pic_path = random.choice(pic_list[1])
                        hn_pic_paths.append(random_pic_path)

                # Read the image and convert it to the CLIP standard input format with a size of [3,224,224]
                input_for_concat = input[sample_id].unsqueeze(0).to(self.device)  # Init
                # Extract prefix of image path: 'img_path_prefix'
                dataset_name = cfg.DATASET.NAME
      
                if dataset_name == "EuroSAT":
                    img_path_prefix, _ = split_img_abs_path(img_path[sample_id], "Highway/Highway_2417.jpg")
                elif dataset_name == "ImageNet":
                    img_path_prefix_cache, _ = split_img_abs_path(img_path[sample_id], "n1234567_1.JPEG")
                    img_path_prefix = reformat_imagenet_path(img_path_prefix_cache)
                else:
                    img_path_prefix, _ = split_img_abs_path(img_path[sample_id], random_pic_path)

                for processing_img in hn_pic_paths:
                    img0 = read_image(img_path_prefix + "/" + processing_img)  # Use ABSOLUTE PATH to read image
                    transformed_img = transform_image(cfg, img0, self.transform_img)["img"].to(
                        self.device)  # Transform image
                    input_for_concat = torch.cat([input_for_concat,
                                                  transformed_img.unsqueeze(0)
                                                  ],
                                                 dim=0)

                label_for_concat = torch.cat([label[sample_id].unsqueeze(0), torch.Tensor(hn_labels)], dim=0)
                label_sple = torch.cat([label_sple, label_for_concat], dim=0)
                input_sple = torch.cat([input_sple.to(self.device), input_for_concat], dim=0)

            # Build final mini-batch
            input_sple = input_sple.to(self.device)
            label_sple = label_sple.type(label.dtype).to(self.device)

        return input_sple, label_sple

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

 
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

 
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]
            if "prompt_learner.token_prefix2" in state_dict:
                del state_dict["prompt_learner.token_prefix2"]
                
            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]
            if "prompt_learner.token_suffix2" in state_dict:
                del state_dict["prompt_learner.token_suffix2"]

            
            cfg = self.cfg
            sple_stack_mode = cfg.SPLE.STACK.MODE
            src_ctx_txt, src_txt_prompts_list, kd_ctx_vpt, kd_vis_prompts_list, kd_layer_params, kd_dict = load_backbone_prompt_vector(cfg)
            
            if self.base2new == "new" and self.cfg.TRAINER.NAME != "PromptKD":
                if "simple" in sple_stack_mode:
                    print("[DPC_PromptKD] Convert mixed prompts in 'simple' mode to tuned parallel prompts.")
                    mixed_ctx = state_dict["prompt_learner.ctx"]
                    stack_ctx = (mixed_ctx - (1 - self.sple_stack_weight) * src_ctx_txt) * (1 / self.sple_stack_weight)
                else:
                    stack_ctx = state_dict["prompt_learner.ctx"]

          
                ctx_vis = state_dict["image_encoder.VPT"]
                txt_prompts_list = []
                vis_prompts_list = []
                for layer_id in range(1, cfg.TRAINER.PROMPTKD.PROMPT_DEPTH_TEXT):
                    txt_prompts_list.append(
                        state_dict["text_encoder.transformer.resblocks." + str(layer_id) + ".VPT_shallow"])
                for layer_id in range(1, cfg.TRAINER.PROMPTKD.PROMPT_DEPTH_VISION):
                    vis_prompts_list.append(
                        state_dict["image_encoder.transformer.resblocks." + str(layer_id) + ".VPT_shallow"])

                print("[StackSPLEForNew] Give weight for tuned unmixed stack prompt to inference on new class.")

                 
                state_dict = kd_dict
                if "prompt_learner.token_prefix" in state_dict:
                    del state_dict["prompt_learner.token_prefix"]
                if "prompt_learner.token_prefix2" in state_dict:
                    del state_dict["prompt_learner.token_prefix2"]
                if "prompt_learner.token_suffix" in state_dict:
                    del state_dict["prompt_learner.token_suffix"]
                if "prompt_learner.token_suffix2" in state_dict:
                    del state_dict["prompt_learner.token_suffix2"]

           
                mixed_ctx_for_n = self.sple_stack_weight_for_new * stack_ctx + (
                        1 - self.sple_stack_weight_for_new) * src_ctx_txt
                state_dict["prompt_learner.ctx"] = mixed_ctx_for_n
                vpt_for_n = self.sple_stack_weight_for_new * ctx_vis + (1 - self.sple_stack_weight_for_new) * kd_ctx_vpt
                state_dict["image_encoder.VPT"] = vpt_for_n

                for layer_id in range(0, cfg.TRAINER.PROMPTKD.PROMPT_DEPTH_TEXT - 1):
                    shallow_p_for_n = (
                            self.sple_stack_weight_for_new * txt_prompts_list[layer_id]
                            + (1 - self.sple_stack_weight_for_new) * src_txt_prompts_list[layer_id]
                            )
                    state_dict["text_encoder.transformer.resblocks." + str(layer_id + 1) + ".VPT_shallow"] = shallow_p_for_n
                for layer_id in range(0, cfg.TRAINER.PROMPTKD.PROMPT_DEPTH_VISION - 1):
                    shallow_p_for_n = (
                            self.sple_stack_weight_for_new * vis_prompts_list[layer_id]
                            + (1 - self.sple_stack_weight_for_new) * kd_vis_prompts_list[layer_id]
                            )
                    state_dict["image_encoder.transformer.resblocks." + str(layer_id + 1) + ".VPT_shallow"] = shallow_p_for_n
               
                for i in range(0, len(kd_layer_params)):
                    if i != 2:
                        state_dict["VPT_image_trans.conv1." + str(i) + ".weight"] = kd_layer_params[i][0]
                        state_dict["VPT_image_trans.conv1." + str(i) + ".bias"] = kd_layer_params[i][1]
                    else:
                        state_dict["VPT_image_trans.conv1.1.running_mean"] = kd_layer_params[i][0]
                        state_dict["VPT_image_trans.conv1.1.running_var"] = kd_layer_params[i][1]
                        state_dict["VPT_image_trans.conv1.1.num_batches_tracked"] = kd_layer_params[i][2]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

 
    @torch.no_grad()
    def build_inference_backbone(self, cfg):
        # Suppress redundant output when loading additional trainers
        def stop_console():
            pass

        original_stdout = sys.stdout
        print("==== [DPC_PromptKD] build a seprate PromptKD backbone model for inference ====")
        sys.stdout = stop_console()

        trainer = build_trainer(cfg, name="PromptKDInfer")
        model_dir = cfg.SPLE.BACK_CKPT_PATH
        trainer.load_model(model_dir, epoch=cfg.SPLE.BACK_CKPT_EPOCH)
        trainer.set_model_mode("eval")
        sys.stdout = original_stdout
        return trainer

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        elif split == "train":
            data_loader = self.train_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")
        
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            image, label = self.parse_batch_test(batch)
            
            with torch.no_grad():
                tea_image_features, tea_text_features, tea_logits = self.model_teacher(image, label)

       
            text_features_sp, image_ft, logit_scale = self.model(image, label)

            
            inference_trainer = self.inference_trainer
            tea_format_text_feats, _, _ = inference_trainer.model_inference(image.type(self.model.dtype).to(self.device))

            if self.train_modal == "base2novel":
                if self.cfg.TRAINER.NAME == "PromptKD":
                    if split == "val":
                        output = logit_scale * image_ft @ tea_text_features[:math.ceil(self.n_cls / 2),:].t()
                    elif split == "test":
                        output = logit_scale * image_ft @ tea_text_features[math.ceil(self.n_cls / 2):,:].t()
                 
                else:
                    if self.base2new == "base":
                        output = logit_scale * image_ft @ text_features_sp.t()
                    elif self.base2new == "new":
                        output = logit_scale * image_ft @ tea_format_text_feats.t()

            elif self.train_modal == "cross":
                output = logit_scale * image_ft @ text_features_sp.t()
            
            self.evaluator.process(output, label) 

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

                                                                                      