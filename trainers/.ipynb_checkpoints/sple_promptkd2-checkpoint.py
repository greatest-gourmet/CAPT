# Independent code for DPC, do not rely on promptkd.py
# trainer name: StackSPLE_PromptKD

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

_tokenizer = _Tokenizer()


# [DPC] In negative sampling stage, convert image data sampled from the image database
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


# [DPC] Split the [absolute path] passed by DataLoader into suffix (relative path only) and prefix (the rest)
# Considered Windows (D:\\XXX\\dolphin/image_0011.jpg) and Linux (/usr/XXX/dolphin/image_0011.jpg)
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


# [DPC] Re-format the absolute path of ImageNet for DPC Dynamic Hard Negative Optimizer
def reformat_imagenet_path(path_str):
    # Windows or Linux path
    return re.sub(r'([\\/])train\1n\d{8}', '', path_str, count=1)


def load_backbone_prompt_vector(cfg):
    """
    [DPC_PromptKD_TO] Read the original prompt of PromptKD, and the text prompt of the teacher model (PromptSRC on ViT-L/14)
    Output 6 params:
    1. Read from [PromptSRC ViT-L/14]: 'prompt_learner.ctx' with size [n_ctx_txt, 768], representing text prompt.
    2. Read from [PromptSRC ViT-L/14]: A list of length PROMPT_DEPTH_TEXT - 1,
        each item is a 'text_encoder.transformer.resblocks.X.VPT_shallow' with size [n_ctx_txt, 768], X starts from 1.
    3. Read from [PromptKD]: 'image_encoder.VPT' with size [n_ctx_txt, 768] as the layer 0 prompt for VPT-shallow.
    4. Read from [PromptKD]: A list of length PROMPT_DEPTH_VISION - 1,
        each item is a 'image_encoder.transformer.resblocks.X.VPT_shallow', X starts from 1.
    5. Read from [PromptKD]: Stores all pre-trained parameters of the VPT_image_trans module.
        Format: [nested list]. See below for specific parameters.
    6. Read from [PromptKD]: Complete 'kd_prompt_learner'.
    """

    # Load model-best.pth.tar fine-tuned by PromptKD
    upper_path = cfg.SPLE.BACK_CKPT_PATH
    ckpt_epoch = cfg.SPLE.BACK_CKPT_EPOCH

    if osp.exists(upper_path + "/VLPromptLearner/model-best.pth.tar"):
        model_path = upper_path + "/VLPromptLearner/model-best.pth.tar"  # path of PromptKD
    else:
        model_path = upper_path + "/VLPromptLearner/model.pth.tar-" + str(ckpt_epoch)  # path of PromptSRC

    # [DPC_PromptKD_TO] Load model-best.pth.tar from teacher model (PromptSRC on ViT-L/14)
    if cfg.TRAINER.MODAL == "base2novel":
        src_model_path = './teacher_model/' + str(cfg.DATASET.NAME) + '/VLPromptLearner/model-best.pth.tar'
    elif cfg.TRAINER.MODAL == "cross":
        src_model_path = './teacher_model/ImageNet-xd/VLPromptLearner_large/model.pth.tar-20'

    # Initialize the pre-trained weights loading of [PromptKD]
    kd_txt_prompts_depth = cfg.TRAINER.PROMPTKD.PROMPT_DEPTH_TEXT  # [PromptKD]: layer depth of text prompts
    kd_vis_prompts_depth = cfg.TRAINER.PROMPTKD.PROMPT_DEPTH_VISION  # [PromptKD]: layer depth of visual prompts
    kd_prompt_learner = torch.load(model_path, map_location="cuda", weights_only=False)["state_dict"]  # Pretrained Weight of PromptKD
    kd_vis_prompts_list = []
    kd_layer_params = [[], [], [], []]  # Store all pre-trained parameters of 'VPT_image_trans' module

    # Initialize the pre-trained weights loading of [PromptSRC]
    src_txt_prompts_depth = cfg.TRAINER.PROMPTKD.PROMPT_DEPTH_TEXT
    src_prompt_learner = torch.load(src_model_path, map_location="cuda", weights_only=False)["state_dict"]  # Pretrained Weight of PromptSRC
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


# [DPC_PromptKD] VLPromptLearner for teacher_model
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
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        # print(f'label is {label}')
        # if label is not None:
        #     prefix = prefix[label]
        #     suffix = suffix[label]

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
        # print(f'ctx size is {ctx.size()}')

        prefix = self.token_prefix
        # print(f'prefix size is {prefix.size()}')

        suffix = self.token_suffix
        # print(f'suffix size is {suffix.size()}')

        if self.trainer_name == "PromptKD" or "PromptKD" in self.trainer_name and self.train_modal == "base2novel":
            # print(f'n_cls is {self.n_cls}')
            prefix = torch.cat([prefix, self.token_prefix2], dim=0)
            suffix = torch.cat([suffix, self.token_suffix2], dim=0)

        prompts = self.construct_prompts(ctx_x, prefix, suffix)

        return prompts


# [DPC_PromptKD] VLPromptLearner for DPC_PromptKD
# [DPC_PromptKD_TO] Use 'PromptSRC ViT-L/14' to init text prompts; Use 'PromptKD backbone' to init VPT prompts.
class VLPromptLearner_SPLE(nn.Module):
    def __init__(self, cfg, classnames, clip_model, is_teacher):
        super().__init__()
        n_cls = len(classnames)
        # Make sure Language depth >= 1
        assert cfg.TRAINER.PROMPTKD.PROMPT_DEPTH_TEXT >= 1, "In Independent VL prompting, Language prompt depth should be >=1" \
                                                        "\nPlease use VPT trainer if you want to learn only vision " \
                                                        "branch"
        n_ctx = cfg.TRAINER.PROMPTKD.N_CTX_TEXT
        ctx_init = cfg.TRAINER.PROMPTKD.CTX_INIT
        sple_init = cfg.SPLE.SPLE_TRAINER.SPLE_INIT  # [DPC] Use DPC method to init

        # [DPC] In base-to-new experiments, force 'SPLE.SPLE_TRAINER.SPLE_BASE_TRAIN = False'
        self.base2new = cfg.DATASET.SUBSAMPLE_CLASSES
        self.sple_stack_weight = cfg.SPLE.STACK.WEIGHT  # [DPC] weight for base
        self.sple_stack_weight_for_new = cfg.SPLE.STACK.WEIGHT_FOR_NEW  # [DPC] weight for new

        # [DPC_PromptKD_TO] Load ViT-L/14 teacher model to build text prompts
        clip_model_teacher = load_clip_to_cpu_teacher(cfg)

        self.sple_stack_mode = cfg.SPLE.STACK.MODE  # [DISCARD] DO NOT CHANGE
        self.sple_stack_depth = cfg.SPLE.STACK.LOOP_DEPTH  # [DISCARD] DO NOT CHANGE
        self.cfg = cfg
        self.text_encoder = TextEncoder(clip_model_teacher)

        # [DPC_PromptKD_TO] Use 'PromptSRC ViT-L/14' to init text prompts; Use 'ViT-B/16 clip_model' to init visual prompts.
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
                # [DPC-TO] Load ViT-L/14 teacher model to build text prompts
                embedding = clip_model_teacher.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        # [DPC] DPC Initialization Method
        '''
        When initializing DPC, please use 'converse' mode only.
        In inference stage, DPC use mixed prompts (mixed_ctx).
        In fine-tuning stage, DPC decouples mixed prompts back to parallel prompts.
        '''
        if "converse" in self.sple_stack_mode or "simple" in self.sple_stack_mode:
            # Load 'ctx_txt / txt_prompts_list / ctx_vpt / vis_prompts_list / kd_layer_params' fine-tuned by PromptKD
            self.pre_ctx_txt, self.pre_txt_prompts_list, self.pre_ctx_vpt, self.pre_vis_prompts_list, self.pre_kd_layer_params, _ = load_backbone_prompt_vector(cfg)

            # Print readme
            if "converse" in self.sple_stack_mode:
                print("[DPC] Construct ctx prompt in --converse-- mode: use parallel prompt for prompt tuning, ",
                      "and save & use mixed prompt for inference.")
            else:
                print("[DPC] Construct ctx prompt by mixed-ctx using --simple-- mode")
            print("[DPC] Stack method:", self.sple_stack_mode, "; Stack weight:", self.sple_stack_weight)
            print("[PromptKD-DPC] Params for Weighting-Decoupling: prompt_learner.ctx")

            if sple_init:
                # [DPC_PromptKD] Initialized parallel prompts in DPC are the same as pre-tuning prompts
                mixed_ctx_txt = self.pre_ctx_txt.type(dtype)  # ViT-L/14: [4, 768]
                print("***** mixed_ctx_txt", mixed_ctx_txt.size())
                mixed_ctx_img = self.pre_ctx_vpt  # ViT-B/16 VPT: [4, 768]
                mixed_shallow_txt = [self.pre_txt_prompts_list[i] for i in
                                     range(0, cfg.TRAINER.PROMPTKD.PROMPT_DEPTH_TEXT - 1)]  # ViT-L/14: [4, 768]
                mixed_shallow_img = [self.pre_vis_prompts_list[i] for i in
                                     range(0, cfg.TRAINER.PROMPTKD.PROMPT_DEPTH_VISION - 1)]  # ViT-B/16 VPT: [4, 768]
            else:
                # [DPC_PromptSRC] If the pre-tuning weights are not used for initialization,
                # the parameters randomly initialized with PromptSRC are weighted with the pre-tuning weights.
                mixed_ctx_txt = (self.sple_stack_weight * ctx_vectors + (
                        1 - self.sple_stack_weight) * self.pre_ctx_txt).type(dtype)

                mixed_ctx_img = self.sple_stack_weight * clip_model.visual.VPT + (
                        1 - self.sple_stack_weight) * self.pre_ctx_vpt

                mixed_shallow_txt = []
                mixed_shallow_img = []
                # item [1-9] in 'clip_model.transformer.ctx_list'
                for i in range(0, cfg.TRAINER.PROMPTKD.PROMPT_DEPTH_TEXT - 1):
                    mixed_shallow_txt.append(self.sple_stack_weight * clip_model.transformer.ctx_list[i + 1]
                                             + (1 - self.sple_stack_weight) * self.pre_txt_prompts_list[i])
                # item [1-9] in 'clip_model.visual.transformer.ctx_list'
                for i in range(0, cfg.TRAINER.PROMPTKD.PROMPT_DEPTH_VISION - 1):
                    mixed_shallow_img.append(self.sple_stack_weight * clip_model.visual.transformer.ctx_list[i + 1]
                                             + (1 - self.sple_stack_weight) * self.pre_vis_prompts_list[i])

            # [DPC_PromptSRC] Use nn.Parameter() to wrap and return all weighted prompt vectors
            self.ctx = nn.Parameter(mixed_ctx_txt)
            clip_model.visual.VPT = nn.Parameter(mixed_ctx_img)
            for i in range(0, cfg.TRAINER.PROMPTKD.PROMPT_DEPTH_TEXT - 1):
                clip_model.transformer.ctx_list[i + 1] = nn.Parameter(mixed_shallow_txt[i])
            for i in range(0, cfg.TRAINER.PROMPTKD.PROMPT_DEPTH_TEXT - 1):
                clip_model.visual.transformer.ctx_list[i + 1] = nn.Parameter(mixed_shallow_img[i])

        # [PromptKD] Discard the weighted method, directly call the original PromptKD method
        else:
            print("[PromptKD-DPC] No stack mode applied, following original PromptKD to init.")
            self.ctx = nn.Parameter(ctx_vectors)

        print(f"[PromptKD-Student-DPC] Independent V-L design")
        print(f'[PromptKD-Student-DPC] Initial text context: "{prompt_prefix}"')
        print(f"[PromptKD-Student-DPC] Number of context words (tokens) for Language prompting: {n_ctx}")
        print(f"[PromptKD-Student-DPC] Number of context words (tokens) for Vision prompting: {cfg.TRAINER.PROMPTKD.N_CTX_VISION}")

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
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

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
        # [DPC] In fine-tuning stage, DPC decouples mixed prompts back to parallel prompts.
        # Please use 'converse' mode only.
        if "converse" in self.sple_stack_mode:
            ctx = (self.ctx - (1 - self.sple_stack_weight) * self.pre_ctx_txt) * (1 / self.sple_stack_weight)
        # original ctx of PromptSRC
        else:
            ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)  # ViT-L/14: [50, 4, 768]
        # print(f'ctx size is {ctx.size()}')

        prefix = self.token_prefix
        # print(f'prefix size is {prefix.size()}')
        
        suffix = self.token_suffix
        # print(f'suffix size is {suffix.size()}')

        if self.trainer_name == "PromptKD" or "PromptKD" in self.trainer_name and self.train_modal == "base2novel":
            # print(f'n_cls is {self.n_cls}')
            prefix = torch.cat([prefix, self.token_prefix2], dim=0)
            suffix = torch.cat([suffix, self.token_suffix2], dim=0)

        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts  # [DPC_PromptKD_TO] Init based on ViT-L/14


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
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model).cuda()
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        # [DPC_PromptKD]
        self.n_cls = len(classnames)

    # [DPC_PromptKD] In DPC, only base classes are used for inference, instead of all categories in PromptKD
    def forward(self, image=None, label=None, sp=False):
        prompts = self.prompt_learner()
        # Compute the prompted image and text features
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts.cuda(), tokenized_prompts.cuda())
        if sp is False:
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        else:
            text_features_sp = text_features[:math.ceil(self.n_cls / 2), :]  # split BASE feat
            text_features = text_features_sp / text_features_sp.norm(dim=-1, keepdim=True)  # norm only on BASE
        
        logit_scale = self.logit_scale.exp()
        
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # Compute the prompted logits
        logits = logit_scale * image_features @ text_features.t()
        
        return image_features, text_features, logits


# [DPC_PromptKD] CustomCLIP for DPC
class CustomSPLECLIP(nn.Module):
    """
    - Since the PromptKD backbone does not pass text features, DPC needs to supplement the text features for L_cl.
    - Since DPC uses the few-shot setting instead of the entire dataset, there is no data leakage when the text features
      are introduced again.
    - Since text prompts uses the same size [4, 768] as the teacher model, contrastive learning in DPC is executed on
      'text_feat' and 'img_feat' after VPT_image_trans().
    """
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

        # [DPC_PromptKD] Use nn.Parameter() to wrap and pass params in 'VPT_image_trans' to PromptKD
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

        # [DPC_PromptKD] Introduce text feature (based on ViT-L/14) to DPC_PromptKD
        self.prompt_learner = VLPromptLearner_SPLE(cfg, classnames, clip_model, False)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        # [DPC_PromptKD_TO] Use teacher model (ViT-L/14) to init TextEncoder
        self.text_encoder = self.prompt_learner.text_encoder

        # [DPC] Introduce extra config
        self.clip_model = clip_model
        self.sple_stack_mode = cfg.SPLE.STACK.MODE  # [DISCARD] DO NOT CHANGE
        self.sple_stack_weight = cfg.SPLE.STACK.WEIGHT  # [DPC] weight for base

    def forward(self, image, label=None):
        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(image.type(self.dtype))  # torch.Size([bs*TopK, 512])
        # PromptKD backbone
        image_features_kd = self.VPT_image_trans(image_features)  # torch.Size([bs*TopK, 768])
        image_features_kd = image_features_kd / image_features_kd.norm(dim=-1, keepdim=True)
        # [DPC_PromptKD_TO] Use transformed image feature for contrastive learning in DPC
        image_features_sp = image_features_kd  # ViT-B/16 + VPT_image_trans: torch.Size([bs*TopK, 768])

        if self.prompt_learner.training:
            # [DPC_PromptKD] Introduce text feature in DPC
            tokenized_prompts = self.tokenized_prompts
            prompts = self.prompt_learner()  # ViT-L/14: [n_cls, 77, 768]
            text_features_sp = self.text_encoder(prompts.cuda(), tokenized_prompts.cuda())  # ViT-L/14: [n_cls, 768]
            '''
            Feature Filtering: In the text_features composed of all classes, according to the id of 'sple_label', 
            extract the tensor with the index corresponding to the hard negative id of each sample in the mini-batch 
            separately to form a new text features with a size of [bs*TopK, 768] (ViT-L/14)
            '''
            text_features_sp = text_features_sp[label.tolist()]
            text_features_sp = text_features_sp / text_features_sp.norm(dim=-1, keepdim=True)  # ViT-L/14

            # Build and calculate the DPC contrastive loss L_cl ('sple_loss') to replace cross-entropy loss
            logits_per_img = logit_scale * image_features_sp @ text_features_sp.t()  # torch.Size(bs*TopK, TopK*bs)
            logits_per_text = logits_per_img.t()  # torch.Size(TopK*bs, bs*TopK)
            # Label the mini-batch: [0,1,2,...,bs*TopK]
            label_ids = torch.arange(label.size(0), device=logits_per_img.device).long()
            # Directly return DPC image-text contrastive loss (L_cl)
            sple_loss = (F.cross_entropy(logits_per_img, label_ids) +
                         F.cross_entropy(logits_per_text, label_ids)
                         ) / 2
            return image_features_kd, logit_scale, sple_loss

        else:
            # [DPC_PromptKD_TO] The text_feature obtained by learning 'self.ctx' needs to be output for inference stage
            tokenized_prompts = self.tokenized_prompts
            prompts = self.prompt_learner()
            text_features_sp = self.text_encoder(prompts.cuda(), tokenized_prompts.cuda())
            text_features_sp = text_features_sp / text_features_sp.norm(dim=-1, keepdim=True)

            # [DPC_PromptKD_TO] The return item increases 'text_features_sp' item obtained by learning 'self.ctx'
            return text_features_sp, image_features_sp, logit_scale


@TRAINER_REGISTRY.register()
class StackSPLE_PromptKD(TrainerX):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.transform_img = build_transform(cfg, is_train=False)  # [DPC] Additional image transform method
        self.base2new = cfg.DATASET.SUBSAMPLE_CLASSES  # [DPC_PromptKD_TI] base/new discrimination
        self.sple_stack_weight = cfg.SPLE.STACK.WEIGHT  # [DPC] weight for base
        self.sple_stack_weight_for_new = cfg.SPLE.STACK.WEIGHT_FOR_NEW  # [DPC] weight for new

        # [DPC_PromptKD_TI2] Init the inference head for Dynamic Hard Negative Optimizer
        cfg.defrost()  # unfreeze the configs
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

            # [DPC_PromptKD_TO] Freeze unused text encoder and clip_model
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
        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

        self.temperature = cfg.TRAINER.PROMPTKD.TEMPERATURE
    
    def forward_backward(self, batch):
        # [DPC] Use Dynamic Hard Negative Optimizer to organize new mini-batch
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
            # [DPC_PromptKD_RE2] Only update DPC loss
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

    # [DPC] Use Dynamic Hard Negative Optimizer to build dynamic mini-batch for DPC
    def parse_sple_batch_train(self, batch, gate=False):
        """
        - For a given batch, perform prompt learner inference on each image, and randomly extract images
          (strictly limited to the base-class training set to avoid data leakage) based on the hard negative
          objects obtained by inference, and finally form a mini-batch containing ground-truth and Top-K
          hard negative objects.
        - Use [non-repeat] setting to ensure that there are no duplicated labels in the sampled mini-batch.
        - Add a 'gate' param to control the source of sampling:
            * In PromptKD backbone (used as the inference head for DPC), set 'gate=True' to sample base class only,
              thus avoiding data leakage.
            * In DPC fine-tuning, set 'gate=False' to load all the passed classes, because that the DataLoader of
              DPC has already been set to read in base classes.
        """
        input = batch["img"]
        label = batch["label"]
        img_path = batch["impath"]  # torch.Size([bs])
        # Read and extend config
        cfg = self.cfg
        topk_sum = cfg.SPLE.INFER_TOPK  # The number (Top-K) of sampled hard negatives
        pic_lib = cfg.SPLE.PIC_LIB  # Path of DPC image database
        trainer_name = cfg.TRAINER.NAME  # Name of trainer

        # To avoid new-class data leakage, under PromptKD trainer, set 'gate=True' (in DPC tuning, use BASE class only);
        # Under DPC trainer, set 'gate=False'(because the data has been read in according to BASE class).
        if trainer_name == "PromptKD":
            gate = True
        if gate is not True:
            class_label = self.dm.dataset.classnames  # Contain ALL classes
        else:
            class_label = self.dm.dataset.classnames[:math.ceil(self.n_cls / 2)]  # Only contain BASE classes

        with torch.no_grad():
            # Init
            input_sple = torch.empty(0, 3, 224, 224)
            label_sple = torch.empty(0)
            objects_in_batch = label.tolist()

            # [DPC_PromptKD] Reuse the teacher model of PromptKD to execute Top-K inference
            '''
            - Since the teacher model of PromptKD performs reasoning on ALL categories, it is necessary to set a gate
              (gate=True) so that PromptKD inference during DPC fine-tuning is only performed on the BASE class to 
              avoid data leakage.
            - text_feats: [n_cls/2, 512]; img_feats: [bs, 512]; batch_similarity: [bs, n_cls/2]
            '''
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

                # Image Sampling: use 'hn_labels' as query to sample positive images in training set
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
                        pic_list = pics_for_selection["train"][0].get(hn_obj_name)  # Search classname to get image list
                        random_pic_path = random.choice(pic_list[1])
                        hn_pic_paths.append(random_pic_path)

                # Read the image and convert it to the CLIP standard input format with a size of [3,224,224]
                input_for_concat = input[sample_id].unsqueeze(0).to(self.device)  # Init
                # Extract prefix of image path: 'img_path_prefix'
                dataset_name = cfg.DATASET.NAME
                # special path format (EuroSAT and ImageNet)
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

        # By default, the best model is loaded
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

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]
            if "prompt_learner.token_prefix2" in state_dict:
                del state_dict["prompt_learner.token_prefix2"]
                
            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]
            if "prompt_learner.token_suffix2" in state_dict:
                del state_dict["prompt_learner.token_suffix2"]

            # [DPC] When performing new-class inference, if use 'simple' mode (mixed_prompt is saved in pth file),
            # model need to restore it to parallel prompt, and then use 'sple_stack_weight_for_new weights' to re-weight
            cfg = self.cfg
            sple_stack_mode = cfg.SPLE.STACK.MODE
            src_ctx_txt, src_txt_prompts_list, kd_ctx_vpt, kd_vis_prompts_list, kd_layer_params, kd_dict = load_backbone_prompt_vector(cfg)
            # [DPC_PromptKD_TI] The weights obtained in PromptKD do not include text prompts, so they are excluded
            if self.base2new == "new" and self.cfg.TRAINER.NAME != "PromptKD":
                if "simple" in sple_stack_mode:
                    print("[DPC_PromptKD] Convert mixed prompts in 'simple' mode to tuned parallel prompts.")
                    mixed_ctx = state_dict["prompt_learner.ctx"]
                    stack_ctx = (mixed_ctx - (1 - self.sple_stack_weight) * src_ctx_txt) * (1 / self.sple_stack_weight)
                else:
                    stack_ctx = state_dict["prompt_learner.ctx"]

                # Load other learnable prompts from the DPC fine-tuned weights
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

                # [DPC_PromptKD_TI] Load weights of PromptKD backbone and execute weighting with DPC learnable params
                state_dict = kd_dict
                if "prompt_learner.token_prefix" in state_dict:
                    del state_dict["prompt_learner.token_prefix"]
                if "prompt_learner.token_prefix2" in state_dict:
                    del state_dict["prompt_learner.token_prefix2"]
                if "prompt_learner.token_suffix" in state_dict:
                    del state_dict["prompt_learner.token_suffix"]
                if "prompt_learner.token_suffix2" in state_dict:
                    del state_dict["prompt_learner.token_suffix2"]

                # [DPC_PromptKD_TI] Weighting learnable params for new tasks
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
                # [DPC_PromptKD_TI] params of 'VPT_image_trans'
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

    # [DPC] Introduce PromptKD inference model ("PromptKD") for Dynamic Hard Negative Optimizer
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


        # ===== 新增：初始化错误预测统计 =====
        wrong_cases = []

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            image, label = self.parse_batch_test(batch)

            with torch.no_grad():
                tea_image_features, tea_text_features, tea_logits = self.model_teacher(image, label)

            # [DPC_PromptKD_TO] Get the 'text_features_sp' obtained by 'self.ctx' in DPC
            text_features_sp, image_ft, logit_scale = self.model(image, label)

            # [DPC_PromptKD_TI] Use 'model_inference' method to generate 'tea_format_text_features' normed on base+new
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

            # ===== 新增：统计预测错误样本 =====
            preds = output.argmax(dim=1).cpu()
            labels = label.cpu()
            for i in range(len(preds)):
                if preds[i] != labels[i]:
                    correct_cls = self.dm.dataset.classnames[labels[i]]
                    pred_cls = self.dm.dataset.classnames[preds[i]]
                    wrong_cases.append({
                        "index": int(i + batch_idx * len(preds)),
                        "true_label": correct_cls,
                        "pred_label": pred_cls
                    })

            # ===== 原始逻辑 =====
            self.evaluator.process(output, label)

        # ===== 保存错误预测统计结果 =====
        if len(wrong_cases) > 0:
            save_path = os.path.join(self.output_dir, "wrong_predictions.json")
            with open(save_path, "w") as f:
                json.dump(wrong_cases, f, indent=4)
            print(f"预测错误样本已保存至: {save_path}")

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]
