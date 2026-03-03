# save PromptKD inference features
# Load all classes (base+new) to generate text_feat for inference
from dassl.engine import TRAINER_REGISTRY

from .promptkd import PromptKD
import math


@TRAINER_REGISTRY.register()
class PromptKDInfer(PromptKD):

    # Use and input the inference head of original tea_text_feature
    def model_inference(self, image):
        # PromptKD: load image features
        image_features = self.model.image_encoder(image.type(self.model.dtype))
        image_features = self.model.VPT_image_trans(image_features)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Following the pipeline of PromptKD backbone, create params for text prompts on base+new classes
        tokenized_prompts = self.model_teacher.tokenized_prompts
        prompts = self.model_teacher.prompt_learner()
        tea_text_features = self.model_teacher.text_encoder(prompts, tokenized_prompts)
        tea_text_features = tea_text_features / tea_text_features.norm(dim=-1, keepdim=True)  # 按 base+new 正则化

        logit_scale = self.model.logit_scale.exp()

        # split base & new
        split = self.cfg.TEST.SPLIT
        if split == "val":
            tea_text_features = tea_text_features[:math.ceil(self.n_cls / 2), :]
        elif split == "test":
            tea_text_features = tea_text_features[math.ceil(self.n_cls / 2):, :]

        return tea_text_features, image_features, logit_scale

    # Using mixed prompt in DPC, construct [sptea_text_feature] with the same format as [tea_text_feature].
    def sp_model_inferene(self, image, sp_prompts=None):
        weight_base = self.cfg.SPLE.STACK.WEIGHT
        weight_new = self.cfg.SPLE.STACK.WEIGHT_FOR_NEW

        # PromptKD: load image features
        image_features = self.model.image_encoder(image.type(self.model.dtype))
        image_features = self.model.VPT_image_trans(image_features)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        if sp_prompts is not None:
            # Use mixed prompts re-weighted by the new class in DPC as new prompts,
            # and process them according to the construction method of [tea_text_feature]
            prompts = sp_prompts

            prompts_o = self.model_teacher.prompt_learner()
            print("====size of DPC prompts:", prompts.size())
            print("====size of PromptKD backbone prompts:", prompts_o.size())

        else:
            prompts = self.model_teacher.prompt_learner()

        # Following the pipeline of PromptKD backbone, create params for text prompts on base+new classes
        tokenized_prompts = self.model_teacher.tokenized_prompts
        sptea_text_features = self.model_teacher.text_encoder(prompts, tokenized_prompts)
        sptea_text_features = sptea_text_features / sptea_text_features.norm(dim=-1, keepdim=True)  # base+new

        logit_scale = self.model.logit_scale.exp()

        # split base & new
        split = self.cfg.TEST.SPLIT
        if split == "val":
            sptea_text_features = sptea_text_features[:math.ceil(self.n_cls / 2), :]
        elif split == "test":
            sptea_text_features = sptea_text_features[math.ceil(self.n_cls / 2):, :]

        return sptea_text_features, image_features, logit_scale


