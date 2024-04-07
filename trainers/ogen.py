#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import os.path as osp
from pathlib import Path	

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


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


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, classnames_new, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_cls_new = len(classnames_new)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_cls_new = n_cls_new
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

        #
        classnames_new = [name.replace("_", " ") for name in classnames_new]
        prompts_new = [prompt_prefix + " " + name + "." for name in classnames_new]
        tokenized_prompts_new = torch.cat([clip.tokenize(p) for p in prompts_new])
        with torch.no_grad():
            embedding_new = clip_model.token_embedding(tokenized_prompts_new).type(dtype)
        self.tokenized_prompts_new = tokenized_prompts_new  # torch.Tensor
        self.register_buffer("token_prefix_new", embedding_new[:, :1, :])  # SOS
        self.register_buffer("token_suffix_new", embedding_new[:, 1 + n_ctx :, :])  # CLS, EOS

        embed_dim = 512
        num_heads = 4
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1, batch_first=True).half()
        self.ffn = MLP(embed_dim, embed_dim, embed_dim, 2).half()
        # self.norm = nn.LayerNorm(embed_dim).half()
        # self.dropout = nn.Dropout(0.1).half()        
        #

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
        else:
            raise ValueError

        return prompts

    def forward_new(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls_new, -1, -1)

        prefix = self.token_prefix_new
        suffix = self.token_suffix_new

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, classnames_new, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, classnames_new, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.tokenized_prompts_new = self.prompt_learner.tokenized_prompts_new
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        name_mapping = {
            'Caltech101': "caltech-101",
            'DescribableTextures': "dtd",
            'EuroSAT': "eurosat",
            'FGVCAircraft': "fgvc_aircraft",
            'Food101': "food-101",
            'ImageNet': "imagenet",
            'OxfordFlowers': "oxford_flowers",
            'OxfordPets': "oxford_pets",
            'StanfordCars': "stanford_cars",
            'SUN397': "sun397",
            'UCF101': "ucf101",
        }
        dataset_name = name_mapping[cfg.DATASET.NAME]

        features_mean_base = torch.load(f'./data/{dataset_name}/features_mean_base.pth')
        features_std_base = torch.load(f'./data/{dataset_name}/features_std_base.pth')
        self.features_mean_base = torch.tensor(features_mean_base).half().cuda()
        self.features_std_base = torch.tensor(features_std_base).half().cuda()

        self.k = 3

        cosine_sim_base = torch.mm(self.features_mean_base, self.features_mean_base.transpose(0, 1))
        cosine_sim_base.fill_diagonal_(-float('inf'))
        _, closest_classes_base = torch.topk(cosine_sim_base, self.k)
        self.closest_classes_base = closest_classes_base

    def feature_generator(self, Q, K, V):
        attn, _ = self.prompt_learner.multihead_attn(Q, K, V)
        Q_attn = self.prompt_learner.ffn(Q) + attn
        Q_attn = Q_attn / Q_attn.norm(dim=-1, keepdim=True)
        Q_attn = Q_attn.flatten(0, 1)
        return Q_attn

    def forward(self, image, label=None, aug=True):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.prompt_learner.forward()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        if not aug:
            return logits
        else:
            w_base = text_features

            prompts_new = self.prompt_learner.forward_new()
            tokenized_prompts_new = self.tokenized_prompts_new
            w_new = self.text_encoder(prompts_new, tokenized_prompts_new)
            w_new = w_new / w_new.norm(dim=-1, keepdim=True)

            label_base_unknown = label
            label_base_known = self.closest_classes_base[label_base_unknown]

            w_base_unknown = w_base[label_base_unknown].detach()
            w_base_unknown = w_base_unknown.unsqueeze(1)
            w_base_known = w_base[label_base_known].detach()

            i_base_unknown = image_features
            i_base_known = self.features_mean_base[label_base_known] + 1 * torch.rand(1, 512).half().cuda() * self.features_std_base[label_base_known]            
            i_base_unknown_pred = self.feature_generator(w_base_unknown, w_base_known, i_base_known)
            logits_base_unknown = logit_scale * i_base_unknown_pred @ w_base.t()

            i_base_known_pred = self.feature_generator(w_base_known, w_base_unknown, i_base_unknown.unsqueeze(1))
            logits_base_known = logit_scale * i_base_known_pred @ w_base.t()

            cosine_sim_new = torch.mm(w_base_unknown.squeeze(1), w_new.transpose(0, 1))
            _, label_new = torch.topk(cosine_sim_new, self.k)
            w_new_unknown = w_new[label_new]
            i_new_pred = self.feature_generator(w_new_unknown, w_base_unknown, i_base_unknown.unsqueeze(1))
            logits_new = logit_scale * i_new_pred @ w_new.t()

            return logits, logits_base_unknown, label_base_unknown, \
                    logits_base_known, label_base_known.flatten(0, 1), \
                    logits_new, label_new.flatten(0, 1), \
                    i_base_unknown_pred, i_base_unknown, i_base_known_pred, i_base_known.flatten(0, 1)


@TRAINER_REGISTRY.register()
class OGEN(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        if self.cfg.DATASET.SUBSAMPLE_CLASSES == 'base':
            classnames_new = self.dm.dataset._classnames_new
        else:
            classnames_new = self.dm.dataset._classnames_new

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, classnames_new, clip_model)
        self.model_teacher = CustomCLIP(cfg, classnames, classnames_new, clip_model)
        self.model_list = []

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)
            load_pretrained_weights(self.model_teacher.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.model_teacher.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.model_teacher.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch, epoch=0):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            raise NotImplementedError
            with autocast():
                pass
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output, output_1, label_1, output_2, label_2, output_3, label_3, \
                    i_pred, i_1, i_pred2, i_2 = self.model(image, label=label)
            if epoch < int(self.cfg.OPTIM.MAX_EPOCH * 0.2):
                loss = F.mse_loss(i_pred, i_1) * 512 + F.mse_loss(i_pred2, i_2) * 512
            else:
                _, _, _, _, _, _, _, _, i_1, _, i_2 = self.model(image, label=label)
                
                loss = F.mse_loss(i_pred, i_1) * 512 + F.mse_loss(i_pred2, i_2) * 512 + F.cross_entropy(output_1, label_1) * 0.25 + F.cross_entropy(output_2, label_2) * 0.25 / self.model.k + F.cross_entropy(output_3, label_3) * 0.5 / self.model.k + F.cross_entropy(output, label) * 0.25 
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
            "acc_1": compute_accuracy(output_1, label_1)[0].item(),
            "acc_2": compute_accuracy(output_2, label_2)[0].item(),
            "acc_3": compute_accuracy(output_3, label_3)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory='', epoch=None):
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
            state_dict_teacher = checkpoint["state_dict_teacher"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]
                del state_dict_teacher["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]
                del state_dict_teacher["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
            self._models_teacher[name].load_state_dict(state_dict_teacher, strict=False)
