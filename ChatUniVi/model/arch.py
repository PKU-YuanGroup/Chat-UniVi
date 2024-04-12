from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from .multimodal_encoder.builder import build_vision_tower
from ChatUniVi.constants import *
from .cluster import CTM, TCBlock
from collections import OrderedDict
from .multimodal_projector.builder import build_vision_projector


class MetaModel:
    def __init__(self, config):
        super(MetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = nn.Linear(config.mm_hidden_size, config.hidden_size)

        if hasattr(config, "config"):
            self.use_cluster = config.config["use_cluster"]
            if self.use_cluster:
                self.ctm0 = CTM(sample_ratio=config.config["spatial_cluster_rate0"], embed_dim=self.config.mm_hidden_size, dim_out=self.config.mm_hidden_size, k=5)
                self.block0 = TCBlock(dim=self.config.mm_hidden_size, num_heads=8)

                self.ctm1 = CTM(sample_ratio=config.config["spatial_cluster_rate1"], embed_dim=self.config.mm_hidden_size, dim_out=self.config.mm_hidden_size, k=3)
                self.block1 = TCBlock(dim=self.config.mm_hidden_size, num_heads=8)

                self.ctm2 = CTM(sample_ratio=config.config["spatial_cluster_rate2"], embed_dim=self.config.mm_hidden_size, dim_out=self.config.mm_hidden_size, k=3)
                self.block2 = TCBlock(dim=self.config.mm_hidden_size, num_heads=8)

                self.ctm3 = CTM(sample_ratio=config.config["temporal_cluster_rate"], embed_dim=self.config.mm_hidden_size, dim_out=self.config.mm_hidden_size, k=5)
                self.block3 = TCBlock(dim=self.config.mm_hidden_size, num_heads=8)
        else:
            self.use_cluster = False

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        vision_tower = build_vision_tower(model_args)

        self.config.use_mm_proj = True
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if fsdp is not None and len(fsdp) > 0:
            self.vision_tower = [vision_tower]
        else:
            self.vision_tower = vision_tower

        if not hasattr(self, 'mm_projector'):
            self.mm_projector = build_vision_projector(self.config)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

    def initialize_cluster_modules(self, model_args):
        self.use_cluster = model_args.use_cluster

        if self.use_cluster and not hasattr(self, 'ctm0'):
            self.ctm0 = CTM(sample_ratio=model_args.spatial_cluster_rate0, embed_dim=self.config.mm_hidden_size, dim_out=self.config.mm_hidden_size, k=5)
            self.block0 = TCBlock(dim=self.config.mm_hidden_size, num_heads=8)

            self.ctm1 = CTM(sample_ratio=model_args.spatial_cluster_rate1, embed_dim=self.config.mm_hidden_size, dim_out=self.config.mm_hidden_size, k=3)
            self.block1 = TCBlock(dim=self.config.mm_hidden_size, num_heads=8)

            self.ctm2 = CTM(sample_ratio=model_args.spatial_cluster_rate2, embed_dim=self.config.mm_hidden_size, dim_out=self.config.mm_hidden_size, k=3)
            self.block2 = TCBlock(dim=self.config.mm_hidden_size, num_heads=8)

            self.ctm3 = CTM(sample_ratio=model_args.temporal_cluster_rate, embed_dim=self.config.mm_hidden_size, dim_out=self.config.mm_hidden_size, k=5)
            self.block3 = TCBlock(dim=self.config.mm_hidden_size, num_heads=8)


class ChatUniViMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images, select_feature="patch")
        return image_features

    def positional_encoding(self, x, num_features=1024, max_len=64):
        p = torch.zeros((1, max_len, num_features))
        _x = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000,
                                                                            torch.arange(0, num_features, 2, dtype=torch.float32) / num_features)

        p[:, :, 0::2] = torch.sin(_x)
        p[:, :, 1::2] = torch.cos(_x)
        x = x + p[:, :x.shape[1], :].to(x.device).to(x.dtype)
        return x

    def project(self, image_features, input_type="image"):
        if self.get_model().use_cluster:
            if input_type == "image":
                cluster_image_features = []
                token_dict = {'x': image_features,
                              'token_num': image_features.size(1),
                              'idx_token': torch.arange(image_features.size(1))[None, :].repeat(
                                  image_features.size(0), 1),
                              'agg_weight': image_features.new_ones(image_features.size(0), image_features.size(1),
                                                                    1),
                              'mask': None}

                token_dict = self.get_model().block0(self.get_model().ctm0(token_dict))
                cluster_image_features.append(token_dict["x"])

                token_dict = self.get_model().block1(self.get_model().ctm1(token_dict))
                cluster_image_features.append(token_dict["x"])

                token_dict = self.get_model().block2(self.get_model().ctm2(token_dict))
                cluster_image_features.append(token_dict["x"])

                image_features = torch.cat(cluster_image_features, dim=1)
                image_features = image_features.to(self.get_model().mm_projector.weight.dtype)
            else:
                cls_features = torch.mean(image_features, dim=1, keepdim=False).unsqueeze(0).clone()
                token_dict = {'x': cls_features,
                              'token_num': cls_features.size(1),
                              'idx_token': torch.arange(cls_features.size(1))[None, :].repeat(
                                  cls_features.size(0), 1),
                              'agg_weight': cls_features.new_ones(cls_features.size(0), cls_features.size(1),
                                                                  1),
                              'mask': None}

                down_dict, token_dict = self.get_model().ctm3(token_dict)
                events = OrderedDict()

                max_len = 0
                for id, i in enumerate(down_dict["idx_token"][0].tolist()):
                    if i not in events:
                        events[i] = [id]
                    else:
                        events[i].append(id)
                    max_len = len(events[i]) if max_len < len(events[i]) else max_len

                cluster_image_features = []
                token_dict = {'x': image_features,
                              'token_num': image_features.size(1),
                              'idx_token': torch.arange(image_features.size(1))[None, :].repeat(
                                  image_features.size(0), 1),
                              'agg_weight': image_features.new_ones(image_features.size(0), image_features.size(1),
                                                                    1),
                              'mask': None}

                token_dict0 = self.get_model().block0(self.get_model().ctm0(token_dict))
                token_dict1 = self.get_model().block1(self.get_model().ctm1(token_dict0))
                token_dict2 = self.get_model().block2(self.get_model().ctm2(token_dict1))

                for id, key in enumerate(events):
                    cur_image_features0 = torch.cat([token_dict0["x"][i] for i in events[key]], dim=0).unsqueeze(0)
                    token_dict = {'x': cur_image_features0,
                                  'token_num': cur_image_features0.size(1),
                                  'idx_token': torch.arange(cur_image_features0.size(1))[None, :].repeat(
                                      cur_image_features0.size(0), 1),
                                  'agg_weight': cur_image_features0.new_ones(cur_image_features0.size(0),
                                                                             cur_image_features0.size(1),
                                                                      1),
                                  'mask': None}

                    cur_token_dict0 = self.get_model().block0(self.get_model().ctm0(token_dict))
                    cluster_image_features.append(cur_token_dict0["x"])

                    cur_image_features1 = torch.cat([token_dict1["x"][i] for i in events[key]], dim=0).unsqueeze(0)
                    token_dict = {'x': cur_image_features1,
                                  'token_num': cur_image_features1.size(1),
                                  'idx_token': torch.arange(cur_image_features1.size(1))[None, :].repeat(
                                      cur_image_features1.size(0), 1),
                                  'agg_weight': cur_image_features1.new_ones(cur_image_features1.size(0),
                                                                             cur_image_features1.size(1),
                                                                             1),
                                  'mask': None}

                    cur_token_dict1 = self.get_model().block1(self.get_model().ctm1(token_dict))
                    cluster_image_features.append(cur_token_dict1["x"])

                    cur_image_features2 = torch.cat([token_dict2["x"][i] for i in events[key]], dim=0).unsqueeze(0)
                    token_dict = {'x': cur_image_features2,
                                  'token_num': cur_image_features2.size(1),
                                  'idx_token': torch.arange(cur_image_features2.size(1))[None, :].repeat(
                                      cur_image_features2.size(0), 1),
                                  'agg_weight': cur_image_features2.new_ones(cur_image_features2.size(0),
                                                                             cur_image_features2.size(1),
                                                                             1),
                                  'mask': None}

                    cur_token_dict2 = self.get_model().block2(self.get_model().ctm2(token_dict))
                    cluster_image_features.append(cur_token_dict2["x"])

                image_features = torch.cat(cluster_image_features, dim=1)
                image_features = image_features.to(self.get_model().mm_projector.weight.dtype)

        else:
            if input_type == "video":
                image_features, cls_features = torch.mean(image_features, dim=0, keepdim=False).unsqueeze(
                    0), torch.mean(image_features, dim=1, keepdim=False).unsqueeze(0)
                image_features = torch.cat([image_features, cls_features], dim=1)

        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.encode_images(images)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = cur_input_embeds + (
                            0. * self.get_model().mm_projector(vision_tower.dummy_feature)).sum()
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]

            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape

            if len(image_token_indices) > 1:
                temp = []
                cur, pre = image_token_indices[0], image_token_indices[0]
                for i in image_token_indices:
                    cur = i
                    if cur - pre == 1:
                        temp[-1] = temp[-1] + [cur]
                    else:
                        temp.append([cur])
                    pre = cur

                for i in temp:
                    image_token_start = image_token_indices[0]
                    image_token_end = image_token_indices[-1]
                    cur_image_features = []

                    for _ in i:
                        cur_image_features.append(image_features[cur_image_idx])
                        cur_image_idx += 1

                    if len(i) > 2:
                        cur_image_features = torch.stack(cur_image_features, dim=0)
                        cur_image_features = self.project(cur_image_features, input_type="video")
                        t, l, n = cur_image_features.size()
                        cur_image_features = cur_image_features.contiguous().view(t * l, n)
                    else:
                        cur_image_features = torch.stack(cur_image_features, dim=0)
                        cur_image_features = self.project(cur_image_features, input_type="image")
                        t, l, n = cur_image_features.size()
                        cur_image_features = cur_image_features.contiguous().view(t * l, n)

                    if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                        cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start - 1]).detach())
                        cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start - 1:image_token_start]))
                        cur_new_input_embeds.append(cur_image_features)
                        cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_end + 1:image_token_end + 2]))
                        if labels is not None:
                            cur_new_labels.append(cur_labels[:image_token_start])
                            cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                            cur_new_labels.append(cur_labels[image_token_end:image_token_end + 1])
                            cur_labels = cur_labels[image_token_end + 2:]
                    else:
                        cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                        cur_new_input_embeds.append(cur_image_features)
                        if labels is not None:
                            cur_new_labels.append(cur_labels[:image_token_start])
                            cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                            cur_labels = cur_labels[image_token_end + 1:]

                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end',
                                                                                  False):
                    cur_input_ids = cur_input_ids[image_token_end + 2:]
                else:
                    cur_input_ids = cur_input_ids[image_token_end + 1:]

            elif image_token_indices.numel() > 0:
                cur_image_features = []
                image_token_start = image_token_indices[0]
                image_token_end = image_token_indices[-1]

                for _ in image_token_indices:
                    cur_image_features.append(image_features[cur_image_idx])
                    cur_image_idx += 1

                cur_image_features = torch.stack(cur_image_features, dim=0)
                cur_image_features = self.project(cur_image_features, input_type="image")
                t, l, n = cur_image_features.size()
                cur_image_features = cur_image_features.contiguous().view(t * l, n)

                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start-1]).detach())
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start-1:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_end+1:image_token_end+2]))
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_new_labels.append(cur_labels[image_token_end:image_token_end+1])
                        cur_labels = cur_labels[image_token_end+2:]
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_labels = cur_labels[image_token_end+1:]

                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_input_ids = cur_input_ids[image_token_end+2:]
                else:
                    cur_input_ids = cur_input_ids[image_token_end+1:]

            if cur_input_ids.numel() > 0:
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            tokenizer.add_tokens([DEFAULT_VIDEO_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False