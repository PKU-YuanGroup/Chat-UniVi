import torch
from .constants import *
from .conversation import conv_templates, SeparatorStyle
from .model.builder import load_pretrained_model
from .utils import disable_torch_init
from .mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from PIL import Image
import os
from decord import VideoReader, cpu
import numpy as np


class Chat:
    def __init__(self, model_path, conv_mode="simple"):
        disable_torch_init()
        self.tokenizer, self.model, self.image_processor, context_len = load_pretrained_model(model_path, None, model_name="ChatUniVi")

        mm_use_im_start_end = getattr(self.model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(self.model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            self.tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            self.tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        self.model.resize_token_embeddings(len(self.tokenizer))

        vision_tower = self.model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()

        self.image_processor = vision_tower.image_processor
        self.conv_mode = conv_mode
        print(self.model)

    def get_prompt(self, qs, state):
        state.append_message(state.roles[0], qs)
        state.append_message(state.roles[1], None)
        return state

    def _get_rawvideo_dec(self, video_path, image_processor, max_frames=MAX_IMAGE_LENGTH, image_resolution=224,
                          video_framerate=1, s=None, e=None):
        if s is None:
            start_time, end_time = None, None
        else:
            start_time = int(s)
            end_time = int(e)
            start_time = start_time if start_time >= 0. else 0.
            end_time = end_time if end_time >= 0. else 0.
            if start_time > end_time:
                start_time, end_time = end_time, start_time
            elif start_time == end_time:
                end_time = start_time + 1

        if os.path.exists(video_path):
            vreader = VideoReader(video_path, ctx=cpu(0))
        else:
            print(video_path)
            raise FileNotFoundError

        fps = vreader.get_avg_fps()
        f_start = 0 if start_time is None else int(start_time * fps)
        f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
        num_frames = f_end - f_start + 1
        if num_frames > 0:
            sample_fps = int(video_framerate)
            t_stride = int(round(float(fps) / sample_fps))

            all_pos = list(range(f_start, f_end + 1, t_stride))
            if len(all_pos) > max_frames:
                sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=max_frames, dtype=int)]
            else:
                sample_pos = all_pos

            patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]
            return patch_images

    @torch.inference_mode()
    def generate(self, images_tensor: list, prompt: str, first_run: bool, state):
        tokenizer, model, image_processor = self.tokenizer, self.model, self.image_processor

        state = self.get_prompt(prompt, state)
        prompt = state.get_prompt()
        print(prompt)

        images_tensor = torch.stack(images_tensor, dim=0)
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        temperature = 0.2
        max_new_tokens = 1024

        stop_str = conv_templates[self.conv_mode].copy().sep if conv_templates[self.conv_mode].copy().sep_style != SeparatorStyle.TWO else \
        conv_templates[self.conv_mode].copy().sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                do_sample=True,
                temperature=temperature,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        print('response', outputs)
        return outputs, state