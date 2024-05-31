from PIL import Image
import math
from decord import VideoReader, cpu
import numpy as np
import os
import torch


def _get_rawvideo_dec(video_path, image_processor, max_frames=64, image_resolution=224, video_framerate=1, s=None, e=None):
    # speed up video decode via decord.
    video_mask = np.zeros(max_frames, dtype=np.int64)
    max_video_length = 0

    # T x 3 x H x W
    video = np.zeros((max_frames, 3, image_resolution, image_resolution), dtype=np.float64)

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
        # T x 3 x H x W
        sample_fps = int(video_framerate)
        t_stride = int(round(float(fps) / sample_fps))

        all_pos = list(range(f_start, f_end + 1, t_stride))
        if len(all_pos) > max_frames:
            sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=max_frames, dtype=int)]
        else:
            sample_pos = all_pos

        patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]

        patch_images = [image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0] for img in patch_images]
        slice_len = len(patch_images)
        return  patch_images, slice_len
        max_video_length = max_video_length if max_video_length > slice_len else slice_len
        if slice_len < 1:
            pass
        else:
            while len(patch_images) < max_frames:
                patch_images.append(torch.zeros((3, image_resolution, image_resolution)))
            # video[:slice_len, ...] = patch_images
    else:
        print("video path: {} error.".format(video_path))

    video_mask[:max_video_length] = [1] * max_video_length

    return patch_images, video_mask