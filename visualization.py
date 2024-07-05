import numpy as np
from PIL import Image
import math
import os
import torch
from ChatUniVi.model.multimodal_encoder.clip_encoder import CLIPVisionTower
from ChatUniVi.model.cluster import CTM, TCBlock


def split(image, patch_size=14, idx=None):
    img = np.asarray(image, dtype=np.uint8).copy()
    h, w, _ = img.shape

    horizontal_lines = [i for i in range(patch_size, h, patch_size)]
    vertical_lines = [i for i in range(patch_size, w, patch_size)]

    for i in horizontal_lines:
        for j in range(w):
            img[i, j, :] = 0

    for j in vertical_lines:
        for i in range(h):
            img[i, j, :] = 0

    image = Image.fromarray(img, 'RGB')
    return image


def merge(image, token_dict, patch_size=14, alpha=0.2, line_color=np.array([200, 200, 200])):
    img = np.asarray(image, dtype=np.uint8).copy()
    h, w, _ = img.shape

    patch_num_h, patch_num_w = h // patch_size, w // patch_size

    color_map = {}
    idx = token_dict["idx_token"].tolist()[0]
    for id, i in enumerate(idx):
        color_map[i] = color_map[i] if i in color_map else {"id": [], "color": []}

        color_map[i]["id"].append(id)

        for _h in range(patch_size):
            for _w in range(patch_size):
                color_map[i]["color"].append(img[_h + patch_size * math.floor(id / patch_num_w),
                                                 _w + patch_size * (id % patch_num_h)])

    for i in color_map:
        color_map[i]["color"] = np.mean(np.stack(color_map[i]["color"], axis=0), axis=0)

        for id in color_map[i]["id"]:
            for _h in range(patch_size):
                for _w in range(patch_size):
                    color = img[_h + patch_size * math.floor(id / patch_num_w), _w + patch_size * (
                                id % patch_num_h)] * alpha + color_map[i]["color"] * (1 - alpha)
                    img[_h + patch_size * math.floor(id / patch_num_w), _w + patch_size * (id % patch_num_h)] = color

    for id, i in enumerate(idx):
        if math.floor(id / patch_num_w) > 0:
            if idx[id - patch_num_w] != i:
                for _w in range(patch_size * (id % patch_num_h), patch_size * (id % patch_num_h + 1)):
                    img[patch_size * math.floor(id / patch_num_w), _w, :] = line_color

        if (id % patch_num_h) > 0:
            if idx[id - 1] != i:
                for _h in range(patch_size * math.floor(id / patch_num_w), patch_size * (math.floor(id / patch_num_w) + 1)):
                    img[_h, patch_size * (id % patch_num_h), :] = line_color

    image = Image.fromarray(img, 'RGB')
    return image


if __name__ == '__main__':
    image_path = "figures/COCO_val2014_000000214293.jpg"
    clip_vit_14_path = ${openai_clip_path}
    output_file = "figures"

    if not os.path.exists(output_file):
        os.makedirs(output_file)

    vision_tower = CLIPVisionTower(clip_vit_14_path)
    image = Image.open(os.path.join(image_path)).resize((224, 224))

    ctm0 = CTM(sample_ratio=64, embed_dim=1024, dim_out=1024, k=32)
    block0 = TCBlock(dim=1024, num_heads=8)

    ctm1 = CTM(sample_ratio=32, embed_dim=1024, dim_out=1024, k=3)
    block1 = TCBlock(dim=1024, num_heads=8)

    ctm2 = CTM(sample_ratio=16, embed_dim=1024, dim_out=1024, k=3)
    block2 = TCBlock(dim=1024, num_heads=8)

    image_processor = vision_tower.image_processor

    img = np.asarray(image, dtype=np.uint8).copy()
    h, w, _ = img.shape
    image = image.resize((math.ceil(h/224) * 224, math.ceil(w/224) * 224))
    image.save("{}/input.jpg".format(output_file))

    img = np.asarray(image, dtype=np.uint8).copy()
    h, w, _ = img.shape
    print(h, w)
    new_image = []
    for i in range(math.ceil(h/224)):
        for j in range(math.ceil(w/224)):
            new_image.append(img[i * 224: (i + 1) * 224, j * 224: (j + 1) * 224, :])

    new_image = [Image.fromarray(img, 'RGB') for img in new_image]
    print(new_image)
    image_tensor = torch.cat([image_processor.preprocess(img, return_tensors='pt')['pixel_values'].half() for img in new_image], dim=0)
    print(image_tensor.size())

    image_features = vision_tower(image_tensor)
    image_features = image_features.view(1, -1, 1024)
    print(image_features.size())

    token_dict = {'x': image_features,
                  'token_num': image_features.size(1),
                  'idx_token': torch.arange(image_features.size(1))[None, :].repeat(
                      image_features.size(0), 1),
                  'agg_weight': image_features.new_ones(image_features.size(0), image_features.size(1),
                                                        1),
                  'mask': None,
                  'init_grid_size': (14, 14)}

    img = merge(image, token_dict, alpha=1, line_color=np.array([255, 255, 255]))
    img.save("{}/vanilla.jpg".format(output_file))

    token_dict0 = block0(ctm0(token_dict))

    img = merge(image, token_dict0, alpha=0.2, line_color=np.array([255, 255, 255]))
    img.save("{}/stage1.jpg".format(output_file))

    token_dict1 = block1(ctm1(token_dict0))
    img = merge(image, token_dict1, alpha=0.2, line_color=np.array([255, 255, 255]))
    img.save("{}/stage2.jpg".format(output_file))

    token_dict2 = block2(ctm2(token_dict1))

    img = merge(image, token_dict2, alpha=0.2, line_color=np.array([255, 255, 255]))
    img.save("{}/stage3.jpg".format(output_file))
