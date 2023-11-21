## Data preparation
### Data for training
We provide the processed data as follows.

<div align="center">
<table border="1" width="100%">
    <tr align="center">
        <th>Datasets</th><th>Hugging Face</th><th>Baidu Disk</th>
    </tr>
    <tr align="center">
        <td>Multimodal Pre-training</td><td><a href="https://huggingface.co/datasets/Chat-UniVi/Chat-UniVi-Instruct/tree/main">Link</a></td><td>-</td>
    </tr>
    </tr>
    <tr align="center">
        <td>Joint Instruction Tuning</td><td><a href="https://huggingface.co/datasets/Chat-UniVi/Chat-UniVi-Instruct/tree/main">Link</a></td><td>-</td>
    </tr>
    <tr align="center">
        <td>ScienceQA</td><td><a href="https://huggingface.co/datasets/Chat-UniVi/Chat-UniVi-Instruct/tree/main">Link</a></td><td>-</td>
    </tr>
</table>
</div>

### Data for validating
We provide the processed data as follows. The annotations are provided in [eval/questions](https://github.com/PKU-YuanGroup/Chat-UniVi/tree/main/ChatUniVi/eval/questions).

<div align="center">
<table border="1" width="100%">
    <tr align="center">
        <th>Datasets</th><th>Hugging Face</th><th>Baidu Disk</th><th>Google Disk</th><th>Peking University Disk</th>
    </tr>
    <tr align="center">
        <td>Image_Understanding</td><td><a href="https://huggingface.co/datasets/Chat-UniVi/Chat-UniVi-Eval/tree/main">Link</a></td><td>-</td><td>-</td><td>-</td>
    </tr>
    <tr align="center">
        <td>Video_Understanding</td><td><a href="https://huggingface.co/datasets/Chat-UniVi/Chat-UniVi-Eval/tree/main">Link</a></td><td>-</td><td>-</td><td>-</td>
    </tr>
    <tr align="center">
        <td>ScienceQA</td><td><a href="https://huggingface.co/datasets/Chat-UniVi/Chat-UniVi-Eval/tree/main">Link</a></td><td>-</td><td>-</td><td>-</td>
    </tr>
    <tr align="center">
        <td>Activitynet_Zero_Shot_QA</td><td><a href="https://huggingface.co/datasets/Chat-UniVi/Chat-UniVi-Eval/tree/main">Link</a></td><td><a href="https://pan.baidu.com/s/1d_AVx9Mz_57nA3exhQZGyA?pwd=9amr ">Link</a></td><td>-</td><td>-</td>
    </tr>
    </tr>
    <tr align="center">
        <td>MSRVTT_Zero_Shot_QA</td><td><a href="https://huggingface.co/datasets/Chat-UniVi/Chat-UniVi-Eval/tree/main">Link</a></td><td><a href="https://pan.baidu.com/s/1QHUtwHXm4Vc-Wc12XFCFsA?pwd=1rj8">Link</a></td><td><a href="https://drive.google.com/file/d/1yXh9lz7flQ5Ui2IRSd6Qi6RqSEeUJwl3/view?usp=drive_link">Link</a></td><td>-</td>
    </tr>
    </tr>
    <tr align="center">
        <td>MSVD_Zero_Shot_QA</td><td><a href="https://huggingface.co/datasets/Chat-UniVi/Chat-UniVi-Eval/tree/main">Link</a></td><td><a href="https://pan.baidu.com/s/1PJSHkjHG2BPl_ddUnBj9AA?pwd=jj34">Link</a></td><td><a href="https://drive.google.com/file/d/1_q4eiSdb7i8P3Hmh4lCfgY1uBGyzU_7X/view?usp=drive_link">Link</a></td><td><a href="https://disk.pku.edu.cn:443/link/8B0D01747D8AA65534820B7E60CBFEFC">Link</a></td>
    </tr>
    </tr>
    <tr align="center">
        <td>TGIF_Zero_Shot_QA</td><td><a href="https://huggingface.co/datasets/Chat-UniVi/Chat-UniVi-Eval/tree/main">Link</a></td><td><a href="https://pan.baidu.com/s/11ubtWbTtubyBmN9UPvAyow?pwd=98yr">Link</a></td><td><a href="https://drive.google.com/file/d/1so6L9rg_gdC8Segur7rKML-ffd4Ix_I6/view?usp=drive_link">Link</a></td><td><a href="https://disk.pku.edu.cn:443/link/B9AB387EFE8817158F181FF3D7A97163">Link</a></td>
    </tr>
    <tr align="center">
        <td>POPE</td><td><a href="https://huggingface.co/datasets/Chat-UniVi/Chat-UniVi-Eval/tree/main">Link</a></td><td>-</td><td>-</td><td>-</td>
    </tr>
</table>
</div>

### Data parameter
Modify the data path in [config/dataset_config.py](https://github.com/PKU-YuanGroup/Chat-UniVi/blob/main/ChatUniVi/config/dataset_config.py):
```python
Pretrain = {
    "chat_path": "${PATH}/CC3M-595K/chat.json",
    "CC3M": "${PATH}/CC3M-595K",
}

VIT = {
    "chat_path": "${PATH}/llava_instruct_150k.json",
    "COCO2017": "${PATH}/COCO2017/train2017",
}

MIMIC_imageonly = {
    "chat_path": "${PATH}/MIMIC-IT-imageonly.json",
    "CDG": "${PATH}/CGD/images",
    "LA": "${PATH}/LA/images",
    "SD": "${PATH}/SD/images",
}

COCO_CAP = {
    "chat_path": "${PATH}/COCO/coco_cap_chat.json",
    "COCO2014": "${PATH}/COCO2014/train2014",
}

COCO_REG = {
    "chat_path": "${PATH}/COCO/coco_reg_chat.json",
    "COCO2014": "${PATH}/COCO2014/train2014",
}

COCO_REC = {
    "chat_path": "${PATH}/COCO/coco_rec_chat.json",
    "COCO2014": "${PATH}/COCO2014/train2014",
}

VIDEO = {
    "chat_path": "${PATH}/video_chat.json",
    "VIDEO": "${PATH}/Activity_Videos",
}

SQA = {
    "chat_path": "${PATH}/llava_train_QCM-LEA.json",
    "ScienceQA": "${PATH}/scienceqa/train",
}
```

## Prepare your own training dataset
### Format of data
All the conversation data is in JSON format, and each conversation has the following content:
* id: Used to distinguish different samples
* "image" or "video": The name of the image or video
* conversations: Conversations data

```json
[
  {
    "id": "COCO_CAP_0",
    "image": "COCO_train2014_000000222016.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nDescribe the main events or objects in the image."
      },
      {
        "from": "gpt",
        "value": "a big red telephone booth that a man is standing in"
      }
    ]
  },
]
```

### Data parameter
Modify the data path in [config/dataset_config.py](https://github.com/PKU-YuanGroup/Chat-UniVi/blob/main/ChatUniVi/config/dataset_config.py):
```python
New_data = {
    "chat_path": "${PATH}/CC3M-595K/chat.json",
    "new_data": "${PATH}/CC3M-595K",
}
```

Then, modify the config in [config/__init__.py](https://github.com/PKU-YuanGroup/Chat-UniVi/blob/main/ChatUniVi/config/__init__.py).
You can also combine different datasets using the list format:

```python
DataConfig = {
    "New": [New_data],
}
```

To use the new dataset when training, you only need to change the parameters of "dataset_use" in the command:
```python
--dataset_use New
```
