from .dataset_config import *
from .model_config import *


ModelConfig = {
    "PRETUNE": model_config_pretune,
    "FINETUNE": model_config_finetune,
}


DataConfig = {
    "Pretrain": [Pretrain, COCO_CAP, COCO_REG, COCO_REC],
    "SQA": [SQA],
    "FINETUNE": [VIT, MIMIC_imageonly, VIDEO],
}