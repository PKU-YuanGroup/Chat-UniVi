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

Pretrain_valley_llava = {
    "chat_path": "${PATH}/valley_llavaimage.json",
    "valley": "${PATH}/Data",
    "llava": "${PATH}/Data",  # from llava v1.5
}

LLaVA = {
    "chat_path": "${PATH}/llavaimage_tune.json",
    "llava": "${PATH}/Data",  # from llava v1.5
}