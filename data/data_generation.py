import argparse
import jsonlines
import json
import random
import os
import copy


TEMPLATE = {
        "id": "",
        "image": "",
        "conversations": [
            {
                "from": "human",
                "value": ""
            },
            {
                "from": "gpt",
                "value": ""
            }
        ]
    }


def read_jsonl(file):
    results = []
    with open(file, "r", encoding='utf-8') as f:
        for item in jsonlines.Reader(f):
            results.append(item)
    return results


def read_json(file):
    with open(file, "r", encoding='utf-8') as f:
        data = json.load(f)
    return data


def write_json(file, data):
    with open(file, "w") as f:
        json.dump(data, f, indent=2)


def COCO_CAP(args):
    save_path = "coco_cap_chat.json"

    COCO_CAP = read_jsonl(f"{args.caption_file}")
    CAP_template = read_json(f"{args.template_path}/image_cap.json")

    template_num = len(CAP_template)

    chat = []
    for j, i in enumerate(COCO_CAP):
        temp = copy.deepcopy(TEMPLATE)
        temp["id"] = "COCO_CAP_"+str(j)
        temp["image"] = i["img_path"]

        template_use = CAP_template[random.randint(0, template_num - 1)]
        if "<image>" in template_use:
            exit()

        temp["conversations"] = [
                {
                    "from": "human",
                    "value":  "<image>\n" + template_use
                },
                {
                    "from": "gpt",
                    "value": i["caption"]
                }
        ]
        chat.append(temp)

    write_json(save_path, chat)
    print(len(chat))


def COCO_REC(args):
    save_path = "coco_rec_chat.json"

    COCO_CAP = read_jsonl(f"{args.ref_file}")
    REG_template = read_json(f"{args.template_path}/COCO_REG.json")

    bbox_template = [
        "defined by <box_start> [<x0>, <y0>, <x1>, <y1>] <box_end>",
        "defined by the coordinates <box_start> [<x0>, <y0>, <x1>, <y1>] <box_end>",
        "bounded by the coordinates <box_start> [<x0>, <y0>, <x1>, <y1>] <box_end>"
    ]

    template_num = len(REG_template)

    chat = []
    for j, i in enumerate(COCO_CAP):
        temp = copy.deepcopy(TEMPLATE)
        temp["id"] = "COCO_REC_"+str(j)
        temp["image"] = i["img_path"]
        [x0, y0, x1, y1] = i["bbox"]
        temp["bbox"] = [round(x0, 2),
                        round(y0, 2),
                        round(x1, 2),
                        round(y1, 2)]


        template_use = copy.deepcopy(REG_template[random.randint(0, template_num - 1)])

        bbox = copy.deepcopy(bbox_template[random.randint(0, 2)])
        bbox = bbox.replace("<x0>", str(temp["bbox"][0]))
        bbox = bbox.replace("<y0>", str(temp["bbox"][1]))
        bbox = bbox.replace("<x1>", str(temp["bbox"][2]))
        bbox = bbox.replace("<y1>", str(temp["bbox"][3]))

        template_use = template_use.replace("<objs>", bbox)
        if "<image>" in template_use:
            exit()

        temp["conversations"] = [
                {
                    "from": "human",
                    "value":  "<image>\n" + template_use
                },
                {
                    "from": "gpt",
                    "value": i["expression"]
                }
        ]
        chat.append(temp)

    write_json(save_path, chat)
    print(len(chat))


def COCO_REG(args):
    save_path = "coco_reg_chat.json"

    COCO_CAP = read_jsonl(f"{args.ref_file}")
    REG_template = read_json(f"{args.template_path}/COCO_REC.json")
    REG_R_template = read_json(f"{args.template_path}/COCO_REC_responce.json")

    bbox_template = [
        "defined by <box_start> [<x0>, <y0>, <x1>, <y1>] <box_end>",
        "defined by the coordinates <box_start> [<x0>, <y0>, <x1>, <y1>] <box_end>",
        "bounded by the coordinates <box_start> [<x0>, <y0>, <x1>, <y1>] <box_end>"
    ]

    hunam_template_num = len(REG_template)
    gpt_template_num = len(REG_R_template)

    chat = []
    for j, i in enumerate(COCO_CAP):
        temp = copy.deepcopy(TEMPLATE)
        temp["id"] = "COCO_REG_"+str(j)
        temp["image"] = i["img_path"]
        [x0, y0, x1, y1] = i["bbox"]
        temp["bbox"] = [round(x0, 2),
                        round(y0, 2),
                        round(x1, 2),
                        round(y1, 2)]


        hunam_template_use = copy.deepcopy(REG_template[random.randint(0, hunam_template_num - 1)])
        gpt_template_use = copy.deepcopy(REG_R_template[random.randint(0, gpt_template_num - 1)])

        bbox = copy.deepcopy(bbox_template[random.randint(0, 2)])
        bbox = bbox.replace("<x0>", str(temp["bbox"][0]))
        bbox = bbox.replace("<y0>", str(temp["bbox"][1]))
        bbox = bbox.replace("<x1>", str(temp["bbox"][2]))
        bbox = bbox.replace("<y1>", str(temp["bbox"][3]))

        hunam_template_use = hunam_template_use.replace("<expr>", i["expression"])
        gpt_template_use = gpt_template_use.replace("<expr>", "object")
        gpt_template_use = gpt_template_use.replace("<objs>", bbox)

        temp["conversations"] = [
                {
                    "from": "human",
                    "value":  "<image>\n" + hunam_template_use
                },
                {
                    "from": "gpt",
                    "value": gpt_template_use
                }
        ]
        chat.append(temp)

    write_json(save_path, chat)
    print(len(chat))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--template_path", type=str, default="template")
    parser.add_argument("--caption_file", type=str, default="coco2014_train.jsonl")
    parser.add_argument("--ref_file", type=str, default="ref3_train.jsonl")
    args = parser.parse_args()

    COCO_REC(args)
    COCO_REG(args)
    COCO_CAP(args)