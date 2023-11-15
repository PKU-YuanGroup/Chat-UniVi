import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from ChatUniVi.constants import *
from ChatUniVi.conversation import conv_templates, SeparatorStyle
from ChatUniVi.model.builder import load_pretrained_model
from ChatUniVi.utils import disable_torch_init
from ChatUniVi.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image
import math
from abc import ABC
import numpy as np
import jsonlines


def get_acc(file):
    acc, num = 0, 0
    yes, no, fail = 0, 0, 0
    tp, fp, fn, tn = 0, 0, 0, 0

    with open(file, "r", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            num += 1
            if "Yes" in item["text"] or "yes" in item["text"]:
                yes += 1
                if "Yes" in item["label"] or "yes" in item["label"]:
                    acc += 1
                    tp += 1
                else:
                    fp += 1

            elif "No" in item["text"] or "no" in item["text"]:
                no += 1
                if "No" in item["label"] or "no" in item["label"]:
                    acc += 1
                    tn += 1
                else:
                    fn += 1
            else:
                fail += 1

    result = {
        "acc": acc / num,
        "yes": yes / num,
        "no": no / num,
        "fail": fail / num,
        "precision": tp / (tp + fp),
        "recall": tp / (tp + fn),
    }
    result["F1-score"] = 2 * result["precision"] * result["recall"] / (result["precision"] + result["recall"])
    print("\n========================================================================")
    print(file)
    print(result)
    print("========================================================================\n")
    return result


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


class LogitsProcessor(ABC):
    """Abstract base class for all logit processors that can be applied during generation."""
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Torch method for processing logits."""
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = "ChatUniVi"
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    image_processor = vision_tower.image_processor

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        try:
            idx = line["question_id"]
            image_file = line["image"]
            qs = line["text"]
            label = line["label"]

            cur_prompt = qs
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            image = Image.open(os.path.join(args.image_folder, image_file))

            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            if args.answer_prompter:
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensor.unsqueeze(0).half().cuda(),
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        max_new_tokens=1024,
                        use_cache=True,
                        stopping_criteria=[stopping_criteria]
                    )

                input_token_len = input_ids.shape[1]
                n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
                if n_diff_input_output > 0:
                    print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
                outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
                outputs = outputs.strip()
                if outputs.endswith(stop_str):
                    outputs = outputs[:-len(stop_str)]
                outputs = outputs.strip()

                outputs_reasoning = outputs
                input_ids = tokenizer_image_token(prompt + outputs_reasoning + ' The answer is ', tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            else:
                outputs_reasoning = ""

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    output_scores=True,
                    return_dict_in_generate=True,
                    max_new_tokens=1024,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria]
                    )

            scores = output_ids.scores[0][0].to(torch.float32)
            label_score = []
            candidates = ["yes", "Yes", "no", "No"]
            for can in candidates:
                can_id = tokenizer.encode(can)[-1]
                label_score.append(scores[can_id].item())
            outputs = candidates[np.argmax(label_score)]

            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "outputs_reasoning": outputs_reasoning + ' The answer is ' + outputs,
                                   "text": outputs,
                                   "label": label,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
            ans_file.flush()
        except Exception as e:
            print(f"Error processing image file '{image_file}': {e}")
    ans_file.close()
    get_acc(answers_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="simpleqa")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--model_use", type=str, default="BASE")
    parser.add_argument("--answer-prompter", action="store_true")
    args = parser.parse_args()

    eval_model(args)
