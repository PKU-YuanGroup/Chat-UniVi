# Chat-UniVi: Unified Visual Representation Empowers Large Language Models with Image and Video Understanding

We introduce Chat-UniVi, a unified vision-language model capable of comprehending and engaging in conversations involving images and videos.
Specifically, Chat-UniVi uniformly represents images and videos using a collection of dynamic visual tokens.
This novel representation framework empowers the model to efficiently utilize a limited number of visual tokens to simultaneously capture the spatial details necessary for images and the comprehensive temporal relationship required for videos.
Besides, we leverage a multi-scale representation that equips large language models to perceive both high-level semantic concepts and low-level visual details.

<div align=center>
<img src="figures/fig0.png" width="800px">
</div>

## 1. Conda environment
Install packages.
```
conda create -n chatunivi python=3.10 -y
conda activate chatunivi
pip install --upgrade pip
pip install -e .
```

Install additional packages for training cases.
```
pip install ninja
pip install flash-attn --no-build-isolation
```

## 2. Pre-trained model

Download the pre-trained model.

|          Name           |                                              Weight                                              |
|:-----------------------:|:------------------------------------------------------------------------------------------------:|
|      Chat-UniVi-7B      | [Download](https://huggingface.co/Chat-UniVi/Chat-UniVi/tree/main) |
| Chat-UniVi-ScienceQA-7B | [Download](https://huggingface.co/Chat-UniVi/Chat-UniVi-ScienceQA/tree/main) |



## 3. Run the demo
Please change the model path on line 15 of the main_demo.py first.

```
# line 15 of main_demo.py
model_path = [model path]
```
Then run the demo:
```
CUDA_VISIBLE_DEVICES=0 uvicorn main_demo:app --host 0.0.0.0 --port 8888
```

<div align=center>
<img src="figures/fig1.png" width="800px">
</div>

## 4. Train the model

### Stage1: Multimodal Pre-training
```
deepspeed \
--include localhost:0,1,2,3,4,5,6,7 \
--master_port=29602 \
ChatUniVi/train/train_mem.py \
--deepspeed scripts/zero3.json \
--model_name_or_path [LLM model path] \
--version v1 \
--model_use PRETUNE \
--dataset_use Pretrain \
--vision_tower openai/clip-vit-large-patch14 \
--tune_mm_mlp_adapter True \
--mm_vision_select_layer -2 \
--mm_use_im_start_end False \
--mm_use_im_patch_token False \
--bf16 True \
--output_dir [stage1 save path] \
--num_train_epochs 1 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 24000 \
--save_total_limit 1 \
--learning_rate 2e-3 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 True \
--model_max_length 2048 \
--gradient_checkpointing True \
--dataloader_num_workers 4 \
--lazy_preprocess True \
--report_to wandb
```

### Stage2: Joint Instruction Tuning
```
deepspeed \
--include localhost:0,1,2,3,4,5,6,7 \
--master_port=29601 \
ChatUniVi/train/train_mem.py \
--deepspeed scripts/zero2.json \
--model_name_or_path [LLM model path] \
--version v1 \
--model_use FINETUNE \
--dataset_use FINETUNE \
--vision_tower openai/clip-vit-large-patch14 \
--pretrain_mm_mlp_adapter [stage1 save path]/mm_projector.bin \
--mm_vision_select_layer -2 \
--mm_use_im_start_end False \
--mm_use_im_patch_token False \
--bf16 True \
--output_dir [stage2 save path] \
--num_train_epochs 2 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 50000 \
--save_total_limit 1 \
--learning_rate 2e-5 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 True \
--model_max_length 2048 \
--gradient_checkpointing True \
--dataloader_num_workers 4 \
--lazy_preprocess True \
--report_to wandb
```


## 5. Evaluate the model

### GPT-based evaluation for image understanding
Our quantitative evaluation protocol follows that of LLaVA. 
Following LLaVA, we employ 90 questions based on 30 COCO validation images, covering various aspects, including conversation, detail description, and complex reasoning.
For more details, please refer to LLaVA.

#### Step 1: Load the model to generate results
```
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python ChatUniVi/eval/model_vqa.py \
--model-path [model path] \
--question-file ChatUniVi/eval/questions/coco2014_val_qa_eval/qa90_questions.jsonl \
--image-folder [image folder] \
--answers-file results/answer-file-vqa.jsonl
```

#### Step 2: GPT evaluation
```
OPENAI_API_KEY=[openai api key] \
python ChatUniVi/eval/evaluate/evaluate_gpt_review_visual.py \
--question ChatUniVi/eval/questions/coco2014_val_qa_eval/qa90_questions.jsonl \
--context ChatUniVi/eval/table/caps_boxes_coco2014_val_80.jsonl \
--answer-list ChatUniVi/eval/questions/coco2014_val_qa_eval/qa90_gpt4_answer.jsonl  results/answer-file-vqa.jsonl \
--rule ChatUniVi/eval/table/rule.json \
--output results/review-file-vqa.json
```

#### Step 3: Calculate score
```
python ChatUniVi/eval/evaluate/summarize_gpt_review.py \
-d  results/review-file-vqa.json
```

### GPT-based evaluation for video understanding
The quantitative evaluation protocol for video understanding follows the methodology introduced by Video-ChatGPT. 
Specifically, Video-ChatGPT curates a test set based on the ActivityNet-200 dataset, which includes videos with rich, dense descriptive captions and associated question-answer pairs from human annotations. 
For more details, please refer to Video-ChatGPT.

It is worth noting that the results span a range from 0 to 5. 
To standardize the metrics, we normalized all scores to a scale of 0 to 100 in the paper.

#### Step 1: Load the model to generate results
```
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python ChatUniVi/eval/model_video_consistency.py \
--model-path [model path] \
--question-file ChatUniVi/eval/questions/video_qa/consistency_qa.json \
--video-folder [video folder] \
--answers-file results/answer-video-consistency.jsonl


CUDA_VISIBLE_DEVICES=0,1,2,3 \
python ChatUniVi/eval/model_video_general.py \
--model-path [model path] \
--question-file ChatUniVi/eval/questions/video_qa/generic_qa.json \
--video-folder [video folder] \
--answers-file results/answer-video-generic.jsonl
```


#### Step 2: GPT evaluation
```
# Correctness of Information
python ChatUniVi/eval/evaluate/evaluate_benchmark_1_correctness.py \
--pred_path results/answer-video-generic.jsonl \
--output_dir results/correctness \
--output_json results/review-video-correctness.jsonl \
--api_key [openai api key] \
--num_tasks 1

# Detail Orientation
python ChatUniVi/eval/evaluate/evaluate_benchmark_2_detailed_orientation.py \
--pred_path results/answer-video-generic.jsonl \
--output_dir results/detailed_orientation \
--output_json results/review-video-detailed_orientation.jsonl \
--api_key [openai api key] \
--num_tasks 1

# Contextual Understanding
python ChatUniVi/eval/evaluate/evaluate_benchmark_3_context.py \
--pred_path results/answer-video-generic.jsonl \
--output_dir results/context \
--output_json results/review-video-context.jsonl \
--api_key [openai api key] \
--num_tasks 1

# Temporal Understanding
python ChatUniVi/eval/evaluate/evaluate_benchmark_4_temporal.py \
--pred_path results/answer-video-generic.jsonl \
--output_dir results/temporal \
--output_json results/review-video-temporal.jsonl \
--api_key [openai api key] \
--num_tasks 1

# Consistency
python ChatUniVi/eval/evaluate/evaluate_benchmark_5_consistency.py \
--pred_path results/answer-video-consistency.jsonl \
--output_dir results/consistency \
--output_json results/review-video-consistency.jsonl \
--api_key [openai api key] \
--num_tasks 1
```


### ScienceQA
ScienceQA is a comprehensive multimodal science question-answering dataset comprising 21k multiple-choice questions.
It covers a wide range of domains, spanning 3 subjects, 26 topics, 127 categories, and 379 skills. 
Each example in ScienceQA contains a visual context, a textual context, a question, multiple options, and the correct answer. 
For the input of Chat-UniVi, we concatenate the question, textual context, and options sequentially into a single sentence.

#### ScienceQA Fine-tuning
```
deepspeed \
--include localhost:0,1,2,3,4,5,6,7 \
--master_port=29603 ChatUniVi/train/train.py \
--deepspeed scripts/zero.json \
--model_name_or_path [LLM model path] \
--version v1 \
--model_use FINETUNE \
--dataset_use SQA \
--vision_tower openai/clip-vit-large-patch14  \
--pretrain_mm_mlp_adapter [stage1 save path]/mm_projector.bin \
--mm_vision_select_layer -2 \
--mm_use_im_start_end False \
--mm_use_im_patch_token False \
--bf16 True \
--output_dir [save path] \
--num_train_epochs 9 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 5000 \
--save_total_limit 1 \
--learning_rate 2e-5 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 True \
--model_max_length 2048 \
--gradient_checkpointing True \
--dataloader_num_workers 4 \
--lazy_preprocess True \
--report_to wandb
```


#### Step 1: Load the model to generate results
```
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python ChatUniVi/eval/model_vqa_scienceqa.py \
--model-path [model path] \
--question-file ChatUniVi/eval/questions/scienceqa/test_QCM-LEA.json \
--image-folder [image folder] \
--answers-file results/answer-scienceqa.jsonl
```

#### Step 2: Calculate score
```
python ChatUniVi/eval/evaluate/evaluate_science_qa.py \
--base-dir ChatUniVi/eval/questions/scienceqa \
--result-file results/answer-scienceqa.jsonl \
--output-file results/output-scienceqa.json \
--output-result results/output-result-scienceqa.json
```


### Zero-shot Video Question Evaluation
Our evaluation protocol follows that of Video-ChatGPT, utilizing GPT-assisted evaluation to assess the capabilities of models.
For more details, please refer to Video-ChatGPT.

#### Step 1: Load the model to generate results
```
# MSRVTT QA
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python ChatUniVi/eval/model_video_qa.py \
--model-path [model path] \
--question-file ChatUniVi/eval/questions/video_qa/msrvtt_qa.json \
--video-folder [video folder] \
--answers-list ChatUniVi/eval/questions/video_qa/msrvtt_a_list.json \
--answers-file results/answer-msrvtt-qa.jsonl

# MSVD QA
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python ChatUniVi/eval/model_video_qa.py \
--model-path [model path] \
--question-file ChatUniVi/eval/questions/video_qa/msvd_qa.json \
--video-folder [video folder] \
--answers-list ChatUniVi/eval/questions/video_qa/msvd_a_list.json \
--answers-file results/answer-msvd-qa.jsonl

# TGIF QA
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python ChatUniVi/eval/model_video_qa.py \
--model-path [model path] \
--question-file ChatUniVi/eval/questions/video_qa/tgif_qa.json \
--video-folder [video folder] \
--answers-list ChatUniVi/eval/questions/video_qa/tgif_a_list.json \
--answers-file results/answer-tgif-qa.jsonl

# ActivityNet QA
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python ChatUniVi/eval/model_video_qa.py \
--model-path [model path] \
--question-file ChatUniVi/eval/questions/video_qa/activitynet_qa.json \
--video-folder [video folder] \
--answers-list ChatUniVi/eval/questions/video_qa/activitynet_a_list.json \
--answers-file results/answer-activitynet-qa.jsonl
```

#### Step 2: Calculate score
```
# MSRVTT QA
python ChatUniVi/eval/evaluate/evaluate_video_qa.py \
--pred_path results/answer-msrvtt-qa.jsonl \
--output_dir results/msrvtt-qa \
--output_json results/review-msrvtt-qa.jsonl \
--api_key [openai api key] \
--num_tasks 1

# MSVD QA
python ChatUniVi/eval/evaluate/evaluate_video_qa.py \
--pred_path results/answer-msvd-qa.jsonl \
--output_dir results/msvd-qa \
--output_json results/review-msvd-qa.jsonl \
--api_key [openai api key] \
--num_tasks 1

# TGIF QA
python ChatUniVi/eval/evaluate/evaluate_video_qa.py \
--pred_path results/answer-tgif-qa.jsonl \
--output_dir results/tgif-qa \
--output_json results/review-tgif-qa.jsonl \
--api_key [openai api key] \
--num_tasks 1

# ActivityNet QA
python ChatUniVi/eval/evaluate/evaluate_video_qa.py \
--pred_path results/answer-activitynet-qa.jsonl \
--output_dir results/activitynet-qa \
--output_json results/review-activitynet-qa.jsonl \
--api_key [openai api key] \
--num_tasks 1
```

### Zero-shot Object Hallucination Evaluation
To quantitatively evaluate the hallucination problem of the model, we adopt the polling-based object probing evaluation (POPE) process.
```
# Random
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python ChatUniVi/eval/model_coco_vqa.py \
--model-path [model path] \
--question-file ChatUniVi/eval/questions/coco_pope/coco_pope_random.jsonl \
--image-folder [image folder] \
--answers-file results/pope-random.jsonl

# Popular
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python ChatUniVi/eval/model_coco_vqa.py \
--model-path [model path] \
--question-file ChatUniVi/eval/questions/coco_pope/coco_pope_popular.jsonl \
--image-folder [image folder] \
--answers-file results/pope-popular.jsonl

# Adversarial
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python ChatUniVi/eval/model_coco_vqa.py \
--model-path [model path] \
--question-file ChatUniVi/eval/questions/coco_pope/coco_pope_adversarial.jsonl \
--image-folder [image folder] \
--answers-file results/pope-adversarial.jsonl
```