## Chat-UniVi-13B
We releas [Chat-UniVi-13B](https://huggingface.co/Chat-UniVi/Chat-UniVi-13B/tree/main).
Our proposed unified visual representation framework greatly reduces the number of visual tokens,
so you can train **13B unified image and video understanding models** in full parameters directly on **8 A100 GPUs** within **3 days**.


## Main Results
### Image understanding

<div align="center">
<table border="1" width="100%">
    <tr align="center">
        <th>Methods</th><th>LLM</th><th>Conversation</th><th>Detail Description</th><th>Complex Reasoning</th><th>All</th>
    </tr>
    <tr align="center">
        <td><a href="https://huggingface.co/Chat-UniVi/Chat-UniVi">Chat-UniVi-7B</a></td><td><a href="https://huggingface.co/lmsys/vicuna-7b-v1.5">Vicuna-7B</a></td><td><b>84.1</b></td><td>74.2</td><td>93.7</td><td>84.2</td>
    </tr>
    </tr>
    <tr align="center">
        <td><a href="https://huggingface.co/Chat-UniVi/Chat-UniVi-13B">Chat-UniVi-13B</a></td><td><a href="https://huggingface.co/lmsys/vicuna-13b-v1.5">Vicuna-13B</a></td><td><b>84.1</b></td><td><b>79.4</b></td><td><b>94.7</b></td><td><b>86.1</b></td>
    </tr>
</table>
</div>

### Video understanding

<div align="center">
<table border="1" width="100%">
    <tr align="center">
        <th>Methods</th><th>LLM</th><th>Correct</th><th>Detail</th><th>Context</th><th>Temporal</th><th>Consistency</th>
    </tr>
    <tr align="center">
        <td><a href="https://huggingface.co/Chat-UniVi/Chat-UniVi">Chat-UniVi-7B</a></td><td><a href="https://huggingface.co/lmsys/vicuna-7b-v1.5">Vicuna-7B</a></td><td>57.8</td><td>58.2</td><td>69.2</td><td>57.8</td><td>56.2</td>
    </tr>
    </tr>
    <tr align="center">
        <td><a href="https://huggingface.co/Chat-UniVi/Chat-UniVi-13B">Chat-UniVi-13B</a></td><td><a href="https://huggingface.co/lmsys/vicuna-13b-v1.5">Vicuna-13B</a></td><td><b>59.4</b></td><td><b>59.8</b></td><td><b>70.5</b></td><td><b>58.0</b></td><td><b>60.6</b></td>
    </tr>
</table>
</div>

### ScienceQA

<div align="center">
<table border="1" width="100%">
    <tr align="center">
        <th>Methods</th><th>LLM</th><th>Average</th><th>Subject</th><th></th><th></th><th>Context Modality</th><th></th><th></th><th>Grade</th><th></th>
    </tr>
    <tr align="center">
        <th></th><th></th><th></th><th>NAT</th><th>SOC</th><th>LAN</th><th>TXT</th><th>IMG</th><th>NO</th><th>G1-6</th><th>G7-12</th>
    </tr>
    <tr align="center">
        <td><a href="https://huggingface.co/Chat-UniVi/Chat-UniVi">Chat-UniVi-7B</a></td><td><a href="https://huggingface.co/lmsys/vicuna-7b-v1.5">Vicuna-7B</a></td><td>88.78</td><td>88.50</td><td>93.03</td><td>85.91</td><td>88.51</td><td>85.97</td><td>88.15</td><td>88.88</td><td>88.60</td>
    </tr>
    </tr>
    <tr align="center">
        <td><a href="https://huggingface.co/Chat-UniVi/Chat-UniVi-13B">Chat-UniVi-13B</a></td><td><a href="https://huggingface.co/lmsys/vicuna-13b-v1.5">Vicuna-13B</a></td><td><b>90.99</b></td><td><b>90.41</b></td><td><b>95.05</b></td><td><b>88.91</b></td><td><b>89.64</b></td><td><b>88.05</b></td><td><b>90.94</b></td><td><b>91.19</b></td><td><b>90.64</b></td>
    </tr>
</table>
</div>

### VideoQA
<div align="center">
<table border="1" width="100%">
    <tr align="center">
        <th>Methods</th><th>LLM</th><th>MSRVTT-QA</th><th></th><th>MSVD-QA</th><th></th><th>TGIF-QA</th><th></th><th>ActivityNet-QA</th><th></th>
    </tr>
    <tr align="center">
        <th></th><th></th><th>Accuracy</th><th>Score</th><th>Accuracy</th><th>Score</th><th>Accuracy</th><th>Score</th><th>Accuracy</th><th>Score</th>
    </tr>
    <tr align="center">
        <td><a href="https://huggingface.co/Chat-UniVi/Chat-UniVi">Chat-UniVi-7B</a></td><td><a href="https://huggingface.co/lmsys/vicuna-7b-v1.5">Vicuna-7B</a></td><td>54.6</td><td>3.1</td><td>65.0</td><td>3.6</td><td>60.3</td><td><b>3.4</b></td><td><b>45.8</b></td><td><b>3.2</b></td>
    </tr>
    </tr>
    <tr align="center">
        <td><a href="https://huggingface.co/Chat-UniVi/Chat-UniVi-13B">Chat-UniVi-13B</a></td><td><a href="https://huggingface.co/lmsys/vicuna-13b-v1.5">Vicuna-13B</a></td><td><b>-</b></td><td><b>-</b></td><td><b>-</b></td><td><b>-</b></td><td><b>-</b></td><td><b>-</b></td><td><b>46.4</b></td><td><b>3.6</b></td>
    </tr>
</table>
</div>

### Hallucination Evaluation (POPE)
Coming soon.

## Train the 13B model
All you have to do is replace the base model with the 13B model.

### Stage1: Multimodal Pre-training
```
deepspeed \
--include localhost:0,1,2,3,4,5,6,7 \
--master_port=29602 \
ChatUniVi/train/train_mem.py \
--deepspeed scripts/zero3.json \
--model_name_or_path ${LLM model path} \
--version v1 \
--model_use PRETUNE \
--dataset_use Pretrain \
--vision_tower openai/clip-vit-large-patch14 \
--tune_mm_mlp_adapter True \
--mm_vision_select_layer -2 \
--mm_use_im_start_end False \
--mm_use_im_patch_token False \
--bf16 True \
--output_dir ${stage1 save path} \
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
--model_name_or_path ${LLM model path} \
--version v1 \
--model_use FINETUNE \
--dataset_use FINETUNE \
--vision_tower openai/clip-vit-large-patch14 \
--pretrain_mm_mlp_adapter ${stage1 save path}/mm_projector.bin \
--mm_vision_select_layer -2 \
--mm_use_im_start_end False \
--mm_use_im_patch_token False \
--bf16 True \
--output_dir ${stage2 save path} \
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
