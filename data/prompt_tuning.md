# Prompt Tuning

For reproducing the fine-tuning results in our paper, we provide the corresponding prompt-tuning configs in `configs/BERT_L12_H768_experiments/prompt_tuning` and `configs/BERT_L12_H768_experiments/moe_prompt_tuning` for Uni-Perceiver-Base and Uni-Perceiver-MoE-Base, respectively.

Specifically, we prompt-tuned the ImageNet-1K dataset with image classification task. For video classification, we fine-tuned Kinetics-400. We also employed image caption and image-text retrieval tasks on MSCOCO caption and FLicker-30K datasets.
In addition, video caption and video-text retrieval tasks are conducted on MSVD dataset.
Please perpare the dataset following [PREPARE_DATA.md](prepare_data.md)

---

In our experiments,  prompt-tuning on all datasets benchmarks is performed on 16 NVIDIA-V100 GPUs with 80GB memory.  
Taking Imagenet-1K as an example, the __Uni-Perceiver-Base__ can be prompt-tuned as
```

sh run.sh configs/BERT_L12_H768_experiments/prompt_tuning/in1k_prompt_tuning_0.01data_lr1e-4.yaml in1k-pt 16 partitionname MODEL.WEIGHTS work_dirs/pretrained_models/uni-perceiver-base-L12-H768-224size-pretrained.pth

```
The __Uni-Perceiver-MoE-Base__ can also be fine-tuned in a similar way:
```
sh run.sh configs/BERT_L12_H768_experiments/moe_prompt_tuning/in1k_prompt_tuning_0.01data_lr1e-4.yaml in1k-moe-pt 16 partitionname MODEL.WEIGHTS work_dirs/pretrained_models/uni-perceiver-moe-base-L12-H768-224size-pretrained.pth
```

