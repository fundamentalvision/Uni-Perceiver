# Fine-tuning

For reproducing the fine-tuning results in our paper, we provide the corresponding fine-tuning configs in `configs/BERT_L12_H768_experiments/finetuning` and `configs/BERT_L12_H768_experiments/moe_finetuning` for Uni-Perceiver-Base and Uni-Perceiver-MoE-Base, respectively.


Specifically, we fine-tuned the ImageNet-1K dataset with image classification task. For video classification, we fine-tuned Kinetics-400. We also employed image caption and image-text retrieval tasks on MSCOCO caption and FLicker-30K datasets.
In addition, language understand tasks are fine-tuned on GLUE benchmarks, and video caption and video-text retrieval tasks are conducted on MSVD dataset.
Please perpare the dataset following [PREPARE_DATA.md](prepare_data.md)

--- 

In our experiments,  fine-tuning on all datasets exception GLUE benchmarks is performed on 16 NVIDIA-V100 GPUs with 80GB memory.  
GLUE tasks are all performed on 1 GPU.
Taking Imagenet-1K as an example, the __Uni-Perceiver-Base__ can be fine-tuned as
```

sh run.sh configs/BERT_L12_H768_experiments/finetuning/in1k_training.yaml in1k-ft 16 partitionname MODEL.WEIGHTS work_dirs/pretrained_models/uni-perceiver-base-L12-H768-224size-pretrained.pth

```
The __Uni-Perceiver-MoE-Base__ can also be fine-tuned in a similar way:
```
sh run.sh configs/BERT_L12_H768_experiments/moe_finetuning/in1k_training.yaml in1k-moe-ft 16 partitionname MODEL.WEIGHTS work_dirs/pretrained_models/uni-perceiver-moe-base-L12-H768-224size-pretrained.pth
```


Note that we used only a few sets of hyperparameters in those task and did not adjust them carefully. Maybe hyper-parameter search can lead to further performance improvement.

