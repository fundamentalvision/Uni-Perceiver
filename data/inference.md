# Inference

Uni-Perceiver models have excellent generalization ability. Thus you can evalute the pre-trained model for any task and dataset.
For any experiment, inference mode can be activated when  argument `--eval-only` is passed.

For example, you can  conduct zero-shot inference on MSVD caption with a pre-trained checkpoint:
```
sh run.sh configs/BERT_L12_H768_experiments/zeroshot_config/msvd_caption.yaml msvd_cap_infer 8 partitionname --eval-only MODEL.WEIGHTS work_dirs/pretrained_models/uni-perceiver-base-L12-H768-224size-pretrained.pth
```


More inference configs are provided in floder `configs/BERT_L12_H768_experiments/zeroshot_config`.