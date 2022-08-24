# Pre-Training

## Preparation
You should prepare training data following [PREPARE_DATA.md](prepare_data.md), and make sure that the environment variable `DATA_PATH` is indeed the location path that stores pre-training data.

```
echo $DATA_PATH                                                         
```

## Training on singe node
```
python -m torch.distributed.launch --nproc_per_node=8 --master_port=$PORT \
main.py --num-gpus $GPUS  --config-file ${CONFIG}  OUTPUT_DIR $WORK_DIR 
```

where `$GPUS` is GPU number, `${CONFIG}` is the configuration file, `$PORT` is the specified available port used for distributed training, and `$WORK_DIR` is the directory used to store checkpoint and training log.

For exmaple, the command for pre-training a Uni-Perceiver-Tiny model with `configs/BERT_L12_H192_experiments/7tasks_berttiny_training.yaml` is as folllowing:
```
python -m torch.distributed.launch --nproc_per_node=8 --master_port=26511 \
main.py --num-gpus 8  --config-file configs/BERT_L12_H192_experiments/7tasks_berttiny_training.yaml  OUTPUT_DIR work_dirs/exp_demo_log
```
Another  training example with gradient accumulation :
```
 python -m torch.distributed.launch --nproc_per_node=4 --master_port=26511 \
  main.py --num-gpus 4 --config-file configs/BERT_L12_H384_experiments/in1k_training.yaml  SOLVER.ACCUM_ITER 2 OUTPUT_DIR work_dirs/deepspeed_moe/BERT_L12_H384_experiments/debug
 ```



## Evaluation without any tuning

You can evaluate the pre-training tasks by adding the `--eval-only` argument.
```
python -m torch.distributed.launch --nproc_per_node=8 --master_port=$PORT \
main.py --num-gpus $GPUS  --config-file ${CONFIG} --eval-only OUTPUT_DIR $WORK_DIR  
```

## Training on multiple nodes
For example, the command for training Uni-Perceiver on 2 nodes of each with 8 GPUs is as following:

On node 1：
```
MASTER_ADDR=<IP address of node 1> NODE_RANK=0 GPUS_PER_NODE=8 
python -m torch.distributed.launch --nproc_per_node=8 --master_port=$PORT \
main.py --num-gpus $GPUS  --config-file ${CONFIG}  OUTPUT_DIR $WORK_DIR 
```

On node 2：
```
MASTER_ADDR=<IP address of node 1> NODE_RANK=1 GPUS_PER_NODE=8 
python -m torch.distributed.launch --nproc_per_node=8 --master_port=$PORT \
main.py --num-gpus $GPUS  --config-file ${CONFIG}  OUTPUT_DIR $WORK_DIR 
```

## Training on slurm cluster

If you are using slurm cluster, you can simply run the following command to train Uni-Perceiver on `GPUS/8` nodes with `GPUS` GPUs:

```
sh run.sh ${CONFIG} ${JOBNAME} ${GPUS} ${PARTITION}
```
* Note: you should change the `DATA_PATH` in the script `./run.sh` before your training.


## Pre-Training of Uni-Perceiver models
To save the computation cost, Uni-Perceiver and Uni-Perceiver-MoE are both pre-trained in a two-stage way:
they are pre-trained with the image resolution of 160x160 firstly, and then are pre-trained for another 10% of total iterations on a higher resolution of 224x224.
The two-stage training strategy makes our training more effective.

### Uni-Perceiver
 Take __Uni-Perceiver-Base__ as an example, the 1-st pre-training stage can be conducted as
```
sh run.sh configs/BERT_L12_H768_experiments/16tasks_training_basedense_stage1_64gpu.yaml base_pretrain_stage1 64 partitionname 
```
After the 1-stage, you can run the 2-nd stage pre-training as
```
sh run.sh configs/BERT_L12_H768_experiments/16tasks_training_basedense_stage2_64gpu.yaml base_pretrain_stage2 64 partitionname MODEL.WEIGHTS work_dirs/BERT_L12_H768_experiments/16tasks_training_basedense_stage1_64gpu/base_pretrain_stage1/model_Epoch_200000_Iter_0199999.pth
```

### Uni-Perceiver-MoE
The __Uni-Perceiver-MoE__ model can also be pre-trained in a similar way, which also follows two-stage pre-training.
```
sh run.sh configs/BERT_L12_H768_experiments/16tasks_training_basemoe_stage1_56gpu.yaml base_moe_pretrain_stage1 56 partitionname 
```

```
sh run.sh configs/BERT_L12_H768_experiments/16tasks_training_basemoe_stage2_56gpu.yaml base_moe_pretrain_stage2 56 partitionname MODEL.WEIGHTS work_dirs/BERT_L12_H768_experiments/16tasks_training_basemoe_stage1_56gpu/base_moe_pretrain_stage1/model_Epoch_200000_Iter_0199999.pth

```
By the way, you should adjust the training iteration and learning scheduler accordingly as you use a different number of GPUs.