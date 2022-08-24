#!/bin/bash

a=$(echo $HOSTNAME | cut  -c12-16)

CONFIG=$1
JOB_NAME=${2:-"experiments"}
GPUS=${3:-8}
  
partition=${4:-'local'} #

GPUS_PER_NODE=${GPUS:-8}
if [ $GPUS_PER_NODE -ge 8 ]; then
  GPUS_PER_NODE=8
fi
CPUS_PER_TASK=${CPUS_PER_TASK:-4}
SRUN_ARGS=${SRUN_ARGS:-""}

PY_ARGS=${@:5}

WORK_DIR=${CONFIG//configs/work_dirs}
WORK_DIR=${WORK_DIR//.yaml//$JOB_NAME}
echo $WORK_DIR
mkdir  -p $WORK_DIR
mkdir -p data/temp

# please change DATA_PATH where you put the training data
export DATA_PATH='/mnt/lustre/share_data/zhujinguo'

srun --partition=${partition}  $SRUN_ARGS \
--job-name=${JOB_NAME} -n$GPUS  --gres=gpu:${GPUS_PER_NODE} \
--ntasks-per-node=${GPUS_PER_NODE} \
--kill-on-bad-exit=1  --cpus-per-task 12 \
python -u main.py --num-gpus $GPUS \
--config-file ${CONFIG} --init_method slurm --resume \
${PY_ARGS} OUTPUT_DIR $WORK_DIR 

