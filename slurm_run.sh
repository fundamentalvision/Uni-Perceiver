#!/bin/bash

a=$(echo $HOSTNAME | cut  -c12-16)

CONFIG=$1
JOB_NAME=${2:-"experiments"}
GPUS=${3:-8}
  
SRUN=${4:-'reserved'} 

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

now=$(date +"%Y%m%d_%H%M%S")

a=$(echo $HOSTNAME | cut  -c12-16)


if [ $a == '140-0' ]; then
  export DATA_PATH='/mnt/lustre/share_data/zhujinguo'
  export LD_LIBRARY_PATH=/mnt/cache/zhujinguo/anaconda3/envs/py36/lib:$LD_LIBRARY_PATH
  export TORCH_EXTENSIONS_DIR='/mnt/lustre/zhujinguo/.cache/torch_extensions'
  export NO_NVRTC=0
  partition='INTERN'
  CEPH_CONFIG='slurm_tools/petreloss_1400.config'
  SRUNreal=${SRUN}

  if [  ${SRUN} == 'vcspot' ]; then
  SRUNreal='spot --async'
  partition=VC
  elif [  ${SRUN} == 'vcauto' ]; then
    SRUNreal='auto --async'
    partition=VC
  elif [  ${SRUN} == 'vcreserved' ]; then
    SRUNreal='reserved'
    partition=VC
  elif [  ${SRUN} == 'spot' ]; then
    SRUNreal='spot --async'
  elif [  ${SRUN} == 'auto' ]; then
    SRUNreal='auto --async'

  fi

elif [ $a == '142-4' ]; then
    # 1424
  export DATA_PATH='/mnt/lustre/share_data/zhujinguo'
  export LD_LIBRARY_PATH=/mnt/cache/zhujinguo/anaconda3/envs/py36/lib:$LD_LIBRARY_PATH
  export TORCH_EXTENSIONS_DIR='/mnt/lustre/zhujinguo/.cache/torch_extensions'
  export NO_NVRTC=0
  partition='vc_research_5'
  CEPH_CONFIG='slurm_tools/petreloss_1424.config'

  SRUNreal=${SRUN}

  if [  ${SRUN} == 'vc4spot' ]; then
  SRUNreal='spot --async'
  partition=vc_research_4
  elif [  ${SRUN} == 'vc4auto' ]; then
    SRUNreal='auto --async -x SH-IDC1-10-142-4-76'
    partition=vc_research_4
  elif [  ${SRUN} == 'vc4reserved' ]; then
    SRUNreal='reserved'
    partition=vc_research_4
  elif [  ${SRUN} == 'spot' ]; then
    SRUNreal='spot --async'
  elif [  ${SRUN} == 'auto' ]; then
    SRUNreal='auto --async'
  fi

else
  echo only SH1424 and SH1400 supported now 

fi

srun --partition=${partition}  $SRUN_ARGS --quotatype=${SRUNreal} -o $WORK_DIR/phoenix-slurm-%j-$now.out \
--job-name=${JOB_NAME} -n$GPUS  --gres=gpu:${GPUS_PER_NODE} \
--ntasks-per-node=${GPUS_PER_NODE} \
--kill-on-bad-exit=1  --cpus-per-task 12 \
python -u main.py --num-gpus $GPUS \
--config-file ${CONFIG} --init_method slurm --resume \
${PY_ARGS} OUTPUT_DIR $WORK_DIR DATALOADER.USE_CEPH True \
DATALOADER.TCS_CONF_PATH $CEPH_CONFIG  SOLVER.CHECKPOINT_PERIOD 10000 SOLVER.CHECKPOINT_MAX_SAVE 1 \
${OTHERARGS} 2>&1

# SOLVER.ACCUM_ITER 2 SOLVER.CHECKPOINT_PERIOD 1000 SOLVER.CHECKPOINT_MAX_SAVE 1 MODEL.BERT.DROP_PATH_PROB 0.1

