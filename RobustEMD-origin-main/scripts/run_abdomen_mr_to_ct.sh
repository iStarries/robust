#!/bin/bash
# 以腹部 MR 作为源域训练，并在腹部 CT 上评估的整套流程。
# 该脚本会依次跑完 5 折训练与测试，并默认读取 ./checkpoints 下的 COCO 预训练权重。

set -euo pipefail

GPUID=0
export CUDA_VISIBLE_DEVICES="${GPUID}"

###### 通用设置 ######
SOURCE_DATASET='ABDOMEN_MR'
TARGET_DATASET='ABDOMEN_CT'
ALL_FOLDS=(0 1 2 3 4)
ALL_SUPP=(2)                 # CHAOS-CT 默认使用第 2 个 support case，可按需改成 0-4
NWORKER=16
N_PART=3

###### 训练超参数 ######
NSTEP=45000
MAX_ITER=3000               # 定义一个 epoch 的 episode 数
SNAPSHOT_INTERVAL=${NSTEP}  # 仅保存最后一次快照，便于自动匹配
DECAY=0.98
SEED=2021
EXCLUDE_LABEL='[1,6]'
TEST_LABEL='[1,2,3,4]'
USE_GT=False

###### 目录设置 ######
LOGDIR="./exps_train_on_${SOURCE_DATASET}"
RESULT_DIR="./results_${SOURCE_DATASET}_to_${TARGET_DATASET}"
CHECKPOINTS_DIR="./checkpoints"
BACKBONE_WEIGHT="${CHECKPOINTS_DIR}/deeplabv3_resnet50_coco-cd0a2569.pth"

mkdir -p "${LOGDIR}" "${RESULT_DIR}" "${CHECKPOINTS_DIR}"

if [ ! -f "${BACKBONE_WEIGHT}" ]; then
  cat <<EOF
[错误] 未找到 ${BACKBONE_WEIGHT}
        请先下载 torchvision 官方提供的 Deeplabv3-ResNet50 COCO 权重：
        wget -P ${CHECKPOINTS_DIR} https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth
EOF
  exit 1
fi

echo "================================================================"
for FOLD in "${ALL_FOLDS[@]}"; do
  echo "[train] ${SOURCE_DATASET} fold ${FOLD}"
  python3 train.py with \
    mode='train' \
    dataset=${SOURCE_DATASET} \
    num_workers=${NWORKER} \
    n_steps=${NSTEP} \
    eval_fold=${FOLD} \
    test_label=${TEST_LABEL} \
    exclude_label=${EXCLUDE_LABEL} \
    use_gt=${USE_GT} \
    max_iters_per_load=${MAX_ITER} \
    seed=${SEED} \
    save_snapshot_every=${SNAPSHOT_INTERVAL} \
    lr_step_gamma=${DECAY} \
    path.log_dir=${LOGDIR}

  CKPT_DIR="${LOGDIR}/CDFS_train_${SOURCE_DATASET}_cv${FOLD}/1/snapshots"
  if [ ! -d "${CKPT_DIR}" ]; then
    echo "[错误] 未找到快照目录：${CKPT_DIR}" >&2
    exit 1
  fi

  CKPT_PATH=$(ls -1v "${CKPT_DIR}"/*.pth | tail -n 1)
  if [ -z "${CKPT_PATH}" ]; then
    echo "[错误] ${CKPT_DIR} 下没有保存任何 .pth 权重" >&2
    exit 1
  fi

  for SUPP_IDX in "${ALL_SUPP[@]}"; do
    echo "[test] ${TARGET_DATASET} fold ${FOLD} support ${SUPP_IDX}"
    python3 test.py with \
      mode='test' \
      dataset=${TARGET_DATASET} \
      num_workers=${NWORKER} \
      n_steps=${NSTEP} \
      eval_fold=${FOLD} \
      max_iters_per_load=${MAX_ITER} \
      supp_idx=${SUPP_IDX} \
      test_label=${TEST_LABEL} \
      seed=${SEED} \
      n_part=${N_PART} \
      reload_model_path="${CKPT_PATH}" \
      save_snapshot_every=${SNAPSHOT_INTERVAL} \
      lr_step_gamma=${DECAY} \
      path.log_dir=${RESULT_DIR}
  done
done

echo "================================================================"
echo "流程结束，训练日志位于 ${LOGDIR}，评估结果位于 ${RESULT_DIR}"

