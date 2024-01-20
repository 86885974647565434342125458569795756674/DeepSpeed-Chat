#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
ACTOR_MODEL_PATH=$1
CRITIC_MODEL_PATH=$2
ACTOR_ZERO_STAGE=$3
CRITIC_ZERO_STAGE=$4
OUTPUT=$5
T_W=$6
MAIN=$7
GEN_BS=$8
GEN_B=$9

if [ "$GEN_B" == "" ]; then
   GEN_B=1
fi

if [ "$GEN_BS" == "" ]; then
   GEN_BS=4
fi

if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output_1.3b-8
fi
if [ "$ACTOR_ZERO_STAGE" == "" ]; then
    ACTOR_ZERO_STAGE=2
fi
if [ "$CRITIC_ZERO_STAGE" == "" ]; then
    CRITIC_ZERO_STAGE=2
fi

# if actor and critic model names are not provided, then use the publicly available AdamG012/chat-opt-1.3b-sft-deepspeed and AdamG012/chat-opt-350m-reward-deepspeed
if [ "$ACTOR_MODEL_PATH" == "" ]; then
    ACTOR_MODEL_PATH=AdamG012/chat-opt-1.3b-sft-deepspeed
fi
if [ "$CRITIC_MODEL_PATH" == "" ]; then
    CRITIC_MODEL_PATH=AdamG012/chat-opt-350m-reward-deepspeed
fi

mkdir -p $OUTPUT
mkdir -p $T_W

Num_Padding_at_Beginning=1 # this is model related

Actor_Lr=9.65e-6
Critic_Lr=5e-6

deepspeed --master_port 12346 $MAIN \
   --actor_model_name_or_path $ACTOR_MODEL_PATH \
   --critic_model_name_or_path $CRITIC_MODEL_PATH \
   --per_device_generation_batch_size $GEN_BS \
   --per_device_training_batch_size 1 \
   --actor_learning_rate ${Actor_Lr} \
   --critic_learning_rate ${Critic_Lr} \
   --actor_dropout 0.0 \
   --deepspeed --seed 1234 \
   --enable_hybrid_engine \
   --actor_zero_stage $ACTOR_ZERO_STAGE \
   --critic_zero_stage $CRITIC_ZERO_STAGE \
   --output_dir $OUTPUT \
   --generation_batches $GEN_B \
   --enable_tensorboard \
   --tensorboard_path $OUTPUT \
   --num_train_epochs 10 \
   --tensorboard_web $T_W \
    &> $OUTPUT/training.log
