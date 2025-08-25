#!/bin/bash
#SBATCH --job-name=RAVEN_affine_arithmetic_training
#SBATCH --qos=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=rtx4090_batch

source scripts/main.sh

run_sweep_and_agent "scripts/training/mnist_dm_medium_eps_train_04_test_04/affine_arithmetic"