#!/bin/sh

DATASET_PATH=../DATASET_CTA

export PYTHONPATH=.././
export RESULTS_FOLDER=../output_CTA_MISSFormer
export unetr_pp_preprocessed="$DATASET_PATH"/unetr_pp_raw/unetr_pp_raw_data/Task501_CTA
export unetr_pp_raw_data_base="$DATASET_PATH"/unetr_pp_raw

python ../unetr_pp/run/run_training.py 2d unetr_pp_trainer_CTA 501 0
