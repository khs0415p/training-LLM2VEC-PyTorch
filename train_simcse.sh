#!/bin/bash

nohup python train_simcse.py --peft_model_name_or_path results/mistral-7b-v0.3-mntp --model_name mistralai/Mistral-7B-v0.3 1> /dev/null 2>&1 &