#!/bin/bash
nohup python train_mntp.py --model_name mistralai/Mistral-7B-v0.3 1> /dev/null 2>&1 && nohup python train_mntp.py --model_name meta-llama/Meta-Llama-3-8B-Instruct 1> /dev/null 2>&1 &