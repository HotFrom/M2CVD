Code Vulnerability Detection with Devign and RV Datasets
This repository contains code for training and evaluating a vulnerability detection model using the Devign and RV datasets. The model is based on RoBERTa and UnixCoder, leveraging pre-trained models for code defect prediction tasks. This repository provides scripts for training, evaluating, and testing the model, along with the setup and hyperparameter configuration used for the experiments.

Requirements
Python 3.8
PyTorch 1.9 or above
Transformers (HuggingFace)
Datasets and other dependencies can be installed via:


pip install -r requirements.txt
Datasets
Devign: A dataset specifically for vulnerability detection in code.
RV (Revised Vulnerabilities): A dataset with revised annotations for vulnerability detection.
The dataset files must be formatted as JSONL files, with one sample per line. The data format for the training, validation, and test sets follows this structure:

train_devign.jsonl
valid_devign.jsonl
test_devign.jsonl
train_data_new_2.json
valid_data_new_2.jsonl
test_data_new_2.jsonl
Model Training
Devign Dataset
To train and evaluate the model using the Devign dataset, use the following command:


CUDA_VISIBLE_DEVICES=1,2,3 python run.py \
  --output_dir=./saved_models \
  --model_type=roberta \
  --tokenizer_name=microsoft/unixcoder-base \
  --model_name_or_path=microsoft/unixcoder-base \
  --do_train \
  --do_eval \
  --do_test \
  --train_data_file=../dataset/data/train_devign.jsonl \
  --eval_data_file=../dataset/data/valid_devign.jsonl \
  --test_data_file=../dataset/data/test_devign.jsonl \
  --epoch 3 \
  --block_size 1024 \
  --train_batch_size 36 \
  --eval_batch_size 64 \
  --learning_rate 2e-5 \
  --max_grad_norm 1.0 \
  --evaluate_during_training \
  --seed 123456
Evaluation Results for Devign
Accuracy: 68.67%
Recall: 56.89%
Precision: 69.39%
F1 Score: 62.52%


RV Dataset
To train and evaluate the model using the RV (Revised Vulnerabilities) dataset, use the following command:


CUDA_VISIBLE_DEVICES=0,1,5 python run.py \
  --output_dir=./saved_models \
  --epoch 5 \
  --block_size 1024 \
  --train_batch_size 6 \
  --eval_batch_size 24 \
  --learning_rate 3e-5 \
  --model_type=roberta \
  --tokenizer_name=/home/wangziliang/unixcoder \
  --model_name_or_path=/home/wangziliang/unixcoder \
  --do_train \
  --do_eval \
  --do_test \
  --train_data_file=../dataset/rv/train_data_new_2.json \
  --eval_data_file=../dataset/rv/valid_data_new_2.jsonl \
  --test_data_file=../dataset/rv/test_data_new_2.jsonl \
  --max_grad_norm 1.0 \
  --evaluate_during_training \
  --seed 123456
  
Model Configuration and Hyperparameters
Model: RoBERTa with UnixCoder
Tokenizer: microsoft/unixcoder-base for Devign, local path for RV
Epochs: 3 for Devign, 5 for RV
Block Size: 1024
Batch Size:
36 for training, 64 for evaluation on Devign
6 for training, 24 for evaluation on RV
Learning Rate:
2e-5 for Devign
3e-5 for RV
Max Gradient Norm: 1.0
Evaluation During Training: Enabled
Random Seed: 123456
Results
For the Devign dataset, the following evaluation metrics were obtained:

 {'Acc': 0.9155672823218998, 'Recall': 0.39035087719298245, 'Precision': 0.6267605633802817, 'F1': 0.4810810810810811}


For the RV dataset, evaluation results will be printed during training.

Usage
To train or test the model, simply modify the paths to the datasets and adjust hyperparameters according to your needs. The run.py script supports both training and evaluation with the --do_train, --do_eval, and --do_test flags.

License
This project is licensed under the MIT License - see the LICENSE file for details.

