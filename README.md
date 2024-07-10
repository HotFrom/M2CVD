# M2CVD

For Devign

CUDA_VISIBLE_DEVICES=1,2,3 python run.py 
--output_dir=./saved_models --model_type=roberta 
--tokenizer_name=microsoft/unixcoder-base
--model_name_or_path=microsoft/unixcoder-base 
--do_train --do_eval --do_test --train_data_file=../dataset/data/train_devign.jsonl 
--eval_data_file=../dataset/data/valid_devign.jsonl 
--test_data_file=../dataset/data/test_devign.jsonl
--epoch 3 --block_size 1024 --train_batch_size 36 --eval_batch_size 64 --learning_rate 2e-5 --max_grad_norm 1.0 --evaluate_during_training --seed 123456

{'Acc': 0.6866764275256223, 'Recall': 0.5689243027888446, 'Precision': 0.6938775510204082, 'F1': 0.6252189141856392}

For rv

CUDA_VISIBLE_DEVICES=0,1,2 python run.py --output_dir=./saved_models  --epoch 5  --block_size 1024  --train_batch_size 6  --eval_batch_size 24 
--learning_rate 2e-5  --model_type=roberta --tokenizer_name=microsoft/unixcoder-base
--model_name_or_path=microsoft/unixcoder-base 
--do_train  --do_eval  --do_test --train_data_file=../dataset/data2/train_data_rv.jsonl --eval_data_file=../dataset/data2/valid_rv.jsonl 
--test_data_file=../dataset/data2/test_rv.jsonl    --max_grad_norm 1.0  --evaluate_during_training  --seed 123456  2>&1 
{'Acc': 0.9230430958663148, 'Recall': 0.2850877192982456, 'Precision': 0.8441558441558441, 'F1': 0.42622950819672123}

