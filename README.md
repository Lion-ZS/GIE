# GIE 
This is the code of paper “Geometry Interaction Knowledge Graph Embeddings”

To preprocess the datasets, run the following commands.

```shell script
cd code
python process_datasets.py
```

```
WN18RR
CUDA_VISIBLE_DEVICES=3 python3 learn.py --dataset WN18RR --model GIE --rank 2000 --optimizer Adagrad --learning_rate 1e-1 --batch_size 2000 --regularizer N3 --reg 1e-1 --max_epochs 100 --valid 5 -train -id 0 -save -weight

 
FB237
CUDA_VISIBLE_DEVICES=7 python3 learn.py --dataset FB237 --model GIE --rank 800 --optimizer Adagrad --learning_rate 1e-1 --batch_size 2000 --regularizer N3 --reg 5e-2 --max_epochs 100 --valid 5 -train -id 0 -save

YAGO3-10
CUDA_VISIBLE_DEVICES=6 python3 learn.py --dataset YAGO3-10 --model GIE --rank 1000 --optimizer Adagrad \
--learning_rate 1e-1 --batch_size 1000 --regularizer N3 --reg 5e-3 --max_epochs 200 \
--valid 5 -train -id 0 -save
```

