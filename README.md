# BTP-code


# Project Title

A brief description of what this project does and who it's for

### DATASET
```
./implicit-hate-corpus
```

### Collab File
This repository can be cloned and easily used on Google Colab
```
collab-file.ipynb
```

### Commands for running the files
#### Training the model
```
./bert/run_ihd_stg1_training_bert.py  --input_files ./implicit-hate-corpus/implicit_hate_v1_stg1_posts.tsv --output_dir ./stage_bert_1
```
     
#### Evaluating the model
```
./bert/run_ihd_stg1_training_bert.py  --input_files ./implicit-hate-corpus/implicit_hate_v1_stg1_posts.tsv --output_dir ./stage_bert_1
```

Above mentioned is an example for running the training and evaluating the model with default arguments and BERT as a transformer 

For DistillBERT according use the run and eval files from *distill-bert* folder

Also here 1 represents stage 1 accordingly similar commands can be written for stage 2 and can be found in the python notebook
     