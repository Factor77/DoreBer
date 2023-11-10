# DoreBer
DoreBer: **Do**cument-Level **R**elation **E**xtraction Method **B**ased on B**er**nNet


# Requirement
```
python==3.9.16
torch==2.0.1+cu117
json5==0.9.6
jsonschema==3.2.0
networkx==2.8.6
numpy==1.21.2
scikit-learn==1.2.1
scipy==1.10.1
spacy==3.4.1
spacy-legacy==3.0.9
spacy-loggers==1.0.3
torch-geometric==2.3.0
```


# Dataset
For the dataset and pretrained embeddings, please download it [here](https://github.com/thunlp/DocRED/tree/master/data), which are officially provided by [DocRED: A Large-Scale Document-Level Relation Extraction Dataset](https://arxiv.org/abs/1906.06127) .

# Data Preprocessing
After you download the dataset, please put the files train_annotated.json, dev.json and test.json to the ./data directory, and files in pre directory to the code/prepro_data. Run:
```
python gen_data.py 
```
For the BERT encoder:
```
python gen_data_bert.py
```

# Training
In order to train the model, run:
```
python train.py
```
For the BERT encoder, Please set the '--model_name' as 'LSR_bert'


# Test
After the training process, we can test the model by:
```
python test.py
```

# Related Repo

# Citation

