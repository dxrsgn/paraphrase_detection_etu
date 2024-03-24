etu_paraphrase_identification
==============================

Paraphrase detection on Quora question pair dataset.  
SPbETU university laboratory work 2 on studying recurrent neural nets.

Project Organization
------------

    ├── README.md          
    │
    │
    ├── models             
    │
    ├── notebooks         
    │   ├── 1.0-tta-dataset-stuff.ipynb <- Some dataset analysyis and stuff
    │   ├── 2.0-tta-lstm.ipynb          <- LSTM training
    │   └── 3.0-tta-transformers.ipynb  <- Transformer training
    │
    ├── requirements.txt   
    │                       
    │
    ├── src               
    │   ├── __init__.py    
    │   │
    │   ├── data           
    │   │   └── tsv_to_csv.py <- Converys original tsv dataset to csv
    │   │   └── datasets.py   <- Dataset classes
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── lstm_models.py        <- LSTM models definition
    │   │   ├── transformer_models.py <- Transformer models definition
    │   │   ├── train_model.py        <- Script for model training
            └── eval_model.py         <- Script for model evaluation


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
