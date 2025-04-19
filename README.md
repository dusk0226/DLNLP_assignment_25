
# DistilBert for sentiment detection
This project fine-tuned the DistilBert Model from [Hugging Face](https://huggingface.co/transformers/v3.0.2/model_doc/distilbert.html#distilberttokenizerfast) Transformers [library](https://github.com/huggingface/transformers) for sentiment detection. The tuned and trained model (parameters) is uploaded to Google drive and shared publicly [here](https://drive.google.com/file/d/1jpYqb6BR0DRSYcWU5_s4mA2cKRhcAo0P/view?usp=sharing). If you run code in this repository locally with 'main.py', it will be downloaded so you do not need to download it manually. 

The database used for training, validation, and test is [Twitter Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis/data) database. This database contains twitter (Now X) messages regarding to multiple topics and shows message entities. The sentiments are labeled in three classes - positive, negative, and neutral, where messages unrelated to their topices are considered neutral. If you run code in this repository locally, download the dataset 'twitter_training.csv' from [kaggle](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis/data?select=twitter_training.csv) and unzip it to the file 'datasets'. 







 


## Run Locally

Clone the project

```bash
  git clone https://github.com/dusk0226/DLNLP_assignment_25.git
```

Go to the project directory

```bash
  cd DLNLP_assignment_25
```

Install dependencies

```bash
  pip install -r env/requirements.txt
```
Run the main file to do training and inference

```bash
  python main.py
```

The file 'main.py' defaultly downloads and uses the trained model to do inference on the test dataset. You can change the global variables in 'main.py' to enable training from beginning ([original model](https://huggingface.co/transformers/v3.0.2/index.html) from hugging face transformers, hyper-parameters tuned but with original model pre-trained parameters) or on the downloaded model.

The defaultly downloaded model achieves approximately 87.7% on the test dataset. It is trained with the default global variable values in 'main.py'.

