""" The main file implementing training and inference. """
import torch
import pandas as pd
from scr.NLP_processing import CreatDataset
import scr.NLP_processing as nlp
from scr.model import DistilBertForSentiment, TransformerTrainer
from scr.model import inference

""" Global Variables. """
fixed_seed = 42
# Special marks to be preserved in NLP.
special_mark_list = ['<e>', '</e>'] 
# Train or use trained model to predict directly.
do_training = False
# If train from  (model parameters has been tuned for this task).
train_on_tuned_model = False
# Path to load or save the model.
model_load_path = "./para_store/DistilBert10e.pth"
model_save_path = "./para_store/DistilBert_new.pth"
# Model training hyper-parameters.
num_epochs=1
batch_size=32
learning_rate=2e-5
max_grad_norm=1.0
patience=2
restore=True
# The number of classes. 
num_labels = 3
# Load data.
data_df = pd.read_csv('./datasets/twitter_training.csv')

""" Environment Setup. """
# Define GPU as device if possible and apply fixed seed to all processes.
nlp.set_seed(fixed_seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Print the used device and torch version.
print("Using device:", device)
print("PyTorch version:", torch.__version__)

def main():    
    """ Data Processing. """
    # Combine entities and contents in wnated format.
    data = nlp.data_preprocess(data_df)
    tokenizer = nlp.tokenize(special_mark_list)
    dataset = CreatDataset(data, tokenizer, max_len=128)
    train_dataset, val_dataset, test_dataset = nlp.split_data(dataset,0.1,0.1)

    """ Training. """  
    if do_training:
        model = DistilBertForSentiment(tokenizer, num_labels)
        if train_on_tuned_model:
            model.load_state_dict(torch.load(model_load_path))

        trainer = TransformerTrainer(model, train_dataset, val_dataset, device,
                                    num_epochs, batch_size, learning_rate,
                                    max_grad_norm, patience, 
                                    restore, model_save_path)
        trained_distilBert = trainer.train()
    else:
        trained_distilBert = DistilBertForSentiment(tokenizer, num_labels)
        trained_distilBert.load_state_dict(torch.load(model_load_path))

    """ Inference. """
    probabilities, _, predictions_text = inference(
    trained_distilBert, device, test_dataset, batch_size)

    print(f'The head of the probability list: {probabilities[:10]}')
    print(f'The head of the sentiment list: {predictions_text[:10]}')  

if __name__ == "__main__":
    main()  

