import os
import nltk
import torch
import numpy as np
from nltk.corpus import stopwords

from my_embeddings    import PretrainedEmbeddings
from my_dataset       import DatasetLoader, Dataset2Index
from part2_classifier import LSTMClassifier, train_rnn, plot_history

UNK       = '<???>'
VOCABSIZE = 25000
TRAINPATH = "data/train.jsonl"
DEVPATH   = "data/dev.jsonl"
EMB_DIM   = 50
EMBEDPATH = "model/embeddings/glove.6B.{}d.txt".format(EMB_DIM)
VERBOSE   = False
EPOCHS    = 100
NLTK_DIR  = 'nltk_data'
DROPOUT   = 0
LR        = 0.001
STOPWORDS = False
PATIENCE  = 5
DOUBLE    = False
LEMMATIZE = 'target'                        # 'nope'  'target'  'all'   
METHOD    = 'baseline'                      # 'baseline'  'lemma2end'  'rm-lemma'
LSTM_AGGREGATION      = 'avg-all'           # 'avg'  'avg-all'  'avg-avg-all'   'concatente'
SENTENCE_AGGREGATION  = 'abs-subtract'      # 'concatenate'  'subtract'  'abs-subtract'
OUTPUT    = 'last'                          # 'last'  'target'
BIDIRECTIONAL  = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nltk.data.path.append('./model/nltk_data')

if LEMMATIZE == 'all':
    nltk.download('wordnet',   download_dir='./nltk_data')                  # requires an internet connection (only once)

if STOPWORDS:
    nltk.download('stopwords', download_dir='./nltk_data')                  # requires an internet connection (only once)
    stopset = set(nltk.corpus.stopwords.words('english'))
else:
    stopset = None


# I load the dataset (taking care of lemmatization and storing the needed information)
train_dataset    = DatasetLoader(datapath=TRAINPATH,
                                lemmatize=LEMMATIZE,
                                verbose=VERBOSE)

# and translate it to indexes of my vocabulary,
# taking the first vocab_size most frequent items
# I also check that each lemma is in the vocabulary I'm building
# by 'adding' or 'replacing' the words
vocab = Dataset2Index(train_dataset,
                    vocab_size=VOCABSIZE,
                    unk=UNK,
                    missing_policy='add',
                    stopwords=stopset,
                    build_onehot=False,
                    verbose=VERBOSE)

# Then, I create an embedding layer loading the pre-trained weights
embeddingLayer = PretrainedEmbeddings(vocab=vocab,
                                    embedpath=EMBEDPATH, 
                                    emb_dim=EMB_DIM,
                                    stopset=stopset,
                                    output_method=OUTPUT,
                                    verbose=VERBOSE)

# and a dataloader instance for the train set 
# (this only exposes __len__ and __getitem__)
train_dataloader = embeddingLayer.prepare_rnn_data(dataset=None, method=METHOD)


# then I repeat the same step for the validation dataset
# without building again the vocabulary
dev_dataset = DatasetLoader(datapath=DEVPATH,
                            lemmatize=LEMMATIZE,
                            verbose=VERBOSE)

dev_dataloader = embeddingLayer.prepare_rnn_data(dataset=dev_dataset, method=METHOD)

# Now I istanciate my classifier
model = LSTMClassifier( rnn_input_dim  = EMB_DIM,
                        drop_prob=DROPOUT,
                        lstm_aggregation=LSTM_AGGREGATION,
                        sentence_aggregation=SENTENCE_AGGREGATION,
                        double=DOUBLE,
                        bidirectional=BIDIRECTIONAL).to(device)

# and train it
optim = torch.optim.Adam(model.parameters(), lr=LR)#, weight_decay=1e-3)
history = train_rnn(model=model, 
                    optimizer=optim, 
                    train_data=train_dataloader,
                    dev_data=dev_dataloader,
                    epochs=EPOCHS,
                    early_stop=True,
                    patience=PATIENCE)


### stats ###

plot_history(history['train_loss_history'], history['dev_loss_history'], metric='Loss',     title=f'Loss with {EMB_DIM} dimensions for embedding')
plot_history(history['train_accu_history'], history['dev_accu_history'], metric='Accuracy', title=f'Accuracy with {EMB_DIM} dimensions for embedding')

print()
print(f">> Vocabulary size:       {           VOCABSIZE}")
print(f">> Method:                {              METHOD}")
print(f">> Removing stopwords:    {           STOPWORDS}")
print(f">> Lemmatization:         {           LEMMATIZE}")
print(f">> Sentence aggregation:  {SENTENCE_AGGREGATION}")
print(f">> LSTM aggregation:      {    LSTM_AGGREGATION}")
print(f">> LSTM output index:     {              OUTPUT}")
print(f">> Double LSTM:           {              DOUBLE}")
print(f">> Bidirectional LSTM:    {       BIDIRECTIONAL}")
print(f">> Dropout:               {             DROPOUT}")
print(f">> Learning Rate:         {                  LR}")

model.print_summary()