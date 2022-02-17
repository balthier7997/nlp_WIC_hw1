import os
import nltk
import torch
import numpy as np
from nltk.corpus      import stopwords

from my_embeddings    import PretrainedEmbeddings
from my_dataset       import DatasetLoader, Dataset2Index
from part1_classifier import MyClassifier, train, plot_history

UNK       = '<???>'
VOCABSIZE = 15000
TRAINPATH = "data/train.jsonl"
DEVPATH   = "data/dev.jsonl"
EMB_DIM   = 50
EMBEDPATH = "embeddings/glove.6B.{}d.txt".format(EMB_DIM)
VERBOSE   = False
EPOCHS    = 200
# 'sum', 'sum-L', 'mean', 'mean-square', 'square-mean', 'mean-L', 'mean-cos', 'mean-cos2', 'mean-weighted'
AGG_FUNC  = 'mean-L'
NLTK_DIR  = 'nltk_data'
# 'nope', 'target', 'all'
LEMMATIZE = 'nope'
LR        = 0.2   
DROP_PROB = 0

'''
    # accuracy usando 50d, 3 hidden layer, early stopping 
    # mean        :   63.70%  66.00%  63.70%    -->     64.46% 

    # accuracy usando 50d, 3 hidden layer, dropout=0.2, early stopping
    # mean        :   63.30%  64.30%  64.30%    -->     63.96% 

    # accuracy usando 50d, stopwords, 3 hidden layer, dropout=0.2, early stopping
    # mean        :   65.70%  64.40%  62.10%    -->     64.06% 

    # accuracy usando 50d, stopwords, 3 hidden layer, dropout=0.1, early stopping
    # mean        :   64.50%  64.60%  64.40%    -->     6% 


    # accuracy usando 50d, stopwords, 3 hidden layer, early stopping
    # sum         :   59.60%  64.20%  61.40%    -->     61.73%
    # sum-L       :   64.50%  63.50%  64.40%    -->     64.13%
    # mean        :   65.10%  66.70%  65.70%    -->     65.83%  <-- #1
    # mean-square :   61.40%  61.90%  54.20%    -->     59.10%
    # square-mean :   52.70%  56.30%  55.60%    -->     54.86%
    # mean-L (10) :   65.60%  64.00%  64.10%    -->     64.56%

    # accuracy usando 50d, stopwords, 3 hidden layer, early stopping e lemmatization della target word
    # mean        :   66.20%  63.80%  66.20%    -->     65.40%  <-- #3
    # mean-L (10) :   62.80%  65.20%  63.10%    -->     63.70%

    # accuracy usando 50d, stopwords, 3 hidden layer, early stopping e lemmatization della frase intera
    # mean        :   64.00%  66.20%  65.60%    -->     65.26%
    # mean-L (10) :   63.40%  64.50%  62.60%    -->     63.50%

    # accuracy usando 50d, stopwords, , 3 hidden layer, early stopping e lr = 0.05
    # mean        :   66.00%  64.50%  64.50%    -->     65.00%

    # accuracy usando 50d, stopwords, , 3 hidden layer, early stopping e lr = 0.1
    # mean        :   64.80%  65.20%  64.30%    -->     64.76%    

    # accuracy usando 50d, stopwords, , 3 hidden layer, early stopping e lr = 0.15
    # mean        :   66.00%  65.40%  63.70%    -->     65.03%

    # accuracy usando 50d, stopwords, , 3 hidden layer, early stopping e lr = 0.25
    # mean        :   65.00%  65.00%  64.90%    -->     64.96%

    # accuracy usando 50d, stopwords, , 3 hidden layer, early stopping, lemmatization target e lr = 0.25
    # mean        :   65.90%  64.40%  66.70%    -->     65.66%  <-- #2

    # accuracy usando 50d, stopwords, , 3 hidden layer, early stopping, lemmatization all e lr = 0.25
    # mean        :   64.60%  65.80%  65.60%    -->     65.33%  

    # accuracy usando 50d, stopwords, , 3 hidden layer, early stopping e lr = 0.21
    # mean        :   64.40%  64.80%  63.20%    -->     64.13% 
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
nltk.download('stopwords', download_dir='./nltk_data')
nltk.download('wordnet',   download_dir='./nltk_data')
nltk.data.path.append('./nltk_data')
stopset = set(nltk.corpus.stopwords.words('english'))
#stopset = None

# I load the dataset and translate it to indexes of my vocabulary, taking the first vocab_size most frequent items
# I also check that each lemma is in the vocabulary I'm building, by 'add' or 'replace' the words
# Then, I create an embedding layer loading the pre-trained weights and a dataloader instance for the train set
train_dataset    = DatasetLoader(datapath=TRAINPATH, lemmatize=LEMMATIZE, verbose=VERBOSE)
vocab            = Dataset2Index(train_dataset, vocab_size=VOCABSIZE, unk=UNK, missing_policy='add',  stopwords=stopset, build_onehot=False, verbose=VERBOSE)
embeddingLayer   = PretrainedEmbeddings(vocab, EMBEDPATH, EMB_DIM, VERBOSE, stopset)
train_dataloader = embeddingLayer.aggregate_data(dataset=None, agg_func=AGG_FUNC, batch_size=32)


# For the dev dataset, I only load the sentences without building again a vocabulary (I use the same as before)
# Then, I create the dataloader istance for the dev set
dev_dataset    = DatasetLoader(datapath=DEVPATH, lemmatize=LEMMATIZE, verbose=VERBOSE)
dev_dataloader = embeddingLayer.aggregate_data(dataset=dev_dataset, agg_func=AGG_FUNC, batch_size=32)


#TODO Now the model ...
model = MyClassifier(input_dim=2*EMB_DIM, hidden_dim=EMB_DIM, output_dim=1, dropout_prob=DROP_PROB)
model.to(device)

optim = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.0)

history = train(model, optim, train_dataloader, dev_dataloader, epochs=EPOCHS, early_stop=True, patience=5)

plot_history(history['train_loss_history'], history['dev_loss_history'], metric='Loss',     title=f'Loss with {EMB_DIM} dimensions for embedding')
plot_history(history['train_accu_history'], history['dev_accu_history'], metric='Accuracy', title=f'Accuracy with {EMB_DIM} dimensions for embedding')





# TODO:
# provare a aumentare il numero di parole nel vocabolario - aumenta ma non di molto
# provare a usare un set di stopword -- DONE
# AUMENTARE RETE -- Fatto, la performance migliora, ma migliora in media?
#                   nel senso, mmi è sembrato più variabile il risultato, da controllare TODO
# provare a togliere il lemma

# TODO cambiare optimizer (adam, sgd, ...)
# TODO aggiungere dropout
# TODO cambiare batch size