import nltk
import torch
import numpy as np

from numpy              import dot
from numpy.linalg       import norm
from torch.nn           import Embedding
from torch.utils.data   import DataLoader
from torch              import zeros, full
from torch.nn.utils.rnn import pad_sequence

try:
    from my_dataset import Index2Embedding
    from torch.cuda import LongTensor, FloatTensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except:
    from torch           import LongTensor, FloatTensor
    from stud.my_dataset import Index2Embedding
    device = torch.device('cpu')

class PretrainedEmbeddings:
    """
        Loads the pre-trained embeddings into an embedding layer.

        Args:
        - vocab     (obj)     : Dataset2Index instance
        - embedpath (str)     : path/to/embed/file
        - emb_dim   (int)     : what file to choose (50, 100, 200, 300)
        - stopset   (set)     : set of stopwords to be removed
        - output_method (str) : whether to take the output at the `last` or `target` index in the lstm
        - verbose   (str)     : whether to display addition info or be silent and quiet
        
        Attributes:
        - layer     (obj) : Embedding instance with the chosen embeddings loaded
    """

    def __init__(self, vocab, embedpath, emb_dim, stopset, output_method, verbose):
        self.vocab     = vocab
        self.embedpath = embedpath
        self.emb_dim   = emb_dim
        self.verbose   = verbose
        self.stopset   = stopset
        self.output_method = output_method
        np.random.seed(1000)                   # needed to get the same results when doing inference

        self.layer = self.load_pretrained_embeddings()
        self.layer.to(device)


    def load_pretrained_embeddings(self):
        # I load the embeddings
        embedding = {}
        with open(self.embedpath) as f:
            for line in f:
                splitted = line.split()
                embedding[splitted[0]] = FloatTensor([float(i) for i in splitted[1:]], device=device)
        weights = zeros(self.vocab.size, self.emb_dim)          # vocab size x embeddings dimensions 
        print(f">> Loaded: {len(embedding)} embeddings from '{self.embedpath}'")
        print(f">> Shape of weights matrix W: {weights.shape}")
        count = 0
        # I save the embedding of all words in my vocabulary
        # if a word doesn't have an embedding, I create a random vector
        for word in self.vocab.word2idx.keys():     
            if word in embedding:
                aux = embedding[word]  
            else:
                aux = FloatTensor(np.random.normal(scale=0.6, size=(self.emb_dim, )), device=device)
                count += 1
            weights[self.vocab.word2idx[word]] = aux
        print(f">> {count} words from my vocab were not present in the embedding (random vector created)")
        embeddingLayer = Embedding.from_pretrained(weights)
        return embeddingLayer


    def words2embedding(self, sntc):
        """
            Translates a single sentence in its embedded form (if a word is not present then it uses the UNK embedding)
            TODO: controlla se bisogna usare UNK oppure non metterla proprio
        """
        result = []
        for word in sntc:
            if word in self.vocab.word2idx.keys():
                aux = self.vocab.word2idx[word]
            else:
                aux = self.vocab.word2idx[self.vocab.unk]
            result.append(aux) 
        return self.layer(LongTensor(result, device=device))   


    ## aggregation methods for PART I

    def aggregate_mean(self, couple):
        """
            Baseline aggregation computing the means of the sentences
            Returns two vectors of shape (EMB_DIM, )
        """
        return np.nanmean(self.words2embedding(couple[0]).cpu(), axis=0), np.nanmean(self.words2embedding(couple[1]).cpu(), axis=0)

    def aggregate_sum(self, couple):
        return np.nansum(self.words2embedding(couple[0]).cpu(), axis=0), np.nansum(self.words2embedding(couple[1]).cpu(), axis=0)

    def aggregate_square_mean(self, couple):
        s0 = self.words2embedding(couple[0]).cpu()
        s1 = self.words2embedding(couple[1]).cpu()
        for i in range(len(s0)):
            s0[i] = np.square(abs(s0[i]))
        for i in range(len(s1)):
            s1[i] = np.square(abs(s1[i]))
        return np.nanmean(s0, axis=0), np.nanmean(s1, axis=0)

    def aggregate_mean_square(self, couple):
        s0, s1 = self.aggregate_mean(couple)
        return np.square(s0), np.square(s1)

    def aggregate_mean_L(self, couple, lemma):
        """
            Aggregates by computing the means and then subtracting the lemma's embedding
            Returns two vectors of shape (EMB_DIM, )
        """
        s0,s1 = self.aggregate_mean(couple)
        lemma = self.words2embedding([lemma]).cpu()
        return np.subtract(12*s0, lemma), np.subtract(12*s1, lemma)

    def aggregate_sum_L(self, couple, lemma):
        s0,s1 = self.aggregate_sum(couple)
        lemma = self.words2embedding([lemma]).cpu()
        return np.subtract(s0, lemma), np.subtract(s1, lemma)

    def aggregate_mean_cos(self, couple, lemma): 
        s0, s1   = self.aggregate_mean(couple)
        lemma    = self.words2embedding([lemma]).cpu().numpy().reshape((50, ))
        cos_sim0 = 1 - dot(s0, lemma)/(norm(s0)*norm(lemma))
        cos_sim1 = 1 - dot(s1, lemma)/(norm(s1)*norm(lemma))
        return cos_sim0*s0, cos_sim1*s1

    def aggregate_mean_cos2(self, couple, lemma):
        s0 = []
        s1 = []
        lemma = self.words2embedding([  lemma]).cpu().numpy().reshape((50, ))
        sntc0 = self.words2embedding(couple[0]).cpu().numpy()
        sntc1 = self.words2embedding(couple[1]).cpu().numpy()
        for vec in sntc0:
            cos_sim = dot(vec, lemma)/(norm(vec)*norm(vec))
            s0.append(cos_sim*vec)
        for word in sntc1:
            cos_sim = dot(vec, lemma)/(norm(vec)*norm(vec))
            s1.append(cos_sim*vec)
        return np.mean(s0, axis=0), np.mean(s1, axis=0)

    def __aggregate_mean_weighted(self, sentence, lemma):
        pos_lemma = 0
        for i in range(len(sentence)):
            if sentence[i] == lemma:
                pos_lemma = i
                break
        s = []
        lemma = self.words2embedding([ lemma]).cpu().numpy().reshape((50, ))
        sntc  = self.words2embedding(sentence).cpu().numpy()
        k = 1.5/len(sntc)
        
        for i in range(len(sntc)):
            weight = max(0.5, k * abs(i - pos_lemma))
            s.append(weight*sntc[i])
        assert len(s) == len(sentence)
        return np.nanmean(s, axis=0)
    
    def aggregate_mean_weighted(self, couple, lemma):
        '''
            pesa le parole a seconda della distanza dalla target (usare con lemmatization = 'target')
        '''
        return self.__aggregate_mean_weighted(couple[0], lemma), self.__aggregate_mean_weighted(couple[1], lemma)

    def aggregate_data(self, dataset=None, agg_func='means', batch_size=32):
        """
        Reads the dataset stored in self.vocab and returns it in a dataloader if dataset is None
        Else, you shall provide a DatasetLoader instance to the function (useful when using the dev set)
        """
        if dataset is None:
            print("Aggregating data in vocab ...")
            data   = self.vocab.data
            lemmas = self.vocab.lemmas
            labels = self.vocab.labels
        else:
            print("Aggregating data in dev dataset ...")
            data   = dataset.data
            lemmas = dataset.lemmas
            labels = dataset.labels
        assert len(data) == len(labels) == len(lemmas)
        print("data shape:", len(data), len(data[0]))

        aggregated_data = []
        for couple, lemma, label in zip(data, lemmas, labels):
            if agg_func == 'sum':
                s0, s1 = self.aggregate_sum(couple)
            elif agg_func == 'sum-L':
                s0, s1 = self.aggregate_sum_L(couple, lemma)
            elif agg_func == 'mean':
                s0, s1 = self.aggregate_mean(couple)
            elif agg_func == 'square-mean':
                s0, s1 = self.aggregate_square_mean(couple)
            elif agg_func == 'mean-square':                                 # provare a aumentare il numero di parole nel vocabolario
                s0, s1 = self.aggregate_mean_square(couple)
            elif agg_func == 'mean-L':
                s0, s1 = self.aggregate_mean_L(couple, lemma)
            elif agg_func == 'mean-cos':
                s0, s1 = self.aggregate_mean_cos(couple, lemma)
            elif agg_func == 'mean-cos2':
                s0, s1 = self.aggregate_mean_cos2(couple, lemma)
            elif agg_func == 'mean-weighted':
                s0, s1 = self.aggregate_mean_weighted(couple, lemma)
            else:
                print("\t\t[ERRO] >> `agg_func` not recognized.")
                exit(1)
            #s0 = np.append(s0, 0)   #separator
            s = FloatTensor(np.append(s0, s1), device=device)
            aggregated_data.append([s, 1 if label == 'True' else 0])
        print("aggregated_data shape:", len(aggregated_data), aggregated_data[0][0].shape, len(aggregated_data[1]))
        embedded_data  = Index2Embedding(aggregated_data)
        return DataLoader(embedded_data, batch_size=batch_size)


    ## aggregation methods for PART II


    def collate_fn(self, data):
        x0 = []
        l0 = []
        x1 = []
        l1 = []
        ys = []
        for elem in data:                   # list of (x0, x1, y) pairs where x is the INDEXED sentence
            x0.append(elem[0])              # first sentence
            x1.append(elem[1])              # second sentence
            ys.append(elem[4])              # labels of the sentences
            
            if self.output_method == 'last':
                l0.append(elem[0].size(0)-1)    # index of the last word of the first sentences
                l1.append(elem[1].size(0)-1)    # index of the last word of the second sentence
            
            elif self.output_method == 'target':
                l0.append(elem[2])    # index of the target word (lemma) in the first sentence
                l1.append(elem[3])    # index of the target word (lemma) in the first sentence
            
            else:
                print(">> Error: output method 404!")
                exit(1)
        
        assert len(x0) == len(l0) == len(x1) == len(l1) == len(ys)
        
        x0 = pad_sequence(x0, batch_first=True, padding_value=0).to(device)      # x has shape (batch_size x max_sentence_lenght)
        x1 = pad_sequence(x1, batch_first=True, padding_value=0).to(device)      # x has shape (batch_size x max_sentence_lenght)
        l0 = LongTensor(l0, device=device)
        l1 = LongTensor(l1, device=device)
        ys = LongTensor(ys, device=device)
        return x0, l0, x1, l1, ys


    def prepare_rnn_data(self, dataset=None, batch_size=32, method='baseline'):
        if dataset is None:
            print("Aggregating data in vocab ...")
            data       = self.vocab.data
            lemmas     = self.vocab.lemmas
            labels     = self.vocab.labels
            lemmas_idx = self.vocab.lemmas_idx
        else:
            print("Aggregating data in dev dataset ...")
            data       = dataset.data
            lemmas     = dataset.lemmas
            labels     = dataset.labels
            lemmas_idx = dataset.lemmas_idx

        print("\ndata shape: dataset_lenght x sentence_pair x sequence_lenght")
        print(f">> {len(data)} x {len(data[0])} x {len(data[0][0])}")
        aggregated_data = []
        for couple, lemma, indexes, label in zip(data, lemmas, lemmas_idx, labels):
            
            if method == 'baseline':
                s0 = self.words2embedding(couple[0])                                # seq1_len x embed_dim
                s1 = self.words2embedding(couple[1])                                # seq2_len x embed_dim
            
            elif method == 'lemma2end':
                s0 = self.move_lemma2end(couple[0], lemma)
                s1 = self.move_lemma2end(couple[1], lemma)
            
            elif method == 'rm-lemma':
                s0 = self.rm_lemma(couple[0], lemma)
                s1 = self.rm_lemma(couple[1], lemma)
            
            else:
                print("[404] >> METHOD not found!")
                exit(1)
            
            aggregated_data.append((s0, s1, indexes[0], indexes[1], 1 if label == 'True' else 0))
        print("\nagg data shape: dataset_lenght x sentence_pair x embed_dim x labels_dim")
        print(f">> {len(aggregated_data)} x {aggregated_data[0][0].shape} x {aggregated_data[0][1].shape} x 1\n")
        print(f">> Using {method} method")
        embedded_data  = Index2Embedding(aggregated_data)
        return DataLoader(embedded_data, batch_size=batch_size, collate_fn=self.collate_fn)


    def move_lemma2end(self, sentence, lemma):
        """
            Moves the lemma of a sentence in the end
            NOTE: must be used in combination with `LEMMATIZE = 'target'`
        """
        assert lemma in sentence
        aux = []
        for i in range(len(sentence)):
            if sentence[i] != lemma:
                aux.append(sentence[i])
        aux.append(lemma)
        return self.words2embedding(aux)
    
    def rm_lemma(self, sentence, lemma):
        """
            Removes the lemma from the sentence
            NOTE: must be used in combination with `LEMMATIZE = 'target'`
        """
        assert lemma in sentence
        aux = []
        for i in range(len(sentence)):
            if sentence[i] != lemma:
                aux.append(sentence[i])
        return self.words2embedding(aux)