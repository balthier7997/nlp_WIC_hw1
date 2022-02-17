import numpy as np
from typing import List, Tuple, Dict

from model import Model

################################
import os
import re
import nltk
import torch
from nltk.corpus        import stopwords
from torch.utils.data   import DataLoader
from torch              import LongTensor, FloatTensor
from torch.nn.utils.rnn import pad_sequence

#from torch.cuda       import LongTensor, FloatTensor

from stud.my_embeddings    import PretrainedEmbeddings
from stud.my_dataset       import DatasetLoader, Dataset2Index
from stud.part2_classifier import LSTMClassifier, train_rnn, plot_history

UNK       = '<???>'
VOCABSIZE = 25000
TRAINPATH = "model/data/train.jsonl"
DEVPATH   = "data/dev.jsonl"
EMB_DIM   = 50
EMBEDPATH = "model/embeddings/glove.6B.{}d.txt".format(EMB_DIM)
VERBOSE   = False
EPOCHS    = 20
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

PATH = "model/part2/try1_part2_4epoch_50d_69.40accu.pt"
#PATH = "model/part2/try1_part2_5epoch_50d_68.70accu.pt"

################################



#TODO Implement the build_model function, initializing your StudentModel class
def build_model(device: str) -> Model:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
    return StudentModel(device)
    #return RandomBaseline()


class RandomBaseline(Model):

    options = [
        ('True', 40000),
        ('False', 40000),
    ]

    def __init__(self):

        self._options = [option[0] for option in self.options]
        self._weights = np.array([option[1] for option in self.options])
        self._weights = self._weights / self._weights.sum()

    def predict(self, sentence_pairs: List[Dict]) -> List[Dict]:
        return [str(np.random.choice(self._options, 1, p=self._weights)[0]) for x in sentence_pairs]



class Index2Embedding():
    """
    Dummy class to expose __len__ and __getitem__ methods for the DataLoader
    """
    def __init__(self, data):
        # data = [ (aggregated_sentences_1, label_1), (aggregated_sentences_2, label_2), ..., (aggregated_sentences_N, label_N) ]
        self.data = data
    
    def __len__(self):
        # returns the number of couples in data
      return len(self.data)

    def __getitem__(self, idx):
        # returns the idx-th couple in data
        return self.data[idx]



class StudentModel(Model):
    
    '''
    - Load your model and use it in the predict method
    - You must respect the signature of the predict method
    - You can add other methods (i.e. the constructor)
    '''

    def __init__(self, device):
        self.device = device
        self.vocab = None
        self.embeddingLayer = None        
        self.model = self.load_student_model()
        self.model.eval()

    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary
    def load_student_model(self):
        nltk.data.path.append('./model/nltk_data')

        if LEMMATIZE == 'all':
            nltk.download('wordnet',   download_dir='./nltk_data')                  # requires an internet connection (only once)
        if STOPWORDS:
            nltk.download('stopwords', download_dir='./nltk_data')                  # requires an internet connection (only once)
            stopset = set(nltk.corpus.stopwords.words('english'))
        else:
            stopset = None


        train_dataset = DatasetLoader(datapath=TRAINPATH,
                                    lemmatize=LEMMATIZE,
                                    verbose=VERBOSE)

        self.vocab = Dataset2Index(train_dataset,
                            vocab_size=VOCABSIZE,
                            unk=UNK,
                            missing_policy='add',
                            stopwords=stopset,
                            build_onehot=False,
                            verbose=VERBOSE)

        self.embeddingLayer = PretrainedEmbeddings(vocab=self.vocab,
                                            embedpath=EMBEDPATH, 
                                            emb_dim=EMB_DIM,
                                            stopset=stopset,
                                            output_method=OUTPUT,
                                            verbose=VERBOSE)
        # Now I istanciate my classifier
        model = LSTMClassifier(rnn_input_dim  = EMB_DIM,
                            drop_prob=DROPOUT,
                            lstm_aggregation=LSTM_AGGREGATION,
                            sentence_aggregation=SENTENCE_AGGREGATION,
                            double=DOUBLE,
                            bidirectional=BIDIRECTIONAL).to(self.device)

        model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
        return model


    def lemmatize_target(self, sentence, start, end, lemma):
        """returns the sentence (str) with the lemma in place of the target word at sentence[start:end]"""
        return sentence[:start] + lemma.lower() + sentence[end:]


    def tokenize_line(self, line, pattern='\W'):
        '''tokenizes a line, returns a list of words'''
        # I insert a space otherwise words such as 'xx-yyyy' becames 'xxyyyy'
        # which, in most cases, isn't proper english and appears only once or twice
        # some are errors or missplelling btw
        line = re.sub('[\.,:;!@#$\(\)\-&\\<>/0-9]', ' ', line)
        return [word.lower() for word in re.split(pattern, line.lower()) if word]


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
        return self.embeddingLayer.layer(LongTensor(result, device=self.device))


    def collate_fn(self, data):
        x0 = []
        l0 = []
        x1 = []
        l1 = []
        for elem in data:                   # list of (x0, x1, y) pairs where x is the INDEXED sentence
            x0.append(elem[0])              # first sentence
            x1.append(elem[1])              # second sentence            
            l0.append(elem[0].size(0)-1)    # index of the last word of the first sentences
            l1.append(elem[1].size(0)-1)    # index of the last word of the second sentence

        assert len(x0) == len(l0) == len(x1) == len(l1)
        
        x0 = pad_sequence(x0, batch_first=True, padding_value=0).to(self.device)      # x has shape (batch_size x max_sentence_lenght)
        x1 = pad_sequence(x1, batch_first=True, padding_value=0).to(self.device)      # x has shape (batch_size x max_sentence_lenght)
        l0 = LongTensor(l0, device=self.device)
        l1 = LongTensor(l1, device=self.device)
        return x0, l0, x1, l1


    @torch.no_grad()
    def predict(self, sentence_pairs: List[Dict]) -> List[str]:
        # STUDENT: implement here your predict function
        # remember to respect the same order of sentences!        

        data = []
        for elem in sentence_pairs:
            sntc1 = self.lemmatize_target(elem['sentence1'], int(elem['start1']), int(elem['end1']), elem['lemma'])
            sntc2 = self.lemmatize_target(elem['sentence2'], int(elem['start2']), int(elem['end2']), elem['lemma'])
            sntc1 = self.words2embedding(self.tokenize_line(sntc1))
            sntc2 = self.words2embedding(self.tokenize_line(sntc2))
            data.append([sntc1, sntc2])
        data = Index2Embedding(data)
        test_dataloader = DataLoader(data, collate_fn=self.collate_fn)

        #with torch.no_grad():
        y_pred = []
        for x0, l0, x1, l1 in test_dataloader:
            out = self.model(x0, l0, x1, l1, y=None, device=self.device)
            y_pred.extend(['True' if round(y.cpu().item()) == 1 else 'False' for y in out['probabilities']])
        return y_pred


