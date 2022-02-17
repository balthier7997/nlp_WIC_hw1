import re
import jsonlines
import collections

from torch.utils.data import Dataset
from nltk.stem        import WordNetLemmatizer

# PoS string for lemmatization in nltk (from nltk.corpus.reader.wordnet)
# ADJ, ADJ_SAT, ADV, NOUN, VERB = "a", "s", "r", "n", "v"

pos_map = { 'ADJ'   :   'a',
            'ADV'   :   'r',
            'NOUN'  :   'n',
            'VERB'  :   'v'  }


class DatasetLoader:

    def __init__(self, datapath, lemmatize='nope', verbose=False):
        """
            Reads the dataset and stores the needed stuff.
            Args:
            - datapath       (str)  : path to dataset in .jsonl format (include the format!)
            - lemmatize      (str)  : whether to lemmatize the target word `target`, everything `all` or nothing `nope`
            - verbose        (bool) : whether to display additional info or not
            
            Attributes:
            - data           (list) : list of processed data containing tuples (sentence pair)
            - labels         (list) : list of labels  
            - lemmas         (list) : list of lemmas
            - lemmas_pos     (list) : list of PoS tag for the lemmas
            - lemmas_idx     (list) : list of tuple containing the indexes of the target word in the sentence
            - sentence_pairs (list) : list of raw sentence pairs
        """
        self.verbose    = verbose
        self.datapath   = datapath
        self.lemmatize  = lemmatize
        self.lemmatizer = WordNetLemmatizer()   if lemmatize == 'all' else None
        
        # populated in `load_dataset`
        self.data           = None
        self.labels         = None
        self.lemmas         = None
        self.lemmas_pos     = None
        self.lemmas_idx     = None
        self.sentence_pairs = None
        self.load_dataset()


    #TODO check strange cases as " he's "
    def tokenize_line(self, line, pattern='\W'):
        '''tokenizes a line, returns a list of words'''
        # I insert a space otherwise words such as 'xx-yyyy' becames 'xxyyyy'
        # which, in most cases, isn't proper english and appears only once or twice
        # some are errors or missplelling btw
        line = re.sub('[\.,:;!@#$\(\)\-&\\<>/0-9]', ' ', line)
        return [word.lower() for word in re.split(pattern, line.lower()) if word]


    def read_dataset(self, path):
        """
        Taken from evaluate.py for simplicity
        """
        sentence_pairs = []
        labels = []
        with jsonlines.open(path) as f:
            for obj in f:
                labels.append(obj.pop('label'))
                sentence_pairs.append(obj)
        assert len(sentence_pairs) == len(labels)
        return sentence_pairs, labels


    def lemmatize_target(self, sentence, start, end, lemma):
        """returns the sentence (str) with the lemma in place of the target word at sentence[start:end]"""
        return sentence[:start] + lemma.lower() + sentence[end:]
        
    def lemmatize_sentence(self, setence, lemma):
        """
        lemmatizes the entire sentence (str) and returns it (list)
        """
        res = []
        for word in setence:
            res.append(self.lemmatizer.lemmatize(word))
        return res

    def load_dataset(self):
        """
        Builds a list of lists containing lists of tokenized words --> data
            [   [tokenized_sentence_1_1, tokenized_sentence_1_2],
                [tokenized_sentence_2_1, tokenized_sentence_2_2], [], ... ]
            Total words = # of tokens --> count
        """
        print(">> Reading dataset '{}'...".format(self.datapath))
        if self.lemmatize == 'target':
            print(">> Lemmatizing targets ...")
        elif self.lemmatize == 'all':
            print(">> Lemmatizing all words (without PoS tagging) ...")
        self.sentence_pairs, self.labels = self.read_dataset(self.datapath)
        if self.verbose:
            print("\nCheck:")
            print('\tLemma:', self.sentence_pairs[0]['lemma'])
            print('\tSntc1:', self.sentence_pairs[0]['sentence1'])
            print('\tSntc2:', self.sentence_pairs[0]['sentence2'])
            print('\tLabel:', self.labels[0], '\n')

        # I take all info I might need (depending on other parameters in the main file)
        count = 0
        self.data       = []
        self.lemmas     = []
        self.lemmas_pos = []
        self.lemmas_idx = []
        for elem in self.sentence_pairs:
            
            if self.lemmatize == 'target':
                s1 = self.lemmatize_target(elem['sentence1'], int(elem['start1']), int(elem['end1']), elem['lemma'])
                s2 = self.lemmatize_target(elem['sentence2'], int(elem['start2']), int(elem['end2']), elem['lemma'])
            else:
                s1 = elem['sentence1']
                s2 = elem['sentence2']

            sntc1 = self.tokenize_line(s1)
            sntc2 = self.tokenize_line(s2)
            
            if self.lemmatize == 'target':
                assert elem['lemma'].lower() in sntc1
                assert elem['lemma'].lower() in sntc2
                idx_lemma1 = sntc1.index(elem['lemma'].lower())
                idx_lemma2 = sntc2.index(elem['lemma'].lower())
            else:
                idx_lemma1 = 0  # dummy value
                idx_lemma2 = 0  # dummy value
            
            if self.lemmatize == 'all':
                sntc1 = self.lemmatize_sentence(sntc1, elem['lemma'])
                sntc2 = self.lemmatize_sentence(sntc2, elem['lemma'])

            self.data.append([sntc1, sntc2])
            self.lemmas.append(elem['lemma'].lower())
            self.lemmas_pos.append(elem['pos'])
            self.lemmas_idx.append([idx_lemma1, idx_lemma2])
            count += len(sntc1) + len(sntc2)
        
        if self.verbose:
            print(">> Total number of words:      {:7d}".format(count))                # 390'174 tokens
            print(">> Distinct lemmas to analyze: {:7d}".format(len(self.lemmas)))     #   3'726 lemmas           


class Dataset2Index:
    
    def __init__(self, DatasetLoader, vocab_size, unk, missing_policy, stopwords=None, build_onehot=False, verbose=False):
        """
            Builds a vocabulary based on the most frequent words in the dataset.
            
            NOTE that target lemmas will always be in the vocabulary
            
            Args:
            - DatasetLoader  (obj)  : DatasetLoader instance
            - vocab_size     (int)  : size of the vocabolary
            - unk            (str)  : token to associate with unknown words
            - missing_policy (str)  : whether to `add` or `replace` words in the dataset with lemmas
            - build_onehot   (bool) : whether to build the onehot encoding or not
            - verbose        (bool) : whether to display additional info or not
            
            Attributes:
            - word2idx       (dict) : mapping from words to indexes  
            - idx2word       (dict) : mapping from indexes to words
            - onehot         (list) : sentences in data translated to their onehot encoding
            - all the attributes of DatasetLoader class
        """
        self.unk            = unk
        self.verbose        = verbose
        self.stopwords      = stopwords
        self.size           = vocab_size
        self.build_onehot   = build_onehot
        self.missing_policy = missing_policy
        self.data           = DatasetLoader.data
        self.labels         = DatasetLoader.labels
        self.lemmas         = DatasetLoader.lemmas
        self.lemmas_pos     = DatasetLoader.lemmas_pos
        self.lemmas_idx     = DatasetLoader.lemmas_idx
        self.sentence_pairs = DatasetLoader.sentence_pairs
        
        # populated in `build_vocab`
        self.word2idx = None
        self.idx2word = None
        self.onehot   = None
        self.build_vocab()


    def convert2id(self, sentence):
        '''converts words in a list of their IDs and returns it'''
        converted = []
        for w in sentence:
            if w in self.word2idx:
                i = self.word2idx[w] 
            else:
                continue
            converted.append(i)
        return converted


    def compute_required_space(self):
        '''returns the set of missing lemmas in the vocab'''
        toadd = set()
        for l in self.lemmas:
            if l not in self.word2idx:
                toadd.add(l)
        return toadd


    def replace_missing(self):
        '''
        Insert missing lemmas in the vocabulary by
        replacing the least common words
        '''
        toadd = self.compute_required_space()
        if len(toadd) > 0:
            if self.verbose:
                print("\t[WARN] >> There are {} target words not in the vocabulary (yet)".format(len(toadd)))
            
            aux = list([key for key in self.word2idx.keys()])
            aux = list(reversed(aux))            
            for key in aux:
                if len(toadd) > 0:
                    if key not in self.lemmas:
                        val = self.word2idx.pop(key)
                        self.word2idx[toadd.pop()] = val
                    else:
                        continue

        for l in self.lemmas:
            assert l in self.word2idx, "\n\n\t\t[ERRO] >> {} is not in the vocab!\n\n".format(l)
        print(">> Missing lemmas added by replacing the least frequent ones.")


    def add_missing(self):
        '''
        Insert missing lemmas in the vocabulary
        '''
        toadd = self.compute_required_space()
        if len(toadd) > 0:
            if self.verbose:
                print("\t[WARN] >> There are {} target words not in the vocabulary (yet)".format(len(toadd)))
            idx = len(self.word2idx)
            for word in toadd:
                self.word2idx[word] = idx
                idx += 1
        self.size = len(self.word2idx)

        for l in self.lemmas:
            assert l in self.word2idx, "\n\n\t\t[ERRO] >> {} is not in the vocab!\n\n".format(l)
        print(">> Missing lemmas added, dict now has size {}.".format(self.size))


    def build_vocab(self):
        """
        Builds a list of lists containing lists of indexed words --> onehot
                [   [tokenized_numerical_sentence_1_1, tokenized_numerical_sentence_1_2],
                    [tokenized_numerical_sentence_2_1, tokenized_numerical_sentence_2_2], [], ... ]
        """
        # Find number of distinct words
        count_list = []
        for couple in self.data:
            for sntc in couple:
                count_list.extend(sntc)
        count = collections.Counter(count_list)
        print(">> Number of distinct words:   {:7d}".format(len(count)))                 # 27'007 distinct words

        # Build a dict that maps words to their ID, considering only the size-1 most common words
        # all the other words will be mapped to UNK
        self.size = max(len(self.lemmas), self.size)    # at least  all lemmas to be in the dict
        print(f">> Building vocabulary with size {self.size}\t({100*self.size/len(count):.2f}% of the original)")
        if self.stopwords is None:
            self.word2idx = {key: index for index, (key, _) in enumerate(count.most_common(self.size - 1))}
        else:
            print(f">> Removing {len(self.stopwords)} words in the stopset")
            self.word2idx = {}
            index = 0
            for key, _ in count.most_common(self.size - 1):
                if key not in self.stopwords:
                    self.word2idx[key] = index
                    index += 1

        self.replace_missing() if self.missing_policy == 'replace' else self.add_missing()
        assert self.unk not in self.word2idx
        self.word2idx[self.unk] = self.size         # I add the unk token
        self.size = len(self.word2idx)              # and update the size
        print(f">> added UNK token -- vocab size is {self.size}")
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        assert len(self.idx2word) == max(self.idx2word.keys())+1

        # Build the ONE-HOT Encoding reverting the previous dict
        # needed to create my own embeddings
        # word2idx --> idx2word
        if self.build_onehot:
            self.onehot = []
            for couple in self.data:
                sntc1 = self.convert2id(couple[0])
                sntc2 = self.convert2id(couple[1])
                self.onehot.append([sntc1, sntc2])

            # check: they must have the same length (in terms of couples, not words)
            assert len(self.data) == len(self.onehot)
        
        # check whether mapping is correct
        if self.verbose:
            print("Check:")
            print("\tWord '{}'\t--> id {}".format("the",        self.word2idx["the"]))
            print("\tWord '{}'\t--> id {}".format("obstacle",   self.word2idx["obstacle"]))
            print("\tWord '{}'\t--> id {}".format("<???>",      self.word2idx["<???>"]))
            print("Check:")
            print("\tWord '{}'\t<-- id {}".format(self.idx2word[self.word2idx["the"]],      self.word2idx["the"]))   
            print("\tWord '{}'\t<-- id {}".format(self.idx2word[self.word2idx["obstacle"]], self.word2idx["obstacle"]))
            print("\tWord '{}'\t<-- id {}".format(self.idx2word[self.word2idx["<???>"]],    self.word2idx["<???>"]))
            print()

            # check: convert the ONEHOT encoding to text again and see what we have left
            if self.build_onehot:
                print("Original:")
                for w in self.data[5][0]:
                    print(w, flush=True, end=' ')
                print("\n")
                print("ONE-HOT encoding:")
                for w in self.onehot[5][0]:
                    print(self.idx2word[w], flush=True, end=' ')
                print("\n")
     


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