#!/usr/bin/python3

import argparse
import sys
import math
import numpy as np

class VocabItem:
    def __init__(self, word):
        self.word = word
        self.count = 0
        self.path = None #PAth (list of indices from the root to the word)
        self.code = None #Huffman encoding

class Vocab:
    def __init__(self, fi, min_count):
        vocab_items = []
        vocab_hash = {}
        word_count = 0

        fi = open(fi, 'r')

        #Add sepcial tokens <bol> (begin of line) and <eol> (end of line) to the list of token
        for token in ['<bol', '<eol>']:
            vocab_hash[token] = len(vocab_items)
            vocab_items.append(VocabItem(token)) #we will increment their count later as we really meet them

        for line in fi:
            tokens = line.split()
            for token in tokens:
                if token not in vocab_hash: #the in is made on the keys of the dictionary
                    vocab_hash[token] = len(vocab_items) # it is a new VocabItem
                    vocab_items.append(VocabItem(token))

                vocab_items[vocab_hash[token]].count += 1 #vocab_hash[token] helps us to find the indice of the word in Vocab_item to help increment its count
                word_count += 1 #total count of words independently they already existed or not!

                if word_count % 1000 == 0:
                    sys.stdout.write("\r Reading %d words", word_count)
                    sys.stdout.flush()

            # add the begin of line and the end of line a token to increment their counts
            vocab_items[vocab_hash['<bol>']].count += 1
            vocab_items[vocab_hash['<eol>']].count += 1
            word_count += 2
        
        #after reading all lines in the file
        self.bytes = fi.tell()
        self.vocab_items = vocab_items #list of VocabItem objects whose index in the list is given by the hash just after
        self.vocab_hash = vocab_hash   #dictionary of words (string) to the index of the corresponding VocabItem object in the list just above
        self.word_count = word_count #the total number of all words found in the file (including duplicates)
        fi.close()

        #the following method does the following
        ## add a special token <unk>
        ## add the word which appear less than min_count times to the <unk> VocabItem
        ## sort Vocab_Items in decending order by frequency in the train file
        self.__sort(min_count)

        #assert(self.word_count == sum([t.count for t in self.vocab_items]), "word_count and sum o t.count do not agree")
        print('Total words in training file %s' % self.word_count)
        print('Total bytes in training file %s' % self.bytes)

    def __sort(self, min_count):
        tmp = [] # the temporary new self.vocab_item
        tmp.append(VocabItem('<unk>'))
        unk_hash = 0 #it is not important that unk has the same hash as <bol> because the folloving sort will recalculate all hash values

        number_of_unknown_words = 0 #only for display at the end of the method
        for vocab_item in self.vocab_items:
            if vocab_item.count < min_count:
                number_of_unknown_words += 1
                tmp[unk_hash].count += vocab_item.count
            else:
                tmp.append(vocab_item) #so the unk_hash for <unk> is only temporatory
            
        tmp.sort(key=lambda vi: vi.count, reverse=True)#custom sort function vi stands for vocable_item see https://learnpython.com/blog/python-custom-sort-function/

        vocab_hash = {} # not the same as the local vocab_hash dict in the __init__ method !
        for i,vi in enumerate(tmp):
            #we redefine the index values as hash so the 0 for unk_hash has no importance it disappear
            vocab_hash[vi.word] = i

        self.vocab_hash = vocab_hash
        self.vocab_items = tmp

        print(f"unknown vocab size {number_of_unknown_words}") #it is only for display

    def __getitem__(self, i): #allow to call vocab[i], vocab[i:j:k]
         return self.vocab_items[i]
    
    def __len__(self): #allow to call len(vocab)
        return len(self.vocab_items)
    
    def __iter__(self): #allow to call for item in vocab
        return iter(self.vocab_items)
    
    #used by the indices method of this class especially the in in self!!!
    def __contains__(self, key): #return True or False depending on key (token:string) in Vocab
        return key in self.vocab_hash #see if key is in the keys of the dictionary
    
    def indices(self, tokens): #token in list is made possible by the special __contains__ method !!!
        return [self.vocab_hash[token] if token in self else self.vocab_hash['<unk>'] for token in tokens] #returns a list
    
    """build a Huffmann tree"""
    def encode_huffmann(self):
        vocab_size = len(self)
        count = [t.count for t in self] + [1e15] * (vocab_size -1) # size vocab_size + vocab_size -1
        parent = [0] * (2 * vocab_size -2) # size vocab_size + vocab_size -1 (different parents) and we exclude the root_idx which has no perent
        binary = [0] * (2 * vocab_size -2) # same as above the root_idx fans no Huffmancode see schema and while node_idx < rott_idx: underneath

        pos1 = vocab_size -1 #count[pos1] the less frequent word
        pos2 = vocab_size # count[pos2] = 1e15

        #see schema on https://towardsdatascience.com/huffman-encoding-python-implementation-8448c3654328
        ## on the link the nodes are classified in descending frequency
        for i in range(vocab_size -1): #from 0 to vocab_size - 2 (we include the root_idx) see schema
            #find min1 the less frequent word in the resulting tree
            if pos1 >= 0: #i = 0 for pos1 = vocab_size-1 i = vocabsize - 2 por pos1 = 1
                if count[pos1] < count[pos2]: # true for the first time
                    min1 = pos1
                    pos1 -= 1 # we take the term a bit more frequent thne the minimal (to calculate min2)
                else: #cela arrive les fois suivantes quand count[vocab_size + i] = count[min1] + count[min2] on ajoute les instervalle soit vocab_size + 1
                    min1 = pos2
                    pos2 += 1
            else: # we only check the upper part count[vocab_size + i] = count[min1] + count[min2]
                min1 = pos2
                pos2 += 1
            
            #find min2 the second less frequent word in the resulting tree pos1 decrease as frequency increase
            if pos1 >= 0:
                if count[pos1] < count(pos2):
                    min2 = pos1
                    pos1  -= 1 #pos1 (leafs) decrease we exclude the one that are after for future iterations
                else:
                    min2 = pos2
                    pos2 += 1 #pos2 (not leafs) increase as frequency increase we exclude the ones that are before for future parent code calculations
            else:
                min2 = pos2
                pos2 += 1

            count[vocab_size + i] = count[min1] + count[min2]
            parent[min1] = vocab_size + i #min 1 varies from vocab_size -1 (i = 0) to 1 or from vocab_size to vocab_size + vocab_size -2
            parent[min2] = vocab_size + i #min2 varies from vocab_size - 2 (i = 0) to 0 or from vocab_size to vocab_size + vocab_size -2
            binary[min2] = 1 #the more frequent word in a subtree is represented by 1 binary[min1] is already 0

            #assign binary code and path pointer to each vocab word
            root_idx = 2 * vocab_size -2 # vocab_size -1 is the last vocab_item + from vocab_size to vocab_size + vocab_size - 2 for the non leaf nodes see schema
            for i, vocab_item in enumerate(self):
                path = [] #list of indices from leaf to the root
                code = [] #binary Huffmann encoding from leaf to the root

                node_idx = i #0 is the most frequent word vocab_size -1 is the less frequent word
                
                while(node_idx < root_idx):
                    if node_idx >= vocab_size: path.append(node_idx) #we start with the first parent then the parent of parent
                    code.append(binary[node_idx])
                    node_idx = parent[node_idx]
                path.append[root_idx]

                vocab_item.path = [j - voca
                b_size for j in path[::-1]] #I chose to begin at 0 first index of the non leaf nodes !
                vocab_item.code = code[::-1]

class UnigramTable:
    """ A list of indices of vocab_items (tokens) in a table following a power distribution
        used to draw negative samples see http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/
    """

    def __init__(self, vocab):
        power = 0.75 #lessen frequent words and augment les frequent words see http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/
        norm = sum([math.pow(t.count, power) for t in vocab]) #normalizing constant to get a probaility, see the __iter__ implemntation of vocab
        
        table_size = 1e8 #length of the unigram table
        table = np.zeros(table_size, dtype=np.uint32) #it is a table of indices in the Vocab object see __get_item__(self, indice)

        print("Filling the Unigram table")
        p = 0 #the cumulative probaility
        i = 0 #the table indice table[i] is the indice in the Vocak object to be repeated until P(wi)

        for j, unigram in enumerate(vocab): #we get the indices of the vocab items in j sorted from the most frequent token to the less frequent one
            p += math.pow(unigram.count, power) / norm
            while i < table_size and float(i) / table_size < p:
                table[i] = j
                i += 1
        
        self.table = table #we need it to take a negative sample

    #we take count negative samples
    def sample(self, count): 
        indices = np.random.randint(0, len(self.table), count)
        return [ self.table[i] for i in indices ]

def sigmoid(z):
    if z > 6:
        return 1
    elif z < -6:
        return 0
    else:
        return 1. / (1. + np.exp(-z))

def train(fi, fo, cbow, neg, dim, alpha, win, min_count, num_processes, binary):
    print(f"the input training file: {fi}")
    print(f"the output model file: {fo}")
    if cbow:
        print("I create a un Continuous Bag Of Words")
    else:
        print("I calculate a Skip Gram Diagram")
    print(f"the number of negative examples (if 0 hierarchival softmax): {neg}")
    print(f"the size of the hidden layer (word embedding): {dim}")
    print(f"coeffcient for gradient descending: {alpha}")
    print(f"the window size of words: {win}")
    print(f"the minimum number of words used for learning the model: {min_count}")
    print(f"the number of parallel processes for learning the model: {num_processes}")
    if binary:
        print(f"the output model file {fo} is in binary format")
    else:
        print(f"the output model file {fo} is in text format")

    #Read train file to init Vocab
    vocab = Vocab(fi, min_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enter the parameters of my Word Model")
    parser.add_argument('-train', help='Training file', dest='fi', required=True)
    parser.add_argument('-model', help='Output model file', dest='fo', required=True)
    parser.add_argument('-cbow', help='1 for CBOW, 0 for Skipgram', dest='cbow', default=1, type=int)
    parser.add_argument('-negative', help='Number of negative examples (> 0) for negative sampling, 0 for hierarchical softmax', dest='neg', default=5, type=int)
    parser.add_argument('-dim', help='Dimensionality of word embedding', dest='dim', default=100, type=int)
    parser.add_argument('-alpha', help='Starting Alpha', dest='alpha', default=0.025, type=float)
    parser.add_argument('-window', help='Max window length', dest='win', default=5, type=int)
    parser.add_argument('-min-count', help='Min count of words used to learn <unk>', dest='min_count', default=5, type=int)
    parser.add_argument('-processes', help='Number of processes (parallel runs)', dest='num_processes', default=1, type=int)
    parser.add_argument('-binary', help='1 for Output model in binary format, 0 otherwise', dest='binary', default=0, type=int)
    #TODO parser.add_argument('-epochs', help='Number of epochs for the training', dest='epochs ', default=1, type=int)
    #input the data
    args = parser.parse_args()
    
    train(args.fi, args.fo, bool(args.cbow), args.neg, args.dim, args.alpha, args.win, args.min_count, args.num_processes, bool(args.binary))