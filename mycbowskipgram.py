#!/usr/bin/python3

import argparse
import sys

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
    
    #used by the indices method of this class
    def __contains__(self, key): #return True or False depending on key (token:string) in Vocab
        return key in self.vocab_hash #see if key is in the keys of the dictionary
    
    def indices(self, tokens): #token in list is made possible by the special __contains__ method !!!
        return [self.vocab_hash[token] if token in self else self.vocab_hash['<unk>'] for token in tokens] #returns a list

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