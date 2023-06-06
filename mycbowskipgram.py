#!/usr/bin/python3

import argparse
import sys
import math
import numpy as np
import time
import struct
import warnings
import os

from multiprocessing import Array, Value, Pool

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
        for token in ['<bol>', '<eol>']:
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
                    sys.stdout.write("\r Reading %d words" % (word_count))
                    sys.stdout.flush()

            # add the begin of line and the end of line a token to increment their counts
            vocab_items[vocab_hash['<bol>']].count += 1
            vocab_items[vocab_hash['<eol>']].count += 1
            word_count += 2
        
        #after reading all lines in the file
        self.bytes = fi.tell() #all the bytes in the file
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
        for i in range(vocab_size -1): #from 0 to vocab_size - 2 (we exclude the root_idx) see schema
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
            parent[min1] = vocab_size + i #min 1 varies from vocab_size -1 (i = 0) to 1 or from vocab_size to vocab_size + vocab_size -2 (because we exclude the root node)
            parent[min2] = vocab_size + i #min2 varies from vocab_size - 2 (i = 0) to 0 or from vocab_size to vocab_size + vocab_size -2 (because we exclude the root node)
            binary[min2] = 1 #the more frequent word in a subtree is represented by 1 binary[min1] is already 0

        #assign binary code and path pointer to each vocab word
        root_idx = 2 * vocab_size -2 # this is the indice of the real root of the Huffmann tree althugout not present th the parent and binary lists
        for i, vocab_item in enumerate(self):
            path = [] #list of indices from leaf to the root
            code = [] #binary Huffmann encoding from leaf to the root

            node_idx = i #0 is the most frequent word vocab_size -1 is the less frequent word
            
            while(node_idx < root_idx):
                if node_idx >= vocab_size: path.append(node_idx) #we start with the first parent then the parent of parent
                code.append(binary[node_idx])
                node_idx = parent[node_idx]
            path.append[root_idx] # the real root of the Huffmann tree

            vocab_item.path = [j - vocab_size for j in path[::-1]] #I chose to begin at 0 first index of the non leaf nodes !
            #vocab_item.path contains the path (list of Vocab' indices) from the root to the parent of vocab_item
            vocab_item.code = code[::-1] #Huffman value from root to me 1 if the more frequent else zero 1 for left (more frequent) 0 for right

class UnigramTable:
    """ A list of indices of vocab_items (tokens) in a table following a power distribution
        used to draw negative samples see http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/
    """

    def __init__(self, vocab):
        power = 0.75 #lessen frequent words and augment les frequent words see http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/
        norm = sum([math.pow(t.count, power) for t in vocab]) #normalizing constant to get a probaility, see the __iter__ implemntation of vocab
        
        table_size = int(1e8) #length of the unigram table
        table = np.zeros(table_size, dtype=np.uint32) #it is a table of indices in the Vocab object see __get_item__(self, indice)

        print("[UnigramTable __init__] Filling the Unigram table", flush=True)
        p = 0 #the cumulative probaility
        i = 0 #the table indice table[i] is the indice in the Vocak object to be repeated until P(wi)
        #enumerate calls the iterator returned from def __iter__(self) of the Vocab class
        for j, unigram in enumerate(vocab): #we get the indices of the vocab items in j sorted from the most frequent token to the less frequent one
            p += math.pow(unigram.count, power) / norm
            while i < table_size and float(i) / table_size < p:
                table[i] = j
                i += 1
        print("[UnigramTable __init__] Unigram table filled", flush=True)
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
    
#for the sense of syn0 and syn1 see the answer at https://stackoverflow.com/questions/53301916/python-gensim-what-is-the-meaning-of-syn0-and-syn0norm
def init_net(dim, vocab_size): #dim is the dimension of the hidden layer, vocab_size is the number of lines at the input
    #init syn0 with a uniform distribution on the interval [-0.5, 0.5]/dim. syn0 represents matrix of the input layer (between the one_hot input and the hidden layer od dimension dim)
    tmp = np.random.uniform(low=-0.5/dim, high=0.5/dim, size=(vocab_size, dim))
    syn0 = np.ctypeslib.as_ctypes(tmp)
    syn0 = Array(syn0._type_, syn0, lock=False)

    #init syn1 with zeros syn1 the transpose matrix of the weights between the hidden and the output layer (which has a vocab size dimension because we output the onehot represnetation of the calculated word)
    tmp = np.zeros(shape=(vocab_size, dim))
    syn1 = np.ctypeslib.as_ctypes(tmp)
    syn1 = Array(syn1._type_, syn1, lock=False)

    #syn0 (vocab_size, dim) coefficients between the hidden_layer and the ouput layer (we don't care of them) initialized randomly
    ## it is in fact the transpose of the output matrix
    #syn1 (vocab_size,dim) coefficients between the input layer and the hidden layer if the inputvetor is a one-hot linevector
    ## then each line in syn1 represent a word in the hidden layer space (the firt lines for the most frequent words) 
    ##these coefficients are 0 initialized
    return (syn0, syn1)

#vocab classified from more frequent to less frequent and the syn0 lines must match (use of zip underneath)
def save(vocab, syn0, fo, binary):
    print(f"[save]Saving output vectors to {fo} in binary({binary}) mode ") # fo is the path of the output file
    dim = len(syn0[0]) # the size of a vector syn0 the size of the hidden layer vectors or word projections
    #what a zip does: https://www.programiz.com/python-programming/methods/built-in/zip
    if binary:
        fo = open(fo, 'wb') #write binary format
        fo.write("%d %d\n" %(len(syn0, dim))) #the number of words and the output dimension of each word (projection)
        fo.write("\n")

        for vocab_item, vector in zip(vocab, syn0): # vocab has a __iter__ method vocab and syn0 must list the words in the same order vocab[0] must correspond to syn[0]
            fo.write("%s " % vocab_item.word)
            for s in vector:
                fo.write(struct.pack('f', s)) #we add a 32bits representation of float s
            fo.write("\n")
    else: #we write in text format
        fo = open(fo, 'w') #write text format
        fo.write("%d %d\n" %(len(syn0), dim)) #the number of words and the output dimension of each word (projection)

        for vocab_item, vector in zip(vocab, syn0):
            word = vocab_item.word
            word_projection = ' '.join([str(s) for s in vector]) # s in vector is a float I onlys join strings
            fo.write('%s %s \n' %(word, word_projection))

    fo.close()

#setting the pool variables as global variables to be shared
def __init_process(*args): 
    global vocab, syn0, syn1, table, cbow, neg, dim, starting_alpha
    global win, num_processes, global_word_count, fi

    vocab, syn0_tmp, syn1_tmp, table, cbow, neg, dim, starting_alpha, win, num_processes, global_word_count = args[:-1]
    fi = open(args[-1], 'r')
    #see https://www.pythonpool.com/suppress-warnings-in-python/ for supressing warnings for specific lines of code
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        syn0 = np.ctypeslib.as_array(syn0_tmp) #syn0 shares the memory with the Pooling unlocked Array syn0_tmp
        syn1 = np.ctypeslib.as_array(syn1_tmp)
    
    num_process =  os.getpid() #see #https://docs.python.org/3/library/multiprocessing.html and tests/shared_arrays_and_values.py
    num_parent_process = os.getppid()
    print(f"[__init_process] initialization of process {num_process} (parent {num_parent_process}) out of {num_processes} processes ended")

def train_process(pid):
    #the global variables (global only to the process i.e. between the init_process and this method) are:
    ## vocab: a Vocabulary object with all the text words of the text already inside
    ## syn0 the (vocab_size x dim) the vectors output (between )
    ## syn1 the (vocab_size x dim) the weights of the hidden layer
    ## table the UnigramTable only used if neg > 0
    ## cbow (program parameter) If True I caclulate the Continuous Bag of Words otherwise I do a SkipGram
    ## neg (program parameter) if positive I use the unigram table to negative sampling: it is the number of negative samples; 0 for softmax OUTPUT
    ## dim (program parameter) the size of the hidden layer
    ## starting_alpha (program parameter) the intial value of alpha used for gradient descent
    ## win (program parameter) the max length of the words' window
    ## num_processes (programm parameter) the total number of processes
    ## global_word_count the shared variable counting the number or processed words during training
    ## the text input's file handle

    #set fi (input file's handle) to point to the right cunk of the training file
    # each process will treat its own chunk of the input file
    start = round(vocab.bytes / num_processes * pid) # vocab.bytes see the end of the __init__ method of the Vocab class
    end = round(vocab.bytes if pid == num_processes -1 else  vocab.bytes / num_processes * (pid + 1)) #this end is the beginning ot the next process (if not the max processes)

    fi.seek(start) #we go from start (inclusive) tho the end (exclusive) position

    print(f"[train_process_{pid}] Worker of number: {pid + 1} out of {num_processes} starting to read the input file at (included) {start} position until (excluded) {end} position")

    alpha = starting_alpha #the alpha parameter used for gradient descent

    word_count = 0
    last_word_count = 0 # to add to the global word count

    while fi.tell < end: #first iteration: we iterate line by line of the file
        line = fi.readline().strip() #the line can start at the middle of a word see tests/File/read.py two processes can overlap the same line
        #Skip blank lines
        if not line:
            continue

        #see Vocab class indices method returns the indices of the word accessible through Vocab.get if it exists in Vocab 
        # otherwithe it returns the indice of <unk> in Vocak (typically for cuted words)
        sent = vocab.indices('[bol]'+ line.split() + '[eol]')

        #second iteration: we iterates on the line
        for sent_pos, token_indice in enumerate(sent):
            if word_count % 10000 == 0: #each 10000 words we update alpha
                global_word_count.value += word_count - last_word_count
                last_word_count = word_count

                #recalculate alpha diminishing it slowly until starting_alpha * 0.0001
                alpha = starting_alpha * (1 - float(global_word_count.value) / vocab.word_count)
                if alpha < starting_alpha * 0.0001: alpha = starting_alpha * 0.0001

                #Print progress info:
                sys.stdout.write('\r [train_process_%d] Process pid %d Alpha: %f Progress: %d of %d (%.2f %%)', pid, pid+1, alpha, 
                                 global_word_count.value, vocab.word_count, float(global_word_count.value) / vocab.word_count * 100)
                sys.stdout.flush()

            #calculate the current window (a random value from 1 -inclusive- to win +1 exclusive)
            current_win = np.random.randint(low=1, high=win+1)
            context_start = max(0, sent_pos - current_win)
            context_end = min(len(sent), sent_pos + current_win + 1) #+1 because the context_end is excluded from the selection
            #token_indice = sent[sent_pos]
            # list of the token_indices from the line's window to be translated in VocabItem(s) through Vocab[token_indice]
            context = sent[context_start:sent_pos] + sent[sent_pos+1, context_end] #we have all the indices of the words of Vocab used (Vocab[context[i]])
            #context contains all the indices of words in Vocab (except for the right word)
            #second iteration: for each line's window we update the weights
            
            #Continuous Bag of Words (CBOW)
            if cbow:
                #neu1 is the computed representation in the hiddent layer of the word Vocab[token_indice] through its windows' neighbors
                ## context contains the vocab_indices which are the same in the Vocab object and in the syn0 traduction 
                ## (0 for the most ferquent vocab_size -1 for the less frequent)
                ##syn0 is the matrice (vocab_size*dim) between the one_hot input (size vocab_size) and the hidden layer
                # a line of syn0 correspond to the hidden representation of the corresponding (1 same indice as the line indice of syn0) one hot representation of the word
                ## (1,0,.....0) is the most frequent word in Vocab
                neu1 = np.mean(np.array([syn0[c] for c in context]), axis = 0) #mean of the rows of the context (only use of the context)
                assert len(neu1) == dim, 'neu1 and dim do not agree'

                ##init neu1e with zeros #error on neu1
                neu1e = np.zeros(dim)
                
                #update neu1e and update syn1 (the matrice (vocab_size x dim) of the weights of the hidden layer)
                if neg > 0: #neg is by default 4
                    #vocab[token_indice] is the right word so 1
                    # we draw 4 negative samples from big Unigram table vocab[target] is a negative sample so 0
                    classifiers = [(token_indice, 1)] + [(target, 0) for target in table.sample(neg)]
                else: #case of hierarchical softmax see https://arxiv.org/abs/1411.2738
                    classifiers = zip(vocab[token_indice].path, vocab[token_indice].code) #see page 10 of https://arxiv.org/abs/1411.2738

                for target, label in classifiers:
                    #forward phase syn1(vocab_size, sim) is the transpose of the weights' matrice between the hidden and the output layers
                    z = np.dot(neu1, syn1[target]) #see h * v'j why do the upper root nodes have a less frequency ? syn1 are the weights of the hidden layer
                    p = sigmoid(z) #z and p are scalars p id the output for target
                    g = alpha * (label - p) #scalar
                    #back propagation 
                    # we do a g * v'i to backpropagate error to syn0
                    neu1e += g * syn1[target] #for np.array it multiplies each element for list it multiply the number of elements
                    syn1[target] += g * neu1 #we recalculate v'i

                #updates syn0 (we have calculated neu1e for each target in the hierarchy of my word or of the negative sampling)
                for context_word in context: #Window in the line: we don not do it for the word itself beacause there is no error
                    syn0[context_word] += neu1e

            else: #SkipGram 
                for context_word in context: #instead of working with neu1 we work with each of context words
                    neu1e = np.zeros(dim) # we initiate the error with zeros

                    #update neu1e and update syn1 (the matrice (vocab_size x dim) of the weights of the hidden layer)
                    if neg > 0:
                        #vocab[token_indice] is the right word so 1
                        # we draw 4 negative samples from big Unigram table vocab[target] is a negative sample so 0
                        classifiers = [(token_indice, 1)] + [(target, 0) for target in table.sample(neg)]
                    else: #case of hierarchical softmax see https://arxiv.org/abs/1411.2738
                        classifiers = zip(vocab[token_indice].path, vocab[token_indice].code) #see page 10 of https://arxiv.org/abs/1411.2738

                    for target, label in classifiers:
                        z = np.dot(syn0[context_word], syn1[target]) #see h * v'j why do the upper root nodes have a less frequency ? syn1 are the weights of the hidden layer
                        p = sigmoid(z) #z and p are scalars
                        g = alpha * (label - p) #scalar
                        # we do a g * v'i to backpropagate error to syn0
                        neu1e += g * syn1[target] #for np.array it multiplies each element for list it multiply the number of elements
                        syn1[target] += g * syn0[context_word] #we recalculate v'i

                    #updates syn0 (we have calculated neu1e for each target in the hierarchy of my word or of the negative sampling)
                    syn0[context_word] += neu1e

            #after having worked on the context of a word of a line we pass to the next
            word_count += 1


def train(fi, fo, cbow, neg, dim, alpha, win, min_count, num_processes, binary):
    print(f"[train] the input training file: {fi}", flush=True)
    print(f"[train] the output model file: {fo}", flush=True)
    if cbow:
        print("[train] I create a un Continuous Bag Of Words", flush=True)
    else:
        print("[train] I calculate a Skip Gram Diagram", flush=True)
    print(f"[train] the number of negative examples (if 0 hierarchival softmax): {neg}", flush=True)
    print(f"[train] the size of the hidden layer (word embedding): {dim}", flush=True)
    print(f"[train] coeffcient for gradient descending: {alpha}", flush=True)
    print(f"[train] the window size of words: {win}", flush=True)
    print(f"[train] the minimum number of words used for learning the model: {min_count}", flush=True)
    print(f"[train] the number of parallel processes for learning the model: {num_processes}", flush=True)
    if binary:
        print(f"[train] the output model file {fo} is in binary format", flush=True)
    else:
        print(f"[train] the output model file {fo} is in text format", flush=True)

    #Read train file to init Vocab
    vocab = Vocab(fi, min_count)

    #init network syn0 are the output words, syn1 are the weight of the hidden layer
    syn0, syn1 = init_net(dim, len(vocab)) # see the defined __len__ method in the Vocab class

    global_word_count = Value('i', 0) # shared integer ('i') value between processes intialized at 0

    table = None
    if neg > 0:
        print("[train] initializing the UNIGRAMM table", flush=True)
        table = UnigramTable(vocab)
    else:
        print("[train] Calculating the HUFFMAN tree used in the hierarchical softmax", flush=True)
        vocab.encode_huffmann()

    #now we are startinf the training job
    print(f"[train] starting the {num_processes} process(es)", flush=True)
    t0 = time.time()
    pool = Pool(processes=num_processes, initializer=__init_process, initargs=(vocab, syn0, syn1, table, cbow, neg, dim, alpha,
                                                                            win, num_processes, global_word_count, fi))
    # issue tasks to the process pool
    pool.map_async(train_process, range(num_processes))

    t1 = time.time()
    duration_in_minutes = round((t1 - t0)/60)
    print(f"[train] the total duration of the training took {duration_in_minutes} minutes using {num_processes} threads")

    #save the model to a file
    ## vocab and syn0 have been calculated syn0 is the vocabulary projected in the new space
    ## fo and binary are arguments introduced manually by the user
    save(vocab, syn0, fo, binary) #fo is the path of the output file


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