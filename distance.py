#!/usr/bin/python3
import argparse
import numpy as np

class WordVector:
    def __init__(self, line):
        self.line = line
        self.word = None
        self.h_vector = None
        self.getItems()

    def getItems(self):
        line_arr= self.line.split()
        if len(line_arr) > 1 and self.word is None:
            self.word = line_arr[0].lower()
            self.h_vector = np.array(line_arr[1:]).astype(np.float64) #each element is a U22 string by default
        return (self.word, self.h_vector)

    @staticmethod
    def calcdistance(wv1, wv2): #takes two word vectors as parameters
        w1,h1 = wv1.getItems()
        w2,h2 = wv2.getItems()
        num = np.dot(h1,h2)
        denom = np.sum(np.sqrt([np.dot(h1,h1), np.dot(h2,h2)]))
        distance = num / denom
        print(f"[WordVector.calcdistance] between the words {w1} and {w2} the distance is {distance}")
        return distance
    
#utility function finding a specific word vector by its word value in the list of word_vectors
## it returns a array of which we expect it to be of size 1
get_word_vector = lambda word, arr_word_vectors: [wv for wv in  arr_word_vectors if word.lower() == wv.word]

def find_words_vectors(file, word1, word2):
    word_vectors = []
    with open(file) as fh:
        lines = fh.readlines()
        for line in lines:
            word_vectors.append(WordVector(line))
    
    print(f"We found {len(word_vectors)} word vectors in the file: {file}")

    wv1 = get_word_vector(word1, word_vectors)
    if wv1 is None or len(wv1) == 0:
        print(f"[find_words_vectors] We could'nt find {word1} in the file {file} exit")
        return None
    
    wv2 = get_word_vector(word2, word_vectors)
    if wv2 is None or len(wv2) == 0:
        print(f"[find_words_vectors] We could'nt find {word2} in the file {file} exit")
        return None
    
    return (wv1[0], wv2[0])



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enter the parameters of my Word Model")
    parser.add_argument('-model', help='The model file', dest='f', required=True)
    parser.add_argument('-mot1', help='The first word', dest='w1', required=True)
    parser.add_argument('-mot2', help='The second word', dest='w2', required=True)
    #input the data
    args = parser.parse_args()

    res = find_words_vectors(args.f, args.w1, args.w2)

    if res is not None:
        wv1, wv2 = res
        #print(f"[main] {dir(wv1)}")
        print(f"[main] between {wv1.word} and {wv2.word} the distance is {WordVector.calcdistance(wv1, wv2)}")
