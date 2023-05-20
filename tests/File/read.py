#!/usr/bin/python3
#copied from https://stacklima.com/fonction-seek-de-python/
# Opening "GfG.txt" text file
f = open("GfG.txt", "r")
# Second parameter is by default 0
# sets Reference point to twentieth
# index position from the beginning
#It is meant to check the way word2vec shares the text between processes
#def indices(self, tokens): #token in list is made possible by the special __contains__ method !!!
#       return [self.vocab_hash[token] if token in self else self.vocab_hash['<unk>'] for token in tokens] #returns a list
## means the cutted word is put in self.vocab_hash['<unk>']
print(f"Je me place à 0")
f.seek(0)
print(f"je lit une ligne")
print(f"|{f.readline().strip()}|")
print(f"la position courante est |{f.tell()}|")
print(f"Je me place à 15")
f.seek(15)
print(f"je lit une ligne")
print(f"|{f.readline().strip()}|")
# prints current position
print(f"la position courante est |{f.tell()}|")

f.close()
