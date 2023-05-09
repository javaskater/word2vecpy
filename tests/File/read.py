#!/usr/bin/python3
#copied from https://stacklima.com/fonction-seek-de-python/
# Opening "GfG.txt" text file
f = open("GfG.txt", "r")
# Second parameter is by default 0
# sets Reference point to twentieth
# index position from the beginning
#f.seek(20)
f.seek(15)
 
# prints current position
print(f.tell())
 
print(f.readline())
f.close()
