#!/usr/bin/python3

import numpy as np

if __name__ == "__main__":
    mot1 = 'comme'
    mot2 = 'que'
    fo = open('files/fleurs_mal_mots.txt', 'r')

    for line in fo:
        stats = line.split()
        if stats[0] == mot1:
            vecteur_mot1 = np.array(stats[1:])
        
        if stats[0] == mot2:
            vecteur_mot2 = np.array(stats[1:])

    print(f"{vecteur_mot1.shape}--{vecteur_mot2.shape}")

    cosine_distance = np.dot(vecteur_mot1, vecteur_mot2)/(np.dot(vecteur_mot1, vecteur_mot1)*np.dot(vecteur_mot2, vecteur_mot2))


    
    print(f"la cosine distance entre les 2 vecteurs est: {cosine_distance}")
        
