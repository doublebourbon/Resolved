# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 09:25:27 2018

@author: jerem
"""

import os
import re
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from keras import optimizers
from keras import regularizers
import matplotlib.pyplot as plt
from tkinter import Tk, Label
from time import time


############################
##### USEFUL FUNCTIONS #####
############################

def mergesort(t):
    """
    Réalise le tri fusion d'une liste t
    # Arguments
        t : List. 
            Liste de couples contenant l'occurence des mots
            et les mots.
    # Returns
        List.
        Renvoit la liste des couples de t ordonnée de façon décroissante selon l'occurence
    """
    
    
    n = len(t)
    if n < 2:
        return t
    else:
        m = n//2
    def fusion(t1,t2):
        i1,i2,n1,n2 = 0,0,len(t1),len(t2)
        t = []
        while i1 < n1 and i2 < n2:
            if t1[i1][0] > t2[i2][0]:
                t.append(t1[i1])
                i1+=1
            else:
                t.append(t2[i2])
                i2+=1
        if i1 == n1:
            t.extend(t2[i2:])
        else:
            t.extend(t1[i1:])
        return t
    return fusion(mergesort(t[:m]),mergesort(t[m:]))


#slicing d'un tableau par colonnes
def slicing(tab,col):
    """
    Réalise le slicing d'un tableau par colonnes
    # Arguments
        tab : List of list.
            Tableau de valeurs
        col : Integer
            Indice de la colonne à sélectionner
            L'indice doit être inférieur à len(tab[0])
    # Returns
        List.
        Renvoit une liste correspondant à la colonne col du tableau tab
    """
    
    if col < len(tab[0]):
        B = []
        for k in range(len(tab)):
            B.append(tab[k][col])
        return(B)
    else:
        return("col is too large")
 

def text_to_tabular(content):
    """
    Transforme la liste content ne contenant qu'un seul élément en une liste de liste
    selon les séparteurs \n et \t
    # Arguments
        content : List. 
            Liste d'un seul élément
    # Returns
        List de list.
        Renvoit la liste séparée selon \n et \t
        
    """     
    cn = content[0].split('\n')
    n = len(cn)-1
    var_list = cn[0].split('\t')
    m = len (var_list)
   
    #initializing tabular tab
    arr = []
    for i in range(n):
        a = []
        for j in range(m):
            a.append(' ')
        arr.append(a)
    
    #putting the strings in the tabular
    for i in range(n):
        line_i = cn[i].split('\t')
        arr[i] = line_i

    return(arr)
    
#the dictionnary wil be sorted by occurence of words in the abstract   
def make_dictionary(data, max_words):
    """
    Renvoit un dictionnaire contenant les mots des textes
    de la colonnes 8 (qui correspond aux résumés)
    # Arguments
        data : List of list.
            Liste contenant toutes les données
        max_words : Integer.
            Taille maximale du dictionnaire
    # Returns
        Dict.
        Renvoit la liste séparée selon \n et \t
        
    """
    #TODO : faire en sorte que les mots avec des majuscules ne soient pas comptés deux fois
    #TODO : gérer la ponctuation
    #TODO : gérer les racines des mots (demander à Loic)
    #TODO : gérer les "how", "to"
    tab = slicing(data,8)
    dict = {}
    dict[None] = 0
    j = 1
    for study in tab:
        l = re.split('\s|[.()/\[\]=#]', study)
        for word in l:
            if word not in dict:
                dict[word] = j
                j = j + 1
                
    occurence = [[0,' ']]*(len(dict)-1)         
    for study in tab:
        l = re.split('\s|[.()/\[\]=#]', study)
        for word in l:
            index = dict.get(word)-1
            occurence[index] = [occurence[index][0] + 1, word]
            
    sorted_dict_list = mergesort(occurence)
    sorted_dict = {}
    for k in range(min(len(dict), max_words)):
        sorted_dict[sorted_dict_list[k][1]] = k
    return(sorted_dict)



def data_selection(data, Delai, Dictionary, maxlen):
    """
    # Arguments
        data : List of list.
            Liste contenant toutes les données
        Delai : Integer.
            Délai à partir duquel on censure les molécules non approuvées
        maxlen : Integer.
            Longueur maximale des résumés.
            Les résumés sont tronqués si leur taille dépasse cette valeur.
            Les résumés sont complétés par la valeur 0 si leur taille est inférieur à cette valeur
        Dictionary : Dict.
            Dictionaire utilsé pour indexer les mots.
            Si un mot n'appartient pas au dictionnaire, il est indexé par 0
    # Returns
        result : List of list.
            Chaque élément de la liste est une liste de taille maxlen
            contenant un résumé dont les mots sont indexés grâce au dictionnaire 
        approval : List.
            L'élément d'indice k de cette liste correspond à la valeur binaire
            de l'approbation par la FDA, dans le délai Delai, 
            de l'étude d'indice k dans la liste result   
        
    """
    delay = slicing(data,16)
    abstract = slicing(data,8)
    fda_approval = slicing(data,14)
    #transforme le texte en nombres grâce au dictionnaire
    nb_abstract = []
    for i in range(len(abstract)):
        l = re.split('\s|[.()/\[\]=#]', abstract[i])
        c = []
        if len(l)< maxlen:
            d = [None]*(maxlen - len(l))
            l = l + d
        else:
            l = l[:maxlen]
        for j in range(len(l)):
            key = Dictionary.get(l[j])
            if key == None:
                key = 0
            c.append(key)
        nb_abstract.append(c)
    
    result = []
    approval = []
    for k in range(1,len(data)):
        if float(delay[k]) > Delai :
            fda_approval[k] = 0
        result.append(nb_abstract[k])
        #result.append([names[k], abstract[k]])
        approval.append(int(fda_approval[k]))
    return result, approval


def test(training_samples = 10000,
         validation_samples = 2000,
         maxlen = 200,
         max_words = 10000,
         delai = 1000,
         epochs = 1,
         embedding_dim = 100,
         regularizer = 0.01,
         learning_rate = 0.001,
         batch_size = 10,
         width = 32,
         layers = 1,
         affichage = False):
    """
    # Arguments
        training_samples : Integer.
            Nombre de données utilisées pour l'entraînement du modèle
        validation_samples : Integer.
            Nombre de données utilisées pour la validation du modèle
        maxlen : Integer.
            Longueur maximale des résumés
        max_words : Integer
            Taille maximale du dictionnaire utilisé
        delai : Integer.
            Délai à partir duquel on censure les molécules non approuvées
        epochs : Integer.
            Nombre d'itération pendant l'entraînement du modèle
        embedding_dim : Integer.
            Dimension des vecteurs pour le Word Embedding.
            Doit être égal à 50, 100, 200 ou 300
        regularizer : Float.
            Coefficient de régularisation permettant de minimiser le problème d'overfitting
        learning_rate : Float.
            Coefficient d'apprentissage par lequel est multiplié le gradient pour mettre à jour les paramètres
        batch_size : Integer.
            Taille des échantillons sélectionnés pour les mises à jour des paramètres
        width : Integer.
            Largeur du réseau de neurones
        layers : Integer.
            Profondeur du réseau de neurone
        Affichage : Boolean.
            True : affiche la progression des résultats
            False : n'affiche rien
    # Returns
        None.
        Écrit dans le fichier Results.txt les paramètres d'entrée, les résultats obtenus et le temps de calcul
        
    """
    t1 = time()
    data_dir = '/Users/jerem/.anaconda/Codes'
    labels = []
    texts = []
    f = open(os.path.join(data_dir, 'data_clean.txt'), encoding = "ISO-8859-1")
    texts.append(f.read())
    f.close()
    #should be equal to 50, 100, 200 or 300
    #preprocessing the data
    data = text_to_tabular(texts)
    Dictionary = make_dictionary(data, max_words)
    Dictionary.pop('')
    Dictionary[None] = 0
    data, labels = data_selection(data, delai, Dictionary, maxlen)
    np_data = np.array(data)
    np_labels = np.array(labels)
    indices = np.arange(np_data.shape[0])
    np.random.shuffle(indices)
    np_data = np_data[indices]
    np_labels = np_labels[indices]
    
    #splitting the data in 3 parts : training, validation and testing
    x_train = np_data[:training_samples]
    x_val = np_data[training_samples: training_samples + validation_samples]
    x_test = np_data[training_samples + validation_samples:]
    y_train = np_labels[:training_samples]
    y_val = np_labels[training_samples: training_samples + validation_samples]
    y_test = np_labels[training_samples + validation_samples:]
    
    
    
    #Preprocessing the Glove information
    #Getting the embedding vector calculated by Glove for every word in our Dictionary
    #TODO : chercher des bases de données adaptées au vocabulaire médicale
    #TODO : Bonus: entraîner nous-mêmes un réseau avec les données fournies
    glove_dir = '/Users/jerem/Downloads/glove.6B'
    embeddings_index = {}
    f = open(os.path.join(glove_dir, 'glove.6B.{}d.txt'.format(embedding_dim)),encoding = 'utf8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, i in Dictionary.items():
        if i < max_words:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
     
    
               
    #Setting up the model
    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
    model.add(Flatten())
    for k in range(layers):
        model.add(Dense(width, kernel_regularizer=regularizers.l2(regularizer), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model.layers[0].set_weights([embedding_matrix])
    #model.layers[0].trainable = False
    
    
    #fixing of the display of the results in the Python console
    verbose = 0
    if affichage:
        verbose = 2
    
    
    #Training the model
    model.compile(optimizers.RMSprop(learning_rate), loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val), verbose = verbose)
    model.save_weights('pre_trained_model.txt')
    
    
    #Plotting the results in the Python console
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epoch = range(1, len(acc) + 1)
    plt.plot(epoch, acc, 'bo', label='Training acc')
    plt.plot(epoch, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epoch, loss, 'bo', label='Training loss')
    plt.plot(epoch, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
                
    #Testing the data and printing the results, depending on the hyperparameters
    loss,accuracy = model.evaluate(x_test, y_test)
    print("La valeur du loss sur les données test est : ",loss)
    print("La valeur de l'accuracy sur les données test est : ",accuracy)
    t2 = time()
    computing_time = t2 - t1
    print("HYPERPARAMETRES:","\n",
          "delay =", delai,"\n",
          "epochs =", epochs,"\n",
          "embedding_dim =", embedding_dim,"\n",
          "regularizer =", regularizer,"\n",
          "learning_rate =", learning_rate,"\n",
          "batch_size =", batch_size,"\n",
          "width =", width, "\n",
          "layers =", layers, "\n",
          "computing_time =", computing_time)
    
    #Saving the results in a text document
    file = open('Results.txt', 'a')
    file.write(str(maxlen))
    file.write("\t")
    file.write(str(max_words))
    file.write("\t")
    file.write(str(delai))
    file.write("\t")
    file.write(str(epochs))
    file.write("\t")
    file.write(str(embedding_dim))
    file.write("\t")
    file.write(str(regularizer))
    file.write("\t")
    file.write(str(learning_rate))
    file.write("\t")
    file.write(str(batch_size))
    file.write("\t")
    file.write(str(width))
    file.write("\t")
    file.write(str(layers))
    file.write("\t")
    file.write(str(maxlen))
    file.write("\t")
    file.write(str(max_words))
    file.write("\t")
    file.write("\t")
    file.write(str(round(loss,4)))
    file.write("\t")
    file.write(str(round(accuracy,4)))
    file.write("\t")
    file.write(str(round(computing_time,1)))
    file.write("\n")
    file.close()
    
    
    #root = Tk()
    #my_text = Label(root, text='Calcul fini !')
    #my_text.pack()
    #root.geometry("500x250+50+50")
    #root.mainloop()
    
