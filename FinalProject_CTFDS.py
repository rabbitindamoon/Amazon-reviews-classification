# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 00:46:42 2018
@author: s171839
"""
#from bitarray import bitarray
#import mmh3
from __future__ import division
import nltk
from bz2 import BZ2File as bzopen
import math 
import string
import re
import numpy as np
from Bloom_Filter import Bloom_Filter
import time
#nltk.download()


# reading files and saving it into lines as byte type, need to decode it.
def bz2_file(filename):
    with bzopen(filename, "r") as bzfin:
        """ Handle lines here """
        lines = []
        for i ,line in enumerate(bzfin):        
            lines.append(line.rstrip())
        return lines
    


#separating the file into good and bad reviews.
#also removing the __label__1, __label__2 from each string.
def remove_labels(lines):
    good=[]
    bad=[]
    for l in lines:
        decoded = l.decode("utf-8")    
        if "label__2" in decoded:
            rep = decoded.replace("__label__2", '')
            good.append(rep)
        elif "label__1" in decoded:
            rep = decoded.replace("__label__1", '')
            bad.append(rep)
    return good, bad


def clean_shingles(document):
    pun = list(string.punctuation)
    words_to=[' ','','?','´´','"', '``']
    words_to_remove = pun + words_to
    #curShingleID = 0
    #docsAsShingleSets = {};
    #t0 = time.time()
    #shinglesInDoc = set()
    shingles=[]
    for i in range(0, len(document)):
        st = document[i]
        #words = st.split(" ")
        tokens = nltk.word_tokenize(st)
        lower = [token.lower() for token in tokens]
        terms = [term for term in lower if term not in words_to_remove]
        for index in range(0, len(terms) - 2):
            # Construct the shingle text by combining three words together.
            shingle = terms[index] + " " + terms[index + 1] + " " + terms[index + 2]
            # Hash the shingle to a 32-bit integer.
    #        crc = binascii.crc32(shingle) & 0xffffffff
            # Add the hash value to the list of shingles for the current document. 
            # Note that set objects will only add the value to the set if the set doesn't already contain it. 
    #        shinglesInDoc.add(crc)
    #    docsAsShingleSets[i] = shinglesInDoc            
            shingles.append(shingle)            
    return shingles  

#This was created to check without the bloom-filter, no longer relevant.
'''
def count_elements(lista, x): 
'''
#    Checking through the list of all shingles in order to see if element is present, time consuming if done for many reviews. 
'''
    start_time = time.time()
    count = 0
    app_big = []
    app_small = []
    for element in lista:
        for ele in x:
            if (element == ele): 
                count = count + 1
                app_big.append(element)
                app_small.append(ele)
    total_time =  time.time() - start_time
    return count, app_big, app_small, total_time
'''      
   
def bf(shingles, p):
    '''
    creating bloom filter table with all elements given in todas.
    '''
    start_time = time.time()
    p=p
    n = len(shingles) #no of items to add 
    #p = 0.01 #false positive probability 
  
    bloomf = Bloom_Filter(n,p)   
    for item in shingles: 
        bloomf.add(item) 
    
    total_time =  time.time() - start_time           
    return bloomf, total_time

def bf_checker(bloomf_buena, bloom_mala, una):
    '''
    Creating bloom filter checker, to see if element might be inside the bloom filter table 
    '''
    
    start_time = time.time()

    count_buena = 0
    count_mala = 0
    prediction= str()
    
    for shingle in una: 
        if bloomf_buena.check(shingle): 
            count_buena+=1
 
    for shingle in una: 
        if  bloom_mala.check(shingle): 
            count_mala+=1
            
    if count_buena>count_mala:
        prediction="good_review"
    if count_mala>count_buena:
        prediction="bad_review"
    total_time =  time.time() - start_time     
    return count_buena, count_mala, prediction, total_time 

def up_train_test(filename):
    '''
    uploading file and dividing the data into train and test.
    '''
    lines = bz2_file(filename)
    train = lines[0:350000]
    test = lines[350000:len(lines)]
    return train, test


# uploading all train data, cleaning it, creating shingles, and making bloomfilter list with each shingle.  
def tokenize(filename):
    '''
    clean the train reviews, and give 2 separte shingles strings list.
    '''
    good, bad = remove_labels(filename)
    good_shingles = clean_shingles(good)
    bad_shingles = clean_shingles(bad)
    total_good_shingles = len(good_shingles)
    total_bad_shingles = len(bad_shingles)
    
    return good_shingles, bad_shingles, total_good_shingles, total_bad_shingles 

def train_reviews(good_shingles, bad_shingles, p):  
    '''
    Takes the train reviews and removes the labels as well as divides the good reviews and bad reviews.
    Cleans the reviews and again separtes them into shingles of 3 words.
    It then creates a bloom filter table with the shingles for both good and bad reviews and returns them.
    '''     
    #lines = bz2_file(filename)
    start_time = time.time()    
#    good, bad = remove_labels(filename)
#    good_shingles = clean_shingles(good)
#    bad_shingles = clean_shingles(bad)
    bloomf_good, tt_bf_good = bf(good_shingles, p)
    bloomf_bad, tt_bf_bad = bf(bad_shingles, p)
   # total_good_shingles = len(good_shingles)
   # total_bad_shingles = len(bad_shingles)
    
    total_time =  time.time() - start_time 
    return bloomf_good, bloomf_bad, total_time


##################################

def test_reviews_upload(filename):
    '''
    creating the lsit of good and bad reviews in order to check them if they are good or bad.
    '''
    good_test, bad_test = remove_labels(filename) 
    return good_test, bad_test


###
#dealing with one review at the time, thus creating shingles of one review in order to check it in hashed table of shingles. 
def one_review(onereview):
    '''
    breaking, cleaning and saving into shingles one review at the time.
    '''
    #review_good_test = good[202]   #this was used as an example, debugging.
    shingle_test = []
    shingles_test = [] 
    pun = list(string.punctuation)
    words_to=[' ','','?','´´','"', '``']
    words_to_remove = pun + words_to
    tokens_test = re.sub("[^\w]", " ",  onereview)
    tokens_test2 = nltk.word_tokenize(tokens_test)       
    lower_test = [token.lower() for token in tokens_test2]
    terms_test = [term for term in lower_test if term not in words_to_remove]
    for index in range(0, len(terms_test) - 2):
    # Construct the shingle text by combining three words together.
        shingle_test = terms_test[index] + " " + terms_test[index + 1] + " " + terms_test[index + 2]
        shingles_test.append(shingle_test) 
    return shingles_test
   
    '''
#this is to check review with brute force(checking all train shingles (good, bad) ato check which one has a higher value.)
#time taken is huge.
how_many_good, big_good, small_good, tt_good = count_elements(good_shingles, shingles_test)
how_many_bad, big_bad, small_bad, tt_bad  = count_elements(bad_shingles, shingles_test)
##
count_good = count_elements(good_shingles, shingles_test)
count_bad = count_elements(bad_shingles, shingles_test)
    '''


def test_reviews(bloomf_good, bloomf_bad, test_reviews, known_type):
    '''
    test_reviews takes the bit array created for the bad and good revies and also takes the test reviews.
    in order to see how it performs, it can also be used for new reviews. it outputs a tab prediction for the test data.
    It saves a one for each 
    
    is divided into 3.
    if you know reviews are good and bad then need to add to known_type good or bad else it will assume you dont know and will give you a one
    tab prediction for a positive review and will be a zero if negrative. and will appen all negative reviews.
    
    output:   tab_prediction: bit array of zeros and ones.
              indexes: index of mislabled reviws, in case that is the review type is known, otherwise index of all negative reviews.
    '''
    
    start_time = time.time()    

    tab_prediction = np.zeros(len(test_reviews))
    indexes = []
    counter=0
    
    if known_type == "good":
        for review in test_reviews:
            review_shingle_test = one_review(review)
            #checking if review is good or bad
            c_g, c_b, prediciton, tt_bf_prediction = bf_checker(bloomf_good, bloomf_bad, review_shingle_test)
            if prediciton == "good_review":
                tab_prediction[counter] = 1
            else:
                indexes.append(counter)
                
            counter += 1
            
    elif known_type == "bad":
        for review in test_reviews:
            review_shingle_test = one_review(review)
            #checking if review is good or bad
            c_g, c_b, prediciton, tt_bf_prediction = bf_checker(bloomf_good, bloomf_bad, review_shingle_test)
            if prediciton == "bad_review":
                tab_prediction[counter] = 1
            else:
                indexes.append(counter)
                
            counter += 1
            
    else:
        for review in test_reviews:
            review_shingle_test = one_review(review)
            #checking if review is good or bad
            c_g, c_b, prediciton, tt_bf_prediction = bf_checker(bloomf_good, bloomf_bad, review_shingle_test)
            if prediciton == "good_review":
                tab_prediction[counter] = 1
            else:
                indexes.append(counter)
                
            counter += 1    
            
    total_time =  time.time() - start_time    
    
    return tab_prediction, indexes, total_time


#######################################################################################
'''
The part below is used to run the code. 
run the entire code and have the files added to it in the same directory.
'''

#def running_all(filename):
train, test = up_train_test("test.ft.txt.bz2") # uploading file and separating a portion of the reviews into train a test.
good_shingles, bad_shingles = tokenize(train)
bloomf_good, bloomf_bad, total_time_BloomFilterTable_both = train_reviews(good_shingles, bad_shingles, 0.01) # creating the bloom filter tables. they wont appear in the variable explorer table.
#file_test = bz2_file("train.ft.txt.bz2")
good_test_reviews, bad_test_reviews = test_reviews_upload(test)   # dividing the test reviews. total 50k reviews, approximate  25k good & 25k bad.


#checking if review is good or bad
#c_g, c_b, prediciton, tt_bf_prediction = bf_checker(bloomf_good, bloomf_bad, test_shingle_review)
tab_prediction_good, indexes_good_miss, tt_g = test_reviews(bloomf_good, bloomf_bad, good_test_reviews, "good") #this is for all the good reviews
tab_prediction_bad, indexes_bad_miss, tt_b = test_reviews(bloomf_good, bloomf_bad, bad_test_reviews, "bad") #this is for all the bad reviews
#########################################################################################





'''
This is created in order to check new review, if user would like to check a new review)
    PLEASE UNCOMMENT LINEs BELOW IN ORDER TO USE FOR A NEW REVIEW, THIS NEW REVIEW HAS TO BE A STRING. After saving it in NEW_REVIEW run
    below it to see if comment is positive or negative. 
    new index will 1 if positive or zero if negatice.
'''
#NEW_REVIEW= str("")
#new_tab_prediction, new_indexe, NEW_tt = test_reviews(bloomf_good, bloomf_bad, NEW_REVIEW, "UNKOWN") 









#########################################################################################
#This part was used only to create the graphs and tables in order to compare different values of p.
'''
#in order to create a table with both P VALUES AND 
train, test = up_train_test("test.ft.txt.bz2") # uploading file and separating a portion of the reviews into train a test.
good_shingles, bad_shingles, t_g_s, t_b_s = tokenize(train)
good_test_reviews, bad_test_reviews = test_reviews_upload(test)

#add
p= [0.01, 0.03, 0.05, 0.07, 0.09]  # p to create bloom filter.
times_add = []
good_rev_number = len(good_test_reviews)
bad_rev_number = len(bad_test_reviews)
#check
times_check_g=[]
times_check_b=[]
accuracy_good = []
accuracy_bad = []
bit_g=[]
bit_b=[]
hash_g=[]
hash_b=[]
for item in p:
    #add#add#add
    bloomf_good, bloomf_bad, total_time_add = train_reviews(good_shingles, bad_shingles,item)
    times_add.append(total_time_add)
    
    #check#check#check
    tab_prediction_good, indexes_good_miss, time_check_g = test_reviews(bloomf_good, bloomf_bad, good_test_reviews, "good") #this is for all the good reviews
    tab_prediction_bad, indexes_bad_miss, time_check_b = test_reviews(bloomf_good, bloomf_bad, bad_test_reviews, "bad") #this is for all the bad reviews    
    times_check_g.append(time_check_g)
    times_check_b.append(time_check_b)
    #need to cehck accuracy...and save it.
    ag = (1 - (len(indexes_good_miss)/good_rev_number))*100
    ab = (1 - (len(indexes_bad_miss)/bad_rev_number))*100
    accuracy_good.append(ag)
    accuracy_bad.append(ab)
    #check size bit array.
    b_g = int(-(t_g_s * math.log(item))/(math.log(2)**2))
    b_b = int(-(t_b_s * math.log(item))/(math.log(2)**2))
    bit_g.append(b_g)
    bit_b.append(b_b)
    #check how many hash functions are needed.
    h_g= int((b_g/item) * math.log(2))
    h_b= int((b_b/item) * math.log(2))
    hash_g.append(h_g)
    hash_b.append(h_b)
    

####################
#graph
import pandas as pd
import matplotlib.pyplot as plt   

###########################
# Plot for good reviews
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Probability of False Positive p')
ax1.set_ylabel('Accuracy', color=color, fontsize=16)
ax1.plot(p, accuracy_good, '-o', color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Time', color=color, fontsize=16)  # we already handled the x-label with ax1
ax2.plot(p, times_check_g, '-o', color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.title('Accuracy(good rev) vs Time for Prob.', fontsize=14)
plt.show()
#########################
# Plot for badd reviews
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Probability of False Positive p')
ax1.set_ylabel('Accuracy', color=color, fontsize=16)
ax1.plot(p, accuracy_bad, '-o', color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Time', color=color, fontsize=16)  # we already handled the x-label with ax1
ax2.plot(p, times_check_b, '-o', color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.title('Accuracy(bad rev) vs Time for Prob.', fontsize=14)
plt.show()



################################
#tables:
d_add = {'p-value': p, 'bloom-Filter-Add': times_add}
df_add = pd.DataFrame(data=d_add)
d_bit= {'p-value': p, 'Bit-Array-size-Good': bit_g, 'Bit-Array-size-Bad': bit_b }   
df_bit =  pd.DataFrame(data=d_bit)

d_hash = {'p-value': p, 'Hash-functions-Good': hash_g, 'Hash-Functions-Bad': hash_b} 
df_hash =  pd.DataFrame(data=d_hash)
'''