# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 12:35:06 2017

@author: Aniket
"""

import pandas as pd
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")
import time
import scipy
from sklearn.preprocessing import normalize
from itertools import cycle
import sys

items_path = r'items.txt'
users_path = r'users.txt'
ratings_path = r'ratings.txt'
subcategories_path = r'subcategories_list.txt'


items_df = pd.read_table(items_path, delimiter = r'::', engine='python')
users_df = pd.read_table(users_path, delimiter = r':', engine='python')
ratings_df = pd.read_table(ratings_path, delimiter = r'::', engine='python')

items = len(items_df)
users = len(users_df)

f = open(subcategories_path)
subcategories = [x.strip() for x in f.readlines()]
                 
def friends_str2list(str_friends):
    return map(int, re.sub(r'{', '', str_friends).split(',')[:-1])

def space2underscore(string):
    return re.sub(r' ', '_', string)
    
def topic_dist(sub_category_list, categories):
    tpc_dist = [0]*len(categories)
    for i in range(len(tpc_dist)):
        if (categories[i] in sub_category_list):
            tpc_dist[i] = 1
    return tpc_dist
       
    
users_df['friends'] = map(friends_str2list, users_df['friends'])
subcategories = map(space2underscore, subcategories)
items_df['sub_category'] = map(space2underscore, items_df['sub_category'])
items_df['topic_distribution'] = map(topic_dist, items_df['sub_category'], [subcategories]*(len(items_df['sub_category'])))
#users_df['topic_distribution'] = [[0]*18]*len(users_df['topic_distribution'])

nil_tpc_dist_items = []
for i in range(len(items_df['topic_distribution'])):
    if(items_df['topic_distribution'][i] == [0] * 18):
        nil_tpc_dist_items.append(i)

###############################################################################

#Building Matrices
#m = number of users, n = number of items

temp, H = [], []
for user, group in ratings_df.groupby('user_id'):
    temp.append(np.mean(list(items_df.loc[list(group['item_id'])]['topic_distribution']), axis = 0))
    H.append(len(group))
    
users_df['topic_distribution'] = temp

#H = np.matrix(H[0])        
H = normalize(H[0], norm ='l1')


# 1) Q (m * n) = Relevance matrix of user 'u' to topic of item 'i'

Q = np.nan_to_num(1 - scipy.spatial.distance.cdist(list(users_df['topic_distribution']), list(items_df['topic_distribution']), 'cosine'))



# 2) S (m * m) = Trust of user 'u' on user 'v'
S = np.zeros((users, users))
S_sym = np.zeros((users, users))

for i in range(users):
    for user, friend in zip([i] * len(users_df['friends'][i]), users_df['friends'][i]):
        S[user, friend] = 1
        S_sym[user,friend] = 1
        S_sym[friend,user] = 1

# 3) W (m * m) = Similarity matrix of user 'u' to topic of user 'v'
W = np.nan_to_num(1 - scipy.spatial.distance.cdist(list(users_df['topic_distribution']), list(users_df['topic_distribution']), 'cosine'))
W = np.multiply(S_sym,W)  
W = normalize(W, norm='l2')  

R = np.empty([len(users_df), len(items_df)], dtype = np.float16)
I = np.empty([len(users_df), len(items_df)], dtype = np.float16)
        
for user, item, rtng in zip(ratings_df['user_id'], ratings_df['item_id'], ratings_df['rating']):
    R[user, item] = rtng
    I[user, item] = 1

###############################################################################

#Parameters for Gradient Descent
global lamda, beta, gamma, eta, l
k = 10                                  #dimension of latent space
lamda = 0.1
beta = 30
gamma = 30
eta = 30
l = 0.000005

U = 0.1 * np.random.randn(users, k)
P = 0.1 * np.random.randn(items, k)

#r = np.mean(ratings_df['rating'])
r = np.empty([len(users_df), len(items_df)], dtype = np.float16)
for user, rating_group in ratings_df.groupby('user_id'):
    r[user][:] = np.mean(rating_group['rating'])
###############################################################################

#Gradient Descent

def cal_pred_rating(r, U, P):
    return (r + np.matmul(U, P.transpose()))
    
def cal_error_der_P(I_, R_, U_, H_, Q_, P_): 
    
    first_fac = np.matmul(np.multiply(I_.transpose(), R_.transpose()), U_)
    
    second_fac = lamda * P_
    
    third_fac = eta * np.matmul(np.multiply(np.multiply(I_.transpose(), np.matlib.repmat(H_, items, 1)), 
                             (np.subtract((np.matmul(U_, P_.transpose())), Q_)).transpose()), U_)
    
    return first_fac + second_fac + third_fac


R_cap = cal_pred_rating(r, U, P)

#print 'Rcap:', R_cap

def cal_error_der_U(I_, R_, P_, H_, Q_, U_, W_, S_):
    
    first_fac = np.matmul(np.multiply(I_, R_), P_)
    
    second_fac = lamda * U_
    
    third_fac = beta * np.subtract(U_, np.matmul(S_,U_))
    
    fourth_fac = -beta *  np.matmul(S_.transpose(),np.subtract(U, np.matmul(S,U))) 
    
    fifth_fac = gamma * np.subtract(U_, np.matmul(W_,U_))
    
    sixth_fac = -gamma * np.matmul(W_.transpose(),np.subtract(U, np.matmul(W,U)))
    
    seventh_fac = eta * np.matmul(np.multiply(np.multiply(I_, (np.matlib.repmat(H_, items, 1)).transpose()), 
                             np.subtract((np.matmul(U_, P_.transpose())), Q_)), P_)
        
    return first_fac + second_fac + third_fac + fourth_fac + fifth_fac + sixth_fac +seventh_fac
    
#error_der_U = map(cal_error_der_U, I, (R_cap - R), [P] * users, H.transpose(), Q, U)

def cal_error_fn(R, R_cap, H, Q, U, P, S, W, I):
    
    first_fac = np.sum(np.sum(np.multiply(R - np.multiply(R_cap,I), R - np.multiply(R_cap,I)), axis = 1), axis = 0) / float(2)
                      
    second_fac = (lamda/float(2)) *(np.linalg.norm(U,ord='fro') + np.linalg.norm(P,ord='fro'))
    
    third_fac_temp = np.subtract(U, np.matmul(S,U))
    third_fac = (beta/float(2)) * ((np.linalg.norm(third_fac_temp,ord='fro'))**2)
        
    fourth_fac_temp = np.subtract(U, np.matmul(W,U))
    fourth_fac = (gamma/float(2)) * ((np.linalg.norm(fourth_fac_temp,ord='fro'))**2)
        
    fifth_fac_temp = np.subtract(Q, np.matmul(U, P.transpose()))
    fifth_fac = (eta/float(2)) * np.sum(np.sum(np.multiply((np.matlib.repmat(H, items, 1)).transpose(), np.multiply(fifth_fac_temp, fifth_fac_temp)), axis = 1), axis = 0)
    
    ####################################
    #third_fac = 0
    #fourth_fac = 0
    ####################################
    
    print 'first:',first_fac , 'second:', second_fac, 'fifth:', fifth_fac
    
    return first_fac + second_fac + third_fac + fourth_fac + fifth_fac

print(cal_error_fn(R, R_cap, H, Q, U, P, S,W,I))
del users_df
del items_df
del ratings_df


U = U.astype(dtype = np.float16)
P = P.astype(dtype = np.float16)

t=0

identifier = (np.arange(users)).transpose() 

while(t<100):
        print t
        #start = time.time()
        error_der_U = cal_error_der_U(I, (R_cap - R), P, H, Q, U, W, S)
        #print('It took {0:0.2f} seconds'.format(time.time() - start))
        
        #start1 = time.time()
        error_der_P = cal_error_der_P(I, (R_cap - R), U, H, Q, P)
        #print('It took {0:0.2f} seconds'.format(time.time() - start1))
        
        U = np.subtract(U, np.multiply(l, error_der_U))
        
        #temp = np.subtract(P, np.multiply(l, error_der_P))
        #P = temp
        for i in range(users):
            for j in range(k):
                P[i][j] -= (l * error_der_P[i][j])
                
        R_cap = cal_pred_rating(r, U, P)
        
        #print R_cap
        
        print(cal_error_fn(R, R_cap, H, Q, U, P,S,W,I))
        t +=1
    
    

