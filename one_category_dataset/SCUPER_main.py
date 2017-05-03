# -*- coding: utf-8 -*-
"""
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
import pickle

items_path = r'items.txt'
users_path = r'users.txt'
ratings_train_path = r'nightlife_training.txt'
ratings_test_path = r'nightlife_test.txt'
subcategories_path = r'subcategories_list.txt'


eps = 10               # epsilon 

#mode = 0               # Simple MF model (No social graph)  
#mode = 1               #  UI (Q) + MF
#mode = 2               #  II (S) + MF
#mode = 3               #  IS (W) + MF
# mode = 4               #  UI + II  (Q,S)
# mode = 5               #  UI + IS  (Q,W)
# mode = 6               #  II + IS  (S,W)
mode = 7               #  ALL   (Q,S,W)

print 'Mode:', mode

items_df = pd.read_table(items_path, delimiter = r'::', engine='python')
users_df = pd.read_table(users_path, delimiter = r':', engine='python')

#ratings_df = pd.read_table(ratings_path, delimiter = r'::', engine='python')

ratings_train_df = pd.read_table(ratings_train_path, delimiter = r'::', engine='python')
ratings_test_df = pd.read_table(ratings_test_path, delimiter = r'::', engine='python')

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
for user, group in ratings_train_df.groupby('user_id'):
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

R_test = np.empty([len(users_df), len(items_df)], dtype = np.float16)
I_test = np.empty([len(users_df), len(items_df)], dtype = np.float16)


num_ratings_train = 0;        
for user, item, rtng in zip(ratings_train_df['user_id'], ratings_train_df['item_id'], ratings_train_df['rating']):
    R[user, item] = rtng
    I[user, item] = 1
    num_ratings_train +=1;

    
num_ratings_test = 0;
for user, item, rtng in zip(ratings_test_df['user_id'], ratings_test_df['item_id'], ratings_test_df['rating']):
    R_test[user, item] = rtng
    I_test[user, item] = 1
    num_ratings_test +=1;
###############################################################################

#Parameters for Gradient Descent
global lamda, beta, gamma, eta, l
k = 10                                  #dimension of latent space
lamda = 0.1
beta = 30
gamma = 30
eta = 30
l = 0.000006

np.random.seed(0)
U = 0.1 * np.random.randn(users, k)
P = 0.1 * np.random.randn(items, k)

######################################################
U = pickle.load( open( "U_0.pckl", "rb" ) )
P = pickle.load( open( "P_0.pckl", "rb" ) )
########################################################

#r = np.mean(ratings_df['rating'])
r = np.empty([len(users_df), len(items_df)], dtype = np.float16)
for user, rating_group in ratings_train_df.groupby('user_id'):
    r[user][:] = np.mean(rating_group['rating'])
###############################################################################

#Gradient Descent

def cal_pred_rating(r, U, P):
    return (r + np.matmul(U, P.transpose()))
    
def cal_error_der_P(I_, R_, U_, H_, Q_, P_):
    first_fac = 0
    second_fac = 0
    third_fac = 0
    
    first_fac = np.matmul(np.multiply(I_.transpose(), R_.transpose()), U_)
    
    second_fac = lamda * P_
    
    if(mode in [1,4,5,7]):
        third_fac = eta * np.matmul(np.multiply(np.multiply(I_.transpose(), np.matlib.repmat(H_, items, 1)), 
                             (np.subtract((np.matmul(U_, P_.transpose())), Q_)).transpose()), U_)
    
    return first_fac + second_fac + third_fac


R_cap = cal_pred_rating(r, U, P)

#print 'Rcap:', R_cap

def cal_error_der_U(I_, R_, P_, H_, Q_, U_, W_, S_):
    
    first_fac = 0
    second_fac = 0
    third_fac = 0
    fourth_fac = 0
    fifth_fac = 0
    sixth_fac = 0
    seventh_fac = 0
    
 
    first_fac = np.matmul(np.multiply(I_, R_), P_)
    
    second_fac = lamda * U_
    
    if(mode in [2,4,6,7]):
        third_fac = beta * np.subtract(U_, np.matmul(S_,U_))
 
    if(mode in [2,4,6,7]):
        fourth_fac = -beta *  np.matmul(S_.transpose(),np.subtract(U, np.matmul(S,U))) 
    
    if(mode in [3,5,6,7]):
        fifth_fac = gamma * np.subtract(U_, np.matmul(W_,U_))
    
    if(mode in [3,5,6,7]):
        sixth_fac = -gamma * np.matmul(W_.transpose(),np.subtract(U, np.matmul(W,U)))
    
    if(mode in [1,4,5,7]):
        seventh_fac = eta * np.matmul(np.multiply(np.multiply(I_, (np.matlib.repmat(H_, items, 1)).transpose()), 
                             np.subtract((np.matmul(U_, P_.transpose())), Q_)), P_)
        
    return first_fac + second_fac + third_fac + fourth_fac + fifth_fac + sixth_fac +seventh_fac
    
#error_der_U = map(cal_error_der_U, I, (R_cap - R), [P] * users, H.transpose(), Q, U)

def cal_error_fn(R, R_cap, H, Q, U, P, S, W, I):
    
    first_fac = 0
    second_fac = 0
    third_fac = 0
    fourth_fac = 0
    fifth_fac = 0
    
    
    first_fac = np.sum(np.sum(np.multiply(R - np.multiply(R_cap,I), R - np.multiply(R_cap,I)), axis = 1), axis = 0) / float(2)
                      
    second_fac = (lamda/float(2)) *(np.linalg.norm(U,ord='fro') + np.linalg.norm(P,ord='fro'))
    
    if(mode in [2,4,6,7]):
        third_fac_temp = np.subtract(U, np.matmul(S,U))
        third_fac = (beta/float(2)) * ((np.linalg.norm(third_fac_temp,ord='fro'))**2)
    
    if(mode in [3,5,6,7]):
        fourth_fac_temp = np.subtract(U, np.matmul(W,U))
        fourth_fac = (gamma/float(2)) * ((np.linalg.norm(fourth_fac_temp,ord='fro'))**2)
    
    if(mode in [1,4,5,7]):
        fifth_fac_temp = np.subtract(Q, np.matmul(U, P.transpose()))
        fifth_fac = (eta/float(2)) * np.sum(np.sum(np.multiply((np.matlib.repmat(H, items, 1)).transpose(), np.multiply(fifth_fac_temp, fifth_fac_temp)), axis = 1), axis = 0)
    
        
    #print 'first:',first_fac , 'second:', second_fac, 'fifth:', fifth_fac
    
    return first_fac + second_fac + third_fac + fourth_fac + fifth_fac

print(cal_error_fn(R, R_cap, H, Q, U, P, S,W,I))
del users_df
del items_df
del ratings_train_df


U = U.astype(dtype = np.float16)
P = P.astype(dtype = np.float16)

t=0

identifier = (np.arange(users)).transpose() 

Error_train_old = 100000
Error_test_old = 100000

Error_list_train = []
Error_list_test = []

RMSE_list_train = []
RMSE_list_test = []

MAE_list_train = []
MAE_list_test = []


while(t<10000):
        print t
        error_der_U = cal_error_der_U(I, (R_cap - R), P, H, Q, U, W, S)        
        error_der_P = cal_error_der_P(I, (R_cap - R), U, H, Q, P)
        
        U = np.subtract(U, np.multiply(l, error_der_U))
        
        for i in range(users):
            for j in range(k):
                P[i][j] -= (l * error_der_P[i][j])
                
        R_cap = cal_pred_rating(r, U, P)
        
         
        print 'before RMSE' 
        RMSE_test = np.sqrt(np.sum(np.square(np.subtract(np.multiply(I_test,R_cap),R_test)))/float(num_ratings_test))
        RMSE_train = np.sqrt(np.sum(np.square(np.subtract(np.multiply(I,R_cap),R)))/float(num_ratings_train))
        
        MAE_test = np.sum(np.abs(np.subtract(np.multiply(I_test,R_cap),R_test))/float(num_ratings_test))
        MAE_train = np.sum(np.abs(np.subtract(np.multiply(I,R_cap),R))/float(num_ratings_train))
        
        print 'RMSE_train:', RMSE_train, 'RMSE_test', RMSE_test
        print 'MAE_train:', MAE_train, 'MAE_test', MAE_test
        #print R_cap
        
        Error_train = cal_error_fn(R, R_cap, H, Q, U, P,S,W,I)
        Error_test = cal_error_fn(R_test, R_cap, H, Q, U, P,S,W,I_test)
        
        print 'Error_train:', Error_train ,'Error_test:', Error_test
        
        if t>0:
            
            if (Error_test-Error_test_old > 0):
                if (Error_train-Error_train_old > 0):
                    print 'Decrease learning rate!'
                    sys.exit(0)
                else:
                   print 'Exiting because of overfitting'
                
                
            if (Error_train-Error_train_old > 0):
                print 'Decrease learning rate!'
                sys.exit(0)
                
            if ((Error_train_old-Error_train)<eps):
                print 'Converged!'
            #    sys.exit(0)    
                
        Error_train_old = Error_train
        Error_test_old = Error_test
        
        Error_list_train.append(Error_train)
        Error_list_test.append(Error_test)
        
        RMSE_list_train.append(RMSE_train)
        RMSE_list_test.append(RMSE_test)
        
        MAE_list_train.append(MAE_train)
        MAE_list_test.append(MAE_test)
        
        
        t +=1
    
    
####################################  RESULTS   #######################################################

'''
Mode:0
    l = 0.007 
    
    RMSE_train: 0.961620057028 RMSE_test 1.02542830799
    Error_train: 36304.7856541 Error_test: 11235.4794905
    

Mode:1
    l= 0.00007
    
    RMSE_train: 0.968688415112 RMSE_test: 1.02522384879
    Error_train: 554949348.92 Error_test: 554923739.508
    
Mode:2
    l = 0.000005 ---> Good convergence
    l = 0.000007 ---> stops after 2 iterations
    
    l = 0.000005   (200 iterations)
    
    RMSE_train: 0.964004723038 RMSE_test 1.02523957393
    Error_train: 49461.2529413 Error_test: 24207.5454704

Mode:3 
    l = 
        
    

    

    
'''