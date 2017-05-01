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


items_path = r'E:\Users\Dell\Desktop\STUDY_SPRING 17\CSCE 670 Information Storage and Retreival\Project\nightlife\items.txt'
users_path = r'E:\Users\Dell\Desktop\STUDY_SPRING 17\CSCE 670 Information Storage and Retreival\Project\nightlife\users.txt'
ratings_path = r'E:\Users\Dell\Desktop\STUDY_SPRING 17\CSCE 670 Information Storage and Retreival\Project\nightlife\ratings.txt'
subcategories_path = r'E:\Users\Dell\Desktop\STUDY_SPRING 17\CSCE 670 Information Storage and Retreival\Project\nightlife\subcategories_list.txt'


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
H = normalize(H, norm ='l2')

# 1) Q (m * n) = Relevance matrix of user 'u' to topic of item 'i'

Q = np.nan_to_num(1 - scipy.spatial.distance.cdist(list(users_df['topic_distribution']), list(items_df['topic_distribution']), 'cosine'))

# 2) W (m * m) = Similarity matrix of user 'u' to topic of user 'v'

W = np.nan_to_num(1 - scipy.spatial.distance.cdist(list(users_df['topic_distribution']), list(users_df['topic_distribution']), 'cosine'))

# 3) S (m * m) = Trust of user 'u' on user 'v'
S = np.zeros((users, users))
for i in range(users):
    for user, friend in zip([i] * len(users_df['friends'][i]), users_df['friends'][i]):
        S[user, friend] = 1
    
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
l = 0.5

U = 0.1 * np.random.randn(users, k)
P = 0.1 * np.random.randn(items, k)

r = np.mean(ratings_df['rating'])

###############################################################################
start = time.time()
#Gradient Descent

def cal_pred_rating(r, U, P):
    return (r + np.matmul(U, P.transpose()))
    
def cal_error_der_P(I_i, R_i, U_, H_, Q_i, P_i): 
    first_fac = np.matmul(np.multiply(I_i, R_i), U_)
    second_fac = lamda * P_i
    third_fac = eta * np.matmul(np.multiply(np.multiply(I_i, H_), 
                             np.subtract((np.matmul(U_, P_i.transpose())).transpose(), Q_i)), U_)
    return first_fac + second_fac + third_fac


R_cap = cal_pred_rating(r, U, P)
error_der_P = map(cal_error_der_P, I.transpose(), (R_cap - R).transpose(),
                  [U] * items, [H] * items, Q.transpose(), P)

def cal_error_der_U(I_u, R_u, P_, H_, Q_u, U_):
    first_fac = np.matmul(np.multiply(I_u, R_u), P_)
    second_fac = lamda * U_
    third_fac = eta * np.matmul(np.multiply(np.multiply(I_u, H_), 
                             np.subtract((np.matmul(U_, P_.transpose())), Q_u)), P_)
    return first_fac + second_fac + third_fac
    
error_der_U = map(cal_error_der_U, I, (R_cap - R), [P] * users, H.transpose(), Q, U)

def cal_error_fn(R, R_cap, H, Q, U, P):
    first_fac = np.sum(np.sum(np.multiply(R - R_cap, R - R_cap), axis = 1), axis = 0) / 2
    temp = np.subtract(Q, np.matmul(U, P.transpose()))
    second_fac = 15 * np.sum(np.sum(np.multiply((np.matlib.repmat(H, items, 1)).transpose(), np.multiply(temp, temp)), axis = 1), axis = 0)
    return (first_fac + second_fac)

print(cal_error_fn(R, R_cap, H, Q, U, P))
del users_df
del items_df
del ratings_df
U = U.astype(dtype = np.float16)
P = P.astype(dtype = np.float16)

temp = np.subtract(U, np.multiply(l, error_der_U))
U = temp
#temp = np.subtract(P, np.multiply(l, error_der_P))
#P = temp
for i in range(21337):
    for j in range(10):
        P[i][j] -= (l * error_der_P[i][0][j])
        
R_cap = cal_pred_rating(r, U, P)
print(cal_error_fn(R, R_cap, H, Q, U, P))


print('It took {0:0.2f} seconds'.format(time.time() - start))
