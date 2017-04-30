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


items_path = r'items.txt'
users_path = r'users.txt'
ratings_path = r'ratings.txt'
subcategories_path = r'subcategories_list.txt'


items_df = pd.read_table(items_path, delimiter = r'::', engine='python')
users_df = pd.read_table(users_path, delimiter = r':', engine='python')
ratings_df = pd.read_table(ratings_path, delimiter = r'::', engine='python')

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

temp = []
for user, group in ratings_df.groupby('user_id'):
    temp.append(np.mean(list(items_df.loc[list(group['item_id'])]['topic_distribution']), axis = 0))
    
users_df['topic_distribution'] = temp


# 1) Q (m * n) = Relevance matrix of user 'u' to topic of item 'i'

Q = np.nan_to_num(1 - scipy.spatial.distance.cdist(list(users_df['topic_distribution']), list(items_df['topic_distribution']), 'cosine'))

# 2) W (m * m) = Similarity matrix of user 'u' to topic of user 'v'

W = np.nan_to_num(1 - scipy.spatial.distance.cdist(list(users_df['topic_distribution']), list(users_df['topic_distribution']), 'cosine'))

R = np.empty([len(users_df), len(items_df)], dtype = np.float64)
I = np.empty([len(users_df), len(items_df)], dtype = np.float64)

start = time.time()
          
for user, item, rtng in zip(ratings_df['user_id'], ratings_df['item_id'], ratings_df['rating']):
    R[user, item] = rtng
    I[user, item] = 1

###############################################################################

#Parameters for Gradient Descent

k = 10                                  #dimension of latent space
lamda = 0.1
beta = 30
gamma = 30
eta = 30

r = np.mean(ratings_df['rating'])

U = np.empty([len(users_df), k], dtype = np.float64)
P = np.empty([len(items_df), k], dtype = np.float64)

#Gradient Descent

#def get_rating_diff(R, R):
def cal_pred_rating(r, U, P):
    return (r + np.matmul(U, P.transpose()))
    
R_cap = cal_pred_rating(r, U, P)
#error_der_P = np.empty([len(items_df), k], dtype = np.float64)

def cal_error_der_P(I_i, R_i, U):
    t = np.multiply(I_i, R_i)
    return np.matmul(t, U)
    
error_der_P = map(cal_error_der_P, I.transpose(), (R_cap - R).transpose(), [U]*(len(items_df)))

print('It took {0:0.2f} seconds'.format(time.time() - start))
