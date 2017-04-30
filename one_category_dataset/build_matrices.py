import pandas as pd
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")
import time

start = time.time()

items_path = r'E:\Users\Dell\Desktop\STUDY_SPRING 17\CSCE 670 Information Storage and Retreival\Project\nightlife\items.txt'
users_path = r'E:\Users\Dell\Desktop\STUDY_SPRING 17\CSCE 670 Information Storage and Retreival\Project\nightlife\users.txt'
ratings_path = r'E:\Users\Dell\Desktop\STUDY_SPRING 17\CSCE 670 Information Storage and Retreival\Project\nightlife\ratings.txt'
subcategories_path = r'E:\Users\Dell\Desktop\STUDY_SPRING 17\CSCE 670 Information Storage and Retreival\Project\nightlife\subcategories_list.txt'


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

Q = np.empty((len(users_df), len(items_df)), dtype = np.float64)

user_len =  range(len(users_df))
item_len =  range(len(items_df))
b = list(users_df['topic_distribution'])
c = list(items_df['topic_distribution'])
for i in range(1):
    a = []
    for j in item_len:
        Q[i,j] = np.dot(b[i][:5], c[j][:5])
        
#map(cosine_similarity, [users_df['topic_distribution'][0]]*21337, items_df['topic_distribution'])
print('It took {0:0.2f} seconds'.format(time.time() - start))
