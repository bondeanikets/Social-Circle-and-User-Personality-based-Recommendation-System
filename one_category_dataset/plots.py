# -*- coding: utf-8 -*-
"""
Created on Wed May 03 12:35:08 2017

@author: Aniket
"""

import matplotlib.pyplot as plt
import seaborn as sb
sb.set_style('darkgrid')
import dill 

filename= r'E:\Users\Dell\Documents\GitHub\Information-Retrieval-Project\one_category_dataset\globalsave_mode_3.pkl'
dill.load_session(filename)

plt.figure()
plt.plot(Error_list_train, color = 'g', label='Training set error')
plt.gca().set_title('Train Error vs Iterations')
plt.gca().set_xlabel('Iterations')
plt.gca().set_ylabel('Train error')
plt.axhline(y=36900, color='red', linestyle='-', label='Threshold Error')
plt.legend(loc='upper right')
plt.savefig( r'E:\Users\Dell\Documents\GitHub\Information-Retrieval-Project\one_category_dataset\Train_error_mode_3.png', 
            dpi=600)


plt.figure()
models = ['Base-MF', 'II', 'IS', 'UI', 'II + IS', 'IS + UI', 'II + UI', 'ALL' ]
night_life_test_rmse = [2.197, 1.241, 1.252, 1.324, 1.093, 1.124, 1.108, 1.024]
ax = sb.barplot(x = night_life_test_rmse, y= models)
ax.set_title('Models vs RMSE')
ax.set_xlabel('RMSE')
ax.set_ylabel('Models')
ax.figure.savefig( r'E:\Users\Dell\Documents\GitHub\Information-Retrieval-Project\one_category_dataset\RMSE_vs_models.png', 
            dpi=600)


plt.figure()
models = ['Base-MF', 'II', 'IS', 'UI', 'II + IS', 'IS + UI', 'II + UI', 'ALL' ]
night_life_test_mae = [1.647, 0.893, 0.902, 0.931, 0.836, 0.842, 0.831, 0.806]
ax = sb.barplot(x = night_life_test_mae, y= models)
ax.set_title('Models vs MAE')
ax.set_xlabel('MAE')
ax.set_ylabel('Models')
ax.figure.savefig( r'E:\Users\Dell\Documents\GitHub\Information-Retrieval-Project\one_category_dataset\MAE_vs_models.png', 
            dpi=600)