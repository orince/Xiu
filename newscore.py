# '''
# Author: Naixin && naixinguo2-c@my.cityu.edu.hk
# Date: 2022-10-05 18:33:47
# LastEditors: Naixin && naixinguo2-c@my.cityu.edu.hk
# LastEditTime: 2022-10-17 13:53:18
# FilePath: /Gtext/xiu/newscore.py
# Description: 

# '''
# '''
# Author: Naixin && naixinguo2-c@my.cityu.edu.hk
# Date: 2022-09-22 12:18:30
# LastEditors: Naixin && naixinguo2-c@my.cityu.edu.hk
# LastEditTime: 2022-10-03 16:05:12
# FilePath: /Gtext/ex.ipynb
# Description: 
# '''

'''
Author: Naixin && naixinguo2-c@my.cityu.edu.hk
Date: 2022-09-22 12:18:30
LastEditors: Naixin && naixinguo2-c@my.cityu.edu.hk
LastEditTime: 2022-10-03 16:05:12
FilePath: /Gtext/ex.ipynb
Description: 
'''
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import fminbound; from scipy.optimize import minimize
from scipy.optimize import minimize_scalar


# WS_index = SESTM1(W[:300,:],r[:300], 100,100, 0.92)
# Ohat = SESTM2(W[:300,:], r[:300], WS_index)

def sentiment_score(W_test, Ohat, WS_index,lam):
    '''
    Input:
        W_test: Out-of-Sample sparse matrix
        Ohat: thetahat obtained from function sestm
        lam: lambda value
    Return:
        p_new: predicted sentiment score
    '''
    #test sample size
    n,S = W_test.shape  
    # n = len(W_test)
    # S = len(W_test[0])
    p_new = []
    # score each article's sentiment
    for i in range(n):   
        
        if not np.sum(W_test[i,WS_index]):   # no sentiment word found
            p_new.append(0.5)
        else:
            def pmle_objectfunction(X):
                # maxmize the object function
           
                # objfun = -np.sum(np.log(X * Ohat[WS_index][0]+ (1-X) * Ohat[WS_index][1])* W_test[i][WS_index] /np.sum(W_test[i,WS_index])) - lam * np.log(X*(1-X))
                objfun = -sum(np.sum(np.log(X * Ohat[j, 0]+ (1-X) * Ohat[j, 1])* W_test[i,j]  for j in WS_index))/np.sum(W_test[i,WS_index])- lam * np.log(X*(1-X))
                
                return objfun
            solution = minimize_scalar(pmle_objectfunction, bounds=(0.0000001,0.9999999), method='bounded')
        
            # append sentiment score
            p_new.append(solution.x)
         
            # optimal_p = fminbound(lambda x: pmle_objectfunction(x),0,1)
            # print(optimal_p)
    return p_new

# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from scipy.optimize import minimize_scalar


# def sentiment_score(W_test, Ohat, WS_index,lam):
#     '''
#     Input:
#         W_test: Out-of-Sample sparse matrix
#         Ohat: thetahat obtained from function sestm
#         lam: lambda value
#     Return:
#         p_new: predicted sentiment score
#     '''
#     #test sample size
#     n,S = W_test.shape  
#     p_new = []
#     # score each article's sentiment
#     for i in range(n):
          
#         if not np.sum(W_test[i,WS_index]):   # no sentiment word found
#             p_new.append(0.5)
#         else:
#             def pmle_objectfunction(x):
#                  # maxmize the object function
#                 objfun = -(sum(np.log(x * Ohat[j,0]+ (1-x) * Ohat[j,1])* W_test[i,j]  for j in WS_index))/sum(W_test[i,WS_index]) - lam * np.log(x*(1-x))
#                 return objfun
#             solution = minimize_scalar(pmle_objectfunction, bounds=(0.00001,0.99999), method='bounded')
            
#             # append sentiment score
#             p_new.append(solution.x)
    
#     return p_new
