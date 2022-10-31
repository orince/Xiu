'''
Author: Naixin && naixinguo2-c@my.cityu.edu.hk
Date: 2022-10-05 18:29:47
LastEditors: Naixin && naixinguo2-c@my.cityu.edu.hk
LastEditTime: 2022-10-31 12:00:56
FilePath: /Gtext/xiu/main.py
Description: xiu

'''
from struct import pack_into
import numpy as np
import heapq 
from validation import Hyperparam
from newscore import sentiment_score
# from dgp import W,r
import pandas as pd
import glob

#################################################################################################################################
#DGP: Input Data
# W_train,r_train = W[:1000,], r[:1000]
# W_vali,r_vali = W[1000:1300,], r[1000:1300]
# W_test,r_test =W[1300:,], r[1300:]


files = glob.glob("/home/yuanzhi/Text_individual/codes/results_primary_news_bag_of_words_train_aggregated/*.csv")
cols = ['newtext_bagofwords', 'threedayreturn'] # 只取这些列
# 列表推导出对象
dflist = [pd.read_csv(i, usecols=cols) for i in files]
df1 = pd.concat(dflist) # 合并

# df1 =pd.read_csv('/home/yuanzhi/Text_individual/codes/results_primary_news_bag_of_words_train/2017_bag_of_words.csv')
df1.newtext_bagofwords = df1.newtext_bagofwords.apply(lambda x: list(eval(x.strip("[]"))))
r_train = np.array(list(df1.threedayreturn))
W_train = np.array(list(df1.newtext_bagofwords))
print('Training Read DONE')
files = glob.glob("/home/yuanzhi/Text_individual/codes/results_primary_news_bag_of_words_vali_aggregated/*.csv")
cols = ['newtext_bagofwords', 'threedayreturn'] # 只取这些列
# 列表推导出对象
dflist = [pd.read_csv(i, usecols=cols) for i in files]
df2 = pd.concat(dflist)
# df2 =pd.read_csv('/home/yuanzhi/Text_individual/codes/results_primary_news_bag_of_words_train/2018_bag_of_words.csv')
df2.newtext_bagofwords = df2.newtext_bagofwords.apply(lambda x: list(eval(x.strip("[]"))))
r_vali = np.array(list(df2.threedayreturn))
W_vali = np.array(list(df2.newtext_bagofwords))
print('Vali Read DONE')
files = glob.glob("/home/yuanzhi/Text_individual/codes/results_primary_news_bag_of_words_test_aggregated/*.csv")
cols = ['newtext_bagofwords', 'return_0'] # 只取这些列
# 列表推导出对象
dflist = [pd.read_csv(i, usecols=cols) for i in files]
df3 = pd.concat(dflist)
# df3 =pd.read_csv('/home/yuanzhi/Text_individual/codes/results_primary_news_bag_of_words_train/2019_bag_of_words.csv')
df3.newtext_bagofwords = df3.newtext_bagofwords.apply(lambda x: list(eval(x.strip("[]"))))
r_test = np.array(list(df3.return_0))
W_test = np.array(list(df3.newtext_bagofwords))
print('Testing Read DONE')


#hyperparameter
Lam = [ 1,  5, 10]
Alpha_posnum = [25, 50, 100]
Alpha_negnum = Alpha_posnum 
Kappa = [0.86,0.88,0.90,0.92,0.94]



(lam, alpha_posnum, kappa),Ohat, WS_index = Hyperparam(W_train, r_train, Lam, Alpha_posnum, Kappa)
print((lam, alpha_posnum, kappa),Ohat, WS_index)
# (lam, alpha_posnum, kappa),Ohat, WS_index = Hyperparam(W_train, r_train, W_vali, r_vali, Lam, Alpha_posnum, Kappa)
# print((lam, alpha_posnum, kappa),Ohat, WS_index)

alpha_negnum = alpha_posnum

# p = sentiment_score(W_test, Ohat, WS_index,lam)
p = sentiment_score(W_train, Ohat, WS_index,lam)
print(p)
max_index = heapq.nlargest(25, enumerate(p), key=lambda x: x[1])
min_index = heapq.nsmallest(25, enumerate(p), key=lambda x: x[1])
# print('max_index =', max_index,'min_index =', min_index)
r_max = heapq.nlargest(25, enumerate(r_train), key=lambda x: x[1])
r_min = heapq.nsmallest(25, enumerate(r_train), key=lambda x: x[1])
for i in range(25):
    print('max_index =',max_index[i][0],'r_max_index =',r_max[i][0])
    print(r_train[max_index[i][0]])

    # print('r_max_index =',r_max[i][0])

for i in range(25):
    print('min_index =',min_index[i][0],'r_min_index =',r_min[i][0])
    print(r_train[min_index[i][0]])

