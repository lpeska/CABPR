import os
import time
import numpy as np
import rank_metrics as rank
import scipy.sparse as sp
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc
from collections import defaultdict

def load_matrix(dataset, folder, shape, first):
    with open(os.path.join(folder, dataset+".csv"), "r") as inf:
        #inf.next()
        int_array = [line.strip("\n").split(";")[0:] for line in inf]
    intMat = np.array(int_array, dtype=np.float64)  
    return intMat
    

def cross_validation(intMat, seed, cv=1, invert=0, fract=0.75):
   
    cv_data = defaultdict(list)

    num_drugs, num_targets = intMat.shape
    prng = np.random.RandomState(seed)
    if cv == 0:
        index = prng.permutation(num_drugs)
    if cv == 1:
        index = prng.permutation(intMat.size)
    step = round(index.size*fract)
        
    ii = index[0:int(step)]
        
    if cv == 0:
        test_data = np.array([[k, j] for k in ii for j in range(num_targets)], dtype=np.int32)
    elif cv == 1:
        test_data = np.array([[k/num_targets, k % num_targets] for k in ii], dtype=np.int32)
    x, y = test_data[:, 0], test_data[:, 1]
    test_label = intMat[x, y]
    W = np.ones(intMat.shape)
    W[x, y] = 0

    return W, test_data, test_label


        
def per_user_rankings(test_data,test_label, scores):
    unique_users = np.unique(test_data[:,0])
    user_array = test_data[:,0]
    ndcg = []   
    aupr_list = [] 
    auc_list = [] 
    p10_list = [] 
    for u in unique_users:
        indices_u =  np.in1d(user_array, [u])
        labels_u = test_label[indices_u].astype(float)
        bool_labels_u = (labels_u > 0.001)
        scores_u = scores[indices_u].astype(float)
        #ndcg is calculated only for the users with some positive examples
        #print(labels_u[0:50])
        if not all(i <= 0.001 for i in labels_u):                        
            tmp = np.c_[labels_u,scores_u]
            tmp = tmp[tmp[:,1].argsort()[::-1],:]
            ordered_labels = tmp[:,0]
            ndcg_u = rank.ndcg_at_k(ordered_labels,ordered_labels.shape[0],1)
            ndcg.append(ndcg_u)  
            
            top10 = tmp[0:9,0]
            presence_at_10 = sum(top10)
            p10_list.append(presence_at_10)  
            
            prec, rec, thr = precision_recall_curve(bool_labels_u, scores_u)
            aupr_val = auc(rec, prec)
            aupr_list.append(aupr_val)
            
            fpr, tpr, thr = roc_curve(bool_labels_u, scores_u)
            auc_val = auc(fpr, tpr)
            auc_list.append(auc_val)     
    return np.array([ndcg, aupr_list, auc_list,p10_list])
        
        
                
        
        

        
        