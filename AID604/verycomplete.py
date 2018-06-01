from sklearn import svm
from sklearn import metrics
import numpy as np
from sklearn.model_selection import KFold
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import BernoulliRBM
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.cluster import DBSCAN
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.model_selection import PredefinedSplit
from sklearn.metrics import silhouette_score
from imblearn.over_sampling import SMOTE
np.set_printoptions(threshold='nan')
tmp = np.loadtxt("AID604.csv", dtype=np.str,skiprows=1, delimiter=",")
total_data=tmp[0:,0:tmp.shape[1]-1].astype(np.float)
total_label=tmp[0:,tmp.shape[1]-1].astype(np.float)

import random
import math
def randomly_pos_points_pick(training_data, training_label):
    positive_points_index_set = np.where(training_label==0)[0]
    num_to_pick = int(len(training_label)*2/3)
    # Need at least two pos points to apply SMOTE
    if num_to_pick < 2:
        num_to_pick = 2
    random.seed(0)
    # Use random.sample to pick up positive points randomly
    random_index_set = random.sample(list(positive_points_index_set), num_to_pick)
    # Use the index to filter out original training data and training label
    print (random_index_set)
    zero_data=training_data[np.where(training_label==1)[0]]
    zero_label=training_label[np.where(training_label==1)[0]]
    training_data_after_sampling = training_data[random_index_set]
    training_label_after_sampling = training_label[random_index_set]
    training_data_after_sampling=np.concatenate((zero_data, training_data_after_sampling), axis=0)
    training_label_after_sampling = np.concatenate((zero_label, training_label_after_sampling), axis=0)
    return training_data_after_sampling, training_label_after_sampling


skf = StratifiedKFold(n_splits=10)
kfold = skf.split(total_data, total_label)
K = -1
svm_matrix = np.zeros((10, 2, 2))
svm_auc = np.zeros((10, 1, 1))
recall = np.zeros((10, 1, 1))
p = np.zeros((10, 1, 1))
F1 = np.zeros((10, 1, 1))

bn_matrix = np.zeros((10, 2, 2))
bn_auc = np.zeros((10, 1, 1))

randomforest_matrix = np.zeros((10, 2, 2))
randomforest_auc = np.zeros((10, 1, 1))

knn_matrix = np.zeros((10, 2, 2))
knn_auc = np.zeros((10, 1, 1))

lr_matrix = np.zeros((10, 2, 2))
lr_auc = np.zeros((10, 1, 1))

gb_matrix = np.zeros((10, 2, 2))
gb_auc = np.zeros((10, 1, 1))

svm_matrix_v = np.zeros((10, 2, 2))
svm_auc_v = np.zeros((10, 1, 1))

bn_matrix_v = np.zeros((10, 2, 2))
bn_auc_v = np.zeros((10, 1, 1))

randomforest_matrix_v = np.zeros((5, 2, 2))
randomforest_auc_v = np.zeros((5, 1, 1))

knn_matrix_v = np.zeros((10, 2, 2))
knn_auc_v = np.zeros((10, 1, 1))

lr_matrix_v = np.zeros((10, 2, 2))
lr_auc_v = np.zeros((10, 1, 1))

gb_matrix_v = np.zeros((10, 2, 2))
gb_auc_v = np.zeros((10, 1, 1))
for traini, testi in kfold:
    K=K+1
    data_train, data_test = total_data[traini], total_data[testi]
    label_train, label_test = total_label[traini], total_label[testi]
    line = int(data_train.size / data_train[0].size)
    svm_best_f1 = -1
    bn_best_f1 = -1
    randomforest_best_f1 = -1
    knn_best_f1 = -1
    lr_best_f1 = -1
    gb_best_f1 = -1
    svm_best_f1_v = -1
    bn_best_f1_v = -1
    randomforest_best_f1_v = -1
    knn_best_f1_v = -1
    lr_best_f1_v = -1
    gb_best_f1_v = -1


    #data_train, label2 = shuffle(data_train, label2, random_state=830)
    while(len(data_train[np.where(label_train==0)[0]]) > len(data_train[np.where(label_train==1)[0]])*100 ):
        print ("Len0: %d",len(data_train[np.where(label_train==0)[0]]))
        print ("Len1: %d", len(data_train[np.where(label_train == 1)[0]]))
        skf2 = StratifiedKFold(n_splits=5)
        kfold2 = skf2.split(data_train, label_train)
        for train1, test1 in kfold2:
            data_train2, data_test_v = data_train[train1], data_train[test1]
            label_train2, label_test_v = label_train[train1], label_train[test1]
            break
        tmp1=data_train.copy()
        tmp2=label_train.copy()
        tmp3=data_test.copy()
        tmp4=label_test.copy()
        data_train=data_train2
        label_train=label_train2



        original_length = data_train[0].size
        original_length_v = data_train[0].size
        n00=0
        n10=0
        n01=0
        n11=0
        lasttrainset=-1
        n00_v = 0
        n10_v = 0
        n01_v=0
        n11_v=0
        lasttrainset_v = -1
        for round in range(1, 2):
            print ("N10 and N00: %0.3f %0.3f" % (n10, n00))
            print ("N01 and N11: %0.3f %0.3f" % (n01, n11))
            print ("N10V and N00V: %0.3f %0.3f" % (n10_v, n00_v))
            print ("N01V and N11V: %0.3f %0.3f" % (n01_v, n11_v))
            length = original_length+round-1
            length_v = original_length + round-1
            print ("%0.3f round" % (round))
            def my_score(X, y):
                return mutual_info_classif(X, y, random_state=724)
            selectedfeatures=SelectKBest(my_score, k=25)
            selectedfeatures.fit(data_train, label_train)
            smalldata_train=selectedfeatures.transform(data_train) #ti qu guo te zheng de 25 wei xun lian shu ju
        #kmeans = KMeans(n_clusters=50, random_state=0).fit_predict(data_train)
            smalldata_train=np.hstack((smalldata_train, data_train[0:,original_length:]))
            best_silhouette_score=-2
            for nclusters in range(100,120,20):
                kmeans_model = KMeans(n_clusters=nclusters, random_state=926)
                kmeans_model.fit(smalldata_train)
                tmp = silhouette_score(smalldata_train,kmeans_model.labels_, sample_size=40000)
                print ("n score:  %0.3f %0.3f" %(nclusters,tmp))
                if(tmp>best_silhouette_score):
                    best_silhouette_score=tmp
                    best_n_clusters=nclusters
            print ("%0.3f best_n_clusters" % (best_n_clusters))
            kmeans_model = KMeans(n_clusters=best_n_clusters, random_state=926)
            kmeans_model.fit(smalldata_train)
            kmeans = kmeans_model.labels_
         #   print kmeans
        #print>>f, kmeans.size
            split=np.zeros(10000)
            nosplit=np.zeros(10000)
            clusters=0
            len1 = int(data_train.size / data_train[0].size) #xun lian shu ju shu liang
            if(len1<150):
                break
            print ("%0.3f len1" % (len1))
            for i in range(0,len1):
                if(label_train[i]==1):
                    split[kmeans[i]]=split[kmeans[i]]+1
                if (label_train[i]==0):
                    nosplit[kmeans[i]] = nosplit[kmeans[i]] + 1
                clusters=max(clusters,kmeans[i])
            data=np.zeros((line,length))
            label=np.zeros((line))
            data_v = np.zeros((line, length))
            label_v = np.zeros((line))
            data1 = np.zeros((line, length))
            label1 = np.zeros((line))
            top=-1
            top_v=-1
            len2=int(data_test.size/data_test[0].size) # ce shi shu ju shu liang
            data_test_tmp=selectedfeatures.transform(data_test) #te zheng xuanze hou de 25 wei ce shi shu ju
            data_test_tmp = np.hstack((data_test_tmp, data_test[0:, original_length:]))
            kmeans2=kmeans_model.predict(data_test_tmp)
            len2_v =int(data_test_v.size / data_test_v[0].size)  # ce shi shu ju shu liang
            data_test_tmp_v = selectedfeatures.transform(data_test_v)  # te zheng xuanze hou de 25 wei ce shi shu ju
            data_test_tmp_v = np.hstack((data_test_tmp_v, data_test_v[0:, original_length:]))
            kmeans2_v = kmeans_model.predict(data_test_tmp_v)
            print ("%0.3f len2"%(len2))
            tmp=np.zeros(1000000)
            tmp_v = np.zeros(1000000)
            print (data.shape[0])
            print (data.shape[1])
            print (data_test.shape[0])
            print (data_test.shape[1])
            for i in range(0,len2):
                if (((split[kmeans2[i]]!=0)and (nosplit[kmeans2[i]]!=0)) or ( round==0)):
                    top=top+1
                    data[top]=data_test[i]
                    label[top]=label_test[i]
                    tmp[top]=(0.0+split[kmeans2[i]])/(split[kmeans2[i]]+nosplit[kmeans2[i]])
                    if (split[kmeans2[i]]+nosplit[kmeans2[i]]==0):
                        tmp[top]=0
                else:  # fen lei wei 0
                    if ((split[kmeans2[i]]==0)and(label_test[i] == 0)):
                        n00 = n00 + 1
                    if ((split[kmeans2[i]] == 0) and (label_test[i] == 1)):
                        n10 = n10 + 1
                    if ((nosplit[kmeans2[i]]==0)and(label_test[i] == 0)):
                        n01 = n01 + 1
                    if ((nosplit[kmeans2[i]] == 0) and (label_test[i] == 1)):
                        n11 = n11 + 1
            for i in range(0,len2_v):
                if (((split[kmeans2_v[i]]!=0) and (nosplit[kmeans2_v[i]]!=0))or( round==0)):
                    top_v=top_v+1
                    data_v[top_v]=data_test_v[i]
                    label_v[top_v]=label_test_v[i]
                    tmp_v[top_v]=(0.0+split[kmeans2_v[i]])/(split[kmeans2_v[i]]+nosplit[kmeans2_v[i]])
                    if (split[kmeans2_v[i]]+nosplit[kmeans2_v[i]]==0):
                        tmp_v[top_v]=0
                else:  # fen lei wei 0
                    if ((split[kmeans2_v[i]]==0)and(label_test_v[i] == 0)):
                        n00_v = n00_v + 1
                    if ((split[kmeans2_v[i]] == 0) and (label_test_v[i] == 1)):
                        n10_v = n10_v + 1
                    if ((nosplit[kmeans2_v[i]]==0)and(label_test_v[i] == 0)):
                        n01_v = n01_v + 1
                    if ((nosplit[kmeans2_v[i]] == 0) and (label_test_v[i] == 1)):
                        n11_v = n11_v + 1
            print  ("%0.3f top" % (top))
            data=data[0:top+1]
            label=label[0:top+1]
            tmp=tmp[0:top+1]
            data=np.column_stack((data, tmp))
            data_v = data_v[0:top_v + 1]
            label_v = label_v[0:top_v + 1]
            tmp_v = tmp_v[0:top_v + 1]
            data_v = np.column_stack((data_v, tmp_v))

            tmp = np.zeros(1000000)
            top_rest_train=-1
            top_split=-1

            for i in range(0, len1):
                if (((split[kmeans[i]]!=0)and (nosplit[kmeans[i]]!=0)) or ( round==0)):
                    top_rest_train=top_rest_train+1
                    data1[top_rest_train]=data_train[i]
                    label1[top_rest_train]=label_train[i]
                    tmp[top_rest_train] = (0.0 + split[kmeans[i]]) / (split[kmeans[i]] + nosplit[kmeans[i]])
                if(nosplit[kmeans[i]]==0):
                    top_split=top_split+1


            if(lasttrainset==top_rest_train):
                break
            lasttrainset=top_rest_train
            data1 = data1[0:top_rest_train + 1]
            label1 = label1[0:top_rest_train + 1]  #qu diao ju lei yihou de xun lian shu ju
            tmp=tmp[0:top_rest_train+1]
            data1=np.column_stack((data1,tmp))
            if(np.count_nonzero(label1)<5):
                break
            if ((np.count_nonzero(label) == 0 )and (round>0)):
                break
            if ((np.count_nonzero(label_v) == 0) and (round > 0)):
                break
           # print data1
          #  print label
            print ("%0.3f top_rest_train" % (top_rest_train))
            train_val_features = np.concatenate((data1, data), axis=0)
            train_val_labels = np.concatenate((label1, label), axis=0)
            test_fold = np.zeros(train_val_features.shape[0])
            test_fold[:data1.shape[0]] = -1 #-1 means tain

            ps = list(PredefinedSplit(test_fold=test_fold).split(train_val_features, train_val_labels))
            #for train_index, test_index in ps.split():
            #    print("TRAIN:", train_index, "TEST:", test_index)
            def f1(C00,C01,C10,C11):
                recall = C11 / (C10 + C11)
                p = C11 / (C01 + C11)
                return 2*(recall*p)/(recall+p)


            label_1 = np.ones(n10+n11)
            label_0 = np.zeros(n00+n01)
            label_fixed=np.concatenate((label_1, label_0), axis=0)
            df_fixed=np.concatenate((np.zeros(n10),np.ones(n11),np.zeros(n00),np.ones(n01) ), axis=0)
            label_fixed = np.concatenate((label_1, label_0), axis=0)
            label_total=np.concatenate((label, label_fixed), axis=0)
            label_1_v = np.ones(n10_v+n11_v)
            label_0_v = np.zeros(n00_v+n01_v)
            label_fixed_v = np.concatenate((label_1_v, label_0_v), axis=0)
            df_fixed_v = np.concatenate((np.zeros(n10_v),np.ones(n11_v),np.zeros(n00_v),np.ones(n01_v) ), axis=0)
            label_total_v = np.concatenate((label_v, label_fixed_v), axis=0)
            '''
            svc = GridSearchCV(svm.SVC(), [{'kernel': ['rbf'], 'gamma': [1e-5,1e-4,1e-3,1e-2,1e-1], 'C': [0.001,0.01,0.1,1, 10,100],'class_weight':[{0:1,1: 1}]}], cv=5,scoring=metrics.make_scorer(metrics.scorer.f1_score, average="binary",pos_label=1),n_jobs=-1)
            svc.fit(data1, label1)
            svc2 = svm.SVC(**svc.best_params_)
            predict_label=svc2.fit(data1, label1).predict(data)
            df=svc2.decision_function(data)
            df = np.concatenate((df, df_fixed), axis=0)
            tmp=np.zeros((10, 2, 2))
            tmp[7]=confusion_matrix(label, predict_label)
            tmp[7][0,0]=tmp[7][0,0]+n00
            tmp[7][1, 0]=tmp[7][1, 0] + n10
            tmp[7][0, 1] = tmp[7][0, 1] + n01
            tmp[7][1, 1] = tmp[7][1, 1] + n11
            print tmp[7]
            svc2_v = svm.SVC(**svc.best_params_)
            predict_label_v = svc2_v.fit(data1, label1).predict(data_v)
            df_v = svc2_v.decision_function(data_v)
            df_v = np.concatenate((df_v, df_fixed_v), axis=0)
            tmp_v = np.zeros((10, 2, 2))
            tmp_v[7] = confusion_matrix(label_v, predict_label_v)
            tmp_v[7][0, 0] = tmp_v[7][0, 0] + n00_v
            tmp_v[7][1, 0] = tmp_v[7][1, 0] + n10_v
            tmp_v[7][0, 1] = tmp_v[7][0, 1] + n01_v
            tmp_v[7][1, 1] = tmp_v[7][1, 1] + n11_v
            if(svm_best_f1_v<f1(C00=tmp_v[7][0,0],C01=tmp_v[7][0,1],C10=tmp_v[7][1,0],C11=tmp_v[7][1,1])):
                svm_best_f1_v = f1(C00=tmp_v[7][0, 0], C01=tmp_v[7][0, 1], C10=tmp_v[7][1, 0], C11=tmp_v[7][1, 1])
                svm_best_f1=f1(C00=tmp[7][0,0],C01=tmp[7][0,1],C10=tmp[7][1,0],C11=tmp[7][1,1])
                svm_matrix[K]=tmp[7]
                svm_matrix_v[K] = tmp_v[7]
                fpr, tpr, thresholds = metrics.roc_curve(label_total, df, pos_label=1)
                svm_auc[K]=metrics.auc(fpr,tpr)
                fpr_v, tpr_v, thresholds_v = metrics.roc_curve(label_total_v, df_v, pos_label=1)
                svm_auc_v[K] = metrics.auc(fpr_v, tpr_v)
            '''
            bb = GridSearchCV(BernoulliNB(), [{ 'alpha': [0.5,1,10,100,1000,10000],'binarize': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}], cv=5,scoring='roc_auc',n_jobs=-1)
            bb.fit(data1, label1)
            bn = BernoulliNB(**bb.best_params_)
            predict_label =bn.fit(data1, label1).predict(data)
            df=bn.predict_proba(data)[:,1]
            df = np.concatenate((df, df_fixed), axis=0)
            tmp = np.zeros((10, 2, 2))
            tmp[7] = confusion_matrix(label, predict_label)
            tmp[7][0, 0] = tmp[7][0, 0] + n00
            tmp[7][1, 0] = tmp[7][1, 0] + n10
            tmp[7][0, 1] = tmp[7][0, 1] + n01
            tmp[7][1, 1] = tmp[7][1, 1] + n11
            bn_v = BernoulliNB(**bb.best_params_)
            predict_label_v = bn_v.fit(data1, label1).predict(data_v)
            df_v = bn_v.predict_proba(data_v)[:, 1]
            df_v = np.concatenate((df_v, df_fixed_v), axis=0)
            tmp_v = np.zeros((10, 2, 2))
            tmp_v[7] = confusion_matrix(label_v, predict_label_v)
            tmp_v[7][0, 0] = tmp_v[7][0, 0] + n00_v
            tmp_v[7][1, 0] = tmp_v[7][1, 0] + n10_v
            tmp_v[7][0, 1] = tmp_v[7][0, 1] + n01_v
            tmp_v[7][1, 1] = tmp_v[7][1, 1] + n11_v
            print (tmp[7].tolist())
            print (tmp_v[7].tolist())
            fpr_v, tpr_v, thresholds_v = metrics.roc_curve(label_total_v, df_v, pos_label=1)
            auc_v = metrics.auc(fpr_v, tpr_v)
            print (auc_v)
            print ("***********")

            if (svm_best_f1_v <= f1(C00=tmp_v[7][0, 0], C01=tmp_v[7][0, 1], C10=tmp_v[7][1, 0], C11=tmp_v[7][1, 1])):
                if(not np.isnan(f1(C00=tmp_v[7][0, 0], C01=tmp_v[7][0, 1], C10=tmp_v[7][1, 0], C11=tmp_v[7][1, 1]))and not np.isnan(f1(C00=tmp[7][0, 0], C01=tmp[7][0, 1], C10=tmp[7][1, 0], C11=tmp[7][1, 1]))):
                    svm_best_f1_v = f1(C00=tmp_v[7][0, 0], C01=tmp_v[7][0, 1], C10=tmp_v[7][1, 0], C11=tmp_v[7][1, 1])
                    svm_best_f1 = f1(C00=tmp[7][0, 0], C01=tmp[7][0, 1], C10=tmp[7][1, 0], C11=tmp[7][1, 1])
                    svm_matrix[K] = tmp[7]
                    svm_matrix_v[K] = tmp_v[7]
                    fpr, tpr, thresholds = metrics.roc_curve(label_total, df, pos_label=1)
                    svm_auc[K] = metrics.auc(fpr, tpr)
                    svm_auc_v[K] = auc_v

            rf = GridSearchCV(RandomForestClassifier(), [{ 'n_estimators': [80,100,120,140,160,180,200,220,1000],'criterion':['gini','entropy'],'class_weight':[{0:1,1: 1}]}],cv=5,scoring='roc_auc',n_jobs=-1)
            rf.fit(data1, label1)
            randomforest = RandomForestClassifier(**rf.best_params_)
            predict_label = randomforest.fit(data1, label1).predict(data)
            df = randomforest.predict_proba(data)[:, 1]
            df = np.concatenate((df, df_fixed), axis=0)
            tmp = np.zeros((10, 2, 2))
            tmp[7] = confusion_matrix(label, predict_label)
            tmp[7][0, 0] = tmp[7][0, 0] + n00
            tmp[7][1, 0] = tmp[7][1, 0] + n10
            tmp[7][0, 1] = tmp[7][0, 1] + n01
            tmp[7][1, 1] = tmp[7][1, 1] + n11

            randomforest_v = RandomForestClassifier(**rf.best_params_)
            predict_label_v = randomforest_v.fit(data1, label1).predict(data_v)
            df_v = randomforest_v.predict_proba(data_v)[:, 1]
            df_v = np.concatenate((df_v, df_fixed_v), axis=0)
            tmp_v = np.zeros((10, 2, 2))
            tmp_v[7] = confusion_matrix(label_v, predict_label_v)
            tmp_v[7][0, 0] = tmp_v[7][0, 0] + n00_v
            tmp_v[7][1, 0] = tmp_v[7][1, 0] + n10_v
            tmp_v[7][0, 1] = tmp_v[7][0, 1] + n01_v
            tmp_v[7][1, 1] = tmp_v[7][1, 1] + n11_v

            print (tmp[7].tolist())
            print (tmp_v[7].tolist())
            fpr_v, tpr_v, thresholds_v = metrics.roc_curve(label_total_v, df_v, pos_label=1)
            auc_v = metrics.auc(fpr_v, tpr_v)
            print (auc_v)
            print ("***********")
            if ((svm_best_f1_v < f1(C00=tmp_v[7][0, 0], C01=tmp_v[7][0, 1], C10=tmp_v[7][1, 0], C11=tmp_v[7][1, 1])) or ((round!=1)and(auc_v==1))):
                if (not np.isnan(f1(C00=tmp_v[7][0, 0], C01=tmp_v[7][0, 1], C10=tmp_v[7][1, 0], C11=tmp_v[7][1, 1])) and not np.isnan(f1(C00=tmp[7][0, 0], C01=tmp[7][0, 1], C10=tmp[7][1, 0], C11=tmp[7][1, 1]))):
                    svm_best_f1_v = f1(C00=tmp_v[7][0, 0], C01=tmp_v[7][0, 1], C10=tmp_v[7][1, 0], C11=tmp_v[7][1, 1])
                    svm_best_f1 = f1(C00=tmp[7][0, 0], C01=tmp[7][0, 1], C10=tmp[7][1, 0], C11=tmp[7][1, 1])
                    svm_matrix[K] = tmp[7]
                    svm_matrix_v[K] = tmp_v[7]
                    fpr, tpr, thresholds = metrics.roc_curve(label_total, df, pos_label=1)
                    svm_auc[K] = metrics.auc(fpr, tpr)
                    svm_auc_v[K] = auc_v

            knn = GridSearchCV(KNeighborsClassifier(), [{ 'n_neighbors': [3,5,7,9]}],  cv=5,scoring='roc_auc',n_jobs=-1)
            knn.fit(data1, label1)
            knn2 = KNeighborsClassifier(**knn.best_params_)
            predict_label =knn2.fit(data1, label1).predict(data)
            df=knn2.predict_proba(data)[:,1]
            df = np.concatenate((df, df_fixed), axis=0)
            tmp = np.zeros((10, 2, 2))
            tmp[7] = confusion_matrix(label, predict_label)
            tmp[7][0, 0] = tmp[7][0, 0] + n00
            tmp[7][1, 0] = tmp[7][1, 0] + n10
            tmp[7][0, 1] = tmp[7][0, 1] + n01
            tmp[7][1, 1] = tmp[7][1, 1] + n11

            knn2_v = KNeighborsClassifier(**knn.best_params_)
            predict_label_v = knn2_v.fit(data1, label1).predict(data_v)
            df_v = knn2_v.predict_proba(data_v)[:, 1]
            df_v = np.concatenate((df_v, df_fixed_v), axis=0)
            tmp_v = np.zeros((10, 2, 2))
            tmp_v[7] = confusion_matrix(label_v, predict_label_v)
            tmp_v[7][0, 0] = tmp_v[7][0, 0] + n00_v
            tmp_v[7][1, 0] = tmp_v[7][1, 0] + n10_v
            tmp_v[7][0, 1] = tmp_v[7][0, 1] + n01_v
            tmp_v[7][1, 1] = tmp_v[7][1, 1] + n11_v
            print (tmp[7].tolist())
            print (tmp_v[7].tolist())
            fpr_v, tpr_v, thresholds_v = metrics.roc_curve(label_total_v, df_v, pos_label=1)
            auc_v = metrics.auc(fpr_v, tpr_v)
            print (auc_v)
            print ("***********")
            if (svm_best_f1_v <= f1(C00=tmp_v[7][0, 0], C01=tmp_v[7][0, 1], C10=tmp_v[7][1, 0], C11=tmp_v[7][1, 1])):
                if (not np.isnan(f1(C00=tmp_v[7][0, 0], C01=tmp_v[7][0, 1], C10=tmp_v[7][1, 0], C11=tmp_v[7][1, 1]))and not np.isnan(f1(C00=tmp[7][0, 0], C01=tmp[7][0, 1], C10=tmp[7][1, 0], C11=tmp[7][1, 1]))):
                    svm_best_f1_v = f1(C00=tmp_v[7][0, 0], C01=tmp_v[7][0, 1], C10=tmp_v[7][1, 0], C11=tmp_v[7][1, 1])
                    svm_best_f1 = f1(C00=tmp[7][0, 0], C01=tmp[7][0, 1], C10=tmp[7][1, 0], C11=tmp[7][1, 1])
                    svm_matrix[K] = tmp[7]
                    svm_matrix_v[K] = tmp_v[7]
                    fpr, tpr, thresholds = metrics.roc_curve(label_total, df, pos_label=1)
                    svm_auc[K] = metrics.auc(fpr, tpr)
                    svm_auc_v[K] = auc_v


            lr = GridSearchCV(LogisticRegression(), [{ 'penalty': ['l1','l2'],'C': [0.001,0.01,0.1,1, 10,100,1000,10000],'class_weight':[{0:1,1: 1}]}], cv=5,scoring='roc_auc',n_jobs=-1)
            lr.fit(data1, label1)
            lr2 = LogisticRegression(**lr.best_params_)
            predict_label =lr2.fit(data1, label1).predict(data)
            df=lr2.predict_proba(data)[:,1]
            df = np.concatenate((df, df_fixed), axis=0)
            tmp = np.zeros((10, 2, 2))
            tmp[7] = confusion_matrix(label, predict_label)
            tmp[7][0, 0] = tmp[7][0, 0] + n00
            tmp[7][1, 0] = tmp[7][1, 0] + n10
            tmp[7][0, 1] = tmp[7][0, 1] + n01
            tmp[7][1, 1] = tmp[7][1, 1] + n11

            lr2_v = LogisticRegression(**lr.best_params_)
            predict_label_v = lr2_v.fit(data1, label1).predict(data_v)
            df_v = lr2_v.predict_proba(data_v)[:, 1]
            df_v = np.concatenate((df_v, df_fixed_v), axis=0)
            tmp_v = np.zeros((10, 2, 2))
            tmp_v[7] = confusion_matrix(label_v, predict_label_v)
            tmp_v[7][0, 0] = tmp_v[7][0, 0] + n00_v
            tmp_v[7][1, 0] = tmp_v[7][1, 0] + n10_v
            tmp_v[7][0, 1] = tmp_v[7][0, 1] + n01_v
            tmp_v[7][1, 1] = tmp_v[7][1, 1] + n11_v
            print (tmp[7].tolist())
            print (tmp_v[7].tolist())
            fpr_v, tpr_v, thresholds_v = metrics.roc_curve(label_total_v, df_v, pos_label=1)
            auc_v = metrics.auc(fpr_v, tpr_v)
            print (auc_v)
            print ("***********")
            if (svm_best_f1_v <= f1(C00=tmp_v[7][0, 0], C01=tmp_v[7][0, 1], C10=tmp_v[7][1, 0], C11=tmp_v[7][1, 1])):
                if (not np.isnan(f1(C00=tmp_v[7][0, 0], C01=tmp_v[7][0, 1], C10=tmp_v[7][1, 0], C11=tmp_v[7][1, 1]))and not np.isnan(f1(C00=tmp[7][0, 0], C01=tmp[7][0, 1], C10=tmp[7][1, 0], C11=tmp[7][1, 1]))):
                    svm_best_f1_v = f1(C00=tmp_v[7][0, 0], C01=tmp_v[7][0, 1], C10=tmp_v[7][1, 0], C11=tmp_v[7][1, 1])
                    svm_best_f1 = f1(C00=tmp[7][0, 0], C01=tmp[7][0, 1], C10=tmp[7][1, 0], C11=tmp[7][1, 1])
                    svm_matrix[K] = tmp[7]
                    svm_matrix_v[K] = tmp_v[7]
                    fpr, tpr, thresholds = metrics.roc_curve(label_total, df, pos_label=1)
                    svm_auc[K] = metrics.auc(fpr, tpr)
                    svm_auc_v[K] =auc_v


            gb = GaussianNB()
            predict_label =gb.fit(data1, label1).predict(data)
            df=gb.predict_proba(data)[:,1]
            df = np.concatenate((df, df_fixed), axis=0)
            tmp = np.zeros((10, 2, 2))
            tmp[7] = confusion_matrix(label, predict_label)
            tmp[7][0, 0] = tmp[7][0, 0] + n00
            tmp[7][1, 0] = tmp[7][1, 0] + n10
            tmp[7][0, 1] = tmp[7][0, 1] + n01
            tmp[7][1, 1] = tmp[7][1, 1] + n11

            gb_v = GaussianNB()
            predict_label_v = gb_v.fit(data1, label1).predict(data_v)
            df_v = gb_v.predict_proba(data_v)[:, 1]
            df_v = np.concatenate((df_v, df_fixed_v), axis=0)
            tmp_v = np.zeros((10, 2, 2))
            tmp_v[7] = confusion_matrix(label_v, predict_label_v)
            tmp_v[7][0, 0] = tmp_v[7][0, 0] + n00_v
            tmp_v[7][1, 0] = tmp_v[7][1, 0] + n10_v
            tmp_v[7][0, 1] = tmp_v[7][0, 1] + n01_v
            tmp_v[7][1, 1] = tmp_v[7][1, 1] + n11_v
            print (tmp[7].tolist())
            print (tmp_v[7].tolist())
            fpr_v, tpr_v, thresholds_v = metrics.roc_curve(label_total_v, df_v, pos_label=1)
            auc_v = metrics.auc(fpr_v, tpr_v)
            print (auc_v)
            print ("***********")
            if (svm_best_f1_v <= f1(C00=tmp_v[7][0, 0], C01=tmp_v[7][0, 1], C10=tmp_v[7][1, 0], C11=tmp_v[7][1, 1])):
                if (not np.isnan(f1(C00=tmp_v[7][0, 0], C01=tmp_v[7][0, 1], C10=tmp_v[7][1, 0], C11=tmp_v[7][1, 1]))and not np.isnan(f1(C00=tmp[7][0, 0], C01=tmp[7][0, 1], C10=tmp[7][1, 0], C11=tmp[7][1, 1]))):
                    svm_best_f1_v = f1(C00=tmp_v[7][0, 0], C01=tmp_v[7][0, 1], C10=tmp_v[7][1, 0], C11=tmp_v[7][1, 1])
                    svm_best_f1 = f1(C00=tmp[7][0, 0], C01=tmp[7][0, 1], C10=tmp[7][1, 0], C11=tmp[7][1, 1])
                    svm_matrix[K] = tmp[7]
                    svm_matrix_v[K] = tmp_v[7]
                    fpr, tpr, thresholds = metrics.roc_curve(label_total, df, pos_label=1)
                    svm_auc[K] = metrics.auc(fpr, tpr)
                    svm_auc_v[K] = auc_v


  


        data_train=tmp1.copy()
        label_train=tmp2.copy()
        data_test=tmp3.copy()
        label_test=tmp4.copy()
        data_train, label_train = randomly_pos_points_pick(data_train, label_train)

    print ("Validation set No. %0.1f " % (K))
    print ("The confusion matrix :")
    print (svm_matrix_v[K].tolist())

    print ("The AUC of is %0.3f " % (svm_auc_v[K]))

    print ("Test set No. %0.1f " % (K))
    print ("The confusion matrix of Best:")
    print (svm_matrix[K].tolist())
    recall[K]=svm_matrix[K][1,1]/(svm_matrix[K][1,0]+svm_matrix[K][1,1])
    p[K]=svm_matrix[K][1,1]/(svm_matrix[K][0,1]+svm_matrix[K][1,1])
    F1[K]=2*(recall[K]*p[K])/(recall[K]+p[K])
    if(recall[K]+p[K]==0):
        F1[K]=0
    print ("The recall of Best is %0.3f " % (recall[K]))
    print ("The p of Best is %0.3f " % (p[K]))
    print ("The F1 of Best is %0.3f " % (F1[K]))
    print ("The AUC of Best is %0.3f " % (svm_auc[K]))

print ("----------------------Finally-------------------------")
print ("The AUC is %0.3f " % (np.mean(svm_auc)))
print ("The recall is %0.3f " % (np.mean(recall)))
print ("The p is %0.3f " % (np.mean(p)))
print ("The F1 is %0.3f " % (np.mean(F1)))