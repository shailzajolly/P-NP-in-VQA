import pickle
import json
import sklearn

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

import numpy as np

def knn_class(k, X_train, Y_train, X_test, test_ids):
    
    y_pred = {}
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, Y_train)
    p = knn.predict(X_test)
    print(p)
    y_pred[test_ids] = knn.predict(X_test)
    
    return y_pred

def get_feats(file):

    feats = []
    q_ids = []
    for key, value in file.items():
        q_ids.append(key)
        feats.append(value)
    return q_ids, feats

def main():
    train = pickle.load(open('data/yn_train_jfeats.pkl', 'rb'))
    _, X_train = get_feats(train)
    print("Train:", len(X_train))

    train_gt = pickle.load(open('data/train_yesno_gt.pkl','rb'))
    Y_train = list(train_gt.values())
    print("YTrain:", len(Y_train))

    test = pickle.load(open('data/non_yn_train_jfeats.pkl','rb'))
    q_ids, X_test = get_feats(test)
    print("Test:", len(X_test))
    
    pred = knn_class(7, X_train, Y_train, X_test, q_ids)
    print("Now Dumping!")
    pickle.dump(pred,open('nonyesno_preds_knn.pkl', 'wb')) 

if __name__ == "__main__":
    main()


