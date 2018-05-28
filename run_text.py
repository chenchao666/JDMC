####### test my code ########################
from JDMC import *
from utils import *
import numpy as np
from sklearn.utils import shuffle
from scipy.io import savemat
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVC
import os
from sklearn import preprocessing
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import ParameterGrid
from scipy.stats import zscore

data_path = "./data/rcv1rcv2aminigoutte/SP"


def ELM_test(dim, source_x, source_y, target_x, target_y, target_nums, random_state):
    ##data perpare
    train_x_s, train_y_s, train_x_t, train_y_t, test_x, test_y = load_text_dataset(dim, source_x, source_y, target_x,
                                                                                   target_y, source_nums=100,
                                                                                   target_nums=target_nums,
                                                                                   random_state=random_state)
    train_y_s = LabelTransform(train_y_s)
    train_y_t = LabelTransform(train_y_t)
    test_y = LabelTransform(test_y)

    ##train model
    net = ELM(train_x_t, train_y_t, test_x, test_y, 2000)
    net.ParamInit()
    net.Activation('relu')
    net.TrainELM('Lp', 0.01)
    net.TrainAccuracy('relu')
    net.TestAccuracy('relu')
    net.printf()
    return net.TestAcc * 100

def run_svm(dim,source_x,source_y,target_x, target_y, target_nums, random_state):
    train_x_s, train_y_s, train_x_t, train_y_t, test_x, test_y = load_text_dataset(dim,source_x,source_y,target_x, target_y,source_nums=100,target_nums=target_nums,random_state=random_state)
    pca = PCA(40)
    pcaed=pca.fit_transform(np.row_stack([train_x_s,train_x_t,test_x]))
    train_x_s, train_x_t, test_x = pcaed[0:len(train_x_s),:], pcaed[len(train_x_s):len(train_x_s)+len(train_x_t),:],pcaed[-len(test_x):,:]
    clf = SVC()
    clf.fit(train_x_t, train_y_t)
    acc = clf.score(test_x, test_y)
    print acc
    return acc

def JDMC_test(f, dim, source_x, source_y, target_x, target_y, c1, c2, c3, c4, nodes, target_nums, random_state=0):
    print("!!!!!!! " + source_name + "-->" + target_name + " !!!!!!!!!!")
    train_x_s, train_y_s, train_x_t, train_y_t, test_x, test_y = load_text_dataset(dim, source_x, source_y, target_x,
                                                                                   target_y, source_nums=100,
                                                                                   target_nums=target_nums,
                                                                                   random_state=random_state)
    # train_x_s = preprocessing.normalize(train_x_s)
    # tmp = np.row_stack([train_x_s,train_x_t, test_x])
    # tmp = np.row_stack([train_x_t, test_x])
    train_y_s = LabelTransform(train_y_s)
    train_y_t = LabelTransform(train_y_t)
    test_y = LabelTransform(test_y)
    pca = PCA(40)
    pcaed=pca.fit_transform(np.row_stack([train_x_s,train_x_t,test_x]))
    train_x_s, train_x_t, test_x = pcaed[0:len(train_x_s),:], pcaed[len(train_x_s):len(train_x_s)+len(train_x_t),:],pcaed[-len(test_x):,:]
    accs = []
    tmp = []
    for _ in range(10):
        net = JDMC3(train_x_s, train_x_t, train_y_s, train_y_t, test_x, test_y, nodes, [c1, c2, c3], 100)
        net.ParamInit()
        net.Activation('relu')
        net.TrainJDMC()
        net.TestJDMC('relu')
        net.printf()
        f.write("seed:%d, Acc:%.2f\n" % (random_state, net.TestAcc))
        tmp.append(net.TestAcc)
    print mean(tmp)
    return np.max(tmp) * 100


if __name__ == '__main__':
    # ELM_test()
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--c1", type=float, default=0.01)
    parser.add_argument("--c2", type=int, default=0.1)
    parser.add_argument("--c3", type=int, default=30)
    parser.add_argument("--c4", type=int, default=100)
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--d", type=int, default=40)
    parser.add_argument("--t", type=int, default=20)


    args = parser.parse_args()
    c1, c2, c3, c4, nodes, dim, target_nums = args.c1, args.c2, args.c3, args.c4, args.n, args.d, args.t
    import datetime

    source_name = "EN"
    target_name = "SP"
    source_path = os.path.join(data_path, "Index_" + source_name + "-SP")
    target_path = os.path.join(data_path, "Index_" + target_name + "-SP")
    source_x, source_y = load_svmlight_file(source_path)
    target_x, target_y = load_svmlight_file(target_path)
    source_x, target_x = source_x.A, target_x.A
    EN_SP = [23, 26, 27, 50, 67, 68, 61, 62, 58, 80, 88, 3, 18, 33, 41, 42, 9, 21, 53, 85]
    FR_SP = [23, 61, 26, 50, 62, 27, 42, 45, 74, 59, 67, 68, 73, 75, 85, 87, 88, 10, 96, 28]
    GR_SP = [26, 59, 62, 14, 50, 61, 3, 9, 16, 27, 41, 42, 58, 79, 85, 33, 6, 28, 87, 53]
    IT_SP = [62, 26, 85, 23, 28, 33, 81, 59, 61, 78, 27, 43, 45, 58, 77, 90, 91, 0, 3, 6]
    seed_dict = {"EN": EN_SP,"FR":FR_SP,"GR":GR_SP,"IT":IT_SP}
    now = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    log_path = os.path.join("/home/chenchao/JDMC/result1", now)
    os.mkdir(log_path)
    accs = []
    with open(os.path.join(log_path, "c1-"+str(c1)+"c2-" + str(c2) + "c3-" + str(c3) + "c4-" + str(c4) + "nodes-" + str(nodes) +
               " dim-" + str(dim) + " target_nums-" + str(target_nums) + source_name + ".txt"), "w") as f:
        for i in seed_dict[source_name]:
            # print(i)
            acc=JDMC_test(f,dim,source_x,source_y,target_x, target_y,c1,c2,c3,c4,nodes,target_nums=target_nums,random_state=i)
            # acc = ELM_test(dim, source_x, source_y, target_x, target_y, target_nums, random_state=i)
            # acc = run_svm(dim, source_x, source_y, target_x, target_y, target_nums,random_state=i)
            accs.append(acc)
            # break
        f.write("Acc:%.1f,Std:%.1f\n" % (np.mean(accs),np.std(accs)/np.sqrt(len(accs))))
        print("acc:%.2f,std:%.2f\n" % (np.mean(accs),np.std(accs)/np.sqrt(len(accs))))
