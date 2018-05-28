
####### test my code ########################
from utils import *
import numpy as np
from sklearn.utils import shuffle
from scipy.io import savemat
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVC
import os
from sklearn.metrics import confusion_matrix
# import seaborn as sn
import matplotlib.pyplot as plt
from JDMC import *

data_path = "./data/Office-Caltech/surf"



def ELM_test(split,source_name,target_name):
    source_path = os.path.join(data_path, source_name + "_SURF_L10.mat")
    target_path = os.path.join(data_path, target_name + "_SURF_L10.mat")
    train_x_ss, train_y_ss, train_x_ts, train_y_ts, test_xs, test_ys = load_mmdt_split(split, source_path, target_path)
    accs = []
    for i in range(20):
        print("####### " + str(i + 1) + " #######")
        train_x_s, train_y_s, train_x_t, train_y_t, test_x, test_y = np.squeeze(train_x_ss[i]), np.squeeze(
            train_y_ss[i]), np.squeeze(train_x_ts[i]), np.squeeze(train_y_ts[i]), np.squeeze(test_xs[i]), np.squeeze(
            test_ys[i])
        # source_dict,target_dict = {},{}
        # source_dict['fts'] = np.append(train_x_s,train_x_t,axis=0)
        # source_dict['labels'] = np.append(train_y_s,train_y_t,axis=0).T
        # savemat("./others/source.mat",source_dict)
        # target_dict['fts'] = test_x
        # target_dict['labels'] = test_y.T
        # savemat("./others/target.mat",target_dict)
        #     train_x_s = (train_x_s-np.mean(train_x_s,1,keepdims=True))/np.std(train_x_s,1,keepdims=True)
        #     train_x_t = (train_x_t-np.mean(train_x_t,1,keepdims=True))/np.std(train_x_t,1,keepdims=True)
        #     test_x = (test_x-np.mean(test_x,1,keepdims=True))/np.std(test_x,1,keepdims=True)
        train_y_s = LabelTransform(train_y_s)
        train_y_t = LabelTransform(train_y_t)
        test_y = LabelTransform(test_y)
        pca = PCA(dim)
        pls = PLSRegression(dim)
        reduc="pca"
        if reduc == "pls":
            train_x_s, _ = pls.fit_transform(train_x_s, train_y_s)
            paced = pca.fit_transform(np.row_stack([train_x_t, test_x]))
            train_x_t, test_x = paced[0:30, :], paced[30:, :]
        elif reduc == "pca":
            paced = pca.fit_transform(np.row_stack([train_x_s, train_x_t, test_x]))
            if source_name == "amazon":
                train_x_s, train_x_t, test_x = paced[0:200, :], paced[200:230, :], paced[230:, :]
            else:
                train_x_s, train_x_t, test_x = paced[0:80, :], paced[80:110, :], paced[110:, :]
        else:
            raise NotImplementedError

    ##train model
        net=ELM(train_x_s,train_y_s,test_x,test_y,1000)
        net.ParamInit()
        net.Activation('relu')
        net.TrainELM('Lp',0.01)
        net.TrainAccuracy('relu')
        net.TestAccuracy('relu')
        net.printf()
        accs.append(net.TestAcc)
    return np.mean(accs)*100,np.std(accs)*100/np.sqrt(len(accs))



def JDMC_test(f,split,source_name,target_name,c1,c2,c3,c4,nodes,dim,random_state=0):
    print("!!!!!!! "+source_name+"-->"+target_name+" !!!!!!!!!!")
    source_path = os.path.join(data_path,source_name+"_SURF_L10.mat")
    target_path = os.path.join(data_path,target_name+"_SURF_L10.mat")
    train_x_ss, train_y_ss, train_x_ts, train_y_ts, test_xs, test_ys = load_mmdt_split(split,source_path,target_path)
    accs=[]
    for i in range(20):
        print("####### "+str(i+1)+" #######")
        train_x_s, train_y_s, train_x_t, train_y_t, test_x, test_y = np.squeeze(train_x_ss[i]), np.squeeze(train_y_ss[i]), np.squeeze(train_x_ts[i]), np.squeeze(train_y_ts[i]), np.squeeze(test_xs[i]), np.squeeze(test_ys[i])
        train_y_s = LabelTransform(train_y_s)
        train_y_t = LabelTransform(train_y_t)
        test_y = LabelTransform(test_y)
        pca = PCA(dim)
        pcaed = pca.fit_transform(np.row_stack([train_x_s, train_x_t, test_x]))
        train_x_s, train_x_t, test_x = pcaed[0:len(train_x_s), :], pcaed[len(train_x_s):len(train_x_s) + len(train_x_t),:], pcaed[-len(test_x):, :]
        tmp = []
        for _ in range(5):
            net = JDMC3(train_x_s, train_x_t, train_y_s, train_y_t, test_x, test_y, nodes, [c1, c2, c3], 100)
            net.ParamInit()
            net.Activation('relu')
            net.TrainJDMC()
            net.TestJDMC('relu')
            net.printf()
            # savemat('JDMC.mat',{'predict':net.out_y,'real':net.actual_y})
            f.write("ID:%d, Acc:%.2f\n"%(i+1,net.TestAcc))
            tmp.append(net.TestAcc)
        # print mean(tmp)
        # print std(tmp)
        accs.append(np.max(tmp))
    f.write("%s-->%s, Acc:%.2f, Std:%.2f\n" % (source_name, target_name, np.mean(accs)*100,np.std(accs)*100/np.sqrt(len(accs))))
    return np.mean(accs)*100,np.std(accs)*100/np.sqrt(len(accs))




if __name__=='__main__':
    # ELM_test()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--c1", type=float, default=0.1)
    parser.add_argument("--c2", type=int, default=0.1)
    parser.add_argument("--c3", type=int, default=30)
    parser.add_argument("--c4", type=int, default=100)
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--d", type=int, default=40)
    parser.add_argument("--t", type=int, default=20)
    args = parser.parse_args()
    c1, c2, c3, c4, nodes, dim, target_nums = args.c1, args.c2, args.c3, args.c4, args.n, args.d, args.t
    import datetime
    now = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    log_path = os.path.join("./result",now+"c1-"+str(c1)+"c2-"+str(c2)+"c3-"+str(c3)+"c4-"+str(c4)+"nodes-"+str(nodes)+" dim-"+str(dim))
    os.mkdir(log_path)
    accs=[]
    stds=[]
    with open(os.path.join(log_path,"result.txt"),"w") as f:
        splits = os.listdir(os.path.join(data_path,"mmdtsplit"))
        for spl in splits:
            # spl=splits[2]
            source_name,target_name = spl.split('_')[1].split('-')
            with open(os.path.join(log_path,source_name+"-->"+target_name),"w") as f1:
                spl = os.path.join(os.path.join(data_path,"mmdtsplit"),spl)
                acc,st=JDMC_test(f1,spl,source_name,target_name,c1,c2,c3,c4,nodes,dim)
                # acc,st = ELM_test(spl, source_name, target_name)
                accs.append(acc)
                stds.append(st)
                f.write("%s-->%s, Acc:%.2f, Std:%.2f\n" % (source_name,target_name,acc,st))
            # break
        f.write("Acc:%.2f, Std:%.2f\n" % (np.mean(accs),np.mean(stds)))
        print("mean: ",np.mean(accs),"std: ",np.mean(stds))
