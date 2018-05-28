
####### test my code ########################
from JDAC import *
from utils import *
from sklearn.decomposition import PCA


def ELM_test():
    ##data perpare
    # data=io.loadmat('mnist.mat')
    # train_x=data.get('train_x')
    # train_x=train_x[0:30000, :]
    # train_y0=data.get('train_y')
    # train_y=train_y0[0:30000, :]
    # #train_y=argmax(train_y0, 1) + 1
    # test_x=data.get('test_x')
    # test_y=data.get('test_y')
    # #test_y=argmax(test_y0, 1) + 1
    # train_x=array(train_x)
    # train_x=train_x.astype('float64')
    # test_x=array(test_x)
    # test_x=test_x.astype('float64')
    data=io.loadmat('D_SURF.mat')
    x=data.get('Xt')
    y=data.get('Yt')
    pca0=PCA(n_components=30)
    x=pca0.fit_transform(x)
    y=LabelTransform(y)
    train_x,train_y,test_x,test_y=RandomSplit(x,y,0.2)
    ##train model
    net=ELM(train_x,train_y,test_x,test_y,500)
    net.ParamInit()
    net.Activation('relu')
    net.TrainELM('Lp',10)
    net.TrainAccuracy('relu')
    net.TestAccuracy('relu')
    net.printf()


def ELM_T():
    train_x_s, train_y_s, train_x_t, train_y_t, test_x, test_y = get_off_cal('D_SURF.mat','W_SURF.mat',8,3,4)
    train_x_s,train_x_t,test_x=PCATransform(train_x_s,train_x_t,test_x)
    train_y_s=LabelTransform(train_y_s)
    train_y_t=LabelTransform(train_y_t)
    # train_y_t=hstack((train_y_s,train_y_t))
    test_y=LabelTransform(test_y)
    net=ELM(train_x_t,train_y_t,test_x,test_y,1000)
    net.ParamInit()
    net.Activation('relu')
    net.TrainELM('Lp',1)
    net.TrainAccuracy('relu')
    net.TestAccuracy('relu')
    net.printf()


def JDAC_test():
    train_x_s, train_y_s, train_x_t, train_y_t, test_x, test_y = get_off_cal('D_SURF.mat','W_SURF.mat',8,3,4)
    train_x_s,train_x_t,test_x=PCATransform(train_x_s,train_x_t,test_x)
    train_y_s=LabelTransform(train_y_s)
    train_y_t=LabelTransform(train_y_t)
    test_y=LabelTransform(test_y)
    net=JDAC(train_x_s,train_x_t,train_y_s,train_y_t,test_x,test_y,600,[0.01,10,50],100)
    net.ParamInit()
    net.Activation('relu')
    net.TrainJDAC()
    net.TestJDAC('relu')
    net.printf()
    return net.TestAcc





if __name__=='__main__':
    # Mean_acc=zeros(5)
    # for i in range(5):
    #     acc=JDAC_test()
    #     Mean_acc[i]=acc
    # print Mean_acc.mean()
    # ELM_test()
     ELM_T()
