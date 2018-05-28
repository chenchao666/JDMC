## By chenchao 2017-12-20
## mail: chench@zju.edu.cn

from numpy import *
from numpy.linalg import *
from scipy.linalg import sqrtm
import time
from IPLearning import *
from numba import jit


class ELM:
    def __init__(self,train_x,train_y,test_x,test_y,NumofHiddenNodes):
        self.train_x=train_x
        self.train_y=train_y
        self.test_x=test_x
        self.test_y=test_y
        self.n_feature=self.train_x.shape[1]
        self.num=NumofHiddenNodes
        self.n_class=self.train_y.shape[1]
        self.runtime=time.time()


    def ParamInit(self):
        self.InputWeighws=2*random.random((self.num,self.n_feature))-1
        self.H=dot(self.InputWeighws,self.train_x.T)


    def Activation(self,type):
        if (type == 'sigmoid'):
            self.H=1/(1+exp(self.H))
        if (type == 'tanh'):
            self.H=(exp(self.H)-exp(-self.H))/(exp(self.H)+exp(-self.H))
        if (type == 'relu'):
            ind =self.H>0
            self.H=self.H*ind
        if (type=='IPL'):
            self.H=IPL(self.H)


    def TrainELM(self,type,coeff):
        if (type=='None'):
            temp=mat(dot(self.H,self.H.T))
            temp=temp.I
            self.OutputWeights=dot(dot(temp,self.H),self.train_y)
            self.OutputWeights=array(self.OutputWeights)
        if (type=='Lp'):
            temp=mat(dot(self.H,self.H.T))
            temp=temp+eye(temp.shape[0])/coeff
            temp=temp.I
            self.OutputWeights = dot(dot(temp,self.H),self.train_y)
            self.OutputWeights = array(self.OutputWeights)
        self.runtime = time.time()-self.runtime


    def TestELM(self,type):
        self.H = dot(self.InputWeighws, self.test_x.T)
        self.Activation(type)
        self.Actual_y=dot(self.H.T,self.OutputWeights)


    def TrainAccuracy(self,type):
        self.H = dot(self.InputWeighws, self.train_x.T)
        self.Activation(type)
        self.Actual_y=dot(self.H.T, self.OutputWeights)
        out_y = argmax(self.Actual_y, 1)+1
        actual_y = argmax(self.train_y,1)+1
        ind = where(out_y==actual_y)
        self.TrainAcc = floor(len(ind[0]))/len(self.train_y)



    def TestAccuracy(self,type):
        self.H=dot(self.InputWeighws,self.test_x.T)
        self.Activation(type)
        self.Actual_y=dot(self.H.T,self.OutputWeights)
        out_y=argmax(self.Actual_y,1)+1
        actual_y = argmax(self.test_y, 1) + 1
        ind=where(out_y==actual_y)
        self.TestAcc=floor(len(ind[0]))/len(self.test_y)

    def printf(self):
        print 'train accuracy:'
        print self.TrainAcc
        print 'test accuracy:'
        print self.TestAcc
        print 'run time(s):'
        print self.runtime


####################################################################################
####################################################################################
####################################################################################

class JDMC0:
    def __init__(self,Xs,Xt,Ys,Yt,test_x,test_y,NumOfHiddenNodes,C=[1,1,1],iter=100):
        self.Xs=Xs
        self.Xt=Xt
        self.Ys=Ys
        self.Yt=Yt
        self.test_x=test_x
        self.test_y=test_y
        self.n_feature=self.Xs.shape[1]
        self.num=NumOfHiddenNodes
        self.n_class=self.Ys.shape[1]
        self.C1=C[0]
        self.C2=C[1]
        self.C3=C[2]
        self.iter=iter


    def ParamInit(self):
        self.runtime=time.time()
        self.InputWeights1=2*random.random((self.num,self.n_feature))-1
        self.InputWeights2=2*random.random((self.num,self.n_feature))-1
        self.Hs=mat(self.InputWeights1)*mat(self.Xs.T)
        self.Hs=self.Hs.T
        self.Ht=mat(self.InputWeights2)*mat(self.Xt.T)
        self.Ht=self.Ht.T
        self.OutputWeights=zeros((self.num,self.n_class))
        self.D=eye(self.num)
        self.Init_M()
        self.Hs=mat(self.Hs)*mat(self.M)
        self.Calculate_U()



    def Init_M(self):
        Cs=mat(cov(self.Hs.T))+100*eye(self.num)
        Ct=mat(cov(self.Ht.T))+100*eye(self.num)
        self.M0=sqrtm(Ct)*pinv(sqrtm(Cs))
        self.M=real(self.M0)
        # self.M=self.D


    def Calculate_U(self):
        self.U=zeros((self.num,self.num))
        self.Us=zeros((self.n_class+1,self.num))
        self.Ut=zeros((self.n_class+1,self.num))
        for i in range(self.Us.shape[0]):
            if (i==0):
                self.Us[i,:]=self.Hs.mean(axis=0)
            else:
                ind0=argmax(self.Ys,1)+1
                ind=where(ind0==i)
                self.Us[i,:]=self.Hs[ind[0],:].mean(axis=0)

        for i in range(self.Ut.shape[0]):
            if (i==0):
                self.Ut[i,:]=self.Ht.mean(axis=0)
            else:
                ind0=argmax(self.Yt,1)+1
                ind=where(ind0==i)
                self.Ut[i,:]=self.Ht[ind[0],:].mean(axis=0)

        delta_U=mat(self.Us-self.Ut)
        for i in range(delta_U.shape[0]):
            self.U=self.U+delta_U[i,:].T*delta_U[i,:]



    def Activation(self,type):
        self.Hs=array(self.Hs)
        self.Ht=array(self.Ht)
        if (type == 'sigmoid'):
            self.Hs=1/(1+exp(self.Hs))
            self.Ht=1/(1+exp(self.Ht))
        if (type == 'tanh'):
            self.Hs=(exp(self.Hs)-exp(-self.Hs))/(exp(self.Hs)+exp(-self.Hs))
            self.Ht=(exp(self.Ht)-exp(-self.Ht))/(exp(self.Ht)+exp(-self.Ht))
        if (type == 'relu'):
            ind1=self.Hs>0
            self.Hs=self.Hs*ind1
            ind2= self.Ht>0
            self.Ht=self.Ht*ind2
        if (type=='IPL'):
            self.Hs=IPL(self.Hs)
            self.Ht=IPL(self.Ht)


    def TrainJDMC(self):
        k=0
        self.Hs=mat(self.Hs)
        self.Ht=mat(self.Ht)
        self.M=mat(self.M)
        self.U=mat(self.U)
        self.Ys=mat(self.Ys)
        self.Yt=mat(self.Yt)
        I=mat(eye(self.num))
        while (k<3):
            n=1
            self.D=I
            while (n<self.iter):
                temp1=self.Ht.T*self.Ht+self.C1*self.M.T*self.Hs.T*self.Hs*self.M+self.C3*self.U*self.C4*self.D
                temp2=self.Ht.T*self.Yt+self.C1*self.M.T*self.Hs.T*self.Ys
                self.OutputWeights=pinv(temp1)*temp2
                self.Update_D()
                n=n+1
            temp3=self.C1*self.Hs.T*self.Hs+self.C2*I
            temp4=self.C1*self.Hs.T*self.Ys*self.OutputWeights.T+self.C2*self.M0
            self.M=pinv(temp3)*temp4
            k+=1
        self.runtime=time.time()-self.runtime


    def Update_D(self):
        for i in range(self.D.shape[0]):
            self.D[i,i]=1.0/(2*norm(self.OutputWeights[i,:])+0.001) #self.D[i,i]=1.0/(2*norm(self.OutputWeights[i,:])+0.001)

    def TestJDMC(self,type):
        self.H=dot(self.InputWeights2,self.test_x.T)
        self.Activation(type)
        self.Actual_y = array(self.H.T*self.OutputWeights)
        self.out_y=argmax(self.Actual_y,1)+1
        self.actual_y=argmax(self.test_y,1)+1
        ind=where(self.out_y==self.actual_y)
        self.TestAcc=floor(len(ind[0]))/len(self.test_y)

    def printf(self):
        print ('Test Accuracy:#######', self.TestAcc)
        print ('run time:', self.runtime)



#################################################################################
class JDMC(JDMC0):

    def TrainJDAC(self):
        k=0
        self.Hs=mat(self.Hs)
        self.Ht=mat(self.Ht)
        self.M=mat(eye(self.num))
        self.U=mat(self.U)
        self.Ys=mat(self.Ys)
        self.Yt=mat(self.Yt)
        I=mat(eye(self.num))
        while (k<1):
            n=1
            self.D=I
            while (n<self.iter):
                temp1=self.Ht.T*self.Ht+self.C1*self.M.T*self.Hs.T*self.Hs*self.M+self.C2*self.U+self.C3*self.D
                temp2=self.Ht.T*self.Yt+self.C1*self.M.T*self.Hs.T*self.Ys
                self.OutputWeights=pinv(temp1)*temp2
                self.Update_D()
                n=n+1
            temp3=self.Hs.T*self.Hs
            temp4=self.OutputWeights*self.OutputWeights.T
            self.M=pinv(temp3)*self.Hs.T*self.Ys*self.OutputWeights.T*pinv(temp4)
            k+=1
        self.runtime=time.time()-self.runtime



###########################################################
class JDAC1(JDMC0):

    def TrainJDMC(self):
        self.Hs = mat(self.Hs)
        self.Ht = mat(self.Ht)
        self.M = mat(eye(self.num))
        self.U = mat(self.U)
        self.Ys = mat(self.Ys)
        self.Yt = mat(self.Yt)
        I = mat(eye(self.num))
        temp1 = self.Ht.T * self.Ht + self.C1 * self.Hs.T * self.Hs + self.C2 * self.U + self.C3 * I
        t = self.Ht.T * self.Ht
        a = self.C1 * self.Hs.T * self.Hs
        b = self.C2 * self.U
        c = self.C3 * I
        temp2 = self.Ht.T * self.Yt + self.C1 * self.Hs.T * self.Ys
        self.OutputWeights=pinv(temp1) * temp2
        # self.OutputWeights = temp1                                                                                                                                                           self.OutputWeights = pinv(temp1) * temp2
        self.runtime = time.time() - self.runtime


################################################################################
class JDMC2(JDMC0):

    def TrainJDMC(self):
        self.Hs = mat(self.Hs)
        self.Ht = mat(self.Ht)
        self.U = mat(self.U)
        self.Ys = mat(self.Ys)
        self.Yt = mat(self.Yt)
        I = mat(eye(self.num))
        n = 1
        self.D = I
        while (n < self.iter):
            temp1 = self.Ht.T * self.Ht + self.C1*self.Hs.T*self.Hs + self.C2 * self.U + self.C3 * self.D
            temp2 = self.Ht.T * self.Yt + self.C1 * self.Hs.T * self.Ys
            temp=self.U
            self.OutputWeights = pinv(temp1) * temp2
            self.Update_D()
            n = n+1
        self.runtime = time.time()-self.runtime



############################################################################################
#################################################################################################
##  main code for train our paper Joint Domain Matching and Classification for Cross-Domain adaptation Via ELM
class JDMC3(JDMC0):

    def TrainJDMC(self):
        k=0
        self.Hs = mat(self.Hs)
        self.Ht = mat(self.Ht)
        self.Pt = mat(eye(self.Ht.shape[0]))
        self.Ps = mat(eye(self.Hs.shape[0]))
        self.U = mat(self.U)
        self.Ys = mat(self.Ys)
        self.Yt = mat(self.Yt)
        I = mat(eye(self.num))
        while (k<3):
            n = 1
            self.D = I
            # OutputWeights_old = self.OutputWeights
            while (n<self.iter):
                temp1 = self.Ht.T * self.Ht + self.C1 * self.Hs.T * self.Hs + self.C2 * self.U + self.C3 * self.D
                temp2 = self.Ht.T * self.Pt * self.Yt + self.C1 * self.Hs.T * self.Ps * self.Ys
                self.OutputWeights=pinv(temp1) * temp2
                self.Update_D()
                n=n+1
            # difference=norm(self.OutputWeights-OutputWeights_old)
            # print difference
            Rs = 100 * eye(self.Hs.shape[0])
            Rt = 100 * eye(self.Ht.shape[0])
            self.Pt = (self.Ht * self.OutputWeights * self.Yt.T + Rt) * pinv(self.Yt * self.Yt.T + Rt)
            self.Ps = (self.Hs * self.OutputWeights *self.Ys.T + Rs) * pinv(self.Hs *self.Hs.T + Rs)
            k+=1
        self.runtime=time.time()-self.runtime

























