from sklearn import svm
import numpy as np
from matplotlib import pyplot as plt
class SVMModel:
    def __init__(self,X,y,gamma='auto',probability=True,kernel='linear',C=1):
        self.X=X
        self.y=y
        self.gamma=gamma
        self.probability=probability
        self.kernel=kernel
        self.C=C
    def fit(self):
        clf = svm.SVC(gamma=self.gamma,probability=self.probability,kernel=self.kernel,C=self.C,tol=1e-5, cache_size=1000)  # 生成svm的分类模型
        self.model=clf.fit(self.X, self.y)  # 利用训练数据建立模型
    def predict(self,Xtest):
        predProb =self.model.predict_proba(Xtest)  # 预报测试集
        yHat =self.model.predict(Xtest)  
        return yHat,predProb
    def plot(self,xAxis=0,yAxis=1):
        X=self.X[:,[xAxis,yAxis]]
        y=self.y
        svc = svm.SVC(kernel=self.kernel, C=self.C, gamma=self.gamma, tol=1e-5, cache_size=1000).fit(X, y)
        
        xlist = np.linspace(X[:, 0].min(), X[:, 0].max(), 200)
        ylist = np.linspace(X[:, 1].min(), X[:, 1].max(), 200)
        XGrid, YGrid = np.meshgrid(xlist, ylist)
        # 预测并绘制结果
        Xtest=np.c_[XGrid.ravel(), YGrid.ravel()]
        Z = svc.predict(Xtest)
        Z = Z.reshape(XGrid.shape)
        plt.contourf(XGrid, YGrid, Z, cmap=plt.cm.hsv)
        #plt.contour(XGrid, YGrid, Z, colors=('k',))
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', linewidth=1.5, cmap=plt.cm.hsv)    
    def plotDecisionFunc(self,xAxis=0,yAxis=1):
        X=self.X[:,[xAxis,yAxis]]
        y=self.y
        #plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', linewidth=1.5, cmap=plt.cm.hsv)    
        plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='summer')

        model = svm.SVC(kernel=self.kernel, C=self.C, gamma=self.gamma, tol=1e-5, cache_size=1000).fit(X, y)
        
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()    
        x = np.linspace(xlim[0], xlim[1], 50)
        y = np.linspace(ylim[0], ylim[1], 50)
        Xgrid, Ygrid = np.meshgrid(x, y)
        xy = np.vstack([Xgrid.ravel(), Ygrid.ravel()]).T
        
        #计算每点到边界的距离(30,30)
        P = model.decision_function(xy)
        print(P.shape)
        if len(P.shape)!=1: # >2类问题，P不是1列
            P=P[:,1]    
        P = P.reshape(Xgrid.shape)
        #绘制等高线（距离边界线为0的实线，以及距离边界为1的过支持向量的虚线）
        ax.contour(Xgrid, Ygrid, P, colors='k',levels=[-1, 0, 1], linestyles=['--', '-', '-.'])    
        # 圈出支持向量
        ax.scatter(model.support_vectors_[:, 0],model.support_vectors_[:, 1],s=200,c='',edgecolors='k')
