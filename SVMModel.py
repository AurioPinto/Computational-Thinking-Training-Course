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