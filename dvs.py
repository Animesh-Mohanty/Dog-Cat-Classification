import os
import pandas as pd
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
print("hello")
dvc = os.path.join('C:\\Users\\anime\\Desktop\\dogvscatp\\train')
categories = ['dogs','cats']
data =[]
label =[]
for cid,category in enumerate(categories):
    print('runing train ',cid)
    for f in os.listdir(os.path.join(dvc,category)):
        imgpath = os.path.join(dvc,category,f)
        img = imread(imgpath)
        img = resize(img,(16,16))
        data.append(img.flatten())
        label.append(cid)
xtrain = np.asarray(data)
ytrain = np.asarray(label)
mod = SVC()
mod.fit(xtrain,ytrain)
print('complete')
dvc2 = os.path.join("C:\\Users\\anime\\Desktop\\dogvscatp\\test")
categories2 = ['dogs','cats']
data2 =[]
label2 =[]
for cid,category in enumerate(categories2):
    print('running test ',cid)
    for f in os.listdir(os.path.join(dvc2,category)):
        imgpath = os.path.join(dvc2,category,f)
        img = imread(imgpath)
        img = resize(img,(16,16))
        data2.append(img.flatten())
        label2.append(cid)
xtest = np.asarray(data2)
ytest = np.asarray(label2)
ypred = mod.predict(xtest)
acc = accuracy_score(ytest,ypred)
print(acc)