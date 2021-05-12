import pandas as pd
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os,ssl,time

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
     ssl._create_default_https_context = ssl._create_unverified_context
X = np.load('image.npz')['arr_0']
y = pd.read_csv("alphabets.csv")["labels"]

print(pd.Series(y).value_counts())
classes = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","p","Q","R","S","T","U","V","W","X","Y","Z"]
nclasses = len(classes)

xtrain,xtest,ytrain,ytest = train_test_split(X,y,random_state = 9,train_size = 7500,test_size = 2500)
xtrainscale = xtrain/255
xtestscale = xtest/255

clf = LogisticRegression(solver = 'saga',multi_class  ='multinomial')
clf.fit(xtrainscale,ytrain)
yprediction = clf.predict(xtestscale)
accuracy = accuracy_score(ytest,yprediction)
print(accuracy)

cap  = cv2.VideoCapture(0)
while(True):
    try:
        ret,frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        height,width = gray.shape
        upperleft = (int(width / 2 - 56),int(height / 2 - 56))
        bottomright = (int(width / 2 +56),int(height / 2 + 56))
        cv2.rectangle(gray,upperleft,bottomright,(0,255,0),2)
        roi = gray[upperleft[1]:bottomright[1],upperleft[0]:bottomright[0]]
        image_pil = Image.fromarray(roi)
        imagebw = image_pil.convert('L')
        Image_bw_resized = imagebw.resize((28,28),Image.ANTIALIAS)
        Image_bw_resized_inverted = PIL.ImageOps.invert(Image_bw_resized)
        pixelfilter = 20
        minpixel = np.percentile(Image_bw_resized_inverted,pixelfilter)
        Image_bw_resized_inverted_scaled = np.clip(Image_bw_resized_inverted-minpixel,0,255)
        maxpixel = np.max(Image_bw_resized_inverted)
        Image_bw_resized_inverted_scaled = np.asarray(Image_bw_resized_inverted_scaled)/maxpixel
        testsample = np.array(Image_bw_resized_inverted_scaled).reshape(1,784)
        testprediction = clf.predict(testsample)
        print('predicted class is',testprediction)
        cv2.imshow('frame',gray)
        if cv2.waitKey(1)& 0XFF == ord('q'):
            break
    except Exception as e:
        pass
cap.release()
cv2.destroyAllWindows(()) 