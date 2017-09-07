
# coding: utf-8

# In[7]:

import numpy as np
import cv2
import os
from scipy import misc
import matplotlib.pyplot as plt
import glob
import pandas as pd
import math
from numpy.linalg import inv


# In[4]:

imgList = []
for filename in glob.glob('C:\\Users\\Fabian\\Desktop\\ML\\3\\positives\\*.png'):
    imgList.append( cv2.imread(filename,cv2.IMREAD_COLOR ))
for filename in glob.glob('C:\\Users\\Fabian\\Desktop\\ML\\3\\negatives\\*.png'):
    imgList.append( cv2.imread(filename,cv2.IMREAD_COLOR ))
    


# In[5]:

#s = pd.Series(imgList)
df = pd.DataFrame([i] for i in imgList)
df.columns = ['data']
lables = [0 if y <30 else 1 for y in range(60)]
df['lables'] = pd.Series(lables)

df.tail(5)


# In[6]:

def getFeatures(imgList):
    featureList = []
    for image in imgList:
        features = [];
        features.append(np.amin(image[:,:,0]))
        features.append(np.amin(image[:,:,1]))
        features.append(np.amin(image[:,:,2]))
        features.append(np.amax(image[:,:,0]))
        features.append(np.amax(image[:,:,2]))
        featureList.append(features)
    return featureList;


# In[7]:

features = getFeatures(df['data'])
df['lowestR'] = pd.Series(i[0] for i in features)
df['lowestG'] = pd.Series(i[1] for i in features)
df['lowestB'] = pd.Series(i[2] for i in features)
df['highestR'] = pd.Series(i[3] for i in features)
df['highestB'] = pd.Series(i[4] for i in features)


# In[8]:

df.head(5)


# In[9]:

positives = df[df.lables == 0]
negatives = df[df.lables == 1]
featuresPositive = positives[['lowestR','lowestG','lowestB','highestR','highestB']]
featuresNegative = negatives[['lowestR','lowestG','lowestB','highestR','highestB']]
featuresNegative.head(10)


# In[10]:

def getCovariance(featureVectors):
    features = np.array(featureVectors)
    mean = np.array(featureVectors.mean());
    m = len(featureVectors);
    covariance = 1/m* sum([((feature-mean).reshape((-1, 1)) * (feature-mean)) for feature in features])
    return covariance


# In[11]:

def getProb (sample, covM, mean):
    exp = -0.5*np.matmul(np.matmul(sample-mean,inv(covM)),(sample-mean).reshape((-1,1)))
    #print(np.exp(exp))
    return (1/((2*math.pi)*np.linalg.det(covM)**0.5))*np.exp(exp)


# In[12]:

features = df[['lowestR','lowestG','lowestB','highestR','highestB']]
lables = df[['lables']]


# In[13]:

covM = getCovariance(np.array(features))
meanP = np.array(featuresPositive.mean());
meanN = np.array(featuresNegative.mean());
phi = 1/len(df) * len(positives)


# In[14]:

features = np.array(features);
lables = np.array(lables);
getProb(features[1],covM,meanP)


# In[15]:


c = 0;
wrong = 0;
for x in features:
    if getProb(x,covM,meanP) > getProb(x,covM,meanN):
        if(lables[c] != 0):
            wrong += 1;
            print('FN',c-29)
    else:
        if(lables[c] != 1):
            wrong += 1; 
            print('FP',c)
    c += 1;

print(wrong*100/len(features),'% wrong')
         


# In[19]:


get_ipython().magic('matplotlib inline')
a = np.array(df.iloc[[35]])
a = a[0,0]
print(a.shape)
plt.imshow(np.asarray(a))


# In[ ]:



