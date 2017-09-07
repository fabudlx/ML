
# coding: utf-8

# In[81]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import uniform
import inspect
import collections


# In[2]:

df = pd.read_csv("C:\\Users\\Fabian\\Desktop\\ML\\5\\dataCircle.txt", delim_whitespace=True);

#df['color'] = df.label.apply(lambda l: 'green' if l == 1 else 'red')
df['X'] = [np.array([x, y]) for (x, y) in zip(df.x1, df.x2)]


# In[76]:

def getErrors(funcList, D): 
    e = []
    for func in funcList:
        errors = np.array([y!=func.calc(x) for y,x in zip(labels,data)])
#             print(errors)
        e.append([(errors*D).sum(),func, func.t,func.dim])
    return e


# In[115]:

def set_rule(funcList, D, ALPHA,RULES):
    for func in funcList:
        errors = np.array([y!=func.calc(x) for y,x in zip(labels,data)])
        e = (errors*D).sum()
        alpha = 0.5 * np.log((1-e)/e)
#             print ('e=%.2f a=%.2f'%(e, alpha))
        w = np.zeros(m)
        for i in range(m):
            if errors[i] == 1: w[i] = D[i] * np.exp(alpha)
            else: w[i] = D[i] * np.exp(-alpha)
        D = w / w.sum()
        RULES.append(func)
        ALPHA.append(alpha)
    return ALPHA,RULES,D


# In[70]:

def evaluate(ALPHA ,RULES):
    NR = len(RULES)
    wrong =[]
    for (x,l) in zip(data,labels):
        hx = [ALPHA[i]*(RULES[i].calc(x)) for i in range(NR)]
        wrong.append(np.sign(l) == np.sign(sum(hx)))
    return wrong      


# In[71]:

class wc_b:
    def __init__(self,t,dim):
        self.t = t
        self.dim = dim
        
    def calc(self,x):
        if x[self.dim] > self.t:
            return 1
        else: return -1
    
class wc_s:
    def __init__(self,t,dim):
        self.t = t
        self.dim = dim
        
    def calc(self,x):
        if x[self.dim] < self.t:
            return 1
        else: return -1


# In[112]:

m = df.shape[0]
D = np.ones(m)/m
RULES = []
ALPHA = []
data = df.X
labels = df.label
winner =[]


# In[118]:


t = -10
funcList_x_b,funcList_x_s,funcList_y_b,funcList_y_s  = [],[],[],[]

for i in range(200):
    funcList_x_b.append(wc_b(t,0))
    funcList_x_s.append(wc_s(t,0))
    funcList_y_b.append(wc_b(t,1))
    funcList_y_s.append(wc_s(t,1))
    t += 0.1
    
#print([i.t for i in funcList])    
    
x_b = sorted(getErrors(funcList_x_b, D), key = lambda x: x[0])
winner.append(x_b[0])
x_s = sorted(getErrors(funcList_x_s, D), key = lambda x: x[0])
winner.append(x_s[0])
y_b = sorted(getErrors(funcList_y_b, D), key = lambda x: x[0])
winner.append(y_b[0])
y_s = sorted(getErrors(funcList_y_s, D), key = lambda x: x[0])
winner.append(y_s[0])

print(winner)
for i in range(5):
    funcListWinners = [i[1] for i in winner]
    ALPHA,RULES,D = set_rule(funcListWinners, D, ALPHA,RULES)
    wrong = evaluate(ALPHA, RULES)
wrongCount = collections.Counter(wrong)
wrongCount[False]/len(wrong)


# In[123]:

weakClass = [[i[2],i[3]] for i in winner]
weakClass


# In[130]:

ALPHA


# In[126]:

# plotError()


# In[90]:

def plotError():
    df['classified'] = wrong
    fig, ax = plt.subplots(num=None, figsize=(6, 6), dpi=100, facecolor='w', edgecolor='k');
    ax.scatter(df.x1, df.x2, c= df.classified.apply(lambda l: 'red' if l == False else 'black'));
    ax.plot()
    ax.set_title('AdaBoost')
    plt.show()


# In[129]:

fig, ax = plt.subplots(num=None, figsize=(6, 6), dpi=100, facecolor='w', edgecolor='k');
ax.scatter(df.x1, df.x2, c= df.label.apply(lambda l: 'green' if l == 1 else 'red'));
for classifier in weakClass:
    if classifier[1] is 0:
        ax.plot([classifier[0],classifier[0]],[-10,10])
    else:
        ax.plot([-10,10],[classifier[0],classifier[0]])
ax.set_title('AdaBoost')
plt.show()


# In[73]:

li = []
func = lambda x: x+y
for i in range(10):
    li.append(func)


# In[ ]:



