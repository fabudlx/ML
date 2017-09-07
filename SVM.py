
# coding: utf-8

# In[1]:

from svmutil import *
import ML_GLM as data;
import numpy as np


# In[2]:

x = np.array(data.df[['lowestR','lowestG','lowestB','highestR','highestB']])
y = np.array(data.df[['lables']])
x = x.tolist()
y = y.tolist()
y = [i[0] for i in y]


# In[9]:

#y, x = svm_read_problem('C:\ProgramData\Anaconda3\Lib\libsvm-3.22\heart_scale')

m = svm_train(y, x, '-t 3')
p_label, p_acc, p_val = svm_predict(y, x, m)
print(p_acc)


# In[ ]:

# options:
# -s svm_type : set type of SVM (default 0)
# 	0 -- C-SVC
# 	1 -- nu-SVC
# 	2 -- one-class SVM
# 	3 -- epsilon-SVR
# 	4 -- nu-SVR
# -t kernel_type : set type of kernel function (default 2)
# 	0 -- linear: u'*v
# 	1 -- polynomial: (gamma*u'*v + coef0)^degree
# 	2 -- radial basis function: exp(-gamma*|u-v|^2)
# 	3 -- sigmoid: tanh(gamma*u'*v + coef0)
# -d degree : set degree in kernel function (default 3)
# -g gamma : set gamma in kernel function (default 1/num_features)
# -r coef0 : set coef0 in kernel function (default 0)
# -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
# -n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
# -p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
# -m cachesize : set cache memory size in MB (default 100)
# -e epsilon : set tolerance of termination criterion (default 0.001)
# -h shrinking: whether to use the shrinking heuristics, 0 or 1 (default 1)
# -b probability_estimates: whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
# -wi weight: set the parameter C of class i to weight*C, for C-SVC (default 1)

# The k in the -g option means the number of attributes in the input data.


# In[33]:

y, x = svm_read_problem('C:\ProgramData\Anaconda3\Lib\libsvm-3.22\heart_scale')
m = svm_train(y, x, '-t 1 -d 4 -g 0.5')
p_label, p_acc, p_val = svm_predict(y, x, m)
print(p_acc)


# In[21]:

y, x = svm_read_problem('C:\ProgramData\Anaconda3\Lib\libsvm-3.22\heart_scale')
m = svm_train(y, x, '-t 1 -d 3')
p_label, p_acc, p_val = svm_predict(y, x, m)
print(p_acc)


# In[37]:

y, x = svm_read_problem('C:\ProgramData\Anaconda3\Lib\libsvm-3.22\heart_scale')
m = svm_train(y, x, '-t 3 -r 3 -g 0.5')
p_label, p_acc, p_val = svm_predict(y, x, m)
print(p_acc)


# In[25]:

y, x = svm_read_problem('C:\ProgramData\Anaconda3\Lib\libsvm-3.22\heart_scale')
m = svm_train(y, x, '-t 3 -g 0.1')
p_label, p_acc, p_val = svm_predict(y, x, m)
print(p_acc)


# In[26]:

y, x = svm_read_problem('C:\ProgramData\Anaconda3\Lib\libsvm-3.22\heart_scale')
m = svm_train(y, x, '-t 2 -g 5')
p_label, p_acc, p_val = svm_predict(y, x, m)
print(p_acc)


# In[ ]:



