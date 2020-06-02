#!/usr/bin/env python
# coding: utf-8

# In[69]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pt


# In[70]:


df=pd.read_csv('bank-additional.csv',delimiter=";")
df.head()


# In[71]:


df.shape


# #DATA PREPROCESSING

# In[72]:


y=df["y"]
x=df.drop(["y"],axis=1)


# In[73]:


from sklearn.preprocessing import LabelEncoder
LEC=LabelEncoder()
for i in range(len(x.columns)):
    x.iloc[:,i]=LEC.fit_transform(x.iloc[:,i])


# In[74]:


x.head()


# In[75]:


df.applymap(np.isreal).head()


# In[76]:


from sklearn.preprocessing import StandardScaler
stc=StandardScaler()
x=stc.fit_transform(x)


# In[77]:


x


# In[78]:


pd.DataFrame(x).head()


# In[79]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)


# In[80]:


#TRAIN MODEL
#from sklearn.linear_model import LogisticRegression
#lre=LogisticRegression()
#lre.fit(x_train,y_train)


# In[81]:


#TEST MODEL AND PREDICT
#y_pred=lre.predict(x_test)


# In[82]:


#CHECKING ACCURACY OF MODEL
from sklearn import metrics
#print("Accuracy",metrics.accuracy_score(y,lre.predict(x)))


# In[83]:


#df3=pd.DataFrame({"Actual":y.values,"Predicted":lre.predict(x)})
#df3.head(15)


# In[84]:


#number_of_unequal_values=0
#for i in range(len(df3)):
#    if df3.iloc[i,0]!=df3.iloc[i,1]:
#     number_of_unequal_values=number_of_unequal_values+1
  
#print(number_of_unequal_values)


# In[85]:


#IMPORTING SUPPORT VECTOR MACHINE
from sklearn.svm import SVC
svc_classifier=SVC(kernel='linear',random_state=0)
svc_classifier.fit(x_train,y_train)


# In[86]:


y_pred_svcclassifier=svc_classifier.predict(x_test)


# In[87]:


print("Accuracy",metrics.accuracy_score(y,svc_classifier.predict(x)))


# In[88]:


svc_classifier_rbf=SVC(kernel='rbf',random_state=0)
svc_classifier_rbf.fit(x_train,y_train)


# In[89]:


y_pred_svcclassifier_rbf=svc_classifier_rbf.predict(x_test)


# In[90]:


print("Accuracy",metrics.accuracy_score(y,svc_classifier_rbf.predict(x)))


# In[ ]:




