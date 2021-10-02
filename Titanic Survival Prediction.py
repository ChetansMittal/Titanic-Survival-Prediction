#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import pandas as pd


# In[21]:


dataset=pd.read_csv("Downloads/train.csv")


# In[22]:


print(dataset.shape)
print(dataset.head(5))


# In[23]:


income_set = set(dataset['Sex'])
dataset['Sex'] =dataset['Sex'].map({'female':0, 'male':1}).astype('int', errors='ignore')
print(dataset.head)


# In[25]:


X = dataset.drop('Survived',axis='columns')
X


# In[27]:


Y=dataset.Survived
Y


# In[42]:


X.columns[X.isna().any()]


# In[43]:


X.Age=X.Age.fillna(X.Age.mean())
X


# In[44]:


X.columns[X.isna().any()] 


# In[46]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25 , random_state=0)


# In[48]:


from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(X_train,Y_train)


# In[50]:


pclassNo = int(input("Enter Person's Pclass number: "))
gender = int(input("Enter Person's Gender 0-female 1-male(0 or 1) : "))
age = int(input("Enter Person's Age: "))
fare = float(input("Enter Person's Fare: "))
person = [[pclassNo, gender , age, fare] ]
result = model . predict(person)

print (result)
if result == 1:
    print("Person might be Survived")
else:
    print( "Person might not be Survived")


# In[52]:


y_pred=model.predict(X_test)
print(np.column_stack((y_pred,Y_test)))


# In[54]:


from sklearn.metrics import accuracy_score
print("Accuracy of the model: {0}%".format(accuracy_score(Y_test,y_pred)*100)) 

