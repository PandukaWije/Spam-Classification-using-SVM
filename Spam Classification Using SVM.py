#!/usr/bin/env python
# coding: utf-8

# 
# # Accuracy enhancing techniques : SMOTE, Grid Search
#  

# In[1]:


import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


# In[2]:


df = pd.read_csv('spam.csv')


# In[3]:


df.head(5)


# # **Checking Class Imbalance**
# 

# In[4]:


df['Label'].value_counts()


# In[5]:


y = df['Label'].values
x = df['EmailText'].values


# In[6]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# In[7]:


print(f'x_train shape: {x_train.shape}')
print(f'y_train shape: {y_train.shape}')


# # Transform the text using Count Vectorizer

# In[8]:


cvec = CountVectorizer() 
x_train = cvec.fit_transform(x_train)
x_test = cvec.transform(x_test)


# # Applying SMOTE Algorithm

# since there is a class imbalance in the dataset

# In[ ]:


#  %pip install imblearn


# In[9]:


from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 1)

x_res, y_res = sm.fit_resample(x_train, y_train)


# In[10]:


print(f'x_train shape (after SMOTE): {x_res.shape}')
print(f'y_train shape (after SMOTE): {y_res.shape}')


# # Traning the Model
# 

# using the resampled data from SMOTE

# In[11]:


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'gamma':['scale', 'auto'],
              'kernel': ['rbf','linear','sigmoid']}  

   
grid = GridSearchCV(SVC(), param_grid, refit = True, n_jobs=-3) 


# In[12]:


# fitting the model for grid search 
grid.fit(x_res, y_res)


# In[13]:


# print best parameter after tuning 
print(grid.best_params_) 


# In[14]:


y_pred = grid.predict(x_test) 


# In[15]:


# print classification report 
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# # Applying the Prediction to the target

# expected output : ['spam', 'ham', 'spam', 'spam', 'ham'] (for the given email set)

# In[16]:


pred_set = ["Hey, you have won a car !!!!. Conrgratzz",
            "Dear applicant, Your CV has been recieved. Best regards",
            "You have received $1000000 to your account",
            "Join with our whatsapp group",
            "Kindly check the previous email. Kind Regards"]


# this line is a test made up by me
# expected output ['spam' 'ham']
pred_set_2 = ["Free calls, Free money, U have won a prize",
              "Mom i will meet your at the gate near the red tree, see you soon."]


# # Transform the target text using Count Vectorizer

# In[17]:


# transformed_pred_set = cvec.fit(pred_set)
pred_set = cvec.transform(pred_set)
pred_set_2 = cvec.transform(pred_set_2)


# # Printing the Predictions

# In[18]:


# this is for the given emails
print(grid.predict(pred_set))


# In[19]:


# this is for the test made up by me 
print(grid.predict(pred_set_2))

