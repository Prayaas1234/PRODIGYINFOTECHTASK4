#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv("C:\\Users\\praya\\OneDrive\\Desktop\\twitter_training.csv")
df


# In[3]:


column_names=['ID','Entity','Sentiments','Remarks']
df_new=pd.read_csv('C:\\Users\\praya\\OneDrive\\Desktop\\twitter_training.csv',names=column_names)
df_new


# In[4]:


df_new.describe()


# In[5]:


df_new.dtypes


# In[6]:


df_new.isnull().sum()


# In[7]:


df_new=df_new.dropna()


# In[8]:


df_new.isnull().sum()


# In[9]:


df_sentiment=df_new['Sentiments'].value_counts()
df_sentiment


# In[10]:


plt.figure(figsize=(10, 10))
plt.pie(df_sentiment, labels=df_sentiment.index, autopct='%1.1f%%',startangle=90)
plt.title('Distribution of Sentiments')
plt.show()


# In[11]:


df_entity=df_new['Entity'].value_counts()
df_entity


# In[12]:


top5_entitycounts=df_new['Entity'].value_counts().sort_values(ascending=False)[:5]
top5_entitycounts


# In[13]:


sentiment_counts = df_new.groupby(['Entity', 'Sentiments']).size().unstack(fill_value=0)
entity_sentiment_counts = sentiment_counts.groupby('Entity').sum()
entity_total_sentiments = entity_sentiment_counts.sum(axis=1)
top_5_entities = entity_total_sentiments.nlargest(5).index
top_5_entity_sentiments = entity_sentiment_counts.loc[top_5_entities]
plt.figure(figsize=(20,16))
top_5_entity_sentiments.plot(kind='bar')
plt.xlabel('Entities')
plt.ylabel('Sentiment Count')
plt.title('Sentiment Distribution among Top 5 Entities')
plt.legend(title='Sentiment')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[14]:


entity_data = df_new[df_new['Entity'] == 'Google']
sentiment_counts = entity_data['Sentiments'].value_counts()
plt.figure(figsize=(12, 6))
sentiment_counts.plot(kind='bar', color=['green', 'red', 'orange','blue'])
plt.xlabel('Sentiment')
plt.ylabel('Frequency')
plt.title('Sentiment Distribution of Google')
plt.show()


# In[15]:


entity_data = df_new[df_new['Entity'] == 'Microsoft']
sentiment_counts = entity_data['Sentiments'].value_counts()
plt.figure(figsize=(10, 10))
labels = sentiment_counts.index  # Labels for pie chart segments
plt.pie(sentiment_counts, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Sentiment Distribution of Microsoft')
plt.show()


# In[ ]:




