#!/usr/bin/env python
# coding: utf-8

# In[115]:


# importing libraries
import numpy as np
import pandas as pd
import datetime as dt

#For Data  Visualization
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


import os
print(os.listdir(r"C:\Users\soura\Desktop\Spring 2020\Machine Learning\Project\input"))

datafield = pd.read_excel(r'C:\Users\soura\Desktop\Spring 2020\Machine Learning\Project\input\Online Retail.xlsx')


# In[116]:


datafield.head(5)


# In[117]:


datafield.info()


# In[118]:


datafield.isnull().sum()


# In[119]:


datafield= datafield.dropna(subset=['CustomerID'])


# In[120]:


datafield.isnull().sum().sum()


# In[121]:


datafield.duplicated().sum()


# In[122]:


datafield = datafield.drop_duplicates()


# In[123]:


datafield.duplicated().sum()


# In[124]:


datafield.describe()


# In[125]:


datafield=datafield[(datafield['Quantity']>0) & (datafield['UnitPrice']>0)]
datafield.describe() 


# In[126]:


datafield.shape


# In[127]:


def get_month(x) : return dt.datetime(x.year,x.month,1)
datafield['InvoiceMonth'] = datafield['InvoiceDate'].apply(get_month)
grouping = datafield.groupby('CustomerID')['InvoiceMonth']
datafield['CohortMonth'] = grouping.transform('min')
datafield.tail()


# In[128]:


def get_month_int (datafieldrame,column):
    year = datafieldrame[column].dt.year
    month = datafieldrame[column].dt.month
    day = datafieldrame[column].dt.day
    return year, month , day 

invoice_year,invoice_month,_ = get_month_int(datafield,'InvoiceMonth')
cohort_year,cohort_month,_ = get_month_int(datafield,'CohortMonth')

year_diff = invoice_year - cohort_year 
month_diff = invoice_month - cohort_month 

datafield['CohortIndex'] = year_diff * 12 + month_diff + 1


# In[129]:


#Counting monthly active customers from each cohort
grouping = datafield.groupby(['CohortMonth', 'CohortIndex'])
cohort_data = grouping['CustomerID'].apply(pd.Series.nunique)

cohort_data = cohort_data.reset_index()
cohort_counts = cohort_data.pivot(index='CohortMonth',columns='CohortIndex',values='CustomerID')
cohort_counts


# In[130]:


# Retention table
cohort_size = cohort_counts.iloc[:,0]
retention = cohort_counts.divide(cohort_size,axis=0) 
retention.round(3) * 100 #vizualizing the number as percentage 


# In[131]:


#Building the heatmap
plt.figure(figsize=(15, 8))
plt.title('Retention rates')
sns.heatmap(data=retention,annot = True,fmt = '.0%',vmin = 0.0,vmax = 0.5,cmap="BuPu_r")
plt.show()


# In[132]:


#Average quantity for each cohort
grouping = datafield.groupby(['CohortMonth', 'CohortIndex'])
cohort_data = grouping['Quantity'].mean()
cohort_data = cohort_data.reset_index()
average_quantity = cohort_data.pivot(index='CohortMonth',columns='CohortIndex',values='Quantity')
average_quantity.round(1)
average_quantity.index = average_quantity.index.date

#Building the heatmap
plt.figure(figsize=(15, 8))
plt.title('Average quantity for each cohort')
sns.heatmap(data=average_quantity,annot = True,vmin = 0.0,vmax =20,cmap="BuGn_r")
plt.show()


# In[133]:


datafield['TotalSum'] = datafield['UnitPrice']* datafield['Quantity']

#Data preparation steps
print('Min Invoice Date:',datafield.InvoiceDate.dt.date.min(),'max Invoice Date:',
       datafield.InvoiceDate.dt.date.max())

datafield.head(3)


# In[134]:


snapshot_date = datafield['InvoiceDate'].max() + dt.timedelta(days=1)
snapshot_date
#The last day of purchase in total is 09 DEC, 2011. To calculate the day periods, 
#let's set one day after the last one,or 
#10 DEC as a snapshot_date. We will count the diff days with snapshot_date.


# In[135]:


# Calculating RFM metrics
rfm = datafield.groupby(['CustomerID']).agg({'InvoiceDate': lambda x : (snapshot_date - x.max()).days,
                                      'InvoiceNo':'count','TotalSum': 'sum'})
#Function Lambda: it gives the number of days between hypothetical today and the last transaction
rfm.rename(columns={'InvoiceDate':'Recency','InvoiceNo':'Frequency','TotalSum':'MonetaryValue'}
           ,inplace= True)

rfm.head()


# In[136]:


#Building RFM segments
r_labels =range(4,0,-1)
f_labels=range(1,5)
m_labels=range(1,5)
r_quartiles = pd.qcut(rfm['Recency'], q=4, labels = r_labels)
f_quartiles = pd.qcut(rfm['Frequency'],q=4, labels = f_labels)
m_quartiles = pd.qcut(rfm['MonetaryValue'],q=4,labels = m_labels)
rfm = rfm.assign(R=r_quartiles,F=f_quartiles,M=m_quartiles)

# Building RFM Segments and RFM Scores
def add_rfm(x) : return str(x['R']) + str(x['F']) + str(x['M'])
rfm['RFM_Segment'] = rfm.apply(add_rfm,axis=1 )
rfm['RFM_Score'] = rfm[['R','F','M']].sum(axis=1)

rfm.head()


# In[137]:


#Selecting bottom RFM segment "111" and viewing top 5 rows
rfm[rfm['RFM_Segment']=='111'].head()


# In[138]:


rfm.groupby('RFM_Score').agg({'Recency': 'mean','Frequency': 'mean',
                             'MonetaryValue': ['mean', 'count'] }).round(1)


# In[139]:


def segments(datafield):
    if datafield['RFM_Score'] > 9 :
        return 'Gold'
    elif (datafield['RFM_Score'] > 5) and (datafield['RFM_Score'] <= 9 ):
        return 'Sliver'
    else:  
        return 'Bronze'

rfm['General_Segment'] = rfm.apply(segments,axis=1)

rfm.groupby('General_Segment').agg({'Recency':'mean','Frequency':'mean',
                                    'MonetaryValue':['mean','count']}).round(1)


# In[140]:


rfm_rfm = rfm[['Recency','Frequency','MonetaryValue']]
print(rfm_rfm.describe())


# In[141]:


# plotting distribution of RFM values
f,ax = plt.subplots(figsize=(10, 12))
plt.subplot(3, 1, 1); sns.distplot(rfm.Recency, label = 'Recency')
plt.subplot(3, 1, 2); sns.distplot(rfm.Frequency, label = 'Frequency')
plt.subplot(3, 1, 3); sns.distplot(rfm.MonetaryValue, label = 'Monetary Value')
plt.style.use('fivethirtyeight')
plt.tight_layout()
plt.show()


# In[142]:


#Unskewing data with log transformations
rfm_log = rfm[['Recency', 'Frequency', 'MonetaryValue']].apply(np.log, axis = 1).round(3)

# plotting distribution of RFM values
f,ax = plt.subplots(figsize=(10, 12))
plt.subplot(3, 1, 1); sns.distplot(rfm_log.Recency, label = 'Recency')
plt.subplot(3, 1, 2); sns.distplot(rfm_log.Frequency, label = 'Frequency')
plt.subplot(3, 1, 3); sns.distplot(rfm_log.MonetaryValue, label = 'Monetary Value')
plt.style.use('fivethirtyeight')
plt.tight_layout()
plt.show()


# In[143]:


#Normalize the variables with StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(rfm_log)
#Store it separately for clustering
rfm_normalized= scaler.transform(rfm_log)


# In[144]:


from sklearn.cluster import KMeans

ks = range(1,8)
inertias=[]
for k in ks :
    # Create a KMeans clusters
    kc = KMeans(n_clusters=k,random_state=1)
    kc.fit(rfm_normalized)
    inertias.append(kc.inertia_)

# Plot ks vs inertias
f, ax = plt.subplots(figsize=(15, 8))
plt.plot(ks, inertias, '-o')
plt.xlabel('Number of clusters, k')
plt.ylabel('Inertia')
plt.xticks(ks)
plt.style.use('ggplot')
plt.title('What is the Best Number for KMeans ?')
plt.show()


# In[145]:


# clustering using k-Means
kc = KMeans(n_clusters= 3, random_state=1)
kc.fit(rfm_normalized)

#Create a cluster label column in the original DataFrame
cluster_labels = kc.labels_

#Calculate average RFM values and size for each cluster:
rfm_rfm_k3 = rfm_rfm.assign(K_Cluster = cluster_labels)

#Calculate average RFM values and sizes for each cluster:
rfm_rfm_k3.groupby('K_Cluster').agg({'Recency': 'mean','Frequency': 'mean',
                                         'MonetaryValue': ['mean', 'count'],}).round(0)


# In[146]:


rfm_normalized = pd.DataFrame(rfm_normalized,index=rfm_rfm.index,columns=rfm_rfm.columns)
rfm_normalized['K_Cluster'] = kc.labels_
rfm_normalized['General_Segment'] = rfm['General_Segment']
rfm_normalized.reset_index(inplace = True)

rfm_melt = pd.melt(rfm_normalized,id_vars=['CustomerID','General_Segment','K_Cluster'],value_vars=['Recency', 'Frequency', 'MonetaryValue'],var_name='Metric',value_name='Value')
rfm_melt.head()


# In[147]:


f, (ax1, ax2) = plt.subplots(1,2, figsize=(15, 8))
sns.lineplot(x = 'Metric', y = 'Value', hue = 'General_Segment', data = rfm_melt,ax=ax1)

# a snake plot with K-Means
sns.lineplot(x = 'Metric', y = 'Value', hue = 'K_Cluster', data = rfm_melt,ax=ax2)

plt.suptitle("Snake Plot of RFM",fontsize=24) #make title fontsize subtitle 
plt.show()


# In[148]:


# The further a ratio is from 0, the more important that attribute is for a segment relative to the total population
cluster_avg = rfm_rfm_k3.groupby(['K_Cluster']).mean()
population_avg = rfm_rfm.mean()
relative_imp = cluster_avg / population_avg - 1
relative_imp.round(2)


# In[149]:


total_avg = rfm.iloc[:, 0:3].mean()
cluster_avg = rfm.groupby('General_Segment').mean().iloc[:, 0:3]
prop_rfm = cluster_avg/total_avg - 1
prop_rfm.round(2)


# In[150]:


# heatmap with RFM
f, (ax1, ax2) = plt.subplots(1,2, figsize=(15, 5))
sns.heatmap(data=relative_imp, annot=True, fmt='.2f', cmap='Blues',ax=ax1)
ax1.set(title = "Heatmap of K-Means")

sns.heatmap(prop_rfm, cmap= 'Oranges', fmt= '.2f', annot = True,ax=ax2)
ax2.set(title = "Heatmap of RFM quantile")

plt.suptitle("Heat Map of RFM",fontsize=20) 

plt.show()


# In[ ]:




