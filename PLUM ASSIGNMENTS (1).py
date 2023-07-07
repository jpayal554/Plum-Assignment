#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


# In[2]:


df = pd.read_csv(r"C:\Users\Vipul\OneDrive\Documents\My Data Sources\Plum\DA Assignment.csv")
pd.set_option('display.max_columns', None)


# In[3]:


# Return top 5 rows
df.head()


# In[4]:


# shows total number of rows and columns in the data
df.shape


# In[5]:


#Data types of the table
df.info()


# In[6]:


# Only work for numerical columns
df.describe()


# In[7]:


df.columns


# In[8]:


df['Requester id'].nunique()


# Conclusion - 
# 1. The data has 16476 Rows and 23 Columns
# 3. Out of 23, 12 are having integer values while 11 have categorical columns
# 4. There are total 6764 unique Requester Id

# # Analyzing the data

# In[9]:


df_hidden = df.drop(['Id', 'Requester id', 'Created at', 'Updated at', 'Assigned at', 'Initially assigned at','Solved at'], axis=1, inplace = True)


# In[10]:


# Checking null values
df['Satisfaction Score'].isnull().sum()


# In[11]:


df['Satisfaction Score'].value_counts()


# In[12]:


df.head(5)


# In[13]:


df['First reply time in minutes within business hours']=df['First reply time in minutes within business hours'].apply(lambda x :x/60)


# In[14]:


df.rename(columns = {'First reply time in minutes within business hours':'First reply time in hours within business hours'}, inplace = True)


# In[15]:


df.head()


# In[16]:


df['First resolution time in hours within business hours']=df['First resolution time in minutes within business hours'].apply(lambda x :x/60)
df['First resolution time in hours']=df['First resolution time in minutes'].apply(lambda x :x/60)
df['Full resolution time in hours within business hours']=df['Full resolution time in minutes within business hours'].apply(lambda x :x/60)
df['Full resolution time in hours']=df['Full resolution time in minutes'].apply(lambda x :x/60)
df['Requester wait time in hours']=df['Requester wait time in minutes'].apply(lambda x :x/60)
df['Requester wait time in hours within business hours']=df['Requester wait time in minutes within business hours'].apply(lambda x :x/60)
df['Resolution time in days']=df['Resolution time'].apply(lambda x :x/24)


# In[17]:


df.head(5)


# In[18]:


df_hidden = df.drop(['First resolution time in minutes', 'First resolution time in minutes within business hours', 'Full resolution time in minutes', 
                     'Full resolution time in minutes within business hours', 'Requester wait time in minutes', 'Requester wait time in minutes within business hours'], axis=1, inplace = True)


# In[19]:


df.head(5)


# In[20]:


df['Status'].value_counts()


# In[21]:


df_hidden = df.drop(['Full resolution time in hours'], axis=1, inplace = True)


# In[22]:


df.head(5)


# In[23]:


df['Manual Tagging of Categories [list]'].value_counts()


# In[70]:


grouped_data = df.groupby('Group')['First reply time in hours within business hours'].mean().reset_index()
print(grouped_data)

plt.figure(figsize=(5, 4))
sns.barplot(data=df, x='Group', y='First reply time in hours within business hours')
plt.xlabel('Group')
plt.ylabel('First reply time in hours within business hours')
plt.title('comparison of First reply Time in hours between group')
plt.xticks(rotation=45)
plt.show()


# Inference - 
# 
#  - Support Team is the fastest in replying to the customer emails. They generally reply to it within 10 hours.
#  - Reimbursement Claims and Onboarding teams replys to the customer emails within 24 hours.
#  - Endorsements Team is the slowest in replying to the customer emails. They take around 70 hours to reply.

# In[71]:


First_resolve_time_hrs = df.groupby('Group')['First resolution time in hours within business hours'].mean().reset_index()
print(First_resolve_time_hrs)


plt.figure(figsize=(5, 4))
sns.barplot(data=df, x='Group', y='First resolution time in hours within business hours')
plt.xlabel('Group')
plt.ylabel('First resolution time in hours within business hours')
plt.title('Comparison of First Resolution Time between Groups')
plt.xticks(rotation=45)
plt.show()


# Inference - 
# 
#  - Support Team is the fastest in giving first resolution to the customer emails.
#  
#  - While Onboarding Team were replying to the first email within 24 hours. They are taking around 270 business hours for first      resolution. 
#  

# In[26]:


result = df.groupby('Group')['First reply time in hours within business hours'].mean().reset_index()


reply = pd.Series(result['First reply time in hours within business hours'].values, index=result['Group'])
reply


# In[27]:


result = df.groupby('Group')['First resolution time in hours within business hours'].mean().reset_index()


first_resolution = pd.Series(result['First resolution time in hours within business hours'].values, index=result['Group'])
first_resolution


# In[28]:


# Avg time taken to reach first resolution by each group
first_resolution - reply


# #### Inference - 
# 
#  - Support Team is the fastest in providing first resolution after giving first reply to the customer emails.
#  
#  - While the Onboardings team are replying to the emails in normal time. They are taking nearly 5 times more than the rest of      the team in reaching first resolution.
#  
#  - While Endorsements team are taking the highest time in giving first reply to the emails comparitively they are quick in          providing first resolution than Onboardings and Reimbursement Claims team.

# In[29]:


Full_resolve_time_hrs = df.groupby('Group')['Full resolution time in hours within business hours'].mean().reset_index()
Full_resolve_time_hrs


plt.figure(figsize=(5, 6))
sns.barplot(data=df, x='Group', y='Full resolution time in hours within business hours')
plt.xlabel('Group')
plt.ylabel('Full resolution time in hours within business hours')
plt.title('Comparison of Full Resolution Time between Group')
plt.xticks(rotation=45)
plt.show()


# In[30]:


result = df.groupby('Group')['Full resolution time in hours within business hours'].mean().reset_index()


Full_resolution = pd.Series(result['Full resolution time in hours within business hours'].values, index=result['Group'])
Full_resolution


# In[31]:


# Time taken to reach full resolution after giving first resolution in business hours.
Full_resolution - first_resolution


# #### Inference - 
# 
#  - Onboardings team took around 250 hrs to reach first resolution but they were the fastest in reaching full resolution after      providing first resolution. They were 11 times faster.
#  
#  - Endorsements team took very little time in reaching full resolution after giving first resolution.
#  
#  - Whereas Support and Reimbursement Claims team were a bit slow in reaching full resolution.

# ## WHICH IS THE MOST & LEAST EFFICIENT GROUP ?
# 
# 

# We can analyze the efficiency of the groups as follows:
# 
# ## Most Efficient Group:
# 
#     Support - With the shortest average times for first reply, first resolution, full resolution, and time taken to reach full       resolution after giving the first resolution.
# 
# ## Least Efficient Group: 
# 
#     Onboardings - With the longest average times for first reply, first resolution, and full resolution.
# 
# 
# ## Efficiency ratio comparison :
#   - Support is approximately 2.96 times more efficient than Endorsements for full resolution time. This means that, on average,     Support takes approximately 2.96 times less time than Endorsements to reach full resolution.
# 
#   - Support is approximately 5.45 times more efficient than Onboardings for full resolution time.This implies that Support takes     approximately 5.45 times less time than Onboardings to reach full resolution, on average.
# 
#   - Support is approximately 2.54 times more efficient than Reimbursement Claims for full resolution time.This suggests that         Support takes approximately 2.54 times less time than Reimbursement Claims to reach full resolution, on average.
# 
# 
# ## Formula used for calculating Efficiency ratio :
# 
#   - Efficiency Ratio = Average Time of Support Group / Average Time of Another Group
# 
# 

# # Which group are reopening the cases most ?

# In[32]:


# count of number of 0 reopens

filtered_data_1 = df[df['Reopens'] == 0 ]

filtered_data_1['Group'].value_counts()


# In[33]:


# % of 0 reopen cases from the total cases of each group

round((filtered_data_1['Group'].value_counts()/df['Group'].value_counts())*100,2)


# In[34]:


# no. of times cases were reopend more than 5 times for each group

filtered_data_2 = df[df['Reopens'] > 5 ]

filtered_data_2['Group'].value_counts()


# ### Inference - 
# 
#  - Endorsements team had least number of reopen cases, so they could reach full resolution in less time from first resolution.
#  
#  - Support team had most number of reopen cases but it reach full resolution in less time as compare to Reimbursement claims        team.It was more efficient in handling reopen cases than any other team.
#  
#  - Whereas Reimbursement Claims team took the highest time to reach full resolution from first resolution.
#  
# It make sense that Reimbursement Claims team took most time in handling reopen cases because doing reimbursement is a tricky business as reaching reimbursement amount would require lot of communication with customers.
# 

# In[74]:


df.head(5)


# # Looking into Requester wait time - 

# In[36]:


Avg_Wait_time = df.groupby('Group')['Requester wait time in hours within business hours'].mean().reset_index()
Avg_Wait_time


# In[37]:


sns.boxplot(x='Group', y='Requester wait time in hours within business hours', data=df);


# In[38]:


# The range of outlier is too much into support and onboardings group.Looking into it. 


support_df_1 = df[df['Group'] == 'Support']
outliers_threshold_1 = support_df_1['Requester wait time in hours within business hours'].quantile(0.75)

outliers_1 = support_df_1[support_df_1['Requester wait time in hours within business hours'] > outliers_threshold_1]



# In[39]:


outliers_1['Requester wait time in hours within business hours'].mean()


# In[40]:


support_df_2 = df[df['Group'] == 'Onboardings']
outliers_threshold_2 = support_df_2['Requester wait time in hours within business hours'].quantile(0.75)

outliers_2 = support_df_2[support_df_2['Requester wait time in hours within business hours'] > outliers_threshold_2]


# In[41]:


outliers_2['Requester wait time in hours within business hours'].mean()


# #### Inference - 
# 
#  - Average Requester wait time is lowest for Support team and highest for Reimbursement Claims team.
#  
#  - It can be seen that average wait time for top 25% of outliers for Support and Onboardings team are over 150 hrs, whereas        their overall mean is around 60 hrs.
#  
#  - We can look into why 25% of the queries are consuming more requester wait time.

# # 	What type of tickets are taking the most time to resolve?

# In[42]:


df.head(5)


# In[60]:


category_list=df['Manual Tagging of Categories [list]'].unique()
category_list


# In[61]:


df['Manual Tagging of Categories [list]'].value_counts()


# In[64]:


df['Manual Tagging of Categories [list]']=df['Manual Tagging of Categories [list]'].apply(lambda x : 'Not mentioned' if x == '-' else x)


# In[65]:


df['Manual Tagging of Categories [list]'].value_counts()


# In[69]:


avg_resolution_time = df.groupby('Manual Tagging of Categories [list]')['Full resolution time in hours within business hours'].mean().sort_values()

table = pd.DataFrame({'Ticket Category': avg_resolution_time.index, 'Average Resolution Time': avg_resolution_time.values})

print(table)

# Create a bar chart to visualize the average resolution time for each ticket category
plt.bar(avg_resolution_time.index, avg_resolution_time.values)
plt.xlabel('Ticket Category')
plt.ylabel('Average Resolution Time within business hours')
plt.title('Average Resolution Time by Ticket Category')
plt.xticks(rotation='vertical')
plt.show()


# ####  Inference - 
# 
#  - HR Queries and Health ID related query taking most time to resolve customer query.

# # Create the different type of data types we can infer from this data

# ### From the given dataset, we can infer several types of data based on the columns provided.
#     Here are the different types of data we can infer:
# 
#  - Categorical Data:
# 
#     - Group
#     - Status
#     - Priority
#     - Via
#         
#   - Numerical Data:
# 
#     - Resolution time
#     - Reopens
#     - Replies
#     - First reply time in minutes within business hours
#     - First resolution time in minutes
#     - First resolution time in minutes within business hours
#     - Full resolution time in minutes
#     - Full resolution time in minutes within business hours
#     - Requester wait time in minutes
#     - Requester wait time in minutes within business hours
# 
#         
#    - Text Data:
# 
#      - Manual Tagging of Categories
#      - Manual Tagging of Categories [list]
# 

# ### Power BI Dashboard Link - 

# https://www.novypro.com/project/plum-assignment

# # Snowflakes table creation and all query and integrated Snowflake with Power BI for visualisation - 

# USE PLUM_ASSIGNMENT;
# 
# CREATE TABLE if not exists DA_Assignment(
# Id INT PRIMARY KEY,
# Requester_id BIGINT,
# Groupss VARCHAR(25),
# Status VARCHAR(20),
# Priority VARCHAR(15),
# Via VARCHAR(50),
# Created_at DATETIME,
# Updated_at DATETIME,
# Assigned_at DATETIME,
# Initially_assigned_at DATETIME,
# Solved_at DATETIME,
# Resolution_time INT,
# Satisfaction_Score VARCHAR(25),
# Reopens INT,
# Replies INT,
# First_reply_time_in_minutes_within_business_hours INT,
# First_resolution_time_in_minutes INT,
# First_resolution_time_in_minutes_within_business_hours INT,
# Full_resolution_time_in_minutes INT,
# Full_resolution_time_in_minutes_within_business_hours INT,
# Requester_wait_time_in_minutes INT,
# Requester_wait_time_in_minutes_within_business_hours INT,
# Manual_Tagging_of_Categories_list VARCHAR(60)
# );
# 
# SELECT * FROM DA_ASSIGNMENT;
# 
# -- Unique Requester id
# SELECT COUNT(DISTINCT Requester_id) FROM DA_ASSIGNMENT;
# 
# SELECT COUNT(DISTINCT Id) FROM DA_ASSIGNMENT;
# 
# -- Each groups total number of id based on their status & Priority
# SELECT GROUPSS, STATUS, PRIORITY, COUNT(ID) FROM DA_ASSIGNMENT
# GROUP BY 1,2,3
# ORDER BY 1;
# 
# -- Calculate Average time taken to solve queries for each group
# CREATE OR REPLACE TABLE OVERALL_DETAILS AS (
# SELECT GROUPSS, COUNT(*) AS total_requests, AVG((Resolution_time)/60) AS avg_resolution_time,
#        AVG((First_reply_time_in_minutes_within_business_hours)/60) AS Avg_First_Reply_Time,
#        AVG((First_resolution_time_in_minutes_within_business_hours)/60) AS Avg_First_resolution_Time,
#        AVG((Full_resolution_time_in_minutes_within_business_hours)/60) AS Avg_Full_resolution_Time,
#        AVG((Requester_wait_time_in_minutes_within_business_hours)/60) AS Avg_Requester_Wait_Time,
#        (Avg_First_resolution_Time - Avg_First_reply_Time) as time_taken_to_reach_first_resolution,
#        (Avg_Full_resolution_Time - Avg_First_resolution_Time) as time_taken_for_full_resolution
# FROM DA_ASSIGNMENT
# GROUP BY 1
# ORDER BY avg_resolution_time DESC);
# 
# 
# -- Calculate efficiency numbers
# CREATE OR REPLACE TABLE OVERALL_VIEW AS (
# SELECT Groupss,
#     COUNT(*) AS TotalTickets,
#     SUM(CASE WHEN (Full_resolution_time_in_minutes_within_business_hours/ 60) <= 60 THEN 1 ELSE 0 END) AS 
#     TicketsResolvedWithin1Hour,
#     SUM(CASE WHEN (Full_resolution_time_in_minutes_within_business_hours/ 60) > 60 AND 
#     (Full_resolution_time_in_minutes_within_business_hours <= 120/60) THEN 1 ELSE 0 END) AS TicketsResolvedWithin2Hours,
#     SUM(CASE WHEN (Full_resolution_time_in_minutes_within_business_hours/60) > 120 AND 
#     (Full_resolution_time_in_minutes_within_business_hours/60) <= 240 THEN 1 ELSE 0 END) AS TicketsResolvedWithin4Hours,
#     SUM(CASE WHEN (Full_resolution_time_in_minutes_within_business_hours/60) > 240 THEN 1 ELSE 0 END) AS 
#     TicketsResolvedAfter4Hours,
#     AVG((Full_resolution_time_in_minutes_within_business_hours)/60) AS AverageResolutionTime
# FROM DA_ASSIGNMENT
# group by 1);
# 
# 
# -- Average Resolution Time by Priority
# CREATE TABLE PRIORITY_CASES AS (
# SELECT Priority, AVG((Full_resolution_time_in_minutes_within_business_hours)/60) AS Avg_Resolution_Time
# FROM DA_ASSIGNMENT
# GROUP BY 1);
# 
# 
# -- Create a table showing average resolution time by ticket category
# CREATE TABLE category_resolution_time AS
# SELECT Manual_Tagging_of_Categories_list AS Category,
#   AVG((Full_resolution_time_in_minutes_within_business_hours)/60) AS Avg_Resolution_Time
# FROM DA_ASSIGNMENT
# GROUP BY 1
# ORDER BY Avg_Resolution_Time DESC;
# 
# 
# -- Count of 0 reopens for each groups
# CREATE TABLE ZERO_REOPEN_CASES AS (
# SELECT Groupss, COUNT(*) AS total_cases
# FROM DA_ASSIGNMENT
# WHERE Reopens = 0
# GROUP BY Groupss);
# 
# 
# -- % of 0 reopen cases from the total cases of each group
# CREATE TABLE ZERO_PERC_REOPEN_CASES AS (
# SELECT Groupss, COUNT(*) AS total_cases,
#   COUNT(CASE WHEN Reopens = 0 THEN 1 END) * 100.0 / COUNT(*) AS percentage_zero_reopens
# FROM DA_ASSIGNMENT
# GROUP BY Groupss);
# 
# 
# -- Count of more than 5 reopens for each groups
# CREATE TABLE MORE_THAN_FIVE_REOPEN_CASES AS (
# SELECT Groupss, COUNT(*) AS count
# FROM DA_ASSIGNMENT
# WHERE Reopens > 5
# GROUP BY Groupss);
# 
