#!/usr/bin/env python
# coding: utf-8

# In[298]:


import simplejson as json
import pandas as pd 
from pandas.io.json import json_normalize
import numpy as np
import fbprophet
import subprocess

# Read the dataset
with open('/var/www/html/google_data_uk.json', encoding='utf-8') as fh:
    df = pd.read_json(fh)
df.head(5)


# In[299]:


with open('/var/www/html/keyword_uk_0.json', encoding='utf-8') as fh:
    keyword_master = pd.read_json(fh)
keyword_master.head(5)


# In[300]:


ctr_df = pd.read_csv('/var/www/html/ml_files/CTR/average_CTR_UK.csv')
ctr_df = ctr_df[ctr_df['position'] <= 10]
ctr_df.head(5)


# In[301]:


# Remove unwanted columns
df = df[['project_id', 'keyword', 'clicks', 'ctr', 'impressions', 'date', 'device', 'position']]

# Remove rows having null position value
df = df[pd.notnull(df['position'])]


# In[302]:


# # Remove bad keywords
keyword_master = keyword_master[~keyword_master['keywords'].isin(["keyword 3","keyword 1","keyword 2" ])]

# # select distinct keywords and projectIDs
keywords_df = keyword_master[['keywords', 'projectID']]
keywords_df = keywords_df.drop_duplicates()
keyword_master = keywords_df[['keywords']]
keyword_master.head(5)


# In[303]:


# Cleaning Keywords
stopword = ["www.", ".com", " .com", ".co.uk"]

# test['tweet_without_stopwords'] = test['tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
df['keyword_cleaned'] = df['keyword'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopword)]))
df.head(5)


# In[304]:


import string 
exclude = set(string.punctuation)
def removePunctuationFix(x):
    try: 
        x = ''.join(ch for ch in x if ch not in exclude) 
    except: 
        pass 
    return x
df['keyword_cleaned'] = df['keyword_cleaned'].apply(removePunctuationFix)
df['keyword_cleaned'] = df['keyword_cleaned'].str.strip()
df['keyword'] = df['keyword_cleaned']
df.head(5)


# In[305]:


keyword_master['keywords'] = keyword_master['keywords'].apply(removePunctuationFix)
keyword_master['keywords'] = keyword_master['keywords'].str.strip()
keyword_master.head(5)


# In[306]:


from pyjarowinkler import distance as dis
def rankApplyElse(key, dt,grouped_data, mobile_keywords, cluster):
    l = []
    for i in range(1,11):
        tempif_grouped_data = grouped_data[grouped_data['keyword'] == key]
        if (i in list(tempif_grouped_data['rank'])):
            temp_grouped_data = grouped_data[(grouped_data['keyword'].isin(list(mobile_keywords[cluster]))) & (grouped_data['rank'] == i)]
            click = round(temp_grouped_data['all_clicks'].mean(), 0)
            impression  = round(temp_grouped_data['all_impressions'].mean(), 0)
            newl = [ key, dt, i, click, impression]
            l.append(newl)
            
        else:
            newl = [ key, dt, i, 0.0, 0.0]
            l.append(newl)

    return l
def rankApplyIf(key, dt, grouped_data, mobile_keywords, cluster):
    l = []
    for i in range(1,11):
        tempif_grouped_data = grouped_data[(grouped_data['keyword'].isin(list(mobile_keywords[cluster])))]
        if (i in list(tempif_grouped_data['rank'])):
            temp_grouped_data = grouped_data[(grouped_data['keyword'].isin(list(mobile_keywords[cluster]))) & (grouped_data['rank'] == i) & (grouped_data['date'] == dt)]
            click = round(temp_grouped_data['all_clicks'].mean(), 0)
            impression  = round(temp_grouped_data['all_impressions'].mean(), 0)
            newl = [ key, dt, i, click, impression]
            l.append(newl)
            
        else:
            newl = [ key, dt, i, 0.0, 0.0]
            l.append(newl)
    return l
                                            
def apply1(dt_list, key_df_list, grouped_data, mobile_keywords, cluster, key):                       
    for dt in dt_list:
        tempif_grouped_data = grouped_data[(grouped_data['keyword'].isin(list(mobile_keywords[cluster])))]
        if (dt in list(tempif_grouped_data['date'].astype(str))):                               
            l = rankApplyIf(key, dt, grouped_data , mobile_keywords, cluster) 
            l = np.array(l)
            l = np.transpose(l)
            l = [l[0].tolist(), l[1].tolist(), l[2].tolist(), l[3].tolist(), l[4].tolist()]
            l = np.array(l)
            l = np.transpose(l)
            l = l.tolist()
            key_df = pd.DataFrame(l, columns =['keyword', 'date', 'rank', 'clicks', 'impressions']) 
            key_df_list[dt] = key_df
        
        else:
            l = rankApplyElse(key, dt, grouped_data , mobile_keywords, cluster)
            l = np.array(l)
            l = np.transpose(l)
            l = [l[0].tolist(), l[1].tolist(), l[2].tolist(), l[3].tolist(), l[4].tolist()]
            l = np.array(l)
            l = np.transpose(l)
            l = l.tolist()
            key_df = pd.DataFrame(l, columns =['keyword', 'date', 'rank', 'clicks', 'impressions']) 
            key_df_list[dt] = key_df


# In[307]:


def round_mean(x):
    return round(x.mean(skipna=True), 0)
def round_max(x):
    return round(x.max(skipna=True), 0)
def modeling_n_prediction(df, device, position):
    # Filter table with keyword from mobile devices and position less than equal to 10
    data = df[(df['device'] == device) & (df['position'] <= position)]
    data['rank'] = data['position'].astype(int)

    # Order the tables
    data = data.sort_values(["keyword", "date", "rank"], ascending = (True, True, True))

    # Group keywords, date and rank and calculate sum of clicks and impressions
    grouped_data = data.groupby(['keyword', 'date', 'rank']).agg(all_clicks=pd.NamedAgg(column='clicks', aggfunc=sum),all_impressions=pd.NamedAgg(column='impressions', aggfunc=sum))
    grouped_data = grouped_data.reset_index()
    grouped_data = grouped_data.sort_values(["keyword", "date", "rank"], ascending = (True, True, True))
    # Get the list of unique keywords in google search console data
    mobile_keywords = grouped_data['keyword'].unique()

    grouped_data['keyword'] = grouped_data['keyword'].astype(str)
    
    key_date_df_list = {}
    count = 1
    for key in list(keyword_master['keywords'].unique()):
        key_df_list = {}
        print(count)
        print('Processing for keyword: ', key)
        print()


        distance = [dis.get_jaro_distance(key,word) for word in mobile_keywords]
        distance = np.array(distance)
        cluster = np.where(distance <= 0.3)
        total_count = len(mobile_keywords[cluster]) - 1


        words = '|'.join(mobile_keywords.tolist())
        key_df = pd.DataFrame(columns=['keyword','date','rank','clicks','impressions'])

        dt_list = list(grouped_data['date'].drop_duplicates().astype(str))
        dt_list.sort()

        apply1(dt_list, key_df_list, grouped_data, mobile_keywords, cluster, key)
        temp_df = pd.DataFrame(columns=['keyword','date','rank','clicks','impressions'])
        for k, val in key_df_list.items():
            temp_df = pd.concat([temp_df, val], ignore_index=True)
        key_date_df_list[key] = temp_df
        count =  count + 1
    t_df = pd.DataFrame(columns=[ 'keyword','date','rank','clicks','impressions'])
    for k, val in key_date_df_list.items():
        t_df =  pd.concat([t_df, val], ignore_index = True)
    all_ranks_df = t_df

    
    
    if (device == 'MOBILE'):
        ctrs = ctr_df[['position', 'mobile_ctr']]
    else:
        ctrs = ctr_df[['position', 'web_ctr']]
    
    all_ranks_df['rank'] = all_ranks_df['rank'].astype(int)
    all_ranks_df['impressions'] = all_ranks_df['impressions'].astype(float)
    all_ranks_df = pd.merge(all_ranks_df, ctrs, left_on = "rank", right_on = "position")
    
    # Calculate the max and avg impressions for the keyword for each date
    temp_all_ranks_df = all_ranks_df.groupby(['keyword', 'date']).agg(avg_impressions=pd.NamedAgg(column='impressions', aggfunc=round_mean),max_impressions=pd.NamedAgg(column='impressions', aggfunc=round_max))
    temp_all_ranks_df = temp_all_ranks_df.reset_index()
    all_ranks_df = pd.merge(all_ranks_df, temp_all_ranks_df, on = ['keyword', 'date'])
    
    # Replace NA values with avg impressions
    all_ranks_df['impressions'] = all_ranks_df['impressions'].fillna(all_ranks_df['avg_impressions'])
    all_ranks_df = all_ranks_df.sort_values(["keyword", "date", "rank"], ascending = (True, True, True))
    #df['First Season'] = np.where(df['First Season'] > 1990, 1, df['First Season'])
    all_ranks_df['impressions'] = np.where(all_ranks_df['impressions'] <= all_ranks_df['avg_impressions'], all_ranks_df['max_impressions'], all_ranks_df['impressions'])


    if (device == 'MOBILE'):
        all_ranks_df['clicks'] = (all_ranks_df['mobile_ctr'] * all_ranks_df['impressions']) / 100
    else:
        all_ranks_df['clicks'] = (all_ranks_df['web_ctr'] * all_ranks_df['impressions']) / 100
    all_ranks_df.clicks = all_ranks_df.clicks.round()
    all_ranks_df['clicks'] = all_ranks_df['clicks'].astype(int)
        
    if (device == 'MOBILE'):
        all_ranks_df['mobile_ctr'] = None
    else:
        all_ranks_df['web_ctr'] = None

    all_ranks_df['avg_impressions'] = None
    all_ranks_df['max_impressions'] = None

    all_ranks_df['keyword'] = all_ranks_df['keyword'].astype(str)
    all_ranks_df['impressions'] = all_ranks_df['impressions'].astype(int)
    all_ranks_df['date'] = all_ranks_df['date'].astype(str)
    
    casted_df = all_ranks_df.pivot_table(index=['keyword', 'date'],  columns='rank', values=['clicks', 'impressions'])
    casted_df.columns = ["{0}_{1}".format(l1, l2) for l1, l2 in casted_df.columns]
    casted_df = casted_df.reset_index()
    casted_df['keyword'] = casted_df['keyword'].astype('category')
    
    key_pred_list = {}

    for key in list(keyword_master['keywords'].unique()):
        print('Forecasting for keyword - ', key)
        print()

        pred_pos_list = {}

        for position in range(1,11):
            print('Position - ', position)
            print()

            key_sub = casted_df[casted_df['keyword'] == key]
            key_sub['date'] = pd.to_datetime(key_sub['date'])
            clicks_trend = key_sub[['clicks_'+str(position), 'date']]
            clicks_trend.columns = ["y", "ds"]

            prediction_days = 14
            pred_len = 0
            totalRow = len(clicks_trend)
            pred_range = [totalRow - pred_len + 1, totalRow]
            pre_views = clicks_trend.head(totalRow - pred_len)
            post_views = clicks_trend.tail(pred_len)

            m = fbprophet.Prophet()
            m.fit(pre_views)
            future = m.make_future_dataframe(periods=prediction_days)
            fcast = m.predict(future)


            pred_df = fcast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(prediction_days)
            pred_df['position'] = position
            pred_df['keyword'] = key
            pred_df.columns = ["date", "clicks", "clicks_lower", "clicks_upper", "position", "keyword"]
            pred_df = pred_df[["keyword", "date", "position", "clicks", "clicks_lower", "clicks_upper"]]
            pred_df.clicks_upper = pred_df.clicks_upper.round()
            pred_df.clicks_lower = pred_df.clicks_lower.round()
            #fig1 = m.plot(fcast)

            pred_pos_list[position] = pred_df
        t1_df = pd.DataFrame(columns=["keyword", "date", "position", "clicks", "clicks_lower", "clicks_upper"])
        for k, val in pred_pos_list.items():
            t1_df = pd.concat([t1_df, val], ignore_index = True)
        key_pred_list[key] = t1_df
    
    print('\n')
    t2_df = pd.DataFrame(columns=["keyword", "date", "position", "clicks", "clicks_lower", "clicks_upper"])
    for k, val in key_pred_list.items():
        t2_df = pd.concat([t2_df, val], ignore_index = True)    
    pred_key_df = t2_df
    casted_pred_df = pred_key_df.pivot_table(index=['keyword', 'date'],  columns='position', values=['clicks', 'clicks_lower', 'clicks_upper'])
    casted_pred_df.columns = ["{0}_{1}".format(l1, l2) for l1, l2 in casted_pred_df.columns]
    casted_pred_df = casted_pred_df.reset_index()
    

    
    casted_pred_df = pd.merge(keywords_df, casted_pred_df, left_on = "keywords", right_on = "keyword")
  #  casted_df['impressions'] = np.where(all_ranks_df['impressions'] <= all_ranks_df['avg_impressions'], all_ranks_df['max_impressions'], all_ranks_df['impressions'])
    
    #print(casted_pred_df['date'])
    casted_pred_df['date'] = casted_pred_df['date'].astype(str)
    casted_pred_df= casted_pred_df.astype(int, errors='ignore')
    #casted_pred_df['date'] = casted_pred_df['date'].astype(str)
    num = casted_pred_df._get_numeric_data()
    num[num<0]=0
    #print(casted_pred_df['date'])
    casted_pred_df.to_json(r'FinalResults_UK_'+device+'.json', orient='records')
    
    return list([casted_df, casted_pred_df])


# In[308]:


results_mobile = modeling_n_prediction(df, 'MOBILE', 10)
results_desktop = modeling_n_prediction(df, 'DESKTOP', 10)

# Update Opportunity Planner With Results
cmd = "php ml_upload_data.php"
#subprocess.call()
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
print("Success and ml upload complete")

