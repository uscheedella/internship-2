#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg as la
from datetime import datetime as dt
import scipy.stats as stat
import seaborn as sb
import sklearn.feature_selection as fs
import sklearn.preprocessing as pp
from statsmodels.tsa.seasonal import seasonal_decompose
from itertools import combinations, product, groupby, cycle


# ## OLD DATA

# In[3]:


dist_df = pd.read_csv(r'/home/uscheedella/Documents/USPS Internship/Data Distribution Tables/Cleaned Data.csv',parse_dates=['ACTIVITY_DATE'], index_col='ACTIVITY_DATE')
dist_df


# In[4]:


def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return pd.DataFrame(diff)


# ## Identifying Abnormal Data

# #### DHL

# #### Nordstrom

# In[5]:


nord_issue_data = dist_df.loc[((dist_df['MAIL_CLASS'] == 'PM') & 
                              (dist_df['MASTER_MID'] == 901139045)) | 
                             ((dist_df['MAIL_CLASS'] == 'PM') & 
                             (dist_df['MASTER_MID'] == 901090935))]

nord_issue_data = nord_issue_data['2020-02-15':'2020-02-27']


# In[6]:


nord = nord_issue_data.reset_index(level=['ACTIVITY_DATE'])


# In[7]:


x = pd.DataFrame(len(nord_issue_data) * ['abnormal'])
x.columns = ["NORMALITY"]
nord_issue_lab = pd.concat([nord.reset_index(drop=True), x.reset_index(drop=True)], axis=1)


# In[8]:


nord_issue_lab = nord_issue_lab.set_index(['ACTIVITY_DATE'])
nord_issue_lab.head()


# #### UPS

# In[9]:


ups_issue_data1 = dist_df.loc[(dist_df['MASTER_MID'] == 902200716) 
                            & (dist_df['RURAL_FLAG'] == 'N') 
                            & (dist_df['MAIL_CLASS'] == 'PS')]
ups_issue_data1 = ups_issue_data1['2020-01-30':'2020-02-14']

ups_issue_data2 = dist_df.loc[(dist_df['MASTER_MID'] == 902200716) 
                            & (dist_df['RURAL_FLAG'] == 'Y') 
                            & (dist_df['MAIL_CLASS'] == 'PS')]
#ups_issue_data2 = ups_issue_data2['2020-02-15':'2020-04-15']
ups_issue_data2 = ups_issue_data2['2020-02-12':'2020-04-15']

ups_issue_data3 = dist_df.loc[(dist_df['MASTER_MID'] == 902200716) 
                            & (dist_df['MAIL_CLASS'] == 'LW')]
ups_issue_data3 = ups_issue_data1['2020-01-30':'2020-02-14']


# In[10]:


ups1 = ups_issue_data1.reset_index(level=['ACTIVITY_DATE'])
ups2 = ups_issue_data2.reset_index(level=['ACTIVITY_DATE'])
ups3 = ups_issue_data3.reset_index(level=['ACTIVITY_DATE'])


# In[11]:


x = pd.DataFrame(len(ups_issue_data1) * ['abnormal'])
x.columns = ["NORMALITY"]
ups_issue_lab1 = pd.concat([ups1.reset_index(drop=True), x.reset_index(drop=True)], axis=1)


# In[12]:


x = pd.DataFrame(len(ups_issue_data2) * ['abnormal'])
x.columns = ["NORMALITY"]
ups_issue_lab2 = pd.concat([ups2.reset_index(drop=True), x.reset_index(drop=True)], axis=1)


# In[13]:


x = pd.DataFrame(len(ups_issue_data3) * ['abnormal'])
x.columns = ["NORMALITY"]
ups_issue_lab3 = pd.concat([ups3.reset_index(drop=True), x.reset_index(drop=True)], axis=1)


# In[14]:


ups_issue_lab = pd.concat([ups_issue_lab1, ups_issue_lab2, ups_issue_lab3])
ups_issue_lab = ups_issue_lab.set_index(['ACTIVITY_DATE'])
ups_issue_lab.head()


# #### Amazon

# In[15]:


amz_issue_data = dist_df.loc[(dist_df['MASTER_MID'] == 899489)
                                  & (dist_df['MAIL_CLASS'] == 'PS')
                                  & (dist_df['RURAL_FLAG'] == 'Y')]
amz_issue_data = amz_issue_data['2020-03-15':'2020-03-30']
amz_issue_data.sort_index()


# #### FedEx

# In[16]:


fed_issue_data1 = dist_df.loc[(dist_df['MASTER_MID'] == 999909356)
                                  & ((dist_df['DRI'] == 'S') | (dist_df['DRI'] == 'D'))
                                  & (dist_df['RI'] == '3D')
                                  & (dist_df['PC'] == '3')
                                  & (dist_df['RURAL_FLAG'] == 'Y')
                                  & (dist_df['MAIL_CLASS'] == 'PS')]
fed_issue_data1 = fed_issue_data1['2020-02-01':'2020-02-05']
fed_issue_data2 = dist_df.loc[(dist_df['MASTER_MID'] == 999909356)
                                  & (dist_df['MAIL_CLASS'] == 'PS')
                                  & (dist_df['PC'] == '3')
                                  & (dist_df['RURAL_FLAG'] == 'Y')
                                  & (dist_df['DRI'] == 'D')]
fed_issue_data2 = fed_issue_data2['2020-02-06':'2020-02-18']


# In[17]:


dist_df_nodate = dist_df.reset_index(level=['ACTIVITY_DATE'])


# In[18]:


dist_abnorm_df = pd.concat([fed_issue_data1, fed_issue_data2, amz_issue_data, ups_issue_data1, ups_issue_data2, ups_issue_data3, nord_issue_data])
dist_abnorm_df
dist_abnorm_df = dist_abnorm_df.reset_index(level=['ACTIVITY_DATE'])
x = pd.DataFrame(len(dist_abnorm_df) * ['abnormal'])
x.columns = ["NORMALITY"]
dist_abnorm_lab = pd.concat([dist_abnorm_df, x], axis=1)


# In[19]:


dist_abnorm_df


# ### Normal Data

# In[20]:


#fed_issue_data1, fed_issue_data2, amz_issue_data, ups_issue_data1, ups_issue_data2, nord_issue_data
dist_norm_df = pd.concat([dist_df_nodate.reset_index(drop=True), dist_abnorm_df.reset_index(drop=True)]).drop_duplicates(keep=False)
dist_norm_df
#dist_norm_df = dist_norm_df.reset_index(level=['ACTIVITY_DATE'])
x = pd.DataFrame(len(dist_norm_df) * ['normal'])
x.columns = ["NORMALITY"]
x
dist_norm_lab = pd.concat([dist_norm_df.reset_index(drop=True), x.reset_index(drop=True)], axis=1)
dist_norm_lab


# In[59]:


type(dist_norm_df.index[0])


# ## Implementation

# In[20]:


dist_lab_df = pd.concat([dist_norm_lab, dist_abnorm_lab])
dist_lab_df
#ind = pd.DataFrame(range(len(dist_lab_df)))
#ind.columns = ['IND']
#dist_lab_df = pd.concat([dist_lab_df.reset_index(drop=True),ind.reset_index(drop=True)],axis=1)
dist_lab_df = dist_lab_df.set_index(['ACTIVITY_DATE'])
dist_lab_df


# In[16]:


## fill in 0's for normal data
## fill in with mean before testing data
## label data as normal or abnormal
## if testing data correctly classified then add to corresponding df
## create bool abnormal = False


# In[22]:


dist_abnorm_date = dist_abnorm_lab.set_index('ACTIVITY_DATE')


# In[23]:


dist_lab_df = dist_lab_df['2020-01-27':].sort_index()


# In[91]:


normm = dist_norm_lab.set_index('ACTIVITY_DATE')
dist_fin_df = pd.concat([normm, dist_abnorm_nodhl])
dist_fin_df


# In[90]:


dist_fin_df['2020-01-27':]


# In[25]:


dist_lab_df


# ## VOLUME CHECKING

# ### CHECKS HELPER FUNCTION

# # FINAL IMPLEMENTATION

# In[60]:


mid_df = pd.DataFrame(columns=['MID'])
mids_ls = ['201114','899489','999909356','901003501','901581565','901000432','898230','963437418',
                 '902200716','901008903','898847','901096548','901000062','901139045','901090935','901267090',
                 '901649158','901958488','901013123','901670915','902215816']
mids_ls = list(map(int, mids_ls))
mid_df['MID'] = mids_ls
names_ls = ['ATFM', 'Amazon', 'Fedex', 'Fedex PRS', 'Fedex Supply Chain', 'DHL', 'Pitney Bowes', 'Pitney Bowes PRS', 'UPS', 'UPS PRS', 
            'UPSMI', 'UPSMI PRS', 'CVS', 'Nordstrom Direct', 'Nordstrom Inc', 'UHG', 'Popout', 'Mercari', 'Walmart', 
            'International Bridge', 'International Bridge Bluebird']
mid_df['COMPANY_NAME'] = names_ls
#mid_df.to_csv('/home/uscheedella/Documents/USPS Internship/Temp 5-26 Data/mid_df.csv')


# In[99]:


mc_list = list(dist_df.MAIL_CLASS.unique())
mc_list.extend(['MR', 'PG'])
mc_df = pd.DataFrame(columns = ['MAIL_CLASS', 'MC_NAME'])
mc_df['MAIL_CLASS'] = mc_list
mc_df['MC_NAME'] = ['PM International', 'Parcel Select', 'Bound Printed Matter', 'Priority Mail', 
                      'Priority Mail Express', 'Media Mail', 'PS Lightweight', 'Marketing Parcels & Nonprofit', 
                      'First Class', 'EX International', 'FC International', 'PS Return', 'Marketing Mail Nonprofit', 
                      'Library Mail', 'PM Return', 'Global Express Guaranteed']
#mc_df.to_csv('/home/uscheedella/Documents/USPS Internship/Temp 5-26 Data/mc_df.csv')


# ### DAY BY DAY CHECK

# ## NEW DATA

# In[21]:


dist_all_df = pd.read_excel(r'/home/uscheedella/Documents/USPS Internship/Data Distribution Tables/Daily Distributions Jan - Jul.xlsx',parse_dates=['ACTIVITY_DATE'], index_col='ACTIVITY_DATE')


# In[22]:


dist_df2 = dist_all_df['2020-05-26':]


# In[23]:


dist_df2['PC'] = dist_df2['PC'].apply(str)


# In[24]:


dist_all_df['PC'] = dist_all_df['PC'].apply(str)


# In[25]:


dist_df2['PC'][5]


# In[26]:


dist_norm_df = pd.concat([dist_df_nodate.reset_index(drop=True), dist_abnorm_df.reset_index(drop=True)]).drop_duplicates(keep=False)
dist_norm_df = dist_norm_df.set_index('ACTIVITY_DATE')


# In[27]:


def check_bf2(newdata):
    print('############################# CHECK BF ##############################')
    #newdata is dist_lab_df
    count = 0
    columns=['MASTER_MID', 'FY', 'MONTH', 'MAIL_CLASS', 'PC', 'DRI', 'RI', 'RURAL_FLAG', 'VOL', 'AVG_WEIGHT', 'AVG_POSTAGE', 'NORMALITY']
    checked_df = pd.DataFrame(columns= columns, index=['ACTIVITY_DATE'])
    for indexi, rowi in newdata.iterrows():
        all_bl_bool = False
        norm_bl_bool = False
        
        print(rowi)
        
        cur_date = newdata.index[count]
        
        # all data baseline
        baseline_df = dist_all_df.loc[(dist_all_df['MASTER_MID'] == rowi.values[0])
                & (dist_all_df['MAIL_CLASS'] == rowi.values[3])
                & (dist_all_df['PC'] == str(rowi.values[4])) & (dist_all_df['DRI'] == rowi.values[5]) 
                & (dist_all_df['RI'] == rowi.values[6]) & (dist_all_df['RURAL_FLAG'] == rowi.values[7])]
        baseline_df.index = pd.to_datetime(baseline_df.index)
        baseline_df = baseline_df.sort_index()
        beg = (newdata.index[count] - pd.DateOffset(months=1))
        beg_mon = "%02d" % beg.month
        beg_day = "%02d" % beg.day
        beg_date = '2020-%s-%s' % (beg_mon, beg_day)
        end = (newdata.index[count] - pd.DateOffset(days=1))
        end_mon = "%02d" % end.month
        end_day = "%02d" % end.day
        end_date = '2020-%s-%s' % (end_mon, end_day)
        print('beg_date: ', beg_date)
        print('end_date: ', end_date)
        bl_df = baseline_df[beg_date:end_date]
        print('len all bl_df: ', len(bl_df))
        if len(bl_df) > 14:
            print('passed dist_all baseline check')
            all_bl_bool = True
        
        # norm data baseline
        baseline2_df = dist_norm_df.loc[(dist_norm_df['MASTER_MID'] == rowi.values[0])
                & (dist_norm_df['MAIL_CLASS'] == rowi.values[3])
                & (dist_norm_df['PC'] == str(rowi.values[4])) & (dist_norm_df['DRI'] == rowi.values[5]) 
                & (dist_norm_df['RI'] == rowi.values[6]) & (dist_norm_df['RURAL_FLAG'] == rowi.values[7])]
        baseline2_df.index = pd.to_datetime(baseline2_df.index)
        baseline2_df = baseline2_df.sort_index()
        bl2_df = baseline2_df[beg_date:end_date]
        print('len of norm bl: ', len(bl2_df))
        if len(bl2_df) > 7:
            print('passed norm baseline check')
            norm_bl_bool = True
        
        if norm_bl_bool and all_bl_bool:
        #if all_bl_bool:
            checked_df = checked_df.append(rowi)
        
        count = count+1
    return checked_df


# In[28]:


def check_vol2(newdata):
    print('####################### CHECK VOL ##########################')
    count = 0
    columns=['MASTER_MID', 'FY', 'MONTH', 'MAIL_CLASS', 'PC', 'DRI', 'RI', 'RURAL_FLAG', 'VOL', 'AVG_WEIGHT', 'AVG_POSTAGE', 'NORMALITY']
    vol_df = pd.DataFrame(columns= columns, index=['ACTIVITY_DATE'])
    good_mc = ['FC', 'EX', 'PM', 'PS', 'LW', 'BB', 'RP']
    for indexi, rowi in newdata.iterrows():
        if rowi.values[3] in good_mc:
            print(rowi)
            ## if ex or bb
            if rowi.values[3] in good_mc:
                print('in good mc')
                if rowi.values[3] == 'EX':
                    mean_vol = 300
                elif rowi.values[3] == 'BB':
                    mean_vol = 50000
                else:
                ## convert to datetime
                    baseline_mc_df = dist_norm_df.loc[(dist_norm_df['MAIL_CLASS'] == rowi.values[3])]
                    baseline_mc_df.index = pd.to_datetime(baseline_mc_df.index)
                    baseline_mc_df = baseline_mc_df.sort_index()
                    ## month baseline
                    beg = (newdata.index[count] - pd.DateOffset(months=1))
                    beg_mon = "%02d" % beg.month
                    beg_day = "%02d" % beg.day
                    beg_date = '2020-%s-%s' % (beg_mon, beg_day)
                    end = (newdata.index[count] - pd.DateOffset(days=1))
                    end_mon = "%02d" % end.month
                    end_day = "%02d" % end.day
                    end_date = '2020-%s-%s' % (end_mon, end_day)
                    print('beg date: ', beg_date)
                    print('end date: ', end_date)
                    print('cur date: ', newdata.index[count])
                    bl_df = baseline_mc_df[beg_date:end_date]
                    if len(bl_df) > 0:
                        mean_vol = bl_df.loc[:,"VOL"].mean()
                        if mean_vol > 10000:
                            mean_vol = 10000
                    else:
                        print('bl_df empty')
                        mean_vol = 1000000000
            print('mail class: ', rowi.values[3])
            print("mean vol: ", mean_vol)
            
            if rowi.values[8] >= mean_vol:
                print('passed vol check')
                temp_row = rowi
                temp_row['VOL_BL'] = mean_vol
                vol_df = vol_df.append(temp_row)
        count = count + 1
    return vol_df


# In[29]:


def mc_day_z2(checked_df):
    print("################################ MC_DAY_Z ########################################")
    mc_dayz_df = pd.DataFrame()
    mc_norm_df = pd.DataFrame()
    
    grp_mid_mc = dist_norm_df.groupby(['MASTER_MID', 'MAIL_CLASS'])
    
    check_df = checked_df.reset_index()
    check_df.rename(columns = {'index':'ACTIVITY_DATE'}, inplace = True)
    chk_grp = check_df.groupby(['ACTIVITY_DATE', 'MASTER_MID', 'MAIL_CLASS'])
    
    max_ind_ls = []
    
    for i in chk_grp:
        grp_key = i[0]
        print('2 and 3rd key: ', grp_key[1], grp_key[2])
        grp_df = i[1]
        grp_df = grp_df.set_index('ACTIVITY_DATE')
        print('date type: ', type(grp_df.index.values[0]))
        pg_prod_slopes = {}
        wt_prod_slopes = {}
        vol_prod_slopes = {}
        z_ls = {}
        if len(grp_df) == 1:
            print('appended bc only one row')
            mc_dayz_df = mc_dayz_df.append(grp_df.iloc[0])
        else:
            print('more than one row')
            for indexi,rowi in grp_df.iterrows():
                count = 0
                end = (grp_df.index[count] - pd.DateOffset(days=1))
                end_mon = "%02d" % end.month
                end_day = "%02d" % end.day
                end_date = '2020-%s-%s' % (end_mon, end_day)
                    
                beg = end - pd.DateOffset(weeks=2) + pd.DateOffset(days=1)
                beg_mon = "%02d" % beg.month
                beg_day = "%02d" % beg.day
                beg_date = '2020-%s-%s' % (beg_mon, beg_day)
                
                print(grp_df.index[count])
                print(beg_date)
                print(end_date)
                
                ## find baseline down to product level
                cur_df = grp_mid_mc.get_group((grp_key[1], grp_key[2]))
                all_prods = cur_df.groupby(['PC', 'DRI', 'RI', 'RURAL_FLAG'])
                cur_prod_df = all_prods.get_group((rowi.values[4], rowi.values[5], rowi.values[6], rowi.values[7]))
                cur_prod_df.index = pd.to_datetime(cur_prod_df.index)
                cur_prod_df = cur_prod_df.sort_index()
                cur_prod_bl = cur_prod_df[beg_date:end_date]
                print('cur_prod_bl len1: ', len(cur_prod_bl))
                
                if len(cur_prod_bl) > 0:
                    print('baseline greater than 0')
                    # fill in normal missing dates with 0's
                    full_bl = cur_prod_bl.sort_index()
                    full_bl = full_bl.drop_duplicates(keep=False)
                    print(full_bl)
                    ### CHANGED THIS ###
                    #r = pd.date_range(start=full_bl.index.min(), end=full_bl.index.max())
                    r = pd.date_range(start=pd.to_datetime(beg_date), end=pd.to_datetime(end_date))
                    print(r)
                    full_bl = full_bl.reindex(r).fillna(0.0).reset_index()
                    full_bl = full_bl.set_index('index')
        
                    # fill in missing dates up to testing data with mean
                    full_bl_new = full_bl.append(rowi)
                    ## CHANGED THIS ##
                    #r_new = pd.date_range(start=full_bl_new.index.min(), end=full_bl_new.index.max())
                    r_new = pd.date_range(start=pd.to_datetime(beg_date), end=pd.to_datetime(end_date))
                    tempdf = full_bl_new.reindex(r_new)
                    tempdf[['VOL']] = tempdf[['VOL']].fillna(np.mean(full_bl['VOL']))
                    tempdf[['AVG_WEIGHT']] = tempdf[['AVG_WEIGHT']].fillna(np.mean(full_bl['AVG_WEIGHT']))
                    tempdf[['AVG_POSTAGE']] = tempdf[['AVG_POSTAGE']].fillna(np.mean(full_bl['AVG_POSTAGE']))
                    #full_bl_new = full_bl_new.reindex(r).fillna(np.mean(full_bl)).reset_index()
                    cur_prod_bl = tempdf
                    print('cur_prod_bl len: ', len(cur_prod_bl))
                
                ## decompose
                if len(cur_prod_bl) > 13:
                    print('baseline greater than 13')
                    vol_decomp = seasonal_decompose(cur_prod_bl['VOL'], model='additive', extrapolate_trend='freq')
                    wt_decomp = seasonal_decompose(cur_prod_bl['AVG_WEIGHT'], model='additive', extrapolate_trend='freq')
                    pg_decomp = seasonal_decompose(cur_prod_bl['AVG_POSTAGE'], model='additive', extrapolate_trend='freq')
        
                    ## find ssn's
                    try:
                        pg_ssn = (np.var(pg_decomp.resid) / np.var(pg_decomp.seasonal + pg_decomp.resid))
                    except:
                        pg_ssn = (np.var(pg_decomp.resid) / 0.000001)
                    pg_ssn = max(0-pg_ssn, 1-pg_ssn)
                    try: 
                        wt_ssn = (np.var(wt_decomp.resid) / np.var(wt_decomp.seasonal + wt_decomp.resid))
                    except:
                        wt_ssn = (np.var(wt_decomp.resid) / 0.000001)
                    wt_ssn = max(0-wt_ssn, 1-wt_ssn)
                    try:
                        vol_ssn = (np.var(vol_decomp.resid) / np.var(vol_decomp.seasonal + vol_decomp.resid))
                    except:
                        vol_ssn = (np.var(vol_decomp.resid) / 0.000001)
                    vol_ssn = max(0-vol_ssn, 1-vol_ssn)
                    print('pg_ssn: ', pg_ssn)
                    print('wt_ssn: ', wt_ssn)
                    print('vol_ssn: ', vol_ssn)
                    if pg_ssn > 0.5:
                        pg_trend = pg_decomp.trend
                    else:
                        pg_trend = cur_prod_bl['AVG_POSTAGE']
                    if wt_ssn > 0.5:
                        wt_trend = wt_decomp.trend
                    else:
                        wt_trend = cur_prod_bl['AVG_WEIGHT']
                    if vol_ssn > 0.5:
                        vol_trend = vol_decomp.trend
                    else:
                        vol_trend = cur_prod_bl['VOL'][0:len(cur_prod_bl['VOL'])-2]
                    
                    xax = range(7)
                    vol_fit = np.polyfit(xax, vol_trend.tail(7), deg = 1)
                    wt_fit = np.polyfit(xax, wt_trend.tail(7), deg = 1)
                    pg_fit = np.polyfit(xax, pg_trend.tail(7), deg = 1)
                    
                    ## find normal dist - mean, stdev
                    vol_mean = np.mean(vol_trend)
                    vol_sd = np.std(vol_trend)
                
                    wt_mean = np.mean(wt_trend)
                    wt_sd = np.std(wt_trend)
                
                    pg_mean = np.mean(pg_trend)
                    pg_sd = np.std(pg_trend)
                    
                    ##pg, wt, and vol adding
                    # added these checks
                    if rowi.values[9] > max(pg_trend.tail(7)):
                        print('pg z max added')
                        try:
                            pg_z = (rowi.values[9] - pg_mean) / pg_sd
                        except:
                            pg_z = (rowi.values[9] - pg_mean) / 0.000001
                        z_ls[pg_z] = rowi.values[13]
                    elif rowi.values[10] > max(wt_trend.tail(7)):
                        print('wt z max added')
                        try: 
                            wt_z = (rowi.values[10] - wt_mean) / wt_sd
                        except:
                            wt_z = (rowi.values[10] - wt_mean) / 0.000001
                        z_ls[wt_z] = rowi.values[13]
                    elif rowi.values[8] > max(vol_trend.tail(7)):
                        if vol_fit[0] > 100:
                            print('vol z max added')
                            try:
                                vol_z = (rowi.values[8] - vol_mean) / vol_sd
                            except:
                                vol_z = (rowi.values[8] - vol_mean) / 0.000001
                            z_ls[vol_z] = rowi.values[13]
                    else:
                        print('no z max found')
                    
                count = count+1
            
            print('z_ls: ', z_ls)
            max_two_z = []
            if z_ls:
                z_sorted = sorted(z_ls)
                print('z_ls two max: ', z_sorted[-3:])
                max_two_z = z_sorted[-2:]
                max_ind_ls.append(z_ls[max_two_z[0]])
                if len(max_two_z) > 1:
                    max_ind_ls.append(z_ls[max_two_z[1]])
                if len(max_two_z) > 2:
                    max_ind_ls.append(z_ls[max_two_z[2]])
                #max_z = max(z_ls)
                #max_ind = z_ls[max_z]
                #max_ind_ls.append(max_ind)
            
            print("max_two_z", max_two_z)
            if max_two_z:
                print(checked_df.loc[checked_df['ind'] == z_ls[max_two_z[0]]])
                mc_dayz_df = mc_dayz_df.append(checked_df.loc[checked_df['ind'] == z_ls[max_two_z[0]]])
                print('appended 1 max')
            if len(max_two_z) > 1:
                print(checked_df.loc[checked_df['ind'] == z_ls[max_two_z[1]]])
                mc_dayz_df = mc_dayz_df.append(checked_df.loc[checked_df['ind'] == z_ls[max_two_z[1]]])
                print('appended 2 max')
            if len(max_two_z) > 2:
                print(checked_df.loc[checked_df['ind'] == z_ls[max_two_z[2]]])
                mc_dayz_df = mc_dayz_df.append(checked_df.loc[checked_df['ind'] == z_ls[max_two_z[2]]])
                print('appended 3 max')
    
    new_rows = []
    for indexi, rowi in checked_df.iterrows():
        if rowi.values[13] not in max_ind_ls:
            new_rows.append(rowi.values)
            
    mc_norm_df = mc_norm_df.append(pd.DataFrame(new_rows))

    
    return [mc_dayz_df, mc_norm_df]        


# In[30]:


def detect_ab_ts_simple2(newdata):
    print('######################### DETECT AB ###########################')
    columns=['MASTER_MID', 'FY', 'MONTH', 'MAIL_CLASS', 'PC', 'DRI', 'RI', 'RURAL_FLAG', 'VOL', 'AVG_WEIGHT', 'AVG_POSTAGE', 'NORMALITY']
    #correct_df = pd.DataFrame(columns= columns, index=['ACTIVITY_DATE'])
    #wrong_df = pd.DataFrame(columns= columns, index=['ACTIVITY_DATE'])
    norm_df = pd.DataFrame(columns= columns, index=['ACTIVITY_DATE'])
    ab_df = pd.DataFrame(columns= columns, index=['ACTIVITY_DATE'])
    return_dict = {}
    count = 0
    ab_bool = False
    ab_bool_week = False
    norm_res = 'normal'
    bl_check_count = 0
    all_check_count = 0
    vol_slopes = []
    wt_slopes = []
    pt_slopes = []
    for indexi, rowi in newdata.iterrows():
        print(rowi)
        ab_bool = False
        ab_bool_week = False
        norm_res = 'normal'
        print(count)
        
        # fetch data from baseline date and product
        cur_date = newdata.index[count]
        ## removed str for pc ##
        baseline_df = dist_norm_df.loc[(dist_norm_df['MASTER_MID'] == rowi.values[0]) 
                & (dist_norm_df['MAIL_CLASS'] == rowi.values[3]) 
                & (dist_norm_df['PC'] == rowi.values[4]) & (dist_norm_df['DRI'] == rowi.values[5]) 
                & (dist_norm_df['RI'] == rowi.values[6]) & (dist_norm_df['RURAL_FLAG'] == rowi.values[7])]
        baseline_df.index = pd.to_datetime(baseline_df.index)
        beg = (newdata.index[count] - pd.DateOffset(months=1))
        beg_mon = "%02d" % beg.month
        beg_day = "%02d" % beg.day
        beg_date = '2020-%s-%s' % (beg_mon, beg_day)
        end = (newdata.index[count] - pd.DateOffset(days=1))
        end_mon = "%02d" % end.month
        end_day = "%02d" % end.day
        end_date = '2020-%s-%s' % (end_mon, end_day)
        print('beg_date: ', beg_date)
        print('end_date: ', end_date)
        baseline_df = baseline_df.sort_index()
        bl_df = baseline_df[beg_date:end_date]
        print("len bl_df: ", len(bl_df))
    
        vol_vals = pd.DataFrame(bl_df['VOL'])
        wt_vals = pd.DataFrame(bl_df['AVG_WEIGHT'])
        pg_vals = pd.DataFrame(bl_df['AVG_POSTAGE'])
        
        # fill in normal missing dates with 0's
        full_bl = bl_df.sort_index()
        full_bl = full_bl.drop_duplicates(keep=False)
        #print('full bl index: ', full_bl.index)
        #print("full bl index min: ", full_bl.index.min())
        #print("full bl index max: ", full_bl.index.max())
        #r = pd.date_range(start=full_bl.index.min(), end=full_bl.index.max())
        #full_bl = full_bl.reindex(r).fillna(0.0).reset_index()
        #full_bl = full_bl.set_index('index')
        
        # fill in missing dates up to testing data with mean
        # later on have to fill with the abnormal extracted values
        full_bl_new = full_bl.append(rowi)
        r_new = pd.date_range(start=full_bl_new.index.min(), end=full_bl_new.index.max())
        #r_new = pd.date_range(start=pd.to_datetime(beg_date), end=pd.to_datetime(end_date))
        tempdf = full_bl_new.reindex(r_new)
        tempdf[['VOL']] = tempdf[['VOL']].fillna(np.mean(full_bl['VOL']))
        tempdf[['AVG_WEIGHT']] = tempdf[['AVG_WEIGHT']].fillna(np.mean(full_bl['AVG_WEIGHT']))
        tempdf[['AVG_POSTAGE']] = tempdf[['AVG_POSTAGE']].fillna(np.mean(full_bl['AVG_POSTAGE']))
        full_bl_new = tempdf
        print('full_bl_means')
        print(full_bl_new)
        bl_mean_dict = {}
        bl_mean_dict['VOL'] = np.mean(full_bl['VOL'])
        bl_mean_dict['AVG_WEIGHT'] = np.mean(full_bl['AVG_WEIGHT'])
        bl_mean_dict['AVG_POSTAGE'] = np.mean(full_bl['AVG_POSTAGE'])
    
        # decompose, if trend is upward then look into that variable
        vol_decomp = seasonal_decompose(full_bl_new['VOL'], model='additive', extrapolate_trend='freq')
        wt_decomp = seasonal_decompose(full_bl_new['AVG_WEIGHT'], model='additive', extrapolate_trend='freq')
        pg_decomp = seasonal_decompose(full_bl_new['AVG_POSTAGE'], model='additive', extrapolate_trend='freq')
        
        pg_ssn = (np.var(pg_decomp.resid) / np.var(pg_decomp.seasonal + pg_decomp.resid))
        pg_ssn = max(0-pg_ssn, 1-pg_ssn)
        wt_ssn = (np.var(wt_decomp.resid) / np.var(wt_decomp.seasonal + wt_decomp.resid))
        wt_ssn = max(0-wt_ssn, 1-wt_ssn)
        vol_ssn = (np.var(vol_decomp.resid) / np.var(vol_decomp.seasonal + vol_decomp.resid))
        vol_ssn = max(0-vol_ssn, 1-vol_ssn)
        print('pg_ssn: ', pg_ssn)
        print('wt_ssn: ', wt_ssn)
        print('vol_ssn: ', vol_ssn)
        if pg_ssn > 0.5:
            pg_trend = pg_decomp.trend
        else:
            pg_trend = full_bl_new['AVG_POSTAGE']
        if wt_ssn > 0.5:
            wt_trend = wt_decomp.trend
        else:
            wt_trend = full_bl_new['AVG_WEIGHT']
        if vol_ssn > 0.55:
            vol_trend = vol_decomp.trend
        else:
            vol_trend = full_bl_new['VOL']
        
        xax = range(7)
        vol_fit = np.polyfit(xax, vol_trend.tail(7), deg = 1)
        wt_fit = np.polyfit(xax, wt_trend.tail(7), deg = 1)
        pg_fit = np.polyfit(xax, pg_trend.tail(7), deg = 1)
        
        if pg_trend[len(pg_trend)-1] > (max(pg_trend[0:len(pg_trend)-2]) + 0.1):
            print("pg wrong")
            ab_bool = True
            var_name = "AVG_POSTAGE"
            final_var = full_bl_new.AVG_POSTAGE.values
            final_dc = seasonal_decompose(full_bl_new['AVG_POSTAGE'], model='additive', extrapolate_trend='freq')
        elif vol_trend[len(vol_trend)-1] > max(vol_trend[0:len(vol_trend)-2]):
            print("vol wrong")
            print('vol_trend: ',vol_trend)
            ab_bool = True
            var_name = "VOL"
            final_var = full_bl_new.VOL.values
            final_dc = seasonal_decompose(full_bl_new['VOL'], model='additive', extrapolate_trend='freq')
            
            ## checking vol increase in all products
            if vol_fit[0] >= 100:
                print('vol up trend')
                cur_grp = grp_mid_mc.get_group((rowi.values[0], rowi.values[3]))
                all_prods = cur_grp.groupby(['PC', 'DRI', 'RI', 'RURAL_FLAG'])
                all_prods_df = pd.DataFrame(all_prods.groups.keys())

                vol_prod_slopes = []

                # finding baseline (1 week)
                for i in all_prods:
                    cur_prod_df = i[1]
                    print('cur_prod_df: ', cur_prod_df)
                    print(cur_prod_df.index)
                    #cur_prod_df.index = pd.to_datetime(cur_prod_df)
                    cur_prod_df = cur_prod_df.sort_index()
                    end = (newdata.index[count] - pd.DateOffset(days=1))
                    end_mon = "%02d" % end.month
                    end_day = "%02d" % end.day
                    end_date = '2020-%s-%s' % (end_mon, end_day)
                    
                    beg = end - pd.DateOffset(weeks=1) + pd.DateOffset(days=1)
                    beg_mon = "%02d" % beg.month
                    beg_day = "%02d" % beg.day
                    beg_date = '2020-%s-%s' % (beg_mon, beg_day)
                    print('cur_prod_bl beg: ', beg_date)
                    print('cur_prod_bl end: ', end_date)
                    cur_prod_bl = cur_prod_df[beg_date:end_date]
    
                # finding slopes
                xax = range(7)
                if len(cur_prod_bl['VOL']) == 7:
                    cur_fit = np.polyfit(xax, cur_prod_bl['VOL'], deg = 1)
                    vol_prod_slopes.append(cur_fit[0])
            
                pos_slopes = [i for i in vol_prod_slopes if i > 0]
                if len(pos_slopes) == len(vol_prod_slopes):
                    ab_bool_week = False

        elif wt_trend[len(wt_trend)-1] > (max(wt_trend[0:len(wt_trend)-2]) + 0.1):
            print("wt wrong")
            ab_bool = True
            var_name = "AVG_WEIGHT"
            final_var = full_bl_new.AVG_WEIGHT.values
            final_dc = seasonal_decompose(full_bl_new['AVG_WEIGHT'], model='additive', extrapolate_trend='freq')
        else:
            print('nothing wrong')
            norm_res = 'normal'
                
        if ab_bool_week:
            print("ab_bool_week true")
            
            trend = final_dc.trend
            seasonal = final_dc.seasonal
            resid = final_dc.resid
    
            # calculate seasonality strength, if > 0.5, remove from final model
            ssn_pt = (np.var(resid) / np.var(seasonal + resid))
            ssn = max(0-ssn_pt, 1-ssn_pt)
            extracted = final_var - trend
    
            if ssn > 0.5:
                extracted = difference(extracted, 7)
                print("Seasonality Removed")
    
            extracted = extracted.tail(8)
            
            ex_std = np.std(extracted)
            if not isinstance(ex_std, float):
                ex_std = ex_std.tolist()[0]
            
            print("ex_std: ", ex_std)
    
            # get interval of normal values, if new data outside --> abnormal
            print("min: ", min(extracted.values[0:len(extracted)-2]) - ex_std)
            print("max: ", max(extracted.values[0:len(extracted)-2]) + ex_std)
            print("cur: ", extracted.iloc[len(extracted)-1].item())
            if (extracted.iloc[len(extracted)-1].item() < min(extracted.values[0:len(extracted)-2]) - ex_std) | (extracted.iloc[len(extracted)-1].item() > max(extracted.values[0:len(extracted)-2]) + ex_std):
                print("Observation at index %d of new data is abnormal!" % count)
                norm_res = "abnormal"
            else:
                print("Normal activity")
                norm_res = "normal"    
        
        elif ab_bool:
            print("ab_bool true")
            # store trend, season and resid values
            trend = final_dc.trend
            seasonal = final_dc.seasonal
            resid = final_dc.resid
    
            # calculate seasonality strength, if > 0.5, remove from final model
            ssn_pt = (np.var(resid) / np.var(seasonal + resid))
            ssn = max(0-ssn_pt, 1-ssn_pt)
            extracted = final_var - trend
    
            if ssn > 0.5:
                extracted = difference(extracted, 7)
                print("Seasonality Removed")
                
            ex_std = np.std(extracted)
            if not isinstance(ex_std, float):
                ex_std = ex_std.tolist()[0]
            print("ex_std: ", ex_std)
    
            # get interval of normal values, if new data outside --> abnormal
            print("min: ", min(extracted.values[0:len(extracted)-2]) - ex_std)
            print("max: ", max(extracted.values[0:len(extracted)-2]) + ex_std)
            print('cur: ', extracted.iloc[len(extracted)-1].item())
            if (extracted.iloc[len(extracted)-1].item() < min(extracted.values[0:len(extracted)-2]) - ex_std) | (extracted.iloc[len(extracted)-1].item() > max(extracted.values[0:len(extracted)-2]) + ex_std):
                print("Observation at index %d of new data is abnormal!" % count)
                norm_res = "abnormal"
            else:
                print("Normal activity")
                norm_res = "normal"
        else:
            # store trend, season and resid values
            final_dc = seasonal_decompose(full_bl_new['AVG_POSTAGE'], model='additive', extrapolate_trend='freq')
            var_name = "AVG_POSTAGE"
            final_var = full_bl_new.AVG_POSTAGE.values
            
            trend = final_dc.trend
            seasonal = final_dc.seasonal
            resid = final_dc.resid
            
            # calculate seasonality strength, if > 0.5, remove from final model
            ssn_pt = (np.var(resid) / np.var(seasonal + resid))
            ssn = max(0-ssn_pt, 1-ssn_pt)
            extracted = final_var - trend
            
            ex_std = np.std(extracted)
            if not isinstance(ex_std, float):
                ex_std = ex_std.tolist()[0]
    
            if ssn > 0.5:
                extracted = difference(extracted, 7)
                print("Seasonality Removed")
            
            norm_res = "normal"
        
        if norm_res == "normal":
            temp_row = rowi
            temp_row['EXTRACTED_CUR'] = extracted.iloc[len(extracted)-1].item()
            print(type((min(extracted.values[0:len(extracted)-2]) - ex_std).item()))
            temp_row['EXTRACTED_MIN'] = (min(extracted.values[0:len(extracted)-2]) - ex_std).item()
            temp_row['EXTRACTED_MAX'] = (max(extracted.values[0:len(extracted)-2]) + ex_std).item()
            norm_df = norm_df.append(rowi)
            print("added to norm_df")
        else:
            temp_row = rowi
            temp_row['AB_VAR'] = var_name
            temp_row['AVG_BL'] = bl_mean_dict[var_name]
            temp_row['EXTRACTED_CUR'] = extracted.iloc[len(extracted)-1].item()
            print(type((min(extracted.values[0:len(extracted)-2]) - ex_std).item()))
            temp_row['EXTRACTED_MIN'] = (min(extracted.values[0:len(extracted)-2]) - ex_std).item()
            temp_row['EXTRACTED_MAX'] = (max(extracted.values[0:len(extracted)-2]) + ex_std).item()
            if var_name == 'AVG_POSTAGE':
                fin = rowi['VOL'] * (rowi['AVG_POSTAGE'] - rowi['AVG_BL'])
            if var_name == 'AVG_WEIGHT':
                fin = 1
            if var_name == 'VOL':
                fin = rowi['AVG_POSTAGE'] * (rowi['VOL'] - rowi['AVG_BL'])
            temp_row['FIN_IMPACT'] = abs(fin)
            ab_df = ab_df.append(temp_row)
            print("added to ab_df")
        
        count = count + 1
    
    return [norm_df, ab_df, vol_slopes, wt_slopes, pt_slopes]


# In[31]:


dist_norm_df = pd.concat([dist_df_nodate.reset_index(drop=True), dist_abnorm_df.reset_index(drop=True)]).drop_duplicates(keep=False)
dist_norm_df


# In[32]:


dist_norm_df = dist_norm_df.set_index('ACTIVITY_DATE')
#dist_norm_df = dist_norm_df['2020-01-27':].sort_index()
#dist_norm_df = dist_norm_df.reset_index(level=['ACTIVITY_DATE'])


# In[33]:


grp_mid_mc = dist_norm_df.groupby(['MASTER_MID', 'MAIL_CLASS'])


# In[34]:


checked2_df = check_bf2(dist_df2.loc[dist_df2.index == '2020-05-26'])
#checked2_df = check_bf2(suplul)
checked2_df = checked2_df[1:]
vol2_df = check_vol2(checked2_df)
vol2_df = vol2_df[1:]
vol2_df['ind'] = range(0, len(vol2_df))
mc_check = mc_day_z2(vol2_df)
final_check_z2_df = mc_check[0]
cols = ['MASTER_MID', 'FY', 'MONTH', 'MAIL_CLASS', 'PC', 'DRI', 'RI', 'RURAL_FLAG', 'VOL', 'AVG_WEIGHT', 'AVG_POSTAGE', 'VOL_BL', 'ind']
final_check_z2_df = final_check_z2_df[cols]
day_res = detect_ab_ts_simple2(vol2_df)


# In[35]:


day_res[1]


# In[66]:


duh = dist_norm_df.loc[(dist_norm_df['MASTER_MID'] == 902200716)
                                  & (dist_norm_df['DRI'] == 'D')
                                  & (dist_norm_df['RI'] == 'SP')
                                  & (dist_norm_df['PC'] == '3')
                                  & (dist_norm_df['RURAL_FLAG'] == 'N')
                                  & (dist_norm_df['MAIL_CLASS'] == 'PS')].sort_index()
duh.index = pd.to_datetime(duh.index)
duh = duh.sort_index()
duh['2020-05-16':'2020-06-15']


# In[40]:


newdatels = []
for i in list(dist_df2.index.unique()):
    date = str(i)[0:10]
    newdatels.append(date)


# In[41]:


newdatels[0:30]


# In[42]:


fin_ab_df2 = pd.DataFrame()

#checked2_df = check_bf2(dist_df2.loc[dist_df2.index == '2020-05-26'])
#checked2_df = check_bf2(suplul)
#checked2_df = checked2_df[1:]
#vol2_df = check_vol2(checked2_df)
#vol2_df = vol2_df[1:]
#vol2_df['ind'] = range(0, len(vol2_df))
#mc_check = mc_day_z2(vol2_df)
#final_check_z2_df = mc_check[0]
#cols = ['MASTER_MID', 'FY', 'MONTH', 'MAIL_CLASS', 'PC', 'DRI', 'RI', 'RURAL_FLAG', 'VOL', 'AVG_WEIGHT', 'AVG_POSTAGE', 'VOL_BL', 'ind']
#final_check_z2_df = final_check_z2_df[cols]
#day_res = detect_ab_ts_simple2(vol2_df)

for date in newdatels[0:30]:
    print(date)
    
    checked2_df = check_bf2(dist_df2.loc[dist_df2.index == date])
    checked2_df = checked2_df[1:]
    vol2_df = check_vol2(checked2_df)
    vol2_df = vol2_df[1:]
    #########
    fil_cols = ['MASTER_MID', 'FY', 'MONTH', 'MAIL_CLASS', 'PC', 'DRI', 'RI', 'RURAL_FLAG', 'VOL', 'AVG_WEIGHT', 'AVG_POSTAGE']
    fil_df = checked2_df[fil_cols]
    nonfil_df = pd.concat([dist_df2.loc[dist_df2.index == date], fil_df]).drop_duplicates(keep=False)
    #########
    vol2_df['ind'] = range(0, len(vol2_df))
    mc_check = mc_day_z2(vol2_df)
    final_check_z2_df = mc_check[0]
    cols = ['MASTER_MID', 'FY', 'MONTH', 'MAIL_CLASS', 'PC', 'DRI', 'RI', 'RURAL_FLAG', 'VOL', 'AVG_WEIGHT', 'AVG_POSTAGE', 'VOL_BL', 'ind']
    final_check_z2_df = final_check_z2_df[cols]
    final_check_z2_df = final_check_z2_df.loc[final_check_z2_df['ind'].notnull()]
    day_res = detect_ab_ts_simple2(final_check_z2_df)
    
    dist_norm_df = pd.concat([dist_norm_df, day_res[0]])
    fin_ab_df2 = pd.concat([fin_ab_df2, day_res[1]])
    ## REMEMBER TO RESET DIST_NORM_DF !!! ##


# In[49]:


final_check_z2_df.loc[final_check_z2_df['ind'].notnull()]


# In[55]:


fin_ab_df = pd.DataFrame()
for date in newdatels:
    print(date)
    
    vol2_df = check_vol2(dist_df2.loc[dist_df2.index == date])
    vol2_df = vol2_df[1:]
    checked2_df = check_bf2(vol2_df)
    checked2_df = checked2_df[1:]
    #########
    fil_cols = ['MASTER_MID', 'FY', 'MONTH', 'MAIL_CLASS', 'PC', 'DRI', 'RI', 'RURAL_FLAG', 'VOL', 'AVG_WEIGHT', 'AVG_POSTAGE']
    fil_df = checked2_df[fil_cols]
    nonfil_df = pd.concat([dist_df2.loc[dist_df2.index == date], fil_df]).drop_duplicates(keep=False)
    #########
    checked2_df['ind'] = range(0, len(checked2_df))
    mc_check = mc_day_z2(checked2_df)
    final_check_z2_df = mc_check[0]
    cols = ['MASTER_MID', 'FY', 'MONTH', 'MAIL_CLASS', 'PC', 'DRI', 'RI', 'RURAL_FLAG', 'VOL', 'AVG_WEIGHT', 'AVG_POSTAGE', 'VOL_BL', 'ind']
    final_check_z2_df = final_check_z2_df[cols]
    day_res = detect_ab_ts_simple2(final_check_z2_df)
    
    dist_norm_df = pd.concat([dist_norm_df, day_res[0], mc_check[1], nonfil_df])
    fin_ab_df = pd.concat([fin_ab_df, day_res[1]])
    ## REMEMBER TO RESET DIST_NORM_DF !!! ##


# In[106]:


checked2_df.loc[checked2_df['MASTER_MID'] == 902200716]


# In[ ]:


## next run remove mc_day filter


# In[43]:


len(fin_ab_df2)


# In[46]:


#898847
#902200716
#901139045
fin_ab_df2.loc[fin_ab_df2['MASTER_MID'] == 901139045]


# In[61]:


fin_clean_ab_df = fin_ab_df2.loc[fin_ab_df2['ind'].notnull()]
ab_cols = ['MASTER_MID', 'FY', 'MONTH', 'MAIL_CLASS', 'PC', 'DRI', 'RI',
       'RURAL_FLAG', 'VOL', 'AVG_WEIGHT', 'AVG_POSTAGE', 'AB_VAR',
       'AVG_BL', 'EXTRACTED_CUR', 'EXTRACTED_MAX', 'EXTRACTED_MIN', 'VOL_BL']
fin_clean_ab_df = fin_clean_ab_df[ab_cols]


# In[62]:


mid_df.columns = ['MASTER_MID', 'COMPANY_NAME']


# In[63]:


fin_clean_ab_df = fin_clean_ab_df.reset_index()


# In[64]:


ab_join = pd.merge(fin_clean_ab_df, mid_df, on='MASTER_MID')
join_cols = ['index', 'MASTER_MID','COMPANY_NAME', 'FY', 'MONTH', 'MAIL_CLASS', 'PC', 'DRI', 'RI',
       'RURAL_FLAG', 'VOL', 'AVG_WEIGHT', 'AVG_POSTAGE', 'AB_VAR', 'AVG_BL', 'VOL_BL']
ab_join = ab_join[join_cols]
ab_join = ab_join.set_index('index')
ab_join


# In[113]:


ret = fin_clean_ab_df.reset_index()
ret.columns = ['ACTIVITY_DATE', 'MASTER_MID', 'FY', 'MONTH', 'MAIL_CLASS', 'PC', 'DRI', 'RI',
       'RURAL_FLAG', 'VOL', 'AVG_WEIGHT', 'AVG_POSTAGE', 'VOL_BL', 'AB_VAR']


# In[121]:


ab_grpby = fin_clean_ab_df.groupby(['MAIL_CLASS', 'MASTER_MID']).count().sort_values(by = 'MONTH', ascending=False)
pd.DataFrame(ab_grpby)


# In[49]:


mid_df = pd.DataFrame(columns=['MID'])
mids_ls = ['201114','899489','999909356','901003501','901581565','901000432','898230','963437418',
                 '902200716','901008903','898847','901096548','901000062','901139045','901090935','901267090',
                 '901649158','901958488','901013123','901670915','902215816']
mids_ls = list(map(int, mids_ls))
mid_df['MID'] = mids_ls
names_ls = ['ATFM', 'Amazon', 'Fedex', 'Fedex PRS', 'Fedex Supply Chain', 'DHL', 'Pitney Bowes', 'Pitney Bowes PRS', 'UPS', 'UPS PRS', 
            'UPSMI', 'UPSMI PRS', 'CVS', 'Nordstrom Direct', 'Nordstrom Inc', 'UHG', 'Popout', 'Mercari', 'Walmart', 
            'International Bridge', 'International Bridge Bluebird']
mid_df['COMPANY_NAME'] = names_ls
#mid_df.to_csv('/home/uscheedella/Documents/USPS Internship/Temp 5-26 Data/mid_df.csv')


# In[19]:


newdates = pd.to_datetime(newdatels)


# In[20]:


def find_missing_data(newdata):
    missing_prod_df = pd.DataFrame()
    all_dates = set(newdata.index)
    #print("all dates: ", all_dates)
    for cur_date in newdates:
        print('cur_date: ',cur_date)
        prev_df = dist_df2
        prev_df = prev_df.reset_index()
        #print(prev_df.columns)
        lul = prev_df.groupby(['MASTER_MID', 'MAIL_CLASS', 'PC', 'DRI', 'RI', 'RURAL_FLAG'])['ACTIVITY_DATE'].max().reset_index()
        #print(lul)
        for indexi, rowi in lul.iterrows():
            print('three days ago: ', cur_date - pd.DateOffset(days=3))
            if rowi.values[6] < cur_date - pd.DateOffset(days=3):
                print('missing data detected')
                temp_row = rowi
                temp_row['Days Missing'] = rowi.values[6] - cur_date
                temp_row['Current Date'] = cur_date
                missing_prod_df = missing_prod_df.append(temp_row)
    return missing_prod_df


# In[21]:


missing_prod_df = find_missing_data(dist_df2)


# In[22]:


#idx = missing_prod_df['Days Missing'].min() == missing_prod_df['Days Missing']
#missing_prod_df[idx]
mp_grpby = missing_prod_df.groupby(['ACTIVITY_DATE', 'DRI', 'MAIL_CLASS', 'MASTER_MID', 'RI', 'RURAL_FLAG', 'PC'])['Days Missing'].min().reset_index()
miss_prod_df = mp_grpby.sort_values(by = 'Days Missing') 


# In[30]:


miss_cols = ['ACTIVITY_DATE', 'MASTER_MID', 'MAIL_CLASS', 'PC', 'DRI', 'RI', 'RURAL_FLAG', 'Days Missing']
miss_prod_df = miss_prod_df[miss_cols]


# In[26]:


tempp = miss_prod_df.groupby(['ACTIVITY_DATE', 'MASTER_MID', 'Days Missing']).count().sort_values(by = 'RURAL_FLAG', ascending= False)
miss_freq = pd.DataFrame(tempp.loc[tempp['RURAL_FLAG'] > 1]['DRI'])
miss_freq.columns = ['# of Missing Products']
miss_freq.index.names = ['STARTING_DATE', 'MASTER_MID', 'Days Missing']
miss_freq


# In[31]:


with pd.ExcelWriter('/home/uscheedella/Documents/USPS Internship/Data Distribution Tables/Miss_Data_July.xlsx') as writer:
    miss_freq.to_excel(writer, sheet_name='Missing Products - Freq', startrow=1, startcol = 1)
    miss_prod_df.to_excel(writer, sheet_name='Missing Products', startrow=1, startcol = 1)


# In[65]:


with pd.ExcelWriter('/home/uscheedella/Documents/USPS Internship/Data Distribution Tables/Ab_Data_July_Temp3.xlsx') as writer:
    ab_join.to_excel(writer, sheet_name='Flagged Abnormal Data', startrow=1, startcol = 1)
    mid_df.to_excel(writer, sheet_name='Master MIDs', startrow=1, startcol = 1)


# In[96]:


fin_clean_ab_df.to_csv('/home/uscheedella/Documents/USPS Internship/Data Distribution Tables/Ab_Data_July.csv')

