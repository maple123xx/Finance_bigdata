import pandas as pd
from datetime import datetime
import numpy as np
#合并第一周和第二周的基金数据
week_pre3=pd.read_csv('./data/week_pre3.csv')
week4=pd.read_csv('./data/week4.csv')
week_pre3.drop('Unnamed: 0', axis=1, inplace=True)
#week4.drop('Unnamed: 0', axis=1, inplace=True)
#week_pre3.rename(columns={'Fund_ID': 'fund_id'}, inplace=True)

# print(week_pre3.columns.values)
# print(week4.columns.values)

# week_pre3['fund_id']=week_pre3['fund_id'].astype(int)
# week4['fund_id']=week4['fund_id'].astype(int)
# print(week_pre3['fund_id'].dtype)
# print(week4['fund_id'].dtype)
week=pd.concat([week_pre3, week4], axis=0, ignore_index=True)#连接两个df,ignore_index=True表示不保留连接轴上的索引，产生一组新索引range(total_length)
week['Date']=pd.to_datetime(week['Date'],format='%Y/%m/%d')#把str变为time,才能比较
week.drop_duplicates(inplace=True)  #删除重复行
week.sort_values(by=['fund_id','Date'],inplace=True)    #先按Fund_ID排序，再按Date排序
week.reset_index(drop=True,inplace=True)    #把索引再重新排一下

print(week.head())
week.to_csv('./data/week_pre4.csv')#有点小问题，没去重