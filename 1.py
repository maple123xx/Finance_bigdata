#测试代码，可随时删掉
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# fund_product = pd.read_csv(r'./data/fundProduct.csv')
# fund144=fund_product[fund_product['Fund_ID'].isin([144])][['Date','close']]
# print(len(fund144))
# plt.plot(range(len(fund144)),fund144['close'])
# plt.plot([330,331,332,333,334],[94.28090797,97.61421877,95.76907428,95.97743284,97.65476814],color='orange')
# plt.show()
# print(fund144)
df_pred=pd.read_csv(r'./data/v1_gui_predict.csv')
df_pred.rename(columns={'Unnamed: 0':'sorted_id'},inplace=True)
df_pred.ix[14:,'diff']=0
df_pred['weight']=df_pred['diff']/(df_pred['diff'].sum())
df_pred.to_csv(r'./data/Ngt5_lgb.csv')
submission=pd.read_csv(r'./data/submission_sample.csv')
submission['weight']=0.0

count=0
for i in df_pred['sorted_id'].tolist():
    submission.ix[i,'weight']=df_pred.ix[count,'weight']
    count+=1
submission.to_csv(r'./data/submission.csv',index=False)