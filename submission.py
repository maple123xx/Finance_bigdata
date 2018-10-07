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
df_pred=pd.read_csv(r'./data/Ngt5_lgb.csv')
df_pred.rename(columns={'Unnamed: 0':'sorted_id'},inplace=True)
df_pred['stock_return']=1/60
df_pred.ix[60:,'stock_return']=0
df_pred.to_csv(r'./data/Ngt5_lgb_release2.csv') #必须提供不同的文件名
submission=pd.read_csv(r'./data/submission_sample.csv')
submission['weight']=0.0

# count=0
# for i in df_pred['sorted_id'].tolist():
#     submission.ix[i,'weight']=df_pred.ix[count,'stock_return']
#     count+=1
# submission['weight'].ix[submission['weight']!=0.0]=1/66
for i in [55,74,84,186,27,5,76,134]:
    submission['weight'].ix[submission['fund_id']==i]=5/100
submission['weight'].ix[submission['fund_id']==435]=60/100
print(submission)
submission.to_csv(r'./data/submission.csv',index=False)