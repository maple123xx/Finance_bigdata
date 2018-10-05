#测试代码，可随时删掉
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
fund_product = pd.read_csv(r'./data/fundProduct.csv')
fund144=fund_product[fund_product['Fund_ID'].isin([144])][['Date','close']]
print(len(fund144))
plt.plot(range(len(fund144)),fund144['close'])
plt.plot([330,331,332,333,334],[94.28090797,97.61421877,95.76907428,95.97743284,97.65476814],color='orange')
plt.show()
print(fund144)