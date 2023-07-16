import pandas as pd
import numpy as np
import os

x_replace=['\'',':','{','}','(',')','learning_rate','n_estimators','num_leaves']

def prep_ResultML(row,way):
 strResult_list=row.split(',')
 new_list=[way]
 for item in strResult_list:
    for x in x_replace:
        if x in item:
          item= item.replace(x,'')
    item=item.strip()
    item=float(item)
    new_list.append(item)
 return  new_list


strategy ='Short'
algoName='lgb'

if algoName=='xgb':
 x_cols=['way','acc_mean','acc_std','learning_rate','n_estimators']
 filepath = f'data\{strategy}_XGBoostTune.xlsx'
elif algoName=='lgb':
 x_cols = ['way', 'acc_mean', 'acc_std', 'learning_rate', 'n_estimators','num_leaves']
 filepath = f'data\{strategy}_LGB.xlsx'

sheetListDict =pd.read_excel(filepath,sheet_name=None,header=None)

list_resultML=[]
for sheet_key,df_data in sheetListDict.items():
    try:

     for col,rw  in df_data.iterrows():
      list_resultML.append(prep_ResultML(rw[0],sheet_key))
    except Exception as err:
        print(f'{sheet_key} ==> {err}')
    print(sheet_key," ok")

print("############################################")
array_resultML = np.array(list_resultML)
print(array_resultML)
if(array_resultML.shape[1]==len(x_cols)) :
 dfx=pd.DataFrame(array_resultML,columns=x_cols)
 dfx.to_csv(f'result\{algoName}_{strategy}-MLResult.csv',index=False)
else:
 print('No.columns is not matched data')

