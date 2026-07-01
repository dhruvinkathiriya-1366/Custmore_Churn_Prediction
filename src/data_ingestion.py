import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split


Data_path="D:/study/Project/Custmore_churn_prediction/data/raw/TelcoCustomerChurn.csv"
 
def load_data():
   data_df=pd.read_csv(Data_path)
   
   return data_df

def clean_data(data_df):
     drop_column=['ZipCode','City','Quarter','State','Country','CustomerID','ChurnScore','ChurnCategory','ChurnReason','CLTV','CustomerStatus','Dependents','ReferredaFriend','InternetService']
     data_clean_df=data_df.drop(columns=drop_column)
     
     return data_clean_df
  
def split_data(data_clean_df):
     x=data_clean_df.drop(columns=['ChurnLabel'])
     y=data_clean_df['ChurnLabel'].map({
    'Yes':1,
    'No':0
    })
     x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
     
     return x_train,x_test,y_train,y_test
  
if __name__ == "__main__":
    
   data_df=load_data() 
   data_clean_df=clean_data(data_df)
   x_train,x_test,y_train,y_test=split_data(data_clean_df)