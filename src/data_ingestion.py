import pandas as pd
from sklearn.model_selection import train_test_split


Data_path="D:/study/Project/Custmore_churn_prediction/data/raw/TelcoCustomerChurn.csv"
 
def load_data():
   data_df=pd.read_csv(Data_path)
   
   return data_df

def clean_data(data_df):
     drop_column=['ZipCode','City','Quarter','State','Country','CustomerID','ChurnScore','ChurnCategory','ChurnReason','CLTV','CustomerStatus','Dependents','ReferredaFriend','InternetService']
     data_clean_df=data_df.drop(columns=drop_column)
     
     return data_clean_df
def fetch_column(x_train):
     cat_col=x_train.select_dtypes(include=["object","str"]).columns
     numeric_col=x_train.select_dtypes(include="number").columns
     ordinal_col=['Contract']
     nominal_col=[col for col in cat_col if col not in ordinal_col]
     not_scallneed=['NumberofDependents','Latitude','Longitude','Number_of_Referrals','SatisfactionScore','ChurnLabel']
     scaling_col=[col for col in numeric_col if col not in not_scallneed]
     robust_col = []
     standard_col = []
     for col in scaling_col:
          Q1 = x_train[col].quantile(0.25)
          Q3 = x_train[col].quantile(0.75)

          IQR = Q3 - Q1

          lower = Q1 - 1.5 * IQR
          upper = Q3 + 1.5 * IQR

          outliers = x_train[
               (x_train[col] < lower) |
               (x_train[col] > upper)
          ]

          if len(outliers) > 0:
               robust_col.append(col)
          else:
               standard_col.append(col)

   
   
     return ordinal_col,nominal_col,robust_col,standard_col

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
   ordinal_col,nominal_col,robust_col,standard_col=fetch_column(x_train)
