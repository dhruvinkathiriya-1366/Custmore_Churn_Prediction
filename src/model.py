import pandas as pd
from sklearn.svm import SVC

def Model():

 model=SVC(
    class_weight="balanced",
    kernel="rbf",
    C=1,
    gamma=0.01
 )

 return model
