import pandas as pd
from sklearn.svm import SVC

def Model1():

 model=SVC(
    kernel="rbf",
    C=1,
    gamma=0.01
 )

 return model
