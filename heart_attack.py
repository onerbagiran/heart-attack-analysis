import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px 

import missingno as msno 

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold, train_test_split,GridSearchCV
from sklearn.metrics import precision_score,confusion_matrix,accuracy_score,roc_curve
from sklearn.preprocessing import StandardScaler

from sklearn import tree
import warnings 
warnings.filterwarnings('ignore')


df=pd.read_csv("heart_atak.csv")
describe=df.describe()

print(df.info())
"""
RangeIndex: 303 entries, 0 to 302
Data columns (total 14 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   age       303 non-null    int64  
 1   sex       303 non-null    int64  
 2   cp        303 non-null    int64  
 3   trestbps  303 non-null    int64  
 4   chol      303 non-null    int64  
 5   fbs       303 non-null    int64  
 6   restecg   303 non-null    int64  
 7   thalach   303 non-null    int64  
 8   exang     303 non-null    int64  
 9   oldpeak   303 non-null    float64
 10  slope     303 non-null    int64  
 11  ca        303 non-null    int64  
 12  thal      303 non-null    int64  
 13  target    303 non-null    int64  
dtypes: float64(1), int64(13)
"""

## missing value problem
print(df.isnull().sum())
"""
age         0
sex         0
cp          0
trestbps    0
chol        0
fbs         0
restecg     0
thalach     0
exang       0
oldpeak     0
slope       0
ca          0
thal        0
target      0
"""

## kategorik ve nümerik analizi
"""
Index(['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'],
      dtype='object')
"""

categorical_list=['sex', 'cp','restecg','exang','slope', 'ca', 'thal', 'target']
df_categorical=df.loc[:,categorical_list]

for i in categorical_list:
    plt.figure()
    sns.countplot(x=i,data=df_categorical,hue="target")
    plt.title(i)
    

numeric_list=['age', 'trestbps', 'chol', 'fbs', 'thalach',
        'oldpeak','target']

df_numeric=df.loc[:,numeric_list]
sns.pairplot(df_numeric,hue='target',diag_kind="kde")
plt.show()


## box, swarm, cat, correlation analysis

scaler=StandardScaler()
scaled_array=scaler.fit_transform(df[numeric_list[:-1]])


df_dummy=pd.DataFrame(scaled_array,columns=numeric_list[:-1])
df_dummy=pd.concat([df_dummy,df.loc[:,"target"]],axis=1)

## boxplot
data_melted=pd.melt(df_dummy,id_vars="target",var_name="features",value_name="value")


plt.figure()
sns.boxplot(x="features",y="value",hue="target",data=data_melted)
plt.show()


#swarmplot
plt.figure()
sns.swarmplot(x="features",y="value",hue="target",data=data_melted)
plt.show()

#catplot
plt.figure()
sns.catplot(x="exang",y="age",hue="target",col="sex",kind="swarm",data=df)
plt.show()

#heatmap
plt.figure()
sns.heatmap(df.corr(),annot=True,fmt=".1f",linewidths=0.7)
plt.show()

#outlier detection
numeric_list=['age', 'trestbps', 'chol', 'fbs', 'thalach',
        'oldpeak','target']

df_numeric=df.loc[:,numeric_list]

for i in numeric_list:
    Q1=np.percentile(df.loc[:,i],25)
    Q3=np.percentile(df.loc[:,i],75)
    IQR=Q3-Q1
    
    print(f"{i}: old shape: {df.loc[:,i].shape}")
    
    upper=np.where(df.loc[:,i]>=(Q3+2.5*IQR))

    lower=np.where(df.loc[:,i]<=(Q1-2.5*IQR))

    try:
        df.drop(upper[0],inplace=True)
    except:print("hata")

    try:
        df.drop(lower[0],inplace=True)
    except:print("hatas")    

    print(f"new shape: {df.shape}")


### modelleme ve hiperparametre
    
df1=df.copy()

df1=pd.get_dummies(df1,columns=categorical_list[:-1],drop_first=True)
X=df1.drop(["target"],axis=1)
y=df1[ "target"]

scaler=StandardScaler()
X[numeric_list[:-1]]=scaler.fit_transform(X[numeric_list[:-1]])

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=42)

logreg=LogisticRegression()
logreg.fit(X_train, y_train)

y_pred_prob=logreg.predict_proba(X_test)
y_pred=np.argmax(y_pred_prob,axis=1)
print(f"test accuracy: {accuracy_score(y_test,y_pred)}")

#roc curve 

fpr,tpr,threshold=roc_curve(y_test,y_pred_prob[:,1])

plt.plot([0,1],[0,1],"k--")
plt.plot(fpr,tpr,label="logistic Regression")
plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.title("logistic regression")
plt.show()


lr=LogisticRegression()
penalty=["l1","l2"]
parameters={"penalty":penalty}

lr_searcher=GridSearchCV(lr,parameters)
lr_searcher.fit(X_train,y_train)
print("best parameters:{}",lr_searcher.best_params_)

y_pred=lr_searcher.predict(X_test)
print("Test accuracy:{} ",accuracy_score(y_test,y_pred))








