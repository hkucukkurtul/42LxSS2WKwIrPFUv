# kütüphaneleri ekleme
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression   
from sklearn.neighbors import KNeighborsClassifier   
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
#%% veri okuma
'''
Bu kısımda ilab.py ile aynı klasörde bulunan term-deposit-marketing-2020 
isimli CSV dosyası "veri" isimle değişkene alınıyor.
Sınıflandırma algoritmalarında kullanabilmek için veri x ve y olarak ikiye ayrılınıyor.
x giriş değerlerini y ise etiketleri barındırmaktadır.
'''
veri=pd.read_csv("term-deposit-marketing-2020.csv")
x=veri.drop(['y'],axis=1,inplace=False)
y=veri.y.values
#%% veri inceleme
'''
verinin bütün sütunları inceleniyor
'''
veri.info()      
veri['job'].describe()
veri['marital'].describe()        
veri['education'].describe()  
veri['default'].describe()
veri['housing'].describe()
veri['loan'].describe()
veri['contact'].describe()
veri['month'].describe()
veri['y'].describe()
#%% nümerik hale getirme
'''
object halinde bulunan bütün sütunlar sınıflandırma algoritmalarına uygun olması için int haline getiriliyor.
'''
x.marital= [1 if each == "married" else 2 if each=="single" else 0 for each in x.marital]
x.education= [1 if each == "primary" else 2 if each=="secondary" else 3 if each=="tertiary" else 0 for each in x.education]
x.default= [1 if each == "yes" else 0 for each in x.default]
x.housing= [1 if each == "yes" else 0 for each in x.housing]
x.loan= [1 if each == "yes" else 0 for each in x.loan]
x.contact= [1 if each == "telephone" else 2 if each=="cellular" else 0 for each in x.contact]
x.month=[1 if each=="jan" else 2 if each=="feb" else 3 if each=="mar" else 4 if each=="apr" else 5 if each=="may" else 6 if each=="jun" else 7 if each=="jul" else 8 if each=="aug" else 10 if each=="oct" else 11 if each=="nov" else 12 for each in x.month]
x.job=[1 if each =="management" else 2 if each=="technician" else 3 if each=="entrepreneur" else 4 if each=="blue-collar" else 5 if each=="retired" else 6 if each=="admin" else 7 if each=="services" else 8 if each=="self-employed" else 9 if each=="housemaid" else 10 if each=="student" else 11 if each=="unemployed" else 0 for each in x.job]
y=[1 if each=="yes" else 0 for each in y]
#%% SVM
'''
Bu ve sonraki sectionlarda sınıflandırma algoritmaları denenmiştir.
'''
svm=SVC()
k = 5
svm_result = cross_val_score(svm,x,y,cv=k) 
print('CV Değeri: ',svm_result)
print('CV Ortalamsı: ',np.sum(svm_result)/k)
#%% NaiveBayes
nb = GaussianNB()
nb_result = cross_val_score(nb,x,y,cv=k)  
print('CV Değeri: ',nb_result)
print('CV Ortalamsı: ',np.sum(nb_result)/k)
#%% Logistic Regression
lr=LogisticRegression() 
lr_result = cross_val_score(lr,x,y,cv=k)  
print('CV Değeri: ',lr_result)
print('CV Ortalamsı: ',np.sum(lr_result)/k)
#%% kNN
knn = KNeighborsClassifier()
knn_result = cross_val_score(knn,x,y,cv=k)  
print('CV Değeri: ',knn_result)
print('CV Ortalamsı: ',np.sum(knn_result)/k)
#%% Random Forest
rf=RandomForestClassifier()
rf_result = cross_val_score(rf,x,y,cv=k)  
print('CV Değeri: ',rf_result)
print('CV Ortalamsı: ',np.sum(rf_result)/k)
#%% Linear Regression
reg = LinearRegression()
k = 5
reg_result = cross_val_score(nb,x,y,cv=k)  
print('CV Değeri: ',reg_result)
print('CV Ortalamsı: ',np.sum(reg_result)/k)
#%% Korelasyon
def islem(x):
    x.marital= [1 if each == "married" else 2 if each=="single" else 0 for each in x.marital]
    x.education= [1 if each == "primary" else 2 if each=="secondary" else 3 if each=="tertiary" else 0 for each in x.education]
    x.default= [1 if each == "yes" else 0 for each in x.default]
    x.housing= [1 if each == "yes" else 0 for each in x.housing]
    x.loan= [1 if each == "yes" else 0 for each in x.loan]
    x.contact= [1 if each == "telephone" else 2 if each=="cellular" else 0 for each in x.contact]
    x.month=[1 if each=="jan" else 2 if each=="feb" else 3 if each=="mar" else 4 if each=="apr" else 5 if each=="may" else 6 if each=="jun" else 7 if each=="jul" else 8 if each=="aug" else 10 if each=="oct" else 11 if each=="nov" else 12 for each in x.month]
    x.job=[1 if each =="management" else 2 if each=="technician" else 3 if each=="entrepreneur" else 4 if each=="blue-collar" else 5 if each=="retired" else 6 if each=="admin" else 7 if each=="services" else 8 if each=="self-employed" else 9 if each=="housemaid" else 10 if each=="student" else 11 if each=="unemployed" else 0 for each in x.job]
    x.y=[1 if each=="yes" else 0 for each in x.y]
    return (x)
veri=islem(veri)
'''
Korelasyon fonksiyonu ile y sütunun diğer sütunlarla olan bağlantısı gösterilmektedir.
'''
corr = veri.corr()
fig, ax = plt.subplots(figsize=(20, 20))
mask = np.triu(np.ones_like(corr, dtype=np.bool))
cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)
sns.heatmap(corr, 
            xticklabels=corr.columns.values, 
            yticklabels=corr.columns.values,
            mask=mask,annot=True, cmap=cmap, vmin=-1, vmax=1)
plt.yticks(rotation=0)
plt.show()