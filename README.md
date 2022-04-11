# Analisis_Exploratorio

Pandas is an open source, BSD-licensed library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language. 

## Pasos 🚀

1. Se importan librerias 
2. Se seleccionan los datos
3. Se convierten datos CSV
4. Se categorizan los dos dataframe
5. Se unifican los dos dataframe
6. Verificar los datos 
7. Descripcion Total dataframe
8. graficas
9. Se identifica correlación
10. Asimetría 
12. KNeighborsClassifier-accuracy_score.

## Importar librerías 🔧

    import pandas as pd 
    import seaborn as sns
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.stats import skew
  
  ## Analisis de regresion from sklearn.linear_model import LogisticRegression
  
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier 
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.model_selection import train_test_split,cross_validate
    from sklearn.preprocessing import MinMaxScaler,StandardScaler,LabelEncoder
    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

## 1. Leer todos los archivos  📋

    url_wine_red='https://raw.githubusercontent.com/terranigmark/curso-analisis-exploratorio-datos-platzi/main/winequality-red.csv'
    url_wine_white='https://raw.githubusercontent.com/terranigmark/curso-analisis-exploratorio-datos-platzi/main/winequality-white.csv'

## 2. Convertir arvhivos ⌨️

    red=pd.read_csv(url_wine_red,delimiter=";")

## 3. Se categorizan los dos dataframe ⚙️

    red['category']='red'
    white['category']='white'

## 4. Se unifican los dos dataframe  📖

    total_wine=red.append(white, ignore_index=True)

## 5. Verificar 🖇️

    total_wine.dtypes

## 6. Descripción 7. Gráficas 📦

    total_wine.describe()
    total_wine.plot()
    total_wine['density'].plot()
    sns.set (rc={'figure.figsize': (14, 8)})
    sns.countplot (total_wine['quality'])
    
## 8. Correlación 🔩

     sns.heatmap(total_wine.corr(), annot=True, fmt='.2f', linewidths=2)

## 9. Asimetría 🔩
    
    skew(total_wine['alcohol'])
    
 ## 10.   KNeighborsClassifier-accuracy_score.
    
    model_names=['KNearestNeighbors']
    acc=[]
    eval_acc={}
    classification_model=KNeighborsClassifier()
    classification_model.fit(x_train,y_train)
    pred=classification_model.predict(x_test)
    acc.append(accuracy_score(pred,y_test))
    eval_acc={'Modelling Algorithm':model_names,'Accuracy':acc}
    eval_acc

## Autor ✒️
    
⭐️ [fradurgo19](https://github.com/fradurgo19)
