#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import random
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV
import itertools
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import string

#%%
# Ejercicio 1
# Analisis Exploratorio de Datos

df = pd.read_csv("sign_mnist_train.csv")

caracteres = list(string.ascii_lowercase)
print(len(caracteres))

def map_labels_with_characters(characters:list, df:pd.DataFrame):
    
    d = dict()
    for i in range(len(characters)):
        d[i] = characters[i]
        
    df['character'] = df['label'].map(d)
    
    # shift column 'Name' to first position 
    first_column = df.pop('character') 
      
    # insert column using insert(position,column_name, 
    # first_column) function 
    df.insert(0, 'character', first_column) 
    
    return df

df = map_labels_with_characters(characters=caracteres, df=df)

print(df.character.value_counts())
#%%
images, labels = df.iloc[:, 2:], df['character']

# Displaying 16 random images
def displayImg(images, title):
    plt.figure(figsize = (15, 10))
    
    for i in range(len(images)):
        plt.subplot(4, 4, i + 1)
        plt.title(title[i])
        plt.imshow(np.reshape(images[i], (28, 28)), cmap = 'gray')
        plt.axis('off')
    plt.show()
    
rand = np.random.randint(0, images.shape[0] - 16)
displayImg(images.iloc[rand:rand + 16].values, labels.iloc[rand:rand + 16].values)

#%%

print(df.head(10))
print("\n")
print(df.shape)

# Hay un total de 27455 imagenes y cada imagen esta compuesta por 784 pixeles -> 28^2 = 784

#%%
# ¿Cuáles parecen ser atributos relevantes para predecir la letra a la que 
# corresponde la seña? ¿Cuáles no? ¿Creen que se pueden descartar atributos?

desvios_por_pixel = df.std()
X = [j for j in range(784)]
fig,ax = plt.subplots()
#for i in range(24):
#    ax.plot(X,promedio_letra.iloc[i,1:])
ax.plot(X,desvios_por_pixel.iloc[1:])
ax.set_title('desvíos estándar de los valores por pixel')
ax.set_xlabel('numero de pixel')
ax.set_ylabel('desvio estandar (tonos de gris)')

# Observamos mucha menor variabilidad en los primeros 100 píxeles que en el resto.
# De esto interpretamos que los primeros 100 píxeles podrían ser eliminados, mientras
# que los otros pueden ser más importantes para predecir la letra.

#%%
# ¿Hay señas que son parecidas entre sí? 
# Por ejemplo, ¿Qué es más fácil de diferenciar: 
# la seña de la E, de la seña de la L o la seña de la E de la seña de la M?

letra_E = df[df['character']=='e']
promedios_letra_e = letra_E.mean()

letra_L = df[df['character']=='l']
promedios_letra_l = letra_L.mean()

letra_M = df[df['character']=='m']
promedios_letra_m = letra_M.mean()

fig,ax = plt.subplots()
ax.plot(X,promedios_letra_e[1:])
ax.plot(X,promedios_letra_m[1:])
ax.set_xlabel('pixel')
ax.set_ylabel('color promedio')
ax.set_title('tono por pixel promedio para las letras E y M')

fig,ax = plt.subplots()
ax.plot(X,promedios_letra_e[1:])
ax.plot(X,promedios_letra_l[1:])
ax.set_xlabel('pixel')
ax.set_ylabel('color promedio')
ax.set_title('tono por pixel promedio para las letras E y L')

# Como hay mucha más superposición entre los gráficos de las letras E y M,
# interpretamos que estas dos letras se parecen mucho más que la E y la L.

#%%
# Tomen una de las clases, por ejemplo la seña correspondiente a la C, 
# ¿Son todas las imágenes muy similares entre sí?

letra_C = df[df['character']=='c']
letra_C.shape #hay 1144 filas

images, labels = letra_C.iloc[0:8, 2:], letra_C['character']
displayImg(images.values, labels.values)

lista_filas = [i for i in range(1144)]
random.seed(100)
muestra = random.choices(lista_filas,k=5)
print(muestra) #elegimos 10 números de fila aleatorios
filas_al_azar_C = letra_C.iloc[muestra,2:]
#filas_al_azar_C

X = [j for j in range(784)]
fig,ax = plt.subplots()
for fila in range(5):
    ax.plot(X,filas_al_azar_C.iloc[fila,:])
ax.set_title('tonalidades por pixel para 5 letras C aleatorias')
ax.set_xlabel('numero de pixel')
ax.set_ylabel('tonalidad de gris')

# Respuesta: En su mayoria si, cambia la forma de la mano en muchos casos, y la altura de la mano, 
# cosa que pueda confundir un algoritmo al buscar pixeles especificos 

# Este dataset está compuesto por imágenes, 
# esto plantea una diferencia frente a los datos que utilizamos en las clases 
# (por ejemplo, el dataset de Titanic). ¿Creen que esto complica la exploración de los datos?

# Respuesta: Si, la complica, sobretodo porque los datos al ser no estructurados 
# no tienen atributos especificos,con lo cual las caracteristicas 
# son la tonalidad de los pixeles individuales. También hay muchos pixeles, lo que
# complica distinguir cuáles son más importantes, y existen
# imágenes con diferentes niveles de iluminación, con las manos de distintos tamaños
# y en diferentes ubicaciones, además de que a veces aparecen elementos aparte de la mano.
# Todo esto dificulta la identificación de la imagen con una letra.

#%%
# Ejercicio 2
# Dada una imagen se desea responder la siguiente pregunta: 
# ¿la imagen corresponde a una seña de la L o a una seña de la A?

# A partir del dataframe original, construir un nuevo dataframe que
# contenga sólo al subconjunto de imágenes correspondientes a señas
# de las letras L o A.

df_ej_2 = df[(df['character'] == 'a') | (df['character'] == 'l')]

#%%
# Sobre este subconjunto de datos, analizar cuántas muestras se tienen
# y determinar si está balanceado con respecto a las dos clases a
# predecir (la seña es de la letra L o de la letra A).

# Printeamos las muestas de cada caracter y su proporcion sobre el total
l_count = df_ej_2.character.value_counts()['l']
a_count = df_ej_2.character.value_counts()['a']
proporcion_l = l_count / (l_count + a_count)
proporcion_a = a_count / (l_count + a_count)
print("Cantidad de muestras 'L': ", l_count, " corresponde a ", "%.3f"%proporcion_l, " del dataset")
print("Cantidad de muestras 'A': ", a_count, " corresponde a ", "%.3f"%proporcion_a, " del dataset")

# Consideramos que esta balanceado ya que casi hay un 50% de muestras de cada clase

#%%

letra_A = df[df['character']=='a']
#descripcion_letra_l = letra_L.describe()
promedios_letra_a = letra_A.iloc[:,2:].mean() #tomamos la fila de promedios

letra_L = df[df['character']=='l']
#descripcion_letra_m = letra_M.describe()
promedios_letra_l = letra_L.iloc[:,2:].mean()

fig,ax = plt.subplots()
ax.plot(X,promedios_letra_a)
ax.plot(X,promedios_letra_l)

#%%
images_A, labels_A = letra_A.iloc[0:8, 2:], letra_A['character']

displayImg(images_A.values, labels_A.values)

#%%
images_L, labels_L = letra_L.iloc[0:8, 2:], letra_L['character']

displayImg(images_L.values, labels_L.values)

#%%
# Separar os datos en conjuntos de train y test.

X = df_ej_2.iloc[:,2:]
y = df_ej_2.iloc[:, 0:1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

#%%
# Ajustar un modelo de KNN considerando pocos atributos, por ejemplo 3. 
# Probar con distintos conjuntos de 3 atributos y comparar resultados.
# Analizar utilizando otras cantidades de atributos.

# Creamos el modelo de KNN con k = 3
clf = KNeighborsClassifier(n_neighbors = 3)

resultados = []
modelo = {}

for i in range(10):
    lista_numero_random = random.sample(range(1, 785), 3)
    
    atributo1 = 'pixel' + str(lista_numero_random[0])
    atributo2 = 'pixel' + str(lista_numero_random[1])
    atributo3 = 'pixel' + str(lista_numero_random[2])
    
    modelo['Atributo1'] = atributo1
    modelo['Atributo2'] = atributo2
    modelo['Atributo3'] = atributo3
    
    X_train_subset = X_train[[atributo1, atributo2, atributo3]]
    
    clf.fit(X_train, y_train.values.ravel())
    clf.predict(X_test)
    
    modelo['Precision_score'] = precision_score(y_test, clf.predict(X_test),average='weighted')
    modelo['Recall_score'] = recall_score(y_test, clf.predict(X_test),average='weighted')
    modelo['R^2_score'] = clf.score(X_test, y_test)
    
    resultados.append(modelo)

print(resultados)

#%%
# Comparar modelos de KNN utilizando distintos atributos y distintos
# valores de k (vecinos). Para el análisis de los resultados, tener en
# cuenta las medidas de evaluación (por ejemplo, la exactitud) y la
# cantidad de atributos.

n_atributos = np.arange(1, 5, 1)

n_neighbors = np.arange(1, 5, 1)

train_score = {}
test_score = {}

for atributo in n_atributos:
    train_score[atributo] = {}
    test_score[atributo] = {}
    
    for neighbor in n_neighbors:

        
        X_train_subset = X_train.iloc[:, :atributo]
        X_test_subset = X_test.iloc[:, :atributo]
        
        knn = KNeighborsClassifier(n_neighbors=neighbor)
        knn.fit(X_train_subset, y_train)
    
        train_score[atributo][neighbor]=accuracy_score(y_train, knn.predict(X_train_subset))
        test_score[atributo][neighbor]=accuracy_score(y_test, knn.predict(X_test_subset))


#%%
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

for i, ax in enumerate(axs.flat):
    ax.plot(n_neighbors, train_score[i+1].values(), label="Train Accuracy")
    ax.plot(n_neighbors, test_score[i+1].values(), label="Test Accuracy")
    ax.set_xlabel("Numero de K")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy variando los k, usando {i+1} pixeles")
    ax.legend()
    ax.grid()

plt.tight_layout()
plt.show()

#%%
# Ejercicio 3
# (Clasificación multiclase) 
# Dada una imagen se desea responder la siguiente pregunta: 
# ¿A cuál de las vocales corresponde la seña en la imagen?

# Vamos a trabajar con los datos correspondientes a las 5 vocales. 
# Primero filtrar solo los datos correspondientes a esas letras. 
# Luego, separar el conjunto de datos en train y test.

df_ej_3 = df[  (df['character'] == 'a') 
             | (df['character'] == 'e')
             | (df['character'] == 'i') 
             | (df['character'] == 'o') 
             | (df['character'] == 'u')]

print(df_ej_3.character.value_counts())

#sSplit del dataset

X = df_ej_3.iloc[:,2:]
y = df_ej_3.iloc[:, 0:1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y['character'])

#%%
# Ajustar un modelo de árbol de decisión. Analizar distintas profundidades.

# Utilizamos Manual Search para analizar tres profundidades distintas 

arbol1 = DecisionTreeClassifier(max_depth = 6)
arbol2 = DecisionTreeClassifier(max_depth = 11)
arbol3 = DecisionTreeClassifier(max_depth = 16)

# Entrenamos los modelos

arbol1.fit(X_train, y_train)
arbol2.fit(X_train, y_train)
arbol3.fit(X_train, y_train)

# Calculamos el score para cada modelo

score_train_1 = arbol1.score(X_train, y_train)
score_test_1 = arbol1.score(X_test, y_test)

score_train_2 = arbol2.score(X_train, y_train)
score_test_2 = arbol2.score(X_test, y_test)

score_train_3 = arbol3.score(X_train, y_train)
score_test_3 = arbol3.score(X_test, y_test)

# Printeamos el score de cada modelos para poder compararlos

print("Arbol con max_depth = 6:", "\n  Score con dataset de Train: ", score_train_1,
      "\n  Score con dataset de Test: ", score_test_1)
print("Arbol con max_depth = 11:", "\n  Score con dataset de Train: ", score_train_2,
      "\n  Score con dataset de Test: ", score_test_2)
print("Arbol con max_depth = 16:", "\n  Score con dataset de Train: ", score_train_3,
      "\n  Score con dataset de Test: ", score_test_3)

#%%
# Para comparar y seleccionar los árboles de decisión,
# utilizar validación cruzada con k-folding.
# Importante: Para hacer k-folding utilizar los datos del conjunto de train.

# Creamos un arbol de decisión
arbol_cv = DecisionTreeClassifier(random_state = 8)


# Definimos que hiperparametros vamos a probar utilizando Grid Search
hyper_params = {'criterion' : ["gini", "entropy"],
                'max_depth' : [i for i in range(6,17,1)]}

# Creamos los modelos que vamos a entrenar con sus respectivos hiperparametros y
# utilizando 5 StratifiedKFolds
clf = GridSearchCV(arbol_cv, hyper_params, cv = 5, verbose = 2, return_train_score = True)

# Entrenamos los modelos
clf.fit(X_train, y_train)

#%%
# ¿Cuál fue el mejor modelo? 
# Evaluar cada uno de los modelos utilizando el conjunto de test. 
# Reportar su mejor modelo en el informe. 
# OBS: Al realizar la evaluación utilizar métricas de clasificación multiclase. 
# Además pueden realizar una matriz de confusión y evaluar los distintos tipos de errores para las clases.

# Asignamos y printeamos los mejores parametros encontrados con el Cross Validation
mejores_parametros = clf.best_params_
print("Mejores parametros encontrados: ", mejores_parametros)

# Finalmente evaluamos el modelo con los mejores parametros
# Calculamos el score
print("Score del arbol con mejores parametros:", "\n  Score con dataset de Train:", clf.best_score_,
      "\n  Score con dataset de Test: ", clf.best_estimator_.score(X_test, y_test))

# Creamos una matriz confusion para visualizar el error por clase
y_true = [i for i in y_test['character']]
y_pred = clf.best_estimator_.predict(X_test)
matriz_confusion = confusion_matrix(y_true, y_pred, labels=['a','e','i','o','u'])

# Hacemos un plot para visualizarla
disp = ConfusionMatrixDisplay(confusion_matrix=matriz_confusion,
                              display_labels=clf.classes_)

disp.plot()
plt.show()

# Asignamos el score a cada parametro del Cross Validation
params = clf.cv_results_['params']
score = clf.cv_results_['mean_test_score']
for i in range(len(params)):
    params[i]['score'] = score[i]
resultados_cv = params
# print("Lista de diccionarios con el score de cada modelo:", *resultados_cv, sep="\n  ")
# Lo dejamos comentado para que no ensucie el output, pueden descomentarlo para
# ver los resultados del Cross Validation

# Creamos el modelo final con los mejores parametros encontrados
arbol_final_modelo = DecisionTreeClassifier(criterion = mejores_parametros['criterion'],
                                     max_depth = mejores_parametros['max_depth'])

# Entrenamos el modelo final con TODO el dataset. Estimamos que su performance es igual
# o ligeramente mejor por haberlo entrenado con un dataset ligeramente mas grande
arbol_final = arbol_final_modelo.fit(X, y)
print(arbol_final)

#%%
# Visualizar el árbol de decisiones
plt.figure(figsize=(30, 20))
plot_tree(arbol_final, feature_names = X.columns, filled=True, fontsize=10)
fig = plt.figure(figsize=(15,7))
plt.show()

#%%
# Visualizar la importancia de las características
importances = arbol_final.feature_importances_
importance_dict = dict()

for i,v in enumerate(importances):
 print('Feature: %0d, Score: %.5f' % (i,v))
 importance_dict[i] = v

num_zeros = sum(1 for valor in importance_dict.values() if valor == 0)

print(f"Número de valores cero en el diccionario de importancias: {num_zeros}, de un total de {len(importance_dict)}")

print(f"Eso quiere decir que el mejor modelo segun el K-Fold no considera relevante el {(num_zeros/len(importance_dict))*100:.2f}% de la features")
# plot feature importance
plt.title("Barplot sobre la importancia de Features segun el DecisionTree")
plt.bar([x for x in range(len(importances))], importances)
plt.xlabel("Pixeles")
plt.ylabel("Importancia %")
plt.show()
