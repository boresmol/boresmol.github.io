---
layout: post
title: The k-Nearest Neighbors (kNN)
---

# Introducción

¡Bienvenidos al segundo post de nuestra serie sobre algoritmos de aprendizaje automático! En esta ocasión, vamos a sumergirnos en el fascinante mundo del algoritmo de clasificación K-Nearest Neighbors (KNN). Si estás buscando una forma intuitiva y efectiva de clasificar datos, estás en el lugar correcto. En este post, exploraremos los conceptos básicos de KNN y cómo se utiliza para realizar clasificaciones basadas en los vecinos más cercanos. Prepárate para descubrir una técnica poderosa y sencilla de entender. ¡Comencemos!

# Indice

1. Idea Intuitiva
2. Explicación de kNN
3. Código 'from scratch'
4. Código con 'scikit-learn'

# Idea Intuitiva

Imagina que tienes una lista de animales, como perros, gatos y conejos. Cada animal tiene diferentes características, como su tamaño, color y sonido que hacen. Ahora, quieres saber a qué tipo de animal pertenece uno nuevo que encontraste, pero no estás seguro.

Aquí es donde entra en juego el algoritmo kNN. KNN significa "vecino más cercano". El concepto es simple: miras a los animales que ya conoces y buscas a los más similares al nuevo animal que encontraste.

Vamos a usar el tamaño y el color como características para clasificar a los animales. Imagina que el nuevo animal es pequeño y de color marrón. Para clasificarlo, observamos los animales que ya conocemos y buscamos los más similares en términos de tamaño y color.

Digamos que encontramos los tres animales más cercanos al nuevo animal. Resulta que dos de ellos son conejos y uno es un gato. Ahora, basándonos en los animales similares, podemos decir que es muy probable que el nuevo animal también sea un conejo.

El algoritmo kNN usa la idea de buscar a los vecinos más cercanos para tomar decisiones. En este caso, los vecinos son los animales que son más similares al nuevo animal. Al observar a los vecinos cercanos, podemos tomar una decisión sobre qué tipo de animal es el nuevo.

Recuerda, el valor de "k" en kNN es el número de vecinos que miramos. En nuestro ejemplo, usamos k=3, lo que significa que buscamos a los tres animales más cercanos. Puedes ajustar el valor de "k" dependiendo de cuántos vecinos desees considerar.

Así es como funciona el algoritmo kNN utilizando animales como ejemplo. ¡Es una forma divertida y sencilla de clasificar cosas basándonos en las características similares!

# Explicación de kNN

A continuación, vamos a dar algunas características sobre kNN para posteriormente, explicar en detalle el funcionamiento de este algoritmo:

## kNN es un algoritmo supervisado de aprendizaje automático.

La primera propiedad determinante de los algoritmos de aprendizaje automático es la distinción entre modelos supervisados y no supervisados. La diferencia entre modelos supervisados y no supervisados radica en el planteamiento del problema.

En modelos supervisados, tienes dos tipos de variables al mismo tiempo:

1. Una variable objetivo, que también se conoce como variable dependiente o variable y.
2. Variables independientes, que también se conocen como variables x o variables explicativas.

La variable objetivo es la variable que deseas predecir. Depende de las variables independientes y no es algo que conozcas de antemano. Las variables independientes son variables que conoces de antemano. Puedes introducirlas en una ecuación para predecir la variable objetivo. De esta manera, es relativamente similar al caso de y = ax + b.

En el gráfico, la variable objetivo es la forma del punto de datos y las variables independientes son la altura y el ancho. Puedes ver la idea detrás del aprendizaje supervisado en el siguiente gráfico:

![aprendizaje sup](https://github.com/boresmol/boresmol.github.io/blob/master/images/knn_02_MLsupervised_wide.aa50e6348ca4.png?raw=true)

En este gráfico, los puntos de datos tienen cada uno una altura, un ancho y una forma. Hay cruces, estrellas y triángulos. A la derecha hay una regla de decisión que un modelo de aprendizaje automático podría haber aprendido.

En este caso, las observaciones marcadas con una cruz son altas pero no anchas. Las estrellas son tanto altas como anchas. Los triángulos son bajos pero pueden ser anchos o estrechos. Básicamente, el modelo ha aprendido una regla de decisión para determinar si una observación es más probable que sea una cruz, una estrella o un triángulo basándose únicamente en su altura y ancho.

En los modelos no supervisados, no hay una división entre variables objetivo y variables independientes. El aprendizaje no supervisado intenta agrupar los puntos de datos evaluando su similitud.

Como se puede ver en el ejemplo, nunca se puede estar seguro de que los puntos de datos agrupados pertenezcan fundamentalmente juntos, pero siempre y cuando el agrupamiento tenga sentido, puede ser muy valioso en la práctica. Puedes ver la idea detrás del aprendizaje no supervisado en el siguiente gráfico:

![knn unsupervised](https://github.com/boresmol/boresmol.github.io/blob/master/images/knn_03_MLunsupervised_wide.a6fd142b42de.png?raw=true)

En este gráfico, las observaciones ya no tienen formas diferentes. Todas son círculos. Sin embargo, aún se pueden agrupar en tres grupos basados en la distancia entre los puntos. En este ejemplo particular, hay tres grupos de puntos que se pueden separar según el espacio vacío entre ellos.

El algoritmo kNN es un modelo de aprendizaje automático supervisado. Esto significa que predice una variable objetivo utilizando una o varias variables independientes.

## kNN es un algoritmo de aprendizaje no lineal.
Una segunda propiedad que marca una gran diferencia en los algoritmos de aprendizaje automático es si los modelos pueden estimar relaciones no lineales.

Los modelos lineales son aquellos que predicen utilizando líneas o hiperplanos. En la imagen, el modelo se representa como una línea dibujada entre los puntos. El modelo y = ax + b es el ejemplo clásico de un modelo lineal. Puedes ver cómo un modelo lineal podría ajustarse a los datos de ejemplo en el siguiente esquema:

![lineal model](https://github.com/boresmol/boresmol.github.io/blob/master/images/modelo_lineal.jpg?raw=true)

En esta imagen, los puntos de datos se representan a la izquierda con estrellas, triángulos y cruces. A la derecha hay un modelo lineal que puede separar los triángulos de los que no son triángulos. La decisión es una línea. Cada punto por encima de la línea es un no triángulo, y todo lo que está por debajo de la línea es un triángulo.

Si quisieras agregar otra variable independiente al gráfico anterior, necesitarías representarla como una dimensión adicional, creando así un cubo con las formas en su interior. Sin embargo, una línea no podría cortar un cubo en dos partes. El equivalente multidimensional de la línea es el hiperplano. Por lo tanto, un modelo lineal se representa mediante un hiperplano, que en el caso del espacio bidimensional resulta ser una línea.

Los modelos no lineales son aquellos que utilizan cualquier enfoque que no sea una línea para separar sus casos. Un ejemplo conocido es el árbol de decisiones, que básicamente es una larga lista de declaraciones if ... else. En el gráfico no lineal, las declaraciones if ... else te permitirían dibujar cuadrados u cualquier otra forma que quisieras dibujar. El siguiente gráfico representa un modelo no lineal aplicado a los datos de ejemplo:

![no lineal](https://github.com/boresmol/boresmol.github.io/blob/master/images/no_lineal.jpg?raw=true)

Este gráfico muestra cómo una decisión puede ser no lineal. La regla de decisión está compuesta por tres cuadrados. La caja en la que cae un nuevo punto de datos definirá su forma predicha. Observa que no es posible ajustar esto de una vez usando una línea: se necesitan dos líneas. Este modelo podría ser recreado con declaraciones if ... else de la siguiente manera:

1. Si la altura del punto de datos es baja, entonces es un triángulo.
2. De lo contrario, si el ancho del punto de datos es bajo, entonces es una cruz.
3. De lo contrario, si ninguna de las condiciones anteriores es verdadera, entonces es una estrella.

kNN es un ejemplo de un modelo no lineal. Más adelante en este tutorial, volveremos a la forma exacta en que se calcula el modelo.

##  kNN es un algoritmo supervisado tanto para clasificación como para regresión.
Los algoritmos de aprendizaje automático supervisado se pueden dividir en dos grupos según el tipo de variable objetivo que pueden predecir:

1. La clasificación es una tarea de predicción con una variable objetivo categórica. Los modelos de clasificación aprenden a clasificar cualquier nueva observación. Esta clase asignada puede ser correcta o incorrecta, no hay un punto intermedio. Un ejemplo clásico de clasificación es el conjunto de datos de iris, en el cual se utilizan medidas físicas de plantas para predecir su especie. Un famoso algoritmo que se puede utilizar para la clasificación es la regresión logística.

2. La regresión es una tarea de predicción en la cual la variable objetivo es numérica. Un ejemplo famoso de regresión es el desafío de precios de viviendas en Kaggle. En este concurso de aprendizaje automático, los participantes intentan predecir los precios de venta de casas basándose en numerosas variables independientes.

En el siguiente gráfico, puedes ver cómo se vería una regresión y una clasificación utilizando el ejemplo anterior:

![regvsclas](https://github.com/boresmol/boresmol.github.io/blob/master/images/knn_06_MLclassificationregression.6029d11323aa.png?raw=true)

La parte izquierda de esta imagen es una clasificación. La variable objetivo es la forma de la observación, que es una variable categórica. La parte derecha es una regresión. La variable objetivo es numérica. Las reglas de decisión podrían ser exactamente las mismas para ambos ejemplos, pero sus interpretaciones son diferentes.

Para una única predicción, las clasificaciones son correctas o incorrectas, mientras que las regresiones tienen un error en una escala continua. Tener una medida de error numérica es más práctico, por lo que muchos modelos de clasificación predicen no solo la clase, sino también la probabilidad de pertenecer a una de las clases.

Algunos modelos solo pueden hacer regresión, otros solo pueden hacer clasificación, y algunos pueden hacer ambas. El algoritmo kNN se adapta sin problemas tanto a la clasificación como a la regresión.

## Desventajas de kNN
La verdadera limitación de kNN es su capacidad para adaptarse a relaciones altamente complejas entre variables independientes y dependientes. Es menos probable que kNN tenga un buen rendimiento en tareas avanzadas como visión por computadora y procesamiento del lenguaje natural.

Puedes intentar mejorar el rendimiento de kNN tanto como sea posible, potencialmente agregando otras técnicas de aprendizaje automático. En la última parte del tutorial, verás una técnica llamada bagging, que es una forma de mejorar el rendimiento predictivo. Sin embargo, llegado a un cierto nivel de complejidad, es probable que kNN sea menos efectivo que otros modelos, independientemente de cómo se haya ajustado.

## Usar kNN para predecir la edad de las babosas de mar
Para seguir con la parte de codificación, usaremos un conjunto de datos de ejemplo para el resto de este tutorial: el conjunto de datos de abulones (Abalone Dataset). Este conjunto de datos contiene mediciones de edad de una gran cantidad de abulones.

## El enunciado del problema de los abulones
La edad de un abulón se puede determinar cortando su concha y contando el número de anillos en la misma. En el conjunto de datos de abulones (Abalone Dataset), puedes encontrar las mediciones de edad de una gran cantidad de abulones junto con muchas otras mediciones físicas.

El objetivo del proyecto es desarrollar un modelo que pueda predecir la edad de un abulón basándose únicamente en las otras mediciones físicas. Esto permitiría a los investigadores estimar la edad del abulón sin tener que cortar su concha y contar los anillos.

Aplicarás el algoritmo kNN para encontrar la predicción más cercana posible.

## Importar el conjunto de datos de abulones
En este tutorial, trabajaremos con el conjunto de datos de abulones (Abalone Dataset). Podrías descargarlo y usar pandas para importar los datos a Python, pero es aún más rápido dejar que pandas importe los datos directamente por ti.


Puedes importar los datos utilizando pandas de la siguiente manera:

```python3
import pandas as pd
url = ("https://archive.ics.uci.edu/ml/machine-learning-databases""/abalone/abalone.data")
abalone = pd.read_csv(url, header=None)
```
Vamos a procesar un poco el dataset:

```python3
abalone.columns = [
"Sex",
"Length",
"Diameter",
"Height",
"Whole weight",
"Shucked weight",
"Viscera weight",
"Shell weight",
"Rings"
]

abalone = abalone.drop("Sex", axis=1)
```

Eliminamos la variable sexo ya que el objetivo es predecir la edad mediante variables puramente físicas y consideramos que esta variable no cumple la condición. 

## Un kNN paso a paso desde cero en Python
En esta parte del tutorial, descubrirás cómo funciona el algoritmo kNN en profundidad. El algoritmo tiene dos componentes matemáticos principales que deberás entender. Para comenzar, realizarás un recorrido en lenguaje sencillo del algoritmo kNN.

### Recorrido en lenguaje sencillo del algoritmo kNN
El algoritmo kNN es un poco atípico en comparación con otros algoritmos de aprendizaje automático. Como viste anteriormente, cada modelo de aprendizaje automático tiene su fórmula específica que debe ser estimada. Lo particular del algoritmo de los vecinos más cercanos (kNN) es que esta fórmula se calcula no en el momento de ajustar el modelo, sino en el momento de hacer una predicción. Esto no es así para la mayoría de los otros modelos.

Cuando llega un nuevo punto de datos, el algoritmo kNN, como su nombre indica, comienza encontrando los vecinos más cercanos de este nuevo punto de datos. Luego, toma los valores de esos vecinos y los utiliza como predicción para el nuevo punto de datos.

Como ejemplo intuitivo de por qué esto funciona, piensa en tus vecinos. Tus vecinos suelen ser relativamente similares a ti. Probablemente estén en la misma clase socioeconómica que tú. Tal vez tengan el mismo tipo de trabajo que tú, tal vez sus hijos vayan a la misma escuela que los tuyos, y así sucesivamente. Pero para algunas tareas, este enfoque no es tan útil. Por ejemplo, no tendría sentido mirar el color favorito de tu vecino para predecir el tuyo.

El algoritmo kNN se basa en la noción de que puedes predecir las características de un punto de datos basándote en las características de sus vecinos. En algunos casos, este método de predicción puede tener éxito, mientras que en otros casos puede no tenerlo. A continuación, verás la descripción matemática de "más cercano" para los puntos de datos y los métodos para combinar múltiples vecinos en una sola predicción.

### Define "Más cercano" utilizando una definición matemática de distancia
Para encontrar los puntos de datos que están más cerca del punto que necesitas predecir, puedes utilizar una definición matemática de distancia llamada distancia euclidiana.

Para llegar a esta definición, primero debes entender lo que se entiende por diferencia de dos vectores. Aquí tienes un ejemplo:

![knn pitagoras](https://github.com/boresmol/boresmol.github.io/blob/master/images/knn_pitagoras.jpg?raw=true)

En esta imagen, puedes ver dos puntos de datos: uno azul en (2,2) y otro verde en (4,4). Para calcular la distancia entre ellos, puedes empezar por sumar dos vectores. El vector a va desde el punto (4,2) hasta el punto (4,4), y el vector b va desde el punto (4,2) hasta el punto (2,2). Sus extremos están indicados por los puntos de colores. Observa que forman un ángulo de 90 grados.

La diferencia entre estos vectores es el vector c, que va desde el extremo del vector a hasta el extremo del vector b. La longitud del vector c representa la distancia entre tus dos puntos de datos.

La longitud de un vector se llama norma. La norma es un valor positivo que indica la magnitud del vector. Puedes calcular la norma de un vector utilizando la fórmula euclidiana:

![formula euclid](https://github.com/boresmol/boresmol.github.io/blob/master/images/euclidean_distance.5b5fe10e9fa0.png?raw=true)

En esta fórmula, la distancia se calcula tomando las diferencias al cuadrado en cada dimensión y luego tomando la raíz cuadrada de la suma de esos valores. En este caso, debes calcular la norma del vector diferencia c para obtener la distancia entre los puntos de datos.

Ahora, para aplicar esto a tus datos, debes entender que tus puntos de datos son en realidad vectores. Luego, puedes calcular la distancia entre ellos calculando la norma del vector diferencia.

Puedes hacer esto en Python utilizando `linalg.norm()` de NumPy. Aquí tienes un ejemplo:

```python3

a = np.array([2, 2])
b = np.array([4, 4])
np.linalg.norm(a - b)
2.8284271247461903

```

En este bloque de código, defines tus puntos de datos como vectores. Luego, calculas `norm()` en la diferencia entre dos puntos de datos. De esta manera, obtienes directamente la distancia entre dos puntos multidimensionales. Aunque los puntos son multidimensionales, la distancia entre ellos sigue siendo un escalar, es decir, un único valor.

### Encontrar los k vecinos más cercanos
Ahora que tienes una forma de calcular la distancia desde cualquier punto a cualquier otro punto, puedes usar esto para encontrar los vecinos más cercanos a un punto en el que deseas hacer una predicción.

Necesitas encontrar un número de vecinos, y ese número está dado por k. El valor mínimo de k es 1. Esto significa usar solo un vecino para la predicción. El valor máximo es el número de puntos de datos que tienes. Esto significa usar todos los vecinos. El valor de k es algo que el usuario define. Las herramientas de optimización pueden ayudarte con esto, como verás en la última parte de este tutorial.

Ahora, para encontrar los vecinos más cercanos en NumPy, vuelve al conjunto de datos de abulones. Como has visto, necesitas definir distancias en los vectores de las variables independientes, por lo que primero debes convertir tu DataFrame de pandas en un arreglo de NumPy utilizando el atributo `.values`:

```python3

X = abalone.drop("Rings", axis=1)
X = X.values
y = abalone["Rings"]
y = y.values

```

Este bloque de código genera dos objetos que ahora contienen tus datos: X e y. X son las variables independientes y y es la variable dependiente de tu modelo. Observa que usas una letra mayúscula para X pero una letra minúscula para y. Esto se hace comúnmente en el código de aprendizaje automático porque la notación matemática generalmente usa una letra mayúscula para las matrices y una letra minúscula para los vectores.

Ahora puedes aplicar kNN con k = 3 en un nuevo abulón que tiene las siguientes mediciones físicas:

| Variable        | Value     |
|-----------------|-----------|
| Length          | 0.569552  |
| Diameter        | 0.446407  |
| Height          | 0.154437  |
| Whole weight    | 1.016849  |
| Shucked weight  | 0.439051  |
| Viscera weight  | 0.222526  |
| Shell weight    | 0.291208  |

Puedes crear este individuo en python de la siguiente forma:

```python3

new_data_point = np.array([
0.569552,
0.446407,
0.154437,
1.016849,
0.439051,
0.222526,
0.291208,
])

```
El siguiente paso es calcular las distancias entre este nuevo punto de datos y cada uno de los puntos de datos en el conjunto de datos de abulones utilizando el siguiente código:

```python3

distances = np.linalg.norm(X - new_data_point, axis=1)

```

Ahora tienes un vector de distancias y necesitas determinar cuáles son los tres vecinos más cercanos. Para hacer esto, necesitas encontrar los ID de las distancias mínimas. Puedes utilizar el método .argsort() para ordenar el arreglo de menor a mayor, y puedes tomar los primeros k elementos para obtener los índices de los k vecinos más cercanos:

```python3


k = 3
nearest_neighbor_ids = distances.argsort()[:k]
nearest_neighbor_ids
array([4045, 1902, 1644], dtype=int32)

```

Esto te indica qué tres vecinos son los más cercanos a tu new_data_point. En el siguiente párrafo, verás cómo convertir esos vecinos en una estimación.

Votación o Promedio de Múltiples Vecinos
Después de identificar los índices de los tres vecinos más cercanos de tu abulón de edad desconocida, ahora necesitas combinar esos vecinos para hacer una predicción para tu nuevo punto de datos.

Como primer paso, debes encontrar los valores reales para esos tres vecinos:

```python3

nearest_neighbor_rings = y[nearest_neighbor_ids]
nearest_neighbor_rings
array([ 9, 11, 10])

```

Ahora que tienes los valores de esos tres vecinos, los combinarás en una predicción para tu nuevo punto de datos. La forma de combinar los vecinos en una predicción varía según si se trata de regresión o clasificación.

### Promedio para Regresión
En problemas de regresión, la variable objetivo es numérica. Combinas varios vecinos en una sola predicción tomando el promedio de sus valores de la variable objetivo. Puedes hacerlo de la siguiente manera:

```python3

prediction = nearest_neighbor_rings.mean()

```

Obtendrás un valor de 10 para la predicción. Esto significa que la predicción de los 3 vecinos más cercanos para tu nuevo punto de datos es 10. Puedes hacer lo mismo para cualquier cantidad de nuevos abulones que desees.

### Moda para Clasificación
En problemas de clasificación, la variable objetivo es categórica. Como se discutió anteriormente, no puedes calcular promedios en variables categóricas. Por ejemplo, ¿cuál sería el promedio de tres marcas de automóviles predichas? Sería imposible decirlo. No se puede aplicar un promedio en predicciones de clase.

En cambio, en el caso de la clasificación, se utiliza la moda. La moda es el valor que ocurre con mayor frecuencia. Esto significa que cuentas las clases de todos los vecinos y retienes la clase más común. La predicción es el valor que ocurre con mayor frecuencia entre los vecinos.

Si hay varias modas, existen múltiples soluciones posibles. Podrías seleccionar al azar un ganador final entre los ganadores. También podrías tomar la decisión final basada en las distancias de los vecinos, en cuyo caso se retendría la moda de los vecinos más cercanos.

Puedes calcular la moda utilizando la función `mode()` de SciPy. Como el ejemplo del abulón no es un caso de clasificación, el siguiente código muestra cómo puedes calcular la moda para un ejemplo simple:

```python3
import scipy.stats
import numpy as np

class_neighbors = np.array(["A", "B", "B", "C"])
scipy.stats.mode(class_neighbors)
ModeResult(mode=array('B', dtype='<U1'), count=array([2]))
```

Como puedes ver, la moda en este ejemplo es "B" porque es el valor que aparece con mayor frecuencia en los datos de entrada

## Ajustar kNN en Python utilizando scikit-learn

Si bien programar un algoritmo desde cero es excelente para fines de aprendizaje, generalmente no es muy práctico cuando se trabaja en una tarea de aprendizaje automático. En esta sección, explorarás la implementación del algoritmo kNN utilizado en scikit-learn, una de las bibliotecas más completas de aprendizaje automático en Python.

### División de los datos en conjuntos de entrenamiento y prueba para la evaluación del modelo

En esta sección, evaluarás la calidad de tu modelo kNN de abalones. En las secciones anteriores, te has centrado en aspectos técnicos, pero ahora adoptarás un enfoque más pragmático y orientado a los resultados.

Existen múltiples formas de evaluar modelos, pero la más común es la división de entrenamiento y prueba. Al utilizar una división de entrenamiento y prueba para evaluar un modelo, divides el conjunto de datos en dos partes:

1. Los datos de entrenamiento se utilizan para ajustar el modelo. Para kNN, esto significa que los datos de entrenamiento se utilizarán como vecinos.
2. Los datos de prueba se utilizan para evaluar el modelo. Esto significa que realizarás predicciones para el número de anillos de cada abalón en los datos de prueba y compararás esos resultados con el número real de anillos conocido.
Puedes dividir los datos en conjuntos de entrenamiento y prueba en Python utilizando la función integrada `train_test_split()` de scikit-learn:

```python3

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=12345
)

```

El parámetro test_size se refiere al número de observaciones que deseas asignar a los datos de entrenamiento y los datos de prueba. Si especificas un `test_size` de 0.2, el tamaño del conjunto de prueba será el 20 por ciento de los datos originales, dejando el 80 por ciento restante como datos de entrenamiento.

`random_state` es un parámetro que te permite obtener los mismos resultados cada vez que se ejecute el código. `train_test_split()` realiza una división aleatoria de los datos, lo cual puede ser problemático para reproducir los resultados. Por lo tanto, es común utilizar `random_state`. La elección del valor de `random_state` es arbitraria.

En el código anterior, separas los datos en conjuntos de entrenamiento y prueba. Esto es necesario para la evaluación objetiva del modelo. Ahora puedes proceder a ajustar un modelo kNN en los datos de entrenamiento utilizando scikit-learn.

### Ajustando una Regresión kNN en scikit-learn para el conjunto de datos de Abalone

Para ajustar un modelo con scikit-learn, comienzas creando un modelo de la clase correcta. En este punto, también necesitas elegir los valores para tus hiperparámetros. Para el algoritmo kNN, debes elegir el valor de k, que se llama `n_neighbors` en la implementación de scikit-learn. Así es como puedes hacerlo en Python:

```python3

from sklearn.neighbors import KNeighborsRegressor
knn_model = KNeighborsRegressor(n_neighbors=3)

```

Creas un modelo no ajustado con knn_model. Este modelo utilizará los tres vecinos más cercanos para predecir el valor de un punto de datos futuro. Para introducir los datos en el modelo, puedes ajustar el modelo en el conjunto de datos de entrenamiento:

```python3

knn_model.fit(X_train, y_train)

```

Usando el método `.fit()`, permites que el modelo aprenda de los datos. En este punto, knn_model contiene todo lo necesario para hacer predicciones sobre nuevos puntos de datos de abalones. ¡Ese es todo el código que necesitas para ajustar una regresión kNN usando Python!

### Usando scikit-learn para inspeccionar el ajuste del modelo

Ajustar un modelo, sin embargo, no es suficiente. En esta sección, verás algunas funciones que puedes utilizar para evaluar el ajuste.

Existen muchas métricas de evaluación disponibles para la regresión, pero utilizarás una de las más comunes, el error cuadrático medio (RMSE). El RMSE de una predicción se calcula de la siguiente manera:

1. Calcula la diferencia entre el valor real y el valor predicho de cada punto de datos.
2. Para cada diferencia, toma el cuadrado de esta diferencia.
3. Suma todas las diferencias al cuadrado.
4. Toma la raíz cuadrada del valor sumado.
Para comenzar, puedes evaluar el error de predicción en los datos de entrenamiento. Esto significa que utilizas los datos de entrenamiento para hacer predicciones, por lo que sabes que el resultado debería ser relativamente bueno. Puedes usar el siguiente código para obtener el RMSE:

```python3

from sklearn.metrics import mean_squared_error
from math import sqrt

train_preds = knn_model.predict(X_train)
mse = mean_squared_error(y_train, train_preds)
rmse = sqrt(mse)
rmse
1.65
```

En este código, calculas el RMSE utilizando el modelo knn_model que ajustaste en el bloque de código anterior. Calculas el RMSE en los datos de entrenamiento por ahora. Para obtener un resultado más realista, debes evaluar el rendimiento en datos que no están incluidos en el modelo. Es por eso que mantuviste separado el conjunto de prueba hasta ahora. Puedes evaluar el rendimiento predictivo en el conjunto de prueba con la misma función que antes:

```python3

test_preds = knn_model.predict(X_test)
mse = mean_squared_error(y_test, test_preds)
rmse = sqrt(mse)
rmse
2.37

```


En este bloque de código, evaluas el error en datos que aún no eran conocidos por el modelo. Este RMSE más realista es ligeramente mayor que antes. El RMSE mide el error promedio de la edad predicha, por lo que puedes interpretar esto como un error promedio de 1.65 años. Si una mejora de 2.37 años a 1.65 años es buena o no depende del caso específico. Al menos te estás acercando a estimar correctamente la edad.

Hasta ahora, solo has utilizado el algoritmo kNN de scikit-learn tal como está. Aún no has ajustado los hiperparámetros y has elegido aleatoriamente un valor para k. Puedes observar una diferencia relativamente grande entre el RMSE en los datos de entrenamiento y el RMSE en los datos de prueba. Esto significa que el modelo sufre de sobreajuste en los datos de entrenamiento: no generaliza bien.

Esto no es motivo de preocupación en este momento. En la próxima parte, verás cómo optimizar el error de predicción o error de prueba utilizando varios métodos de ajuste.

### Visualización del ajuste de tu modelo

Antes de comenzar a mejorar el modelo, es importante analizar el ajuste real del mismo. Para comprender lo que el modelo ha aprendido, puedes visualizar cómo se han realizado las predicciones utilizando Matplotlib. Aquí tienes un ejemplo de cómo hacerlo:

```python3

import seaborn as sns
import matplotlib.pyplot as plt

cmap = sns.cubehelix_palette(as_cmap=True)
f, ax = plt.subplots()
points = ax.scatter(
X_test[:, 0], X_test[:, 1], c=test_preds, s=50, cmap=cmap
)
f.colorbar(points)
plt.show()

```

En este bloque de código, utilizas Seaborn para crear un gráfico de dispersión de las primeras dos columnas de X_test seleccionando los arrays X_test[:,0] y X_test[:,1]. Recuerda que las primeras dos columnas son Longitud y Diámetro. Están fuertemente correlacionadas, como has visto en la tabla de correlaciones.

Utilizas el parámetro c para especificar que los valores predichos (test_preds) deben usarse como una barra de color. El argumento s se utiliza para especificar el tamaño de los puntos en el gráfico de dispersión. Utilizas cmap para especificar el mapa de colores cubehelix_palette. Para obtener más información sobre cómo trazar con Matplotlib, consulta "Python Plotting With Matplotlib".

Con el código anterior, obtendrás el siguiente gráfico:

![plot](https://github.com/boresmol/boresmol.github.io/blob/master/images/Captura%20de%20pantalla%202023-07-06%20134709.jpg?raw=true)

En este gráfico, cada punto representa un abalón del conjunto de prueba, con su longitud real en el eje X y su diámetro real en el eje Y. El color del punto refleja la edad predicha. Puedes observar que cuanto más largo y grande es un abalón, mayor es su edad predicha. Esto es lógico y es una señal positiva. Significa que tu modelo está aprendiendo algo que parece correcto.

Esta visualización es una vista bidimensional de un conjunto de datos de siete dimensiones. Si experimentas con ellas, te dará una gran comprensión de lo que el modelo está aprendiendo y, tal vez, de lo que no está aprendiendo o está aprendiendo incorrectamente.

### Ajustar y optimizar kNN en Python utilizando scikit-learn

Existen numerosas formas de mejorar tu puntuación predictiva. Algunas mejoras se pueden lograr trabajando con los datos de entrada mediante técnicas de manipulación de datos, pero en este tutorial, nos enfocaremos en el algoritmo kNN. A continuación, veremos cómo mejorar la parte del algoritmo en el proceso de modelado.

#### Mejorando el rendimiento de kNN en scikit-learn utilizando GridSearchCV

Hasta ahora, siempre hemos trabajado con k=3 en el algoritmo kNN, pero el mejor valor para k es algo que debes encontrar empíricamente para cada conjunto de datos.

Cuando usas pocos vecinos, la predicción será mucho más variable que cuando usas más vecinos:

1. Si utilizas solo un vecino, la predicción puede cambiar drásticamente de un punto a otro. Cuando piensas en tus propios vecinos, es posible que uno sea bastante diferente de los demás. Si vivieras al lado de un valor atípico, tu predicción con 1-NN sería incorrecta.

2. Si tienes varios puntos de datos, el impacto de un vecino extremadamente diferente será mucho menor.

3. Si utilizas demasiados vecinos, la predicción de cada punto puede ser muy similar. Digamos que utilizas todos los vecinos para una predicción. En ese caso, todas las predicciones serían iguales.

Para encontrar el mejor valor de k, vamos a utilizar una herramienta llamada GridSearchCV. Esta es una herramienta que se utiliza frecuentemente para ajustar hiperparámetros de modelos de aprendizaje automático. En tu caso, te ayudará a encontrar automáticamente el mejor valor de k para tu conjunto de datos.

GridSearchCV está disponible en scikit-learn, y tiene la ventaja de que se utiliza de manera muy similar a los modelos de scikit-learn:

```python3

from sklearn.model_selection import GridSearchCV

parameters = {"n_neighbors": range(1, 50)}
gridsearch = GridSearchCV(KNeighborsRegressor(), parameters)
gridsearch.fit(X_train, y_train)

GridSearchCV(estimator=KNeighborsRegressor(),
param_grid={'n_neighbors': range(1, 50)})

```

Here, you use GridSearchCV to fit the model. In short, GridSearchCV repeatedly fits kNN regressors on a part of the data and tests the performances on the remaining part of the data. Doing this repeatedly will yield a reliable estimate of the predictive performance of each of the values for k. In this example, you test the values from 1 to 50.

In the end, it will retain the best performing value of k, which you can access with `.best_params_`:

```python3

gridsearch.best_params_
{'n_neighbors': 25, 'weights': 'distance'}

```

En este código, puedes imprimir los parámetros que tienen el menor puntaje de error. Con `.best_params_`, puedes ver que elegir 25 como valor para k brindará el mejor rendimiento predictivo. Ahora que sabes cuál es el mejor valor de k, puedes ver cómo afecta tus rendimientos de entrenamiento y prueba:

```python3

train_preds_grid = gridsearch.predict(X_train)
train_mse = mean_squared_error(y_train, train_preds_grid)
train_rmse = sqrt(train_mse)
test_preds_grid = gridsearch.predict(X_test)
test_mse = mean_squared_error(y_test, test_preds_grid)
test_rmse = sqrt(test_mse)
train_rmse
2.0731294674202143
test_rmse
2.1700197339962175

```

Con este código, ajustas el modelo en los datos de entrenamiento y evalúas los datos de prueba. Puedes ver que el error de entrenamiento es peor que antes, pero el error de prueba es mejor que antes. Esto significa que tu modelo se ajusta de manera menos cercana a los datos de entrenamiento. El uso de GridSearchCV para encontrar un valor para k ha reducido el problema de sobreajuste en los datos de entrenamiento.

### Añadiendo el promedio ponderado de vecinos basado en la distancia

Utilizando GridSearchCV, lograste reducir el RMSE de prueba de 2.37 a 2.17. En esta sección, verás cómo mejorar aún más el rendimiento.

A continuación, probarás si el rendimiento de tu modelo mejora al predecir utilizando un promedio ponderado en lugar de un promedio regular. Esto significa que los vecinos que están más lejos tendrán menos influencia en la predicción.

Puedes hacer esto estableciendo el hiperparámetro de pesos (weights) en el valor "distance". Sin embargo, establecer este promedio ponderado podría tener un impacto en el valor óptimo de k. Por lo tanto, nuevamente utilizarás GridSearchCV para determinar qué tipo de promedio debes usar:

```python3

parameters = {
"n_neighbors": range(1, 50),
"weights": ["uniform", "distance"],
}
gridsearch = GridSearchCV(KNeighborsRegressor(), parameters)
gridsearch.fit(X_train, y_train)
gridsearch.best_params_
{'n_neighbors': 25, 'weights': 'distance'}
test_preds_grid = gridsearch.predict(X_test)
test_mse = mean_squared_error(y_test, test_preds_grid)
test_rmse = sqrt(test_mse)
test_rmse
2.163426558494748
```

Aquí, pruebas si tiene sentido utilizar un peso diferente utilizando tu GridSearchCV. Al aplicar un promedio ponderado en lugar de un promedio regular, has reducido el error de predicción de 2.17 a 2.1634. Aunque no es una mejora enorme, sigue siendo mejor, lo que lo hace valioso

### Mejora usando Bagging


Para mejorar aún más el ajuste de kNN en scikit-learn, puedes utilizar el método de Bagging como tercer paso. Bagging es un método de conjunto, o un método que toma un modelo de aprendizaje automático relativamente sencillo y ajusta una gran cantidad de esos modelos con ligeras variaciones en cada ajuste. Bagging a menudo utiliza árboles de decisión, pero kNN también funciona perfectamente.

Los métodos de conjunto suelen ser más efectivos que los modelos individuales. Un modelo puede estar equivocado de vez en cuando, pero el promedio de cien modelos debería estar equivocado menos veces. Los errores de diferentes modelos individuales tienden a promediarse entre sí, y la predicción resultante será menos variable.

Puedes utilizar scikit-learn para aplicar bagging a tu regresión kNN siguiendo los siguientes pasos. Primero, crea el KNeighborsRegressor con las mejores opciones para k y weights que obtuviste de GridSearchCV:

```python3

best_k = gridsearch.best_params_["n_neighbors"]
best_weights = gridsearch.best_params_["weights"]
bagged_knn = KNeighborsRegressor(n_neighbors=best_k, weights=best_weights)

```

Luego importa la clase BaggingRegressor de scikit-learn y crea una nueva instancia con 100 estimadores utilizando el modelo `bagged_knn`:

```python3

from sklearn.ensemble import BaggingRegressor
bagging_model = BaggingRegressor(bagged_knn, n_estimators=100)

```

Ahora puedes hacer una predicción y calcular el RMSE para ver si mejoró:

```python3
test_preds_grid = bagging_model.predict(X_test)
test_mse = mean_squared_error(y_test, test_preds_grid)
test_rmse = sqrt(test_mse)
test_rmse
2.1616
```

Como vemos, gracias al Bagging hemos conseguido un error ligeramente menor.

En tres pasos incrementales, has mejorado el rendimiento predictivo del algoritmo. La siguiente tabla muestra un resumen de los diferentes modelos y sus rendimientos:

| Modelo                            | Error    |
|----------------------------------|----------|
| K arbitraria                     | 2.37     |
| GridSearchCV para k              | 2.17     |
| GridSearchCV para k y pesos      | 2.1634   |
| Bagging y GridSearchCV           | 2.1616   |

En esta tabla, se muestran los cuatro modelos desde el más simple hasta el más complejo. El orden de complejidad corresponde con el orden de las métricas de error. El modelo con un k aleatorio tuvo el peor rendimiento, mientras que el modelo con bagging y GridSearchCV tuvo el mejor rendimiento.

Es posible que se puedan lograr más mejoras en las predicciones de los abalones. Por ejemplo, sería posible buscar formas de manipular los datos de manera diferente o encontrar otras fuentes de datos externas.

## Conclusión
Ahora que conoces todo sobre el algoritmo kNN, estás listo para comenzar a construir modelos predictivos eficientes en Python. Se necesitan algunos pasos para pasar de un modelo básico de kNN a un modelo completamente ajustado, ¡pero el aumento en el rendimiento lo vale completamente!

En este tutorial aprendiste cómo:

1, Comprender los fundamentos matemáticos detrás del algoritmo kNN.
2. Codificar el algoritmo kNN desde cero utilizando NumPy.
3. Utilizar la implementación de scikit-learn para ajustar un kNN con una cantidad mínima de código.
4. Utilizar GridSearchCV para encontrar los mejores hiperparámetros de kNN.
5. Mejorar al máximo el rendimiento de kNN utilizando bagging.

Una gran ventaja de las herramientas de ajuste de modelos es que muchas de ellas no solo se aplican al algoritmo kNN, sino que también se aplican a muchos otros algoritmos de aprendizaje automático.

Espero que esta entrada te sea útil. ¡Hasta la próxima!

