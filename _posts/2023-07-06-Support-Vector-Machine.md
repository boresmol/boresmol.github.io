---
layout: post
title: Support Vector Machine - SVM
---

# Introducción

En esta primera entrada del blog sobre ciencia de datos y machien learning, vamos a tratar de explicar el algoritmo SVM (Support Vector Machines) orientado a tareas de clasificación binarias. Se tratará de aportar una explicación sencilla a la par de rigurosa, comenzando por dar una idea sencilla y aproximada de que es y que hace este algoritmo, para seguir con la explicación del SVM lineal, implementando este algoritmo 'from scratch' y finalizaremos el post explicando como lidiar con conjuntos de datos no lineales. ¡Espero que os guste!

# Indice



1.   Idea intuitiva.
2.   El SVM lineal e impkementación 'from scratch'.
3.   SVM para datos no lineales. El kernel trick.
4.   Comparación de los distintos algoritmos.
5.   Conclusiones.




## Idea intuitiva

Imaginemos que estamos en un zoológico y tenemos dos tipos de animales: leones y jirafas. Queremos encontrar una forma de separarlos en diferentes grupos utilizando el SVM lineal.

El SVM lineal nos ayuda a trazar una línea imaginaria en el suelo del zoológico para separar los leones de las jirafas. ¿Cómo lo hace? Imagina que tienes una cuerda larga . Tu objetivo es colocarla en el suelo de tal manera que los leones estén en un lado de la cuerda y las jirafas estén en el otro lado.

Para hacer esto, primero necesitamos observar algunas características de los animales. Por ejemplo, podríamos medir la altura de los animales y la longitud de sus patas. Estas características nos ayudarán a distinguir entre leones y jirafas.

Ahora, supongamos que hemos recolectado los datos y hemos creado un gráfico con la altura en el eje vertical y la longitud de las patas en el eje horizontal. En el gráfico, cada león y cada jirafa se representa como un punto.

El SVM lineal tratará de encontrar la mejor línea recta que pueda separar los puntos de los leones y las jirafas. Querrá dibujar esa línea de tal manera que los leones queden en un lado y las jirafas queden en el otro.

Pero hay algo especial sobre esta línea. El SVM lineal también se asegurará de que la línea esté lo más alejada posible de los animales más cercanos a ella. Esto significa que habrá una "zona segura" a cada lado de la línea, donde no habrá animales.

Entonces, en nuestro ejemplo del zoológico, el SVM lineal buscará la mejor línea recta que separe a los leones y las jirafas utilizando las características de altura y longitud de las patas. Dibujará una línea en el suelo y se asegurará de que los leones estén en un lado y las jirafas estén en el otro. También se asegurará de que la línea esté lo más alejada posible de los animales más cercanos.



# SVM Lineal y el Margen Máximo

En el campo del aprendizaje automático, las Máquinas de Vectores de Soporte (SVM, por sus siglas en inglés) son un poderoso algoritmo de clasificación. En particular, el SVM lineal se utiliza para separar dos clases diferentes de objetos en un espacio dimensional mediante un hiperplano lineal. A continuación exploraremos en detalle el concepto del SVM lineal y su enfoque de maximización del margen.

## La Función del Hiperplano

En el caso del SVM lineal, el objetivo es encontrar un hiperplano que logre separar óptimamente las clases de objetos. Matemáticamente, el hiperplano se define como:

`f(x) = w· x + b`


donde:

- `f(x)` es la función que clasifica los objetos.
- `w` es el vector de pesos.
- `x` es el vector de características del objeto.
- `b` es el sesgo (también conocido como término de sesgo o término de intercepción).

El hiperplano divide el espacio dimensional en dos regiones, una para cada clase. Los objetos se asignan a una clase en función de qué lado del hiperplano se encuentren.

## Maximizando el Margen

En el contexto del SVM lineal, maximizar el margen es una parte fundamental para lograr una buena separación entre las clases. El margen se define como la distancia más corta entre el hiperplano de separación y los vectores de soporte, que son los objetos más cercanos al hiperplano y pertenecen a diferentes clases.

La fórmula para calcular el margen máximo es `Margen máximo = 2 / ||w||`, donde `||w||` representa la norma euclidiana del vector de pesos `w`. La norma euclidiana se calcula como la raíz cuadrada de la suma de los cuadrados de los elementos del vector. La magnitud de `||w||` está inversamente relacionada con el margen: cuanto mayor sea el valor de `||w||`, más estrecho será el margen, y viceversa.

Para maximizar el margen, es necesario encontrar los vectores de soporte más cercanos al hiperplano. Estos vectores de soporte influyen en la posición y orientación del hiperplano de separación. Al ajustar el hiperplano de manera que la distancia a los vectores de soporte sea máxima, se logra un margen óptimo.

Los vectores de soporte desempeñan un papel crucial en el SVM lineal, ya que definen el límite de decisión entre las clases. Son los objetos que se encuentran más próximos al hiperplano y, por lo tanto, son los más difíciles de clasificar correctamente. Al maximizar el margen, se busca minimizar el riesgo de clasificar incorrectamente los vectores de soporte.

El concepto de maximización del margen en el SVM lineal es importante porque proporciona una buena generalización y capacidad de clasificación para nuevos datos. Al aumentar el margen, se reduce la probabilidad de sobreajuste y se mejora la capacidad del modelo para separar las clases de manera más robusta.

En resumen, maximizar el margen en el SVM lineal implica encontrar los vectores de soporte más cercanos al hiperplano y ajustar el hiperplano para que su distancia a estos vectores sea máxima. Esto se logra calculando la norma euclidiana del vector de pesos y utilizando la fórmula `Margen máximo = 2 / ||w||`. Al maximizar el margen, se mejora la capacidad del modelo para clasificar correctamente los nuevos datos y se reduce el riesgo de sobreajuste.

![Imagen de un ejemplo gráfico de SVM lineal](https://3.bp.blogspot.com/-Zt5ab9bdQ64/UxsBSEH-3_I/AAAAAAAADRE/caQodexaP2c/s1600/36.png)

## Función de Pérdida y Optimización

En el SVM lineal, se utiliza la función de pérdida conocida como Bisagra (Hinge) para medir el error de clasificación. La función de pérdida se define como:

`L(y, f(x)) = max(0, 1 - y · f(x))`


donde `y` es la etiqueta de clase verdadera `(-1 o 1)` y `f(x)` es la salida del hiperplano.

El objetivo de entrenar el SVM lineal es minimizar la función de pérdida, al tiempo que se busca maximizar el margen. Esto se logra a través de la optimización de un problema de programación cuadrática.


## Implementación from scratch:

A continuación se realiza una implementación del algoritmo 'from scratch'. Esto quiere decir que no se usa ninguna librería donde ya esté implementado el algoritmo como por ejemplo scikit-learn, si no que se implementa de 0 usando numpy.

El proceso de maximización del margen en SVM lineal implica encontrar los parámetros óptimos del hiperplano de separación. A continuación se presenta un resumen del algoritmo en pseudocódigo para maximizar el margen:

1. Comenzar con un valor inicial grande para `w`, por ejemplo, `(w0, w0)`. Este valor se disminuirá más adelante.
2. Seleccionar un tamaño de paso como `w0 * 0.1`.
3. Establecer un valor pequeño para `b`, que se aumentará más adelante. El rango de `b` será `(-b0 < b < +b0)`, con un paso de incremento de `b_multiple`. Es importante seleccionar sabiamente el valor de `b0` para evitar un costo computacional excesivo.
4. Se realizará el siguiente chequeo para cada punto `xi` en el conjunto de datos:
    - Comprobar para todas las transformaciones de `w`, como `(w0, w0), (-w0, w0), (w0, -w0), (-w0, -w0)`.
    - Si `yi(xi.w + b)` no es mayor o igual a 1 para todos los puntos, detener el proceso.
    - De lo contrario, calcular `|w|` y agregarlo a un diccionario como clave, y `(w, b)` como valor.
    - Si `w <= 0`, se ha completado el paso actual y se pasa al siguiente paso.
    - De lo contrario, disminuir `w` como `(w0 - step, w0 - step)` y continuar con el paso 3.
5. Repetir este proceso hasta que el paso se convierta en `w0 * 0.001`, ya que más adelante será un punto costoso:
    - Reducir el tamaño del paso como `step = step * 0.1`.
    - Volver al paso 3.
6. Seleccionar `(w, b)` que tenga la menor `|w|` del diccionario como resultado final del proceso de maximización del margen.

Este algoritmo permite encontrar los parámetros de `w` y `b` que maximizan el margen en SVM lineal. Al maximizar el margen, se mejora la capacidad de clasificación del modelo y se reduce el riesgo de sobreajuste. Es importante ajustar adecuadamente los valores iniciales y los tamaños de paso para obtener resultados óptimos.

```python3

%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np

```

```python3
class SVM(object):
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1: 'r', -1: 'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    def fit(self, data):
        self.data = data
        opt_dict = {}
        transforms = [[1, 1], [-1, 1], [-1, -1], [1, -1]]
        all_data = np.array([])

        for yi in self.data:
            all_data = np.append(all_data, self.data[yi])

        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None

        step_sizes = [self.max_feature_value * 0.1, self.max_feature_value * 0.01, self.max_feature_value * 0.001]
        b_range_multiple = 5
        b_multiple = 5
        latest_optimum = self.max_feature_value * 10

        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            optimized = False

            while not optimized:
                for b in np.arange(-1 * self.max_feature_value * b_range_multiple,
                                   self.max_feature_value * b_range_multiple,
                                   step * b_multiple):
                    for transformation in transforms:
                        w_t = w * transformation
                        found_option = True

                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi * (np.dot(w_t, xi) + b) >= 1:
                                    found_option = False

                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]

                if w[0] < 0:
                    optimized = True
                else:
                    w = w - step

            norms = sorted([n for n in opt_dict])
            opt_choice = opt_dict[norms[0]]

            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0] + step * 2

    def visualize(self):
        [[self.ax.scatter(x[0], x[1], s=100, c=self.colors[i]) for x in data_dict[i]] for i in data_dict]

        def hyperplane(x, w, b, v):
            return (-w[0] * x - b + v) / w[1]

        hyp_x_min = self.min_feature_value * 0.9
        hyp_x_max = self.max_feature_value * 1.1

        pav1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        pav2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max], [pav1, pav2], 'k')

        nav1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nav2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nav1, nav2], 'k')

        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2], 'y--')
```

```python3
data_dict = {-1:np.array([[1,7],[2,8],[3,8]]),1:np.array([[5,1],[6,-1],[7,3]])}
svm = SVM() 
svm.fit(data=data_dict)
svm.visualize()
```

![Resultado script](https://github.com/boresmol/boresmol.github.io/blob/master/images/descargar.png?raw=true)

Como vemos, este es un ejemplo muy simple. 3 puntos en cada clase perfectamente separables por una recta. Tristemente, los datos reales suelen ser de naturaleza no lineal, por lo que el algoritmo descrito e implementado anteriormente no nos sirve para separar clases. Para solucionar este problema, tenemos que hacer uso de lo que se conoce como 'kernel trick' que explicaremos a continuación.

# SVM para datos no lineales. El kernel trick.

En casos donde los datos no pueden ser separados perfectamente por un hiperplano lineal, se utiliza una técnica llamada "kernel trick". Esta técnica permite mapear los datos a un espacio de características de mayor dimensión donde los objetos se pueden separar de manera lineal. Se introduce una función de kernel que realiza la transformación de los datos. El kernel más utilizado es el kernel radial, pero, ¿qué es esto del kernel?

## El kernel en SVM

## Funciones de Kernel en SVM

Las funciones de kernel son una herramienta muy poderosa para explorar espacios de alta dimensión. Nos permiten realizar discriminantes lineales en variedades no lineales, lo que puede llevar a una mayor precisión y robustez que los modelos lineales tradicionales por sí solos.

La función de kernel es simplemente una función matemática que convierte un espacio de entrada de baja dimensión en un espacio de dimensión superior. Esto se logra mediante la asignación de los datos a un nuevo espacio de características. En este espacio, los datos serán linealmente separables. Esto significa que se puede utilizar una máquina de vectores de soporte para encontrar un hiperplano que separe los datos.

Por ejemplo, si la entrada 𝑥 es bidimensional, la función de kernel la mapeará a un espacio tridimensional. En este espacio, los datos serán linealmente separables.

![Mapeo kernel](https://github.com/boresmol/boresmol.github.io/blob/master/images/descargar%20(1).png?raw=true)

Además, las funciones de kernel proporcionan más características que otros algoritmos como redes neuronales o conjuntos de árboles en algunos tipos de problemas que involucran reconocimiento de escritura a mano, detección de caras, etc., porque extraen propiedades intrínsecas de los puntos de datos a través de una función de kernel.

## El kernel radial

RBF, que significa Función de Base Radial (por sus siglas en inglés, Radial Basis Function), es un kernel muy poderoso utilizado en SVM. A diferencia de los kernels lineales o polinómicos, RBF es más complejo y eficiente al mismo tiempo, ya que puede combinar múltiples kernels polinómicos de diferentes grados varias veces para proyectar los datos no linealmente separables en un espacio de dimensionalidad superior, de modo que puedan ser separables utilizando un hiperplano.

![no lineal](https://github.com/boresmol/boresmol.github.io/blob/master/images/descargar%20(2).png?raw=true)

El kernel RBF funciona mapeando los datos a un espacio de alta dimensión mediante la búsqueda de los productos escalares y los cuadrados de todas las características en el conjunto de datos, y luego realizando la clasificación utilizando la idea básica del SVM lineal. Para proyectar los datos en un espacio de dimensionalidad superior, el kernel RBF utiliza la llamada función de base radial, que puede escribirse como:

![formula1](https://github.com/boresmol/boresmol.github.io/blob/master/images/formula1.png?raw=true)

Aquí, `||X1 - X2||^2` se conoce como la Distancia Euclidiana al Cuadrado y `σ` es un parámetro libre que se puede utilizar para ajustar la ecuación.

Cuando se introduce un nuevo parámetro `ℽ = 1 / (2σ^2)`, la ecuación será:

![formula2](https://github.com/boresmol/boresmol.github.io/blob/master/images/formula2.png?raw=true)

La ecuación es realmente simple : la Distancia Euclidiana al Cuadrado se multiplica por el parámetro gamma y luego se encuentra el exponente de todo el resultado. Esta ecuación puede encontrar los productos internos transformados para mapear los datos a dimensiones superiores directamente sin necesidad de transformar todo el conjunto de datos, lo que conduce a una ineficiencia. Y es por esto que se conoce como la función de kernel RBF (Radial Basis Function).

La gráfica de distribución del RBF Kernel se verá así:

![distribucion](https://github.com/boresmol/boresmol.github.io/blob/master/images/distribucion.png?raw=true)

Como puedes ver, la gráfica de distribución del RBF kernel se asemeja a la curva de la distribución gaussiana, que es conocida como una curva en forma de campana. Por lo tanto, el kernel RBF también se conoce como el kernel de base radial gaussiana.

El kernel RBF se utiliza de manera muy popular con algoritmos como K-Nearest Neighbors (K-Vecinos más Cercanos) y Support Vector Machines (Máquinas de Vectores de Soporte).

Por último, esta visualización final resume muy bien todo lo explicado en esta sección del post:

![kernel](https://github.com/boresmol/boresmol.github.io/blob/master/images/kernel.png?raw=true)

# Comparación de algoritmos

A continuación vamos a hacer una pequeña experimentación para acabar de entender la importancia del kernel trick en datos no lineales:

1. Generaremos un conjunto de datos artificial no lineal de dos clases
2. Crearemos un SVM Lineal para tratar de separar las clases
3. Usaremos el kernel trick para tratar de separar las clases
4. Compararemos el rendimiento

```python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_circles

X, y = make_circles(n_samples=500, noise=0.06, random_state=42)

df = pd.DataFrame(dict(x1=X[:, 0], x2=X[:, 1], y=y))

```

```python3

colors = {0:'blue', 1:'yellow'}
fig, ax = plt.subplots()
grouped = df.groupby('y')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x1', y='x2', label=key, color = colors[key])
plt.show()
```

![Resultado codigo](https://github.com/boresmol/boresmol.github.io/blob/master/images/resultado_codigo.png?raw=true)

```python3

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

clf = SVC(kernel="linear")

clf.fit(X, y)

pred = clf.predict(X)

print("El accuracy del modelo lineal es: ",accuracy_score(pred, y))
```
**El accuracy del modelo lineal es: 0.496**

```python3

# Definimos el kernel radial con la fórmula antes explicada

def RBF(X, gamma):

    # Free parameter gamma
    if gamma == None:
        gamma = 1.0/X.shape[1]

    # RBF kernel Equation
    K = np.exp(-gamma * np.sum((X - X[:,np.newaxis])**2, axis = -1))

    return K

```

```python3
# Aplicamos el kernel al dataset

X = RBF(X, gamma=None)
```

```python3
# Alicamos el kernel lineal a un dataset al que se le ha aplicado el kernel trick

clf = SVC(kernel="linear")

clf.fit(X, y)

pred = clf.predict(X)

print("El accuracy del SVM con kernel radial es: ",accuracy_score(pred, y))
```
**El accuracy del SVM con kernel radial es:  0.94**


Como podemos ver, el accuracy del SVM con kernel radial es significativamente mayor al SVM lineal simple, lo que demuestra la suma importancia del kernel a la hora de enfrentarnos a datos no lineales.

# Conclusiones
SVM es un algoritmo de machine learning altamente efectivo y versátil. De hecho, aunque actualmente no es el estado del arte, ha sido muy utilizado durante décadas en el campo de la visión por computador, más concretamente en el reconocimiento de dígitos manuscritos y de rostros.

Las ventajas de las máquinas de vectores de soporte son las siguientes:

- Efectivas en espacios de alta dimensionalidad.
- Siguen siendo efectivas en casos donde el número de dimensiones es mayor que el número de muestras.
- Utilizan un subconjunto de puntos de entrenamiento en la función de decisión (llamados vectores de soporte), por lo que también son eficientes en memoria.
- Versátiles: se pueden especificar diferentes funciones de kernel para la función de decisión. Se proporcionan kernels comunes, pero también es posible especificar kernels personalizados.

Las desventajas de las máquinas de vectores de soporte incluyen:

- Si el número de características es mucho mayor que el número de muestras, es crucial evitar el sobreajuste al elegir funciones de kernel y términos de regularización.
- Las SVM no proporcionan directamente estimaciones de probabilidad. Estas se calculan utilizando una costosa validación cruzada de cinco pliegues (ver Puntuaciones y probabilidades).
- Las Máquinas de Vectores de Soporte son herramientas poderosas, pero sus requisitos computacionales y de almacenamiento aumentan rápidamente con el número de vectores de entrenamiento. El núcleo de una SVM es un problema de programación cuadrática.


Mencionar que existen versiones de [SVM para regresión](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) como para [clasificación multiclase](https://scikit-learn.org/stable/modules/svm.html#classification)



