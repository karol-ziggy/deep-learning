# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.24.0",
#     "matplotlib>=3.8",
#     "numpy>=1.26",
#     "scikit-learn>=1.4",
# ]
# ///

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    from sklearn.datasets import make_moons

    return ListedColormap, make_moons, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Límites de una sola neurona

    En la parte anterior habíamos construido un perceptrón con una función de activación (sigmoide) para resolver un problema de clasificación binaria. Recordamos que la salida del perceptron se calculaba como,

    \begin{equation}
    A = \sigma (WX + b)
    \end{equation}

    donde $\sigma (z) = \frac{1}{1+e^{-z}}$ es la función sigmoide.

    Normalmente se toma como umbral de decisión cuando A > 0.5, es decir, cuando la $WX + b = 0$. Por lo tanto, **una sola neurona únicamente pude construir una frontera de decision lineal**
    """)
    return


@app.cell
def _(mo):
    # Definimos algunos widgets con marimo

    conjunto = mo.ui.dropdown(
        options={
            "Dos grupos separables": "lineal",
            "Compuerta XOR": "xor",
            "Dos lunas": "lunas",
        },
        value="Compuerta XOR",
        label="Conjunto de datos",
    )
    ruido = mo.ui.slider(
        start=0.0,
        stop=0.35,
        step=0.01,
        value=0.08,
        label="Ruido",
    )
    iteraciones = mo.ui.slider(
        start=100,
        stop=5000,
        step=100,
        value=1500,
        label="Iteraciones",
    )
    learning_rate = mo.ui.slider(
        start=0.01,
        stop=2.0,
        step=0.01,
        value=0.5,
        label="Learning rate",
    )

    mo.vstack(
        [
            mo.md("### Controles del experimento"),
            conjunto,
            ruido,
            iteraciones,
            learning_rate,
        ]
    )
    return conjunto, iteraciones, learning_rate, ruido


@app.cell
def _(conjunto, make_moons, np, ruido):
    def construir_datos(nombre, ruido, semilla=42):
        """
        Construye un problema de clasificación binaria con observaciones en columnas.
        """
        generador = np.random.default_rng(semilla)

        if nombre == "lineal":
            clase_0 = generador.normal((-1.2, -1.0), ruido + 0.25, (80, 2))
            clase_1 = generador.normal((1.2, 1.0), ruido + 0.25, (80, 2))
            muestras = np.vstack((clase_0, clase_1))
            etiquetas = np.hstack((np.zeros(80), np.ones(80)))
        elif nombre == "xor":
            centros = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
            muestras = np.repeat(centros, 40, axis=0)
            muestras = muestras + generador.normal(0, ruido, muestras.shape)
            etiquetas = np.repeat([0, 1, 1, 0], 40)
        else:
            muestras, etiquetas = make_moons(
                n_samples=200,
                noise=ruido,
                random_state=semilla,
            )

        return muestras.T, etiquetas.reshape(1, -1)

    X, y = construir_datos(conjunto.value, ruido.value)
    return X, y


@app.cell(hide_code=True)
def _(X, mo, y):
    mo.md(rf"""
    ## Forma de los datos

    Organizamos los datos utilizando la misma convención que en los notebooks anteriores

    \[
    X\in\mathbb{{R}}^{{n_x\times m}},\qquad
    Y\in\mathbb{{R}}^{{1\times m}}
    \]

    Para el conjunto seleccionado:

    - `X.shape = {X.shape}`
    - `y.shape = {y.shape}`
    - tenemos \(m={X.shape[1]}\) observaciones de entrenamiento.
    """)
    return


@app.cell
def _(np):
    def sigmoid(Z):
        """
        Aplica la función sigmoide elemento por elemento.
        """
        Z_seguro = np.clip(Z, -500, 500)
        return 1.0 / (1.0 + np.exp(-Z_seguro))

    def initialize_parameters(n_x, semilla=42):
        """
        Inicializa los pesos de una única neurona sigmoide.
        """
        generador = np.random.default_rng(semilla)

        W = generador.normal(0, 0.01, (1, n_x))
        b = np.zeros((1, 1))

        parameters = {
            "W":W,
            "b":b
        }
    
        return parameters

    def forward_propagation(X, parameters):
        """
        Calcula Z=WX+b y posteriormente A=sigmoid(Z).
        """
        W = parameters["W"]
        b = parameters["b"]

        Z = np.matmul(W,X) + b

        y_hat = sigmoid(Z)
        
        return y_hat

    def loss(y_hat, y):
        """
        Calcula la entropía cruzada binaria promedio.
        """
    
        A_segura = np.clip(y_hat, 1e-10, 1 - 1e-10)

        loss = -np.mean(y * np.log(A_segura) + (1-y) * np.log(1-A_segura))
    
        return loss

    def backward_propagation(y_hat, X, y):
        """
        Calcula los gradientes de W y b.
        """
        m = X.shape[1]
        dZ = y_hat - y

        dW = np.matmul(dZ,X.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        grads = {
            "dW": dW,
            "db":db
        }
        return grads

    def update_parameters(parameters, grads, learning_rate=0.5):
        """
        Actualiza los parámetros utilizando la regla de descenso del gradiente
        """

        # pesos w y sesgo b
        W = parameters["W"]
        b = parameters["b"]

        # gradiente
        dW = grads["dW"]
        db = grads["db"]

        # descenso del gradiente
        W = W - learning_rate*dW
        b = b - learning_rate*db

        parameters = {
            "W":W,
            "b":b
        }

        return parameters

    def nn_model(X, y, iterations=1000, learning_rate=0.5):
        """
        Integra forward, costo, backward y actualización de parámetros.
        """
        parameters = initialize_parameters(X.shape[0])
        costs = []

        # ciclo de entrenamiento
        for i in range(iterations):
            y_hat = forward_propagation(X=X, parameters=parameters)
            error = loss(y_hat=y_hat, y=y)
            grads = backward_propagation(y_hat=y_hat, X=X, y=y)
            parameters = update_parameters(parameters=parameters, grads=grads, learning_rate=learning_rate)

            if i % 10 == 0:
                costs.append((i, error))

        return parameters, costs

    return forward_propagation, nn_model


@app.cell
def _(X, forward_propagation, iteraciones, learning_rate, nn_model, np, y):
    parameters, costs = nn_model(
        X,
        y,
        iterations=iteraciones.value,
        learning_rate=learning_rate.value,
    )
    probabilities = forward_propagation(X, parameters)
    predictions = (probabilities > 0.5).astype(int)
    accuracy = float(np.mean(predictions == y))
    return accuracy, costs, parameters


@app.cell
def _(ListedColormap, X, costs, np, parameters, plt, y):
    x1_min, x1_max = X[0].min() - 0.5, X[0].max() + 0.5
    x2_min, x2_max = X[1].min() - 0.5, X[1].max() + 0.5
    eje_x1 = np.linspace(x1_min, x1_max, 250)
    eje_x2 = np.linspace(x2_min, x2_max, 250)
    malla_x1, malla_x2 = np.meshgrid(eje_x1, eje_x2)
    puntos_malla = np.vstack((malla_x1.ravel(), malla_x2.ravel()))
    Z_malla = parameters["W"] @ puntos_malla + parameters["b"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].contourf(
        malla_x1,
        malla_x2,
        (Z_malla.reshape(malla_x1.shape) > 0).astype(int),
        alpha=0.22,
        cmap=ListedColormap(["royalblue", "tomato"]),
    )
    axes[0].contour(
        malla_x1,
        malla_x2,
        Z_malla.reshape(malla_x1.shape),
        levels=[0],
        colors="black",
        linewidths=2,
    )
    axes[0].scatter(
        X[0], X[1], c=y.ravel(),
        cmap=ListedColormap(["blue", "red"]), edgecolor="white",
    )
    axes[0].set(title="Frontera aprendida", xlabel="$x_1$", ylabel="$x_2$")

    pasos, valores_cost = zip(*costs)
    axes[1].plot(pasos, valores_cost, color="purple")
    axes[1].set(title="Loop de entrenamiento", xlabel="Iteración", ylabel="Costo")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()

    fig
    return


@app.cell(hide_code=True)
def _(accuracy, conjunto, mo):
    interpretacion = (
        "En este caso una recta es suficiente para separar las clases."
        if conjunto.value == "lineal"
        else "Aunque el costo disminuya, una sola recta no puede representar la geometría completa del problema."
    )
    mo.callout(
        mo.md(
            rf"""
            **Exactitud obtenida: {accuracy:.1%}**

            {interpretacion}

            La limitación no se debe necesariamente al número de iteraciones. Se
            debe a la estructura del modelo: la frontera \(WX+b=0\) siempre es una
            recta. Esto motiva la incorporación de una **capa oculta con varias
            neuronas**.
            """
        ),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preguntas sobre el tema

    1. Selecciona los tres conjuntos y compara la exactitud.
    2. Aumenta el número de iteraciones en XOR. ¿Desaparece la limitación?
    3. Cambia el *learning rate*. ¿Qué parte del resultado corresponde a la
       optimización y qué parte corresponde a la capacidad del modelo?
    4. Explica por qué agregar más variables de entrada no resolvería por sí solo
       el problema XOR.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Respuestas

    > Los valores numéricos corresponden a los parámetros por defecto del notebook
    > (ruido = 0.08, 1500 iteraciones, learning rate = 0.5, semilla = 42).

    ### 1. Comparación de la exactitud en los tres conjuntos

    | Conjunto | Exactitud | Costo final |
    |---|---|---|
    | Dos grupos separables | **100 %** | ≈ 0.001 |
    | Compuerta XOR | **50 %** | ≈ 0.693 |
    | Dos lunas | **≈ 86.5 %** | ≈ 0.266 |

    - **Dos grupos separables:** una recta separa perfectamente las clases, por lo que la neurona alcanza el 100 %.
    - **XOR:** la exactitud es de 50 %, equivalente a adivinar al azar. Además, el costo se queda en $\ln 2 \approx 0.693$, que es el costo de predecir siempre $A=0.5$. La neurona ni siquiera logra aprender algo útil.
    - **Dos lunas:** la recta captura la tendencia global y acierta la mayoría de los puntos, pero las lunas son curvas y se entrelazan, así que siempre quedan puntos mal clasificados cerca de la frontera.

    ### 2. Más iteraciones en XOR: ¿desaparece la limitación?

    **No.** Con 100, 500, 1500 y 5000 iteraciones la exactitud se mantiene en 50 % y el costo en ≈ 0.693. El descenso del gradiente ya está en un punto donde el gradiente es prácticamente cero: por la simetría del problema, los pesos quedan cerca de $W\approx 0$ y $b\approx 0$. Entrenar más tiempo no cambia nada, porque **no existe ninguna recta** que resuelva el problema; el mínimo del costo para este modelo *es* 0.693.

    ### 3. Efecto del *learning rate*: optimización vs. capacidad del modelo

    | Conjunto | lr = 0.01 | lr = 0.1 | lr = 0.5 | lr = 2.0 |
    |---|---|---|---|---|
    | Dos grupos | 100 % (costo 0.035) | 100 % (0.005) | 100 % (0.001) | 100 % (0.0004) |
    | XOR | 50 % (0.693) | 50 % (0.693) | 50 % (0.693) | 50 % (0.693) |
    | Dos lunas | 82.5 % (0.360) | 87.5 % (0.270) | 86.5 % (0.266) | 86.5 % (0.266) |

    - **Parte de optimización (lo que controla el learning rate):** la *velocidad* con la que el costo baja y qué tan cerca se llega del mejor valor posible. Con un lr muy pequeño (0.01) el modelo no alcanza a converger en 1500 iteraciones (se ve en dos lunas: 82.5 % y costo 0.360). Con un lr mayor converge más rápido, y a partir de cierto valor ya no mejora.
    - **Parte de capacidad (lo que NO cambia con el learning rate):** el *techo* del resultado. En XOR el resultado es el mismo con cualquier lr, y en dos lunas se estabiliza en ≈ 86–87 % sin importar cuánto se aumente. Ese techo lo fija la forma del modelo: una frontera lineal.

    En resumen: el learning rate decide *qué tan rápido y qué tan bien llegamos al mejor modelo lineal*, pero no puede hacer que el modelo deje de ser lineal.

    ### 4. ¿Por qué agregar más variables de entrada no resuelve XOR por sí solo?

    La neurona calcula $z = w_1x_1 + w_2x_2 + b$, y la clase depende del signo de $z$. Para XOR necesitamos:

    - Clase 0: $(-1,-1)$ y $(1,1)$ → $-w_1-w_2+b<0$ y $w_1+w_2+b<0$. Sumando ambas: $b<0$.
    - Clase 1: $(-1,1)$ y $(1,-1)$ → $-w_1+w_2+b>0$ y $w_1-w_2+b>0$. Sumando ambas: $b>0$.

    Esto es una **contradicción** ($b<0$ y $b>0$), así que ningún valor de $W$ y $b$ resuelve el problema.

    El problema no es la *cantidad* de variables, sino que la neurona solo combina las entradas de forma **lineal**. Si agregamos variables que son combinaciones lineales de $x_1$ y $x_2$ (o variables irrelevantes), el modelo sigue siendo lineal en la información que realmente importa y la contradicción persiste. Solo ayudaría una variable **no lineal** en las entradas, como $x_1x_2$ (que vale $+1$ para la clase 0 y $-1$ para la clase 1), pero eso equivale a diseñar a mano la transformación que necesitamos. Una **capa oculta con varias neuronas y activación no lineal** aprende esa transformación automáticamente, y por eso es el siguiente paso natural.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
