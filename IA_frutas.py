# 1. Importamos el algoritmo (Árbol de Decisión)
from sklearn import tree

# 2. Datos de entrenamiento (Entrada: [Peso, Textura])
# Textura: 0 = Suave, 1 = Rugosa
caracteristicas = [
    [140, 0],  # Manzana
    [130, 0],  # Manzana
    [150, 1],  # Naranja
    [170, 1]   # Naranja
]

# Etiquetas (Salida esperada: 0 = Manzana, 1 = Naranja)
etiquetas = [0, 0, 1, 1]

# 3. Creamos e instanciamos la IA (el clasificador)
clasificador = tree.DecisionTreeClassifier()

# 4. Entrenamos la IA con nuestros datos
clasificador.fit(caracteristicas, etiquetas)

# 5. Le pedimos una predicción para una fruta nueva de 160g y textura rugosa (1)
resultado = clasificador.predict([[160, 1]])

if resultado[0] == 0:
    print("La IA dice que es una Manzana 🍎")
else:
    print("La IA dice que es una Naranja 🍊")