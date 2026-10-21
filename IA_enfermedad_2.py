import pandas as pandas
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

datos = {
    'Temperatura': [36.5, 38.5, 39.0, 37.0, 38.8, 36.2, 39.5, 37.2, 38.1, 36.8, 39.2, 36.1],
    'Tos':         [0,    1,    1,    0,    1,    0,    1,    1,    1,    0,    1,    0],
    'Fatiga':      [0,    1,    1,    1,    0,    0,    1,    0,    1,    0,    1,    0],
    'Tiene_Gripe': [0,    1,    1,    0,    1,    0,    1,    0,    1,    0,    1,    0]
}

df = pd.DataFrame(datos)

X = df[['Temperatura', 'Tos', 'Fatiga']]
y = df['Tiene_Gripe']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

modelo = DecisionTreeClassifier()
modelo.fit(X_train, y_train)

predicciones_examen = modelo.predict(X_test)

precision = accuracy_score(y_test, predicciones_examen)

print(f"--- RESULTADO DEL EXAMEN ---")
print(f"Precisión del modelo: {precision * 100:.1f}%")