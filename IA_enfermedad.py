import pandas as pd
from sklearn.tree import DecisionTreeClassifier

datos = {
    'Temperaturas': [36.5, 38.5, 39.0, 37.0, 38.8, 36.2, 39.5, 37.2],
    'Tos': [0,    1,    1,    0,    1,    0,    1,    1],
    'Fatiga': [0,    1,    1,    1,    0,    0,    1,    0],
    'Tiene_Gripe': [0,    1,    1,    0,    1,    0,    1,    0]
}

df = pd.DataFrame(datos)

X = df[['Temperaturas', 'Tos', 'Fatiga']]

y = df['Tiene_Gripe']

modelo = DecisionTreeClassifier()
modelo.fit(X, y)

temperatura = float(input('¿Cuánta fiebre tienes? (ej. 38.5): '))
tos = int(input('¿Tienes tos? (1=Sí, 0=No): '))
fatiga = int(input('¿Te sientes cansado? (1=Sí, 0=No): '))

paciente_nuevo = [[temperatura, tos, fatiga]]

prediccion = modelo.predict(paciente_nuevo)

print("--- RESULTADO DEL DIAGNÓSTICO ---")
if prediccion[0] == 1:
    print("La IA detecta patrones compatibles con GRIPE")
else:
    print("La IA indica que el paciente está SANO / RESFRIADO COMÚN")