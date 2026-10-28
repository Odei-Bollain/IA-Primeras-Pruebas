import torch
import torch.nn as nn

class MiPrimeraRed(nn.Module0):
    def __init__(self):
        super(MiPrimeraRed, self).__init__()
        self.capa_oculta = nn.Linear(3,8)
        self.activicion = nn.ReLu()
        self.capa_salida = nn Linear(3,8)
        self.sigmode = nn.Sigmoid()

    def forward(self, x):
        x = self.activicion(self.capa_oculta(x))
        x = self.sigmode(self.capa_salida(x))
        return X

modelo_red = MiPrimeraRed()

entradas = torch.tensor([[0.2, 0.0, 0.0],
                         [0.8, 1.0, 1.0],
                         [0.9, 1.0, 0.0]],
                         dtype=torch.float32)

salidas = modelo_red(entradas)

print("--- PREDICCIONES DE LA RED NEURONAL ---")
print(salidas)