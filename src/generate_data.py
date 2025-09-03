# src/generate_data.py
import numpy as np
import pandas as pd

np.random.seed(42)
n = 1000

temperatura = np.random.normal(25, 5, n)
hora_dia = np.random.randint(0, 24, n)
dia_semana = np.random.randint(0, 7, n)

demanda = (50 
        + 2 * temperatura 
        + 5 * ((hora_dia >= 9) & (hora_dia <= 18)).astype(int) 
        + 3 * (dia_semana < 5).astype(int)
        + np.random.normal(0, 5, n))

df = pd.DataFrame({
    "temperatura": temperatura,
    "hora_dia": hora_dia,
    "dia_semana": dia_semana,
    "demanda": demanda
})

df.to_csv("data/consumo_energia.csv", index=False)
print("✅ Arquivo salvo em data/consumo_energia.csv")
