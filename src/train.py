import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# =========================
# 1. Carregar dados
# =========================
df = pd.read_csv("data/consumo_energia.csv")

X = df[["temperatura", "hora_dia", "dia_semana"]]
y = df["demanda"]

# =========================
# 2. Divisão treino/teste
# =========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =========================
# 3. Modelos
# =========================
model_lr = LinearRegression()
model_rf = RandomForestRegressor(random_state=42)

model_lr.fit(X_train, y_train)
model_rf.fit(X_train, y_train)

y_pred_lr = model_lr.predict(X_test)
y_pred_rf = model_rf.predict(X_test)

# =========================
# 4. Função de métricas
# =========================
def regression_metrics(y_true, y_pred, X):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    p = X.shape[1]
    adj_r2 = 1 - (1-r2)*(n-1)/(n-p-1)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R²": r2, "Adj R²": adj_r2}

metrics_lr = regression_metrics(y_test, y_pred_lr, X_test)
metrics_rf = regression_metrics(y_test, y_pred_rf, X_test)

print("\n📊 Métricas - Regressão Linear:")
for k, v in metrics_lr.items():
    print(f"{k}: {v:.4f}")

print("\n📊 Métricas - Random Forest:")
for k, v in metrics_rf.items():
    print(f"{k}: {v:.4f}")

# =========================
# 5. Visualização
# =========================
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred_lr, alpha=0.6, label="LinearRegression", color="blue")
plt.scatter(y_test, y_pred_rf, alpha=0.6, label="RandomForest", color="orange")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2)
plt.xlabel("Valores Reais (Demanda)")
plt.ylabel("Predições")
plt.legend()
plt.title("Comparação de Modelos - Predição de Demanda Energética")
plt.show()
