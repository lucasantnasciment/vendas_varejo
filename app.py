from prophet import Prophet
import pandas as pd

# Carregar dados
df = pd.read_csv("vendas.csv", parse_dates=["data"])

# Agrupar por data
df_diario = df.groupby("data")["valor_total"].sum().reset_index()
df_diario = df_diario.rename(columns={"data": "ds", "valor_total": "y"})

# Criar modelo Prophet
modelo = Prophet(daily_seasonality=True, yearly_seasonality=False)
modelo.fit(df_diario)

# Prever 30 dias à frente
futuro = modelo.make_future_dataframe(periods=30)
forecast = modelo.predict(futuro)

# Resultado
forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail()
