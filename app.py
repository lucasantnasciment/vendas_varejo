import pandas as pd
import joblib
from datetime import timedelta
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
import plotly.express as px

# ====== Carregar modelo e dados ======
modelo = joblib.load("models/modelo_vendas.pkl")
df = pd.read_csv("data/vendas.csv", parse_dates=["data"])

# Features derivadas
df["mes"] = df["data"].dt.month
df["ano"] = df["data"].dt.year
df["dia_semana"] = df["data"].dt.weekday

# ====== Criar previsões futuras ======
def prever_futuro(df, dias=30):
    ultima_data = df["data"].max()
    datas_futuras = [ultima_data + timedelta(days=i) for i in range(1, dias+1)]
    
    df_future = pd.DataFrame({"data": datas_futuras})
    df_future["mes"] = df_future["data"].dt.month
    df_future["ano"] = df_future["data"].dt.year
    df_future["dia_semana"] = df_future["data"].dt.weekday
    # Para exemplo, vamos usar a média de quantidade vendida
    df_future["quantidade"] = df["quantidade"].mean()
    
    previsoes = modelo.predict(df_future[["mes", "ano", "dia_semana", "quantidade"]])
    df_future["valor_previsto"] = previsoes
    return df_future

df_future = prever_futuro(df)

# ====== Iniciar app Dash ======
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Dashboard de Vendas"

# ====== Layout ======
app.layout = dbc.Container([
    html.H1("Dashboard de Vendas", className="text-center my-4"),
    
    dbc.Row([
        dbc.Col([
            html.H5("Vendas Históricas"),
            dcc.Graph(
                figure=px.line(df.groupby("data")["valor_vendas"].sum().reset_index(),
                               x="data", y="valor_vendas",
                               title="Vendas ao Longo do Tempo")
            )
        ], md=6),

        dbc.Col([
            html.H5("Top 10 Produtos"),
            dcc.Graph(
                figure=px.bar(df.groupby("produto")["quantidade"].sum()
                              .sort_values(ascending=False).head(10).reset_index(),
                              x="produto", y="quantidade", title="Mais Vendidos")
            )
        ], md=6)
    ]),

    dbc.Row([
        dbc.Col([
            html.H5("Previsão de Vendas para os Próximos 30 Dias"),
            dcc.Graph(
                figure=px.line(df_future, x="data", y="valor_previsto", markers=True,
                               title="Previsão de Vendas")
            )
        ])
    ])
], fluid=True)

# ====== Rodar ======
if __name__ == "__main__":
    app.run(debug=True)
