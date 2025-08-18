import pandas as pd
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet

# =====================
# 1. Carregar dados
# =====================
df = pd.read_csv("vendas.csv", parse_dates=["data"])

# =====================
# 2. Previsão com Prophet
# =====================
df_forecast = df.groupby("data")["valor_total"].sum().reset_index()
df_forecast = df_forecast.rename(columns={"data": "ds", "valor_total": "y"})

modelo = Prophet(daily_seasonality=True, yearly_seasonality=False)
modelo.fit(df_forecast)

futuro = modelo.make_future_dataframe(periods=30)
forecast = modelo.predict(futuro)

faturamento_previsto = forecast.tail(30)["yhat"].sum()

# =====================
# 3. Inicializar app
# =====================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Dashboard de Vendas"

# =====================
# 4. Layout
# =====================
app.layout = dbc.Container([
    html.H1("📊 Dashboard de Vendas", className="text-center my-3"),

    # Filtros
    dbc.Row([
        dbc.Col([
            html.Label("Produto"),
            dcc.Dropdown(
                id="filtro-produto",
                options=[{"label": p, "value": p} for p in df["produto"].unique()],
                multi=True,
                placeholder="Selecione produto(s)"
            )
        ], width=4),
        dbc.Col([
            html.Label("Categoria"),
            dcc.Dropdown(
                id="filtro-categoria",
                options=[{"label": c, "value": c} for c in df["categoria"].unique()],
                multi=True,
                placeholder="Selecione categoria(s)"
            )
        ], width=4),
        dbc.Col([
            html.Label("Período"),
            dcc.DatePickerRange(
                id="filtro-data",
                start_date=df["data"].min(),
                end_date=df["data"].max(),
                display_format="DD/MM/YYYY"
            )
        ], width=4),
    ], className="mb-4"),

    # KPIs
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("💰 Faturamento Total"),
                html.H2(id="kpi-faturamento")
            ])
        ], className="shadow-lg rounded-4"), width=4),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("🛒 Ticket Médio"),
                html.H2(id="kpi-ticket")
            ])
        ], className="shadow-lg rounded-4"), width=4),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("📦 Nº de Vendas"),
                html.H2(id="kpi-vendas")
            ])
        ], className="shadow-lg rounded-4"), width=4),
    ], className="mb-4"),

    # Gráficos
    dbc.Row([
        dbc.Col(dcc.Graph(id="grafico-vendas"), width=6),
        dbc.Col(dcc.Graph(id="grafico-categorias"), width=6),
    ], className="mb-4"),

    # Previsão
    html.H2("🔮 Previsão de Vendas (30 dias)", className="mt-4"),
    html.H4(f"Faturamento Previsto: R$ {faturamento_previsto:,.2f}",
            style={"margin": "20px", "fontSize": "22px"}),

    dcc.Graph(
        figure=go.Figure([
            go.Scatter(x=df_forecast["ds"], y=df_forecast["y"], mode="lines", name="Histórico"),
            go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Previsão", line=dict(dash="dash"))
        ]).update_layout(title="Histórico vs Previsão de Vendas")
    )
], fluid=True)

# =====================
# 5. Callbacks
# =====================
@app.callback(
    [Output("kpi-faturamento", "children"),
     Output("kpi-ticket", "children"),
     Output("kpi-vendas", "children"),
     Output("grafico-vendas", "figure"),
     Output("grafico-categorias", "figure")],
    [Input("filtro-produto", "value"),
     Input("filtro-categoria", "value"),
     Input("filtro-data", "start_date"),
     Input("filtro-data", "end_date")]
)
def atualizar_dashboard(produto, categoria, start_date, end_date):
    df_filtrado = df.copy()

    if produto:
        df_filtrado = df_filtrado[df_filtrado["produto"].isin(produto)]
    if categoria:
        df_filtrado = df_filtrado[df_filtrado["categoria"].isin(categoria)]
    if start_date and end_date:
        df_filtrado = df_filtrado[(df_filtrado["data"] >= start_date) & (df_filtrado["data"] <= end_date)]

    # KPIs
    faturamento = df_filtrado["valor_total"].sum()
    ticket = df_filtrado["valor_total"].mean()
    vendas = len(df_filtrado)

    # Gráfico 1: Vendas ao longo do tempo
    fig_vendas = px.line(df_filtrado.groupby("data")["valor_total"].sum().reset_index(),
                         x="data", y="valor_total", title="Faturamento ao longo do tempo")

    # Gráfico 2: Faturamento por Categoria
    fig_categorias = px.bar(df_filtrado.groupby("categoria")["valor_total"].sum().reset_index(),
                            x="categoria", y="valor_total", title="Faturamento por Categoria")

    return (f"R$ {faturamento:,.2f}",
            f"R$ {ticket:,.2f}" if not pd.isna(ticket) else "R$ 0,00",
            f"{vendas:,}",
            fig_vendas, fig_categorias)

# =====================
# 6. Rodar
# =====================
if __name__ == "__main__":
    app.run(debug=True)
