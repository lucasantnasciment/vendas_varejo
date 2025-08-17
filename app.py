import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd

# Carregar dataset
df = pd.read_csv("C:\\Users\\Lara\\OneDrive\\Área de Trabalho\\12-python-projects\\vendas_varejo\\vendas_varejo\\data\\vendas.csv", parse_dates=["data"])

# Inicializar app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Dashboard de Vendas"

# Layout
app.layout = dbc.Container([
    html.H1("📊 Dashboard de Vendas", className="text-center my-4"),

    # Filtros
    dbc.Row([
        dbc.Col([
            html.Label("Produto:"),
            dcc.Dropdown(
                id="filtro_produto",
                options=[{"label": p, "value": p} for p in sorted(df["produto"].unique())],
                multi=True,
                placeholder="Selecione produto(s)"
            )
        ], md=4),

        dbc.Col([
            html.Label("Categoria:"),
            dcc.Dropdown(
                id="filtro_categoria",
                options=[{"label": c, "value": c} for c in sorted(df["categoria"].unique())],
                multi=True,
                placeholder="Selecione categoria(s)"
            )
        ], md=4),

        dbc.Col([
            html.Label("Período:"),
            dcc.DatePickerRange(
                id="filtro_data",
                start_date=df["data"].min(),
                end_date=df["data"].max(),
                display_format="DD/MM/YYYY"
            )
        ], md=4),
    ], className="mb-4"),

    # KPIs
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("💰 Faturamento Total"),
                html.H2(id="kpi_faturamento", className="text-success")
            ])
        ], className="shadow-sm"), md=4),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("🛒 Ticket Médio"),
                html.H2(id="kpi_ticket", className="text-primary")
            ])
        ], className="shadow-sm"), md=4),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("📦 Quantidade Vendida"),
                html.H2(id="kpi_quantidade", className="text-warning")
            ])
        ], className="shadow-sm"), md=4),
    ], className="mb-4"),

    # Gráficos
    dbc.Row([
        dbc.Col(dcc.Graph(id="grafico_produto"), md=6),
        dbc.Col(dcc.Graph(id="grafico_tempo"), md=6),
    ])
], fluid=True)


# Callbacks
@app.callback(
    [Output("kpi_faturamento", "children"),
     Output("kpi_ticket", "children"),
     Output("kpi_quantidade", "children"),
     Output("grafico_produto", "figure"),
     Output("grafico_tempo", "figure")],
    [Input("filtro_produto", "value"),
     Input("filtro_categoria", "value"),
     Input("filtro_data", "start_date"),
     Input("filtro_data", "end_date")]
)
def atualizar_dashboard(produtos, categorias, data_ini, data_fim):
    df_filtrado = df.copy()

    if produtos:
        df_filtrado = df_filtrado[df_filtrado["produto"].isin(produtos)]
    if categorias:
        df_filtrado = df_filtrado[df_filtrado["categoria"].isin(categorias)]
    if data_ini and data_fim:
        df_filtrado = df_filtrado[(df_filtrado["data"] >= data_ini) & (df_filtrado["data"] <= data_fim)]

    # KPIs
    faturamento = df_filtrado["valor_total"].sum()
    quantidade = df_filtrado["quantidade"].sum()
    ticket = faturamento / quantidade if quantidade > 0 else 0

    # Gráfico por produto
    fig_produto = px.bar(df_filtrado.groupby("produto")["valor_total"].sum().reset_index(),
                         x="produto", y="valor_total", title="Faturamento por Produto")

    # Gráfico no tempo
    fig_tempo = px.line(df_filtrado.groupby("data")["valor_total"].sum().reset_index(),
                        x="data", y="valor_total", title="Evolução das Vendas")

    return (f"R$ {faturamento:,.2f}",
            f"R$ {ticket:,.2f}",
            f"{quantidade:,}",
            fig_produto,
            fig_tempo)


if __name__ == "__main__":
    app.run(debug=True)
