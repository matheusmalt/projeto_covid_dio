# Imports
from pmdarima.arima import auto_arima
import numpy as np
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import re
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# Funções:
def corrigir_colunas(col_name):
    return re.sub(r"[/| ]", "", col_name).lower()

def crescimento(data, variable, data_inicio=None, data_fim=None):
    if data_inicio == None:
        data_inicio = data.observationdate.loc[data[variable] > 0].min()
    else:
        data_inicio = pd.to_datetime(data_inicio)
    
    if data_fim == None:
        data_fim = data.observationdate.iloc[-1]
    else:
        data_fim = pd.to_datetime(data_fim)
    
    passado = data.loc[data.observationdate == data_inicio, variable].values[0]
    presente = data.loc[data.observationdate == data_fim, variable].values[0]
    numero_pontos = (data_fim - data_inicio).days

    taxa = ((presente/passado)**(1/numero_pontos) - 1)

    return taxa

def crescimento_diario(data, variable, data_inicio=None):
    if data_inicio == None:
        data_inicio = data.observationdate.loc[data[variable] > 0].min()
    else:
        data_inicio = pd.to_datetime(data_inicio)

    data_fim = data.observationdate.max()

    n = (data_fim - data_inicio).days

    taxas = list(map(
        lambda x: (data[variable].iloc[x] - data[variable].iloc[x - 1]) / data[variable].iloc[x - 1],
      range(1, n + 1)
    ))

    return np.array(taxas) * 100

# Corpo:
fig = go.Figure()
df = pd.read_csv("covid_19_data.csv", parse_dates = ["ObservationDate", "Last Update"])

df.columns = [corrigir_colunas(col) for col in df.columns]

brasil = df.loc[(df.countryregion == "Brazil") & (df.confirmed > 0)]
confirmados_covid = px.line(brasil, "observationdate", "confirmed", title="Casos confirmados de COVID Brasil")

brasil["novoscasos"] = list(map(
    lambda x: 0 if (x == 0) else brasil["confirmed"].iloc[x] - brasil["confirmed"].iloc[x - 1],
    np.arange(brasil.shape[0])
))

casos_dia = px.line(brasil, "observationdate", "novoscasos", title="Novos casos por dia de COVID Brasil")

fig.add_trace(
    go.Scatter(x=brasil.observationdate, y=brasil.deaths, name="Mortes",
    mode="lines+markers", line={"color": "red"})
)

fig.update_layout(title="Mortes por COVID no Brasil")

taxa_diario = crescimento_diario(brasil, "confirmed")

primeiro_dia = brasil.observationdate.loc[brasil.confirmed > 0].min()

crescimento_confirmado = px.line(x=pd.date_range(primeiro_dia, brasil.observationdate.max())[1:],
        y=taxa_diario, title="Taxa de crescimento de casos confirmados no Brasil")


# Mostrar na tela Plotly
crescimento_confirmado.show()
fig.show()
casos_dia.show()
confirmados_covid.show()

# Matplot
confirmados = brasil.confirmed
confirmados.index = brasil.observationdate

res = seasonal_decompose(confirmados)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))

ax1.plot(res.observed)
ax2.plot(res.trend)
ax3.plot(res.seasonal)
ax4.plot(confirmados.index, res.resid)
ax4.axhline(0, linestyle="dashed", c="black")
plt.show()

# Arima

modelo = auto_arima(confirmados)
fig = go.Figure(go.Scatter(
    x=confirmados.index, y=confirmados, name="Observados"
))

fig.add_trace(go.Scatter(
    x=confirmados.index, y=modelo.predict_in_sample(), name="Preditos"
))

fig.add_trace(go.Scatter(
    x=pd.date_range("2020-05-20", "2020-06-20"), y=modelo.predict(31), name="Forecast"
))

fig.update_layout(title="Previsão de casos confirmados no Brasil para os próximos 30 dias")
fig.show()
