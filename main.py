import pandas as pd
import numpy as np
import plotly.graph_objs as go
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import streamlit as st
from shillelagh.backends.apsw.db import connect
import warnings
################################################
# FUNCTIONS START
def currency_selector():
    st.session_state['currency_selector'] = st.session_state['currency_selector']
# FUNCTIONS END

# connect to gsheets
conn = connect(":memory:")
cursor = conn.cursor()
def run_query(query):
    rows = cursor.execute(query)
    rows = rows.fetchall()
    return rows

# data URL
sheets_url = st.secrets["public_gsheets_url"]
rows = run_query(f'SELECT * FROM "{sheets_url}"')
#
st.set_page_config(
    page_title='Exchange Rate Forecasting Tool',
    layout='wide'
)
df_exchange = pd.DataFrame(rows, columns=['Date','ZAR','MUR','KSH','NGN','MZM','BPU'])
df_exchange['Date'] = pd.to_datetime(df_exchange['Date'])
# st.write(df_exchange)
df_exchange.set_index('Date',inplace=True)
currency_list = ['ZAR','MUR','KSH','NGN','MZM','BPU']
arima_orders_dic = {
    'ZAR':(2,1,2),
    'MUR': (2,1,1),
    'KSH': (2,1,3),
    'NGN': (2,1,3),
    'MZM': (1,1,1),
    'BPU': (3,1,2)

}
# initialize an empty df for weekly resample
# st.write(df_exchange)
week_exchange_df = pd.DataFrame()
for i in range(len(currency_list)):
    week_series = df_exchange[currency_list[i]].resample('W').mean()
    week_exchange_df[currency_list[i]] =  week_series.values
    if i== len(currency_list) -1 :
        week_exchange_df['Date'] = week_series.index
        week_exchange_df.set_index('Date',inplace=True)
# for n, currency in enumerate(currency_list):
#   week_exchange_df[f'{currency}_log'] = np.log(week_exchange_df[currency])


# create header
header =  st.container()
# summary metrics container
summary_metrics = st.container()
# forecast container
forecast_container = st.container()
with header:
    # create the title for the dashboard
    st.title('Exchange Rate Forecast Tool')
with summary_metrics:
    st.header('Current Exchange Rates')
    st.write(f"Last updated: {df_exchange.index.tolist()[-1]}")
    # create 6 columns to display each currency
    col1,col2,col3,col4,col5,col6 = st.columns(6)
    with col1:
        # present the exchange rate for ZAR
        zar_rate = df_exchange['ZAR'].tolist()[-1]
        zar_rate_delta = round(df_exchange['ZAR'].tolist()[-1] - df_exchange['ZAR'].tolist()[-2],3)
        st.metric('ZAR/USD',f"R{round(zar_rate,3)}", zar_rate_delta, 'inverse')
    with col2:
        # present the exchange rate for MUR
        mur_rate = df_exchange['MUR'].tolist()[-1]
        mur_rate_delta = round(df_exchange['MUR'].tolist()[-1] - df_exchange['MUR'].tolist()[-2],3)
        st.metric('MUR/USD', f"MUR{round(mur_rate,3)}", mur_rate_delta, 'inverse')
    with col3:
        # present the exchange rate for KSH
        ksh_rate = df_exchange['KSH'].tolist()[-1]
        ksh_rate_delta = round(df_exchange['KSH'].tolist()[-1]-df_exchange['KSH'].tolist()[-2],3)
        st.metric('KSH/USD',f"KSH{round(ksh_rate,3)}", ksh_rate_delta,'inverse')
    with col4:
        # present the exchange rate for NGN
        ngn_rate  = df_exchange['NGN'].tolist()[-1]
        ngn_rate_delta = round(df_exchange['NGN'].tolist()[-1]-df_exchange['NGN'].tolist()[-2],3)
        st.metric('NGN/USD',f"NGN{round(ngn_rate,3)}", ngn_rate_delta, 'inverse')
    with col5:
        # present the exchange rate for MZM
        mzm_rate = df_exchange['MZM'].tolist()[-1]
        mzm_rate_delta = round(df_exchange['MZM'].tolist()[-1]-df_exchange['MZM'].tolist()[-2],3)
        st.metric('MZM',f"MZM{round(mzm_rate,3)}", mzm_rate_delta, 'inverse')
    with col6:
        # present the exchange rate for BPU
        bpu_rate = df_exchange["BPU"].tolist()[-1]
        bpu_rate_delta = round(df_exchange["BPU"].tolist()[-1]-df_exchange["BPU"].tolist()[-2],3)
        st.metric('BPU',f"BPU{round(bpu_rate,3)}", bpu_rate_delta, 'inverse')
with forecast_container:
    st.header('Forecast Exchange Rate with ARIMA model')
    currency_select = st.selectbox('Currency', ['ZAR','MUR','KSH','NGN','MZM','BPU'], key='currency_selector',
                              on_change=currency_selector)
    week_exchange_df = week_exchange_df[currency_select]
    # st.write(week_exchange_df)
    # split data into train and test set
    length = week_exchange_df.shape[0]
    split = int(0.8*length)
    train_set, test_set = week_exchange_df[0:split], week_exchange_df[split-3:]
    model = ARIMA(week_exchange_df[:-3], order=arima_orders_dic[currency_select])
    result = model.fit()
    forecast = result.get_forecast(4).predicted_mean
    conf_ = result.get_forecast(4).conf_int(alpha=0.05)
    # st.write(conf_)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=week_exchange_df.index.values, y=week_exchange_df.values, mode='lines+markers',name=f'past-{currency_select}/USD'))
    fig.add_trace(go.Scatter(x=forecast.index.values, y=forecast.values, mode='lines+markers', name='forecast'))
    fig.add_trace(go.Scatter(x=conf_.index.values,y=conf_[f'upper {currency_select}'], mode='lines', showlegend=False))
    fig.add_trace(go.Scatter(x=conf_.index.values, y =conf_[f'lower {currency_select}'], mode='lines', fillcolor='rgba(68, 68, 68, 0.3)',
                             fill='tonexty', showlegend=False))
    fig.update_layout(title=f"{currency_select} 4 week forecast")
    fig.update_xaxes(title='Date')
    fig.update_yaxes(title=f'{currency_select}/USD')

    st.plotly_chart(fig, use_container_width=True)
