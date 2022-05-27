from dash import Dash, dcc, html, Input, Output
from binance import Client
from enums.timeframes import TimeframeTypes, Timeframes, TimeframesGap
import plotly.graph_objs as go
from algorithm.lstm_close import LSTMCloseAlgorithm
import pandas as pd
import utils.load as loader

symbols = loader.load_symbol_data()
intervals = []
selected_timeframe = Timeframes[TimeframeTypes.TF_1MIN]
selected_tf_interval = TimeframesGap[TimeframeTypes.TF_1MIN]

app = Dash()


app.layout = html.Div([
    html.H4("Time Series", style={"textAlign": "left"}),
    html.Div([
        html.Div([
            html.Label('Currency Symbol'),
            dcc.Dropdown(
                options=symbols,
                value=loader.DEFAULT_SYMBOL,
                # multi=True
                id='select-symbol'
            ),
        ], style={'padding': 10, 'flex': 2}),
        html.Div([
            html.Label('Features'),
            dcc.Dropdown(['Close', 'Rate of Changes'], 'Close'),
        ], style={'padding': 10, 'flex': 1}),
        html.Div([
            html.Label('Algorithm'),
            dcc.Dropdown(['LSTM', 'RNN', 'XGBoost'], 'LSTM'),
        ], style={'padding': 10, 'flex': 1}),
        html.Div([

        ], style={'padding': 10, 'flex': 1}),
    ], style={'display': 'flex'}),
    dcc.Loading(
        className='loading_wrapper',
        children=[
            dcc.Graph(
                id='data-graph',
            ),
        ]
    ),
    dcc.Interval(
        id='graph-interval-update',
        interval=selected_tf_interval,  # (in milliseconds), call update
        n_intervals=0
    )
])


@app.callback(
    Output('data-graph', 'figure'),
    Input('select-symbol', 'value'),
    Input('graph-interval-update', 'n_intervals')
)
def update_graph(new_selected_symbol, n):
    selected_symbol = new_selected_symbol

    df = pd.DataFrame(loader.load_data(
        symbol=selected_symbol,
        interval=selected_timeframe
    ))

    df = df.sort_values('open_time')

    algorithm = LSTMCloseAlgorithm()
    predict = algorithm.predict(df)
    df_predict = pd.DataFrame({
        "open_time": predict[0],
        "close": predict[1]
    })
    # print(df_predict)
    # print(len(df_predict))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df_predict['open_time'], y=df_predict['close'], name=f'{selected_symbol} Predict closing price',
                   line=dict(color='red'))
    )
    fig.add_trace(
        go.Scatter(x=df['open_time'], y=df['close'], name=f'{selected_symbol} closing price',
                   line=dict(color='royalblue'))
    )
    fig.add_candlestick(x=df['open_time'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
                        name=f'{selected_symbol} candlestick')
    # # Update graph to follow latest
    last_row = df.iloc[-1, :]
    last_x = last_row['open_time']
    last_y = float(last_row['close'])
    last_dx_prev = pd.Timedelta(milliseconds=17*selected_tf_interval)
    last_dx_next = pd.Timedelta(milliseconds=3*selected_tf_interval)
    last_dy = last_y / 1000
    fig.update_layout(
        xaxis={
            'title': 'Time',
            'range': [last_x - last_dx_prev, last_x + last_dx_next]
        },
        yaxis={
            'title': 'Price',
            'range': [last_y - 5*last_dy, last_y + 3*last_dy]
        },
        xaxis_rangeslider_visible=False
    )

    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
