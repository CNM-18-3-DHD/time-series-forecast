from dash import Dash, dcc, html, Input, Output
from enums.timeframes import TimeframeTypes, Timeframes, TimeframesGap
import plotly.graph_objs as go
from algorithm import AlgorithmFactory, Algorithm
import pandas as pd
import utils.load as loader

symbols = loader.load_symbol_data()
intervals = []
selected_timeframe = Timeframes[TimeframeTypes.TF_1MIN]
selected_tf_interval = TimeframesGap[TimeframeTypes.TF_1MIN]
algorithms = {
    'LTSM': Algorithm.LTSM,
    'RNN': Algorithm.RNN,
    'XGBOOST': Algorithm.LTSM   # TODO: Change later, this is just for not creating bug
}


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
            html.Label('Feature'),
            dcc.Dropdown(
                [
                    {'label': 'Close Price', 'value': 'Close'},
                    {'label': 'Rate of Change', 'value': 'Rate of Change'},
                ], 'Close',
                id='select-feature',
            ),
        ], style={'padding': 10, 'flex': 1}),
        html.Div([
            html.Label('Algorithm'),
            dcc.Dropdown(
                options=[
                    {'label': 'LSTM', 'value': 'LTSM'},
                    {'label': 'RNN', 'value': 'RNN'},
                    {'label': 'XGBoost', 'value': 'XGBOOST'},
                ],
                value='LTSM',
                id='select-algorithm',
            ),
        ], style={'padding': 10, 'flex': 1}),
        html.Div([

        ], style={'padding': 10, 'flex': 1}),
    ], style={'display': 'flex'}),
    dcc.Loading(
        parent_className='loading_wrapper',
        children=[
            dcc.Graph(
                id='data-graph',
            ),
        ]
    ),
    # dcc.Interval(
    #     id='graph-interval-update',
    #     interval=selected_tf_interval,  # (in milliseconds), call update
    #     n_intervals=0
    # )
])


def handle_feature_roc(df, selected_symbol, selected_algorithm):
    df_roc = df[['open_time', 'close']].copy()
    df_roc['close'] = df_roc['close'].pct_change()
    df_roc['close'] = df_roc['close'].fillna(0)

    algorithm = AlgorithmFactory.get(algorithms.get(selected_algorithm))
    algorithm.fit(df_roc)
    df_predict = algorithm.predict(n=1)
    last_row = pd.DataFrame({'open_time': df_roc['open_time'].tail(1), 'close': df_roc['close'].tail(1)})
    df_predict = pd.concat([last_row, df_predict], ignore_index=True)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df_predict['open_time'], y=df_predict['close'], name=f'{selected_symbol} Predict rate of change',
                   line=dict(color='red', dash='dot'))
    )
    fig.add_trace(
        go.Scatter(x=df['open_time'], y=df_roc['close'], name=f'{selected_symbol} rate of change close price',
                   line=dict(color='royalblue'))
    )

    # Update graph to follow latest
    last_row = df.iloc[-1, :]
    last_x = last_row['open_time']
    last_dx_prev = pd.Timedelta(milliseconds=27*selected_tf_interval)
    last_dx_next = pd.Timedelta(milliseconds=3*selected_tf_interval)
    fig.update_layout(
        xaxis={
            'title': 'Time',
            'range': [last_x - last_dx_prev, last_x + last_dx_next]
        },
        xaxis_rangeslider_visible=False,
        height=600
    )

    return fig


def handle_feature_close(df, selected_symbol, selected_algorithm):
    algorithm = AlgorithmFactory.get(algorithms.get(selected_algorithm))
    algorithm.fit(df)
    df_predict = algorithm.predict(n=1)
    last_row = pd.DataFrame({'open_time': df['open_time'].tail(1), 'close': df['close'].tail(1)})
    df_predict = pd.concat([last_row, df_predict], ignore_index=True)
    # print(df_predict)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df_predict['open_time'], y=df_predict['close'], name=f'{selected_symbol} Predict closing price',
                   line=dict(color='red', dash='dot'))
    )
    fig.add_trace(
        go.Scatter(x=df['open_time'], y=df['close'], name=f'{selected_symbol} closing price',
                   line=dict(color='royalblue'))
    )
    fig.add_candlestick(x=df['open_time'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
                        name=f'{selected_symbol} candlestick')

    # Update graph to follow latest
    last_row = df.iloc[-1, :]
    last_x = last_row['open_time']
    last_y = float(last_row['close'])
    last_dx_prev = pd.Timedelta(milliseconds=27*selected_tf_interval)
    last_dx_next = pd.Timedelta(milliseconds=8*selected_tf_interval)
    last_dy = last_y / 1000
    fig.update_layout(
        xaxis={
            'title': 'Time',
            'range': [last_x - last_dx_prev, last_x + last_dx_next]
        },
        yaxis={
            'title': 'Price',
            'range': [last_y - 20*last_dy, last_y + 20*last_dy]
        },
        xaxis_rangeslider_visible=False,
        height=600
    )

    return fig


@app.callback(
    Output('data-graph', 'figure'),
    Input('select-symbol', 'value'),
    Input('select-feature', 'value'),
    Input('select-algorithm', 'value')
    # Input('graph-interval-update', 'n_intervals')
)
def update_graph(new_selected_symbol, new_selected_feature, selected_algo):
    selected_symbol = new_selected_symbol

    df = pd.DataFrame(loader.load_data(
        symbol=selected_symbol,
        interval=selected_timeframe
    ))

    # print(df)

    df = df.sort_values('open_time')

    if new_selected_feature == 'Rate of Change':
        return handle_feature_roc(df, selected_symbol, selected_algo)

    return handle_feature_close(df, selected_symbol, selected_algo)


if __name__ == "__main__":
    app.run_server(debug=True)
