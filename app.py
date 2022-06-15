from dash import Dash, dcc, html, Input, Output, State, ALL, ctx
import dash
from enums.timeframes import TimeframeTypes, Timeframes, TimeframesGap
import utils
from algorithm import AlgorithmFactory, Algorithm
import pandas as pd
import utils.figure as fig_utils
import utils.load as loader
from utils.ws_message import format_binance_message
from dash_extensions import WebSocket

symbols = loader.load_symbol_data()
selected_timeframe = Timeframes[TimeframeTypes.TF_1MIN]
selected_tf_interval = TimeframesGap[TimeframeTypes.TF_1MIN]
algorithms = {
    'LTSM': Algorithm.LTSM,
    'RNN': Algorithm.RNN,
    'XGBOOST': Algorithm.LTSM   # TODO: Change later, this is just for not creating bug
}

# Shared server state, not safe for using multiple tabs/browser
g_is_loading = False
g_is_initial = False
g_df = None
g_df_predict = None
g_selected_model = None
g_selected_feature = None
g_selected_symbol = None
g_current_ws_index = 0
g_current_step = 0

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
    html.Div(
        id='ws-wrapper',
        children=[
            WebSocket(id={'type': 'ws-data', 'index': '0'})
        ]
    ),
    html.P(id='initial-debug'),
    html.P(id='ws-debug')
    # dcc.Interval(
    #     id='graph-interval-update',
    #     interval=selected_tf_interval,  # (in milliseconds), call update
    #     n_intervals=0
    # )
])


def handle_feature_roc(df, selected_symbol, algorithm):
    df_roc = df[['open_time', 'close']].copy()
    df_roc['close'] = df_roc['close'].pct_change()
    df_roc['close'] = df_roc['close'].fillna(0)

    algorithm.fit(df_roc)
    df_predict = algorithm.predict(n=1)
    last_row = pd.DataFrame({'open_time': df_roc['open_time'].tail(1), 'close': df_roc['close'].tail(1)})
    df_predict = pd.concat([last_row, df_predict], ignore_index=True)
    global g_df_predict
    g_df_predict = df_predict

    return fig_utils.get_fig_roc(df, df_roc, df_predict, selected_symbol, selected_tf_interval)


def handle_feature_close(df, selected_symbol, algorithm):
    algorithm.fit(df)
    df_predict = algorithm.predict(n=1)
    last_row = pd.DataFrame({'open_time': df['open_time'].tail(1), 'close': df['close'].tail(1)})
    df_predict = pd.concat([last_row, df_predict], ignore_index=True)
    # print(df_predict)
    global g_df_predict
    g_df_predict = df_predict

    return fig_utils.get_fig_close(df, df_predict, selected_symbol, selected_tf_interval)


def fit_algorithm(df, selected_feature, selected_algo):
    algorithm = AlgorithmFactory.get(algorithms.get(selected_algo))
    if selected_feature == 'Rate of Change':
        df_roc = df[['open_time', 'close']].copy()
        df_roc['close'] = df_roc['close'].pct_change()
        df_roc['close'] = df_roc['close'].fillna(0)
        algorithm.fit(df_roc)
    else:
        df_copy = df[['open_time', 'close']].copy()
        algorithm.fit(df_copy)
    return algorithm


@app.callback(
    Output('initial-debug', 'children'),
    Output('ws-wrapper', 'children'),
    Input('select-symbol', 'value'),
    Input('select-feature', 'value'),
    Input('select-algorithm', 'value'),
)
def update_initial(selected_symbol, selected_feature, selected_algo):
    global g_is_loading, g_df, g_selected_feature, \
        g_selected_model, g_is_initial, g_current_ws_index, g_selected_symbol, g_current_step
    g_is_loading = True

    df = pd.DataFrame(loader.load_data(
        symbol=selected_symbol,
        interval=selected_timeframe
    ))
    df = df.sort_values('open_time')

    algorithm = fit_algorithm(df, selected_feature, selected_algo)

    g_df = df
    g_selected_symbol = selected_symbol
    g_selected_feature = selected_feature
    g_selected_model = algorithm
    g_is_initial = True
    g_is_loading = False
    g_current_step = 0  # reset
    g_current_ws_index += 1

    ws_url = loader.get_ws_url(selected_symbol, selected_timeframe)
    ws_component = WebSocket(id={
        'type': 'ws-data',
        'index': g_current_ws_index
    }, url=ws_url)

    return f"Loaded {selected_symbol} {selected_feature} {selected_algo}", [ws_component]


def handle_ws_roc(df, selected_symbol, algorithm):
    global g_current_step
    df_roc = df[['open_time', 'close']].copy()
    df_roc['close'] = df_roc['close'].pct_change()
    df_roc['close'] = df_roc['close'].fillna(0)

    df_predict = algorithm.predict_step(g_current_step + 1)
    # print(df_predict)

    return fig_utils.get_fig_roc(df, df_roc, df_predict, selected_symbol, selected_tf_interval)


def handle_ws_close(df, selected_symbol, algorithm):
    global g_current_step
    df_predict = algorithm.predict_step(g_current_step + 1)
    # print(df_predict)

    return fig_utils.get_fig_close(df, df_predict, selected_symbol, selected_tf_interval)


def find_ws_input(ctx_args, ws_id):
    return next((item for item in ctx_args if item['id']['index'] == ws_id), None)


# Multiple callback still present but filter out only the last callback
@app.callback(
    Output('data-graph', 'figure'),
    Input({'type': 'ws-data', 'index': ALL}, 'message'),
)
def update_graph_ws(message):
    if ctx.triggered_id is None:
        return dash.no_update

    global g_current_ws_index
    if ctx.triggered_id['index'] != g_current_ws_index:
        return dash.no_update

    global g_is_loading, g_is_initial, g_selected_model, g_df, g_selected_feature, g_selected_symbol
    if g_is_loading or g_selected_model is None:
        return dash.no_update
    # print(ctx.args_grouping)
    current_input = find_ws_input(ctx.args_grouping, g_current_ws_index)  # cannot use input message, multiple inputs
    message = current_input['value']

    data, is_closed, success = format_binance_message(message)
    df = g_df
    if success:
        df_last_row = utils.get_last_row(df)
        data_last_row = utils.get_last_row(data)
        if df_last_row['open_time'] == data_last_row['open_time']:
            df = df.iloc[:-1, :]  # Replace last row: last is not final
        df = pd.concat([df, data], ignore_index=True)
        if is_closed:
            global g_current_step
            g_current_step += 1
            g_df = df

    if g_selected_feature == 'Rate of Change':
        fig = handle_ws_roc(df, g_selected_symbol, g_selected_model)
    else:
        fig = handle_ws_close(df, g_selected_symbol, g_selected_model)

    return fig


@app.callback(
    Output('ws-debug', 'children'),
    Input({'type': 'ws-data', 'index': ALL}, 'message'),
    suppress_callback_exceptions=True,
)
def update_ws_message(message):
    if ctx.triggered_id is None:
        return dash.no_update

    global g_current_ws_index
    if ctx.triggered_id['index'] != g_current_ws_index:
        return dash.no_update

    current_input = find_ws_input(ctx.args_grouping, g_current_ws_index)  # cannot use input message, multiple inputs
    message = current_input['value']

    data, is_closed, success = format_binance_message(message)
    return [f"{data}"]


if __name__ == "__main__":
    app.run_server(debug=True)
