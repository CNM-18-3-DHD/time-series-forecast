from dash import Dash, dcc, html, Input, Output, dash_table, ALL, ctx
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
    'XGBOOST': Algorithm.XGBOOST,
    'TRANSFORMER': Algorithm.TRANSFORMER,
}
data_table_columns = [
    {'id': 'open_time', 'name': 'Time'},
    {'id': 'open', 'name': 'Open'},
    {'id': 'low', 'name': 'Low'},
    {'id': 'high', 'name': 'High'},
    {'id': 'close', 'name': 'Close'},
]

# Shared server state, not safe for using multiple tabs/browser
g_is_loading = False
g_df = None
g_selected_model = None
g_selected_feature = None
g_selected_symbol = None
g_current_ws_index = 0
g_current_step = 0

app = Dash()


app.layout = html.Div([
    html.Div([
        html.H4("[DACK-CNM]Time Series", className='top_bar_title'),
        html.H5('Current view: Loading... ', id='initial-debug'),
        html.P('18120304 - 18120312 - 18120355', className='top_bar_title'),
    ], className='top_bar'),
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
                    {'label': 'Transformer', 'value': 'TRANSFORMER'},
                ],
                value='LTSM',
                id='select-algorithm',
            ),
        ], style={'padding': 10, 'flex': 1}),
        html.Div([

        ], style={'padding': 10, 'flex': 1}),
    ], style={'display': 'flex'}),
    html.Div([
        html.H5('Current data', className='title'),
        dash_table.DataTable(
            id='ws-current-data',
            columns=data_table_columns,
            style_cell={
                'height': 'auto',
                'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
                'whiteSpace': 'normal'
            },
            style_header={
                'fontFamily': 'sans-serif',
                'backgroundColor': 'rgb(220, 220, 220)',
                'fontWeight': 'bold'
            },
            style_data={
                'fontFamily': 'sans-serif',
                'fontSize': '24px'
            },
        ),
    ], id='ws-debug', style={'padding': '0 10px'}),
    dcc.Loading(
        children=[
            html.Div([
                dcc.Loading(
                    parent_className='loading_wrapper',
                    children=[
                        dcc.Graph(
                            id='data-graph',
                        ),
                    ]
                ),
            ], id='graph-wrapper')
        ]
    ),
    html.Div(
        id='ws-wrapper',
        children=[
            WebSocket(id={'type': 'ws-data', 'index': '0'})
        ]
    ),
])


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
    Output('graph-wrapper', 'children'),  # Abuse this to make loading initial
    Input('select-symbol', 'value'),
    Input('select-feature', 'value'),
    Input('select-algorithm', 'value'),
)
def update_initial(selected_symbol, selected_feature, selected_algo):
    global g_is_loading, g_df, g_selected_feature, \
        g_selected_model, g_current_ws_index, g_selected_symbol, g_current_step
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
    g_is_loading = False
    g_current_step = 0  # reset
    g_current_ws_index += 1

    ws_url = loader.get_ws_url(selected_symbol, selected_timeframe)
    ws_component = WebSocket(id={
        'type': 'ws-data',
        'index': g_current_ws_index
    }, url=ws_url)

    return f"Current view: {selected_symbol} - Feature: {selected_feature} - Algorithm: {selected_algo}", \
           [ws_component], dash.no_update


# WebSocket rate of change
def handle_ws_roc(df, selected_symbol, algorithm):
    global g_current_step
    df_roc = df[['open_time', 'close']].copy()
    df_roc['close'] = df_roc['close'].pct_change()
    df_roc['close'] = df_roc['close'].fillna(0)

    df_predict = algorithm.predict_step(g_current_step + 1, df_roc)
    # print(df_predict)

    return fig_utils.get_fig_roc(df, df_roc, df_predict, selected_symbol, selected_tf_interval)


# WebSocket close price
def handle_ws_close(df, selected_symbol, algorithm):
    global g_current_step
    df_predict = algorithm.predict_step(g_current_step + 1, df)
    # print(df_predict)

    return fig_utils.get_fig_close(df, df_predict, selected_symbol, selected_tf_interval)


def find_ws_input(ctx_args, ws_id):
    # https://stackoverflow.com/questions/8653516/python-list-of-dictionaries-search
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

    global g_is_loading, g_selected_model, g_df, g_selected_feature, g_selected_symbol
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
    Output('ws-current-data', 'data'),
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

    if not success:
        return dash.no_update

    return data.to_dict('records')


if __name__ == "__main__":
    app.run_server(debug=True)
