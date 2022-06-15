import plotly.graph_objs as go
import pandas as pd


def get_fig_roc(df, df_roc, df_predict, selected_symbol, selected_tf_interval):
    df_last_row = pd.DataFrame({'open_time': df_roc['open_time'].tail(1), 'close': df_roc['close'].tail(1)})
    df_predict_display = pd.concat([df_last_row, df_predict], ignore_index=True)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_predict_display['open_time'], y=df_predict_display['close'],
            name=f'{selected_symbol} predict rate of change',
            line=dict(color='red', dash='dot')
        )
    )
    fig.add_trace(
        go.Scatter(x=df['open_time'], y=df_roc['close'], name=f'{selected_symbol} rate of change close price',
                   line=dict(color='royalblue'))
    )

    # Update graph to follow latest
    last_row = df.iloc[-1, :]
    last_x = last_row['open_time']
    last_dx_prev = pd.Timedelta(milliseconds=27*selected_tf_interval)
    last_dx_next = pd.Timedelta(milliseconds=8*selected_tf_interval)
    fig.update_layout(
        xaxis={
            'title': 'Time',
            'range': [last_x - last_dx_prev, last_x + last_dx_next]
        },
        xaxis_rangeslider_visible=False,
        height=580
    )

    return fig


def get_fig_close(df, df_predict, selected_symbol, selected_tf_interval):
    df_last_row = pd.DataFrame({'open_time': df['open_time'].tail(1), 'close': df['close'].tail(1)})
    df_predict_display = pd.concat([df_last_row, df_predict], ignore_index=True)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_predict_display['open_time'], y=df_predict_display['close'],
            name=f'{selected_symbol} Predict closing price',
            line=dict(color='red', dash='dot')
        )
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
        height=580
    )

    return fig
