import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Store(id='local', storage_type='local'),
    html.H6("Change the value in the text box to see callbacks in action!"),
    html.Div(["Input: ",
              dcc.Input(id='my-input', type='text')]),
    html.Br(),
    html.Div(id='my-output'),
    html.Div(id='live-update-graph'),
    dcc.Interval(
        id='interval-component',
        interval=1 * 1000,  # in milliseconds
        n_intervals=0
    )

])


# add a click to the appropriate store.
@app.callback(Output('local', 'data'),
              Input(component_id='my-input', component_property='value'),
              State('local', 'data'))
def on_click(value, data):
    if value is None:
        # prevent the None callbacks is important with the store component.
        # you don't want to update the store for nothing.
        raise PreventUpdate

    # Give a default data dict with 0 clicks if there's no data.
    data = data or {'input_value': 'init_val'}

    data['input_value'] = value
    return data


# output the stored clicks in the table cell.
@app.callback(Output('my-output', 'children'),
              # Since we use the data prop in an output,
              # we cannot get the initial data on load with the data prop.
              # To counter this, you can use the modified_timestamp
              # as Input and the data as State.
              # This limitation is due to the initial None callbacks
              # https://github.com/plotly/dash-renderer/pull/81
              Input('local', 'modified_timestamp'),
              State('local', 'data'))
def on_data(ts, data):
    if ts is None:
        raise PreventUpdate

    data = data or {}

    return data.get('input_value', 0)

import datetime
# Multiple components can update everytime interval gets fired.
@app.callback(Output('live-update-graph', 'children'),
              Input('interval-component', 'n_intervals'))
def update_graph_live(n):
    time = datetime.datetime.now()
    return time




if __name__ == '__main__':
    app.run_server(debug=True)