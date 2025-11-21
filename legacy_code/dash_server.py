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
    html.Button('Create new experiment', id='create-exp', n_clicks=0),
    html.Button('Resume existing experiment', id='resume-exp', n_clicks=0),
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


if __name__ == '__main__':
    app.run_server(debug=True)