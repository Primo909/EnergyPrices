import pandas as pd
import geopandas as gpd
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pickle

from plots import *

countries = ["BG", "GR", "HR", "RO", "RS", "SI"]
country_dict = {
"BG":"Bulgaria",
"GR":"Greece",
"HR":"Croatia",
"RO":"Romania",
"RS":"Republic of Serbia",
"SI":"Slovenia",
}

# Create the app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Define the layout
app.layout = html.Div([
    html.H1("GIS Dashboard demo"),
    dcc.Tabs(id="tabs_main", value="tabs_main",
        children = [
            dcc.Tab(label="MapsTab",
                value="tabs_main"),
            dcc.Tab(label="CountryTab",
                value="tabs_country"),
            ]),
        html.Div(id="tabs_content"),
])

@app.callback(Output("tabs_content", "children"),
        Input("tabs_main","value"))
def render_tabs(tab):
    if tab=="tabs_main":
        return html.Div([
            dcc.Dropdown(id='dropdown',
                options=[{'label': i, 'value': i} for i in ['LoadActual', 'DayAheadPrice']],
                value='DayAheadPrice'
            ),
            dcc.Graph(id='map', figure={}),
            ])
    if tab=="tabs_country":
        row = html.Div([
            dbc.Row(dbc.Col(html.Div("A single column"))),
            dbc.Row([
                dbc.Col(html.Div("One of three columns")),
                dbc.Col(html.Div("One of three columns")),
                dbc.Col(html.Div("One of three columns")),
            ]
            ),
        ]
)
        return html.Div([row,
            dcc.Dropdown(id='country-select',
                options=[{'label': country_dict[code], 'value': code} for code in countries],
                value='SI'
            ),
            #html.Br(),
                html.P(id="used_features"),
                html.Div([
                    dcc.Graph(id='power_gen_stacked', figure={}),
                    ], style={"width":"45%","display":"inline-block"}),
                html.Div([
                dcc.Graph(id='power_price_raw', figure={}),
                    ], style={"width":"45%","display":"inline-block"}),
                html.Div([
                dcc.Graph(id='power_price_pred', figure={}),
                    ], style={"width":"45%","display":"inline-block"}),
                html.Div(id='metrics_table',
                    style={"display":"inline-block", "width":"45%"}
                    ),
                html.Div([
                dcc.Slider(0, 12, 4,
                    value=10,
                    id='n_features'
                    ),
                    ], style={"width":"45%","display":"inline-block"}),
            ])
    

# Define the callback
@app.callback(
    Output(component_id='map', component_property='figure'),
    Input(component_id='dropdown', component_property='value')
)
def update_map_callback1(column):
    return update_map(column)

@app.callback(
    Output(component_id='power_gen_stacked', component_property='figure'),
    Input(component_id='country-select', component_property='value')
)
def update_plotGeneration(code, roll_window=24):
    return plotGeneration(code, roll_window=24)
@app.callback(
    Output(component_id='power_price_raw', component_property='figure'),
    Input(component_id='country-select', component_property='value')
)
def update_plotPrice(code, roll_window=24):
    return plotPrice(code, roll_window=24)

@app.callback(
    [Output(component_id='power_price_pred', component_property='figure'),
        Output(component_id="used_features", component_property="children"),
        Output('metrics_table', 'children'),
        ],
    [Input(component_id='country-select', component_property='value'),
        Input(component_id="n_features", component_property="value")])
def update_plotPricePrediction(code,n_features, roll_window=24):
    fig, selected_features, err= plotPricePrediction(code, n_features, roll_window)
    html_P = "This model uses the features: "+', '.join(selected_features)
    html_table = generate_table(err)
    return fig, html_P, html_table

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
