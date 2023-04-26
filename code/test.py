from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px

from plots import * 

# Iris bar figure

def drawFigure():
    return  html.Div([
        dbc.Card(
            dbc.CardBody([
                dcc.Graph(
                    figure=px.bar(
                        df, x="sepal_width", y="sepal_length", color="species"
                    ).update_layout(
                        template='plotly_dark',
                        plot_bgcolor= 'rgba(0, 0, 0, 0)',
                        paper_bgcolor= 'rgba(0, 0, 0, 0)',
                    ),
                    config={
                        'displayModeBar': False
                    }
                ) 
            ])
        ),  
    ])

# Text field
def drawText(text):
    return html.Div([
        dbc.Card(
            dbc.CardBody([
                html.Div([
                    html.H2(text),
                ], style={'textAlign': 'center'}) 
            ])
        ),
    ])

def drawSlider():
    return html.Div([
        dbc.Card(
            dbc.CardBody([
                html.Div([
                html.Label("Number of features used for training"),
                dcc.Slider(0, 12, 1,
                    value=10,
                    id='n_features'
                    ),
                ],
                )
            ])
        ),
    ])
def drawSliderPercent():
    return html.Div([
        dbc.Card(
            dbc.CardBody([
                html.Div([
                html.Label("Percent of Data used for training"),
                dcc.Slider(10, 90,
                    value=70,
                    id='test_size'
                    ),
                ],
                )
            ])
        ),
    ])

def drawRadioModel():
    return html.Div([
        dbc.Card(
            dbc.CardBody([
                html.Div([
                html.Label("Percent of Data used for training"),
                dcc.RadioItems(id='model_type',
                    options=[
                    {'label': 'Random Forest', 'value': 'rf'},
                    {'label': 'Neural Network', 'value': 'nn'},
                    ],
                    value='rf', inline=True),
                    ],
                    )
            ])
        ),
    ])
def drawDropdown():
    return html.Div([
        dbc.Card(
            dbc.CardBody([
                html.Div([
                    dcc.Dropdown(id='country-select',
                                options=[{'label': country_dict[code], 'value': code} for code in countries],
                                value='SI'
                                ),
                ],
                )
            ])
        ),
        ])

def drawMapDrowdown():
    return html.Div([
        dbc.Card(
            dbc.CardBody([
                html.Div([
                    dcc.Dropdown(id='dropdown',
                        options=[{'label': i, 'value': i} for i in ['LoadActual', 'DayAheadPrice']],
                        value='DayAheadPrice'
                    ),
                ],
                )
            ])
        ),
        ])
# Data
df = px.data.iris()

# Build App
app = Dash(external_stylesheets=[dbc.themes.CERULEAN])

tabs = html.Div(
    [
        dbc.Tabs(
            [
                dbc.Tab(label="Tab 1", tab_id="tab-1"),
                dbc.Tab(label="Tab 2", tab_id="tab-2"),
            ],
            id="tabs",
            active_tab="tab-1",
        ),
        html.Div(id="content"),
    ]
)



app.layout = dbc.Card(dbc.CardBody(dbc.Row([dbc.Col([tabs],width=12),])))

tab1_content=html.Div([
    dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    drawText("Choose country")
                ], width=3),
                dbc.Col([
                    drawDropdown()
                ], width=3),
            ], align='center'), 
            html.Br(),
            dbc.Row([
                #
                dbc.Col([
                    dbc.Card(dbc.CardBody([
                        dcc.Graph(id="power_gen_stacked")
                        ]))
                ], width=6),
                #
                dbc.Col([
                    dbc.Card(dbc.CardBody([dcc.Graph(id="power_price_raw")]))
                ], width=6),
                #
            ], align='center'), 
            html.Br(),
            dbc.Row([
                dbc.Col([
                    drawRadioModel(),
                    drawSlider(),
                    drawSliderPercent(),
                    dbc.Card(dbc.CardBody([html.Div(id='metrics_table')])),
                    dbc.Card(dbc.CardBody([
                    html.P(id="used_features")
                    ]))
                ], width=6),
                #
                dbc.Col([
                    dbc.Card(dbc.CardBody([dcc.Graph(id="power_price_pred")]))
                    
                ], width=6),
            ], align='center'),      
        ]), color = 'white'
    )
])

tab2_content=html.Div([
    dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    drawText("Choose Feature")
                ], width=3),
                dbc.Col([
                    drawMapDrowdown()
                ], width=3),
            ], align='center'), 
            html.Br(),
            dbc.Row([
                dbc.Col([
                    dbc.Card(dbc.CardBody([dcc.Graph(id="map")]))
                ], width=6),
            ], align='center'),      
            ]), color = 'white'
    )
])
@app.callback(Output("content", "children"), [Input("tabs", "active_tab")])
def switch_tab(at):
    if at == "tab-1":
        return tab1_content
    elif at == "tab-2":
        return tab2_content
    return html.P("This shouldn't ever be displayed...")

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
        Input(component_id="n_features", component_property="value"),
        Input(component_id="model_type", component_property="value"),
    Input(component_id='test_size', component_property='value')])
def update_plotPricePrediction(code,n_features,model_type, test_size, roll_window=24):
    test_size=1-test_size/100
    fig, selected_features, err= plotPricePrediction(code, n_features, model_type, test_size, roll_window)
    html_P = "This model predicts the energy price one day in advance. Therefore we don't use Power-1 but only Power-24. The list of features reads: "+', '.join(selected_features)
    html_table = generate_table(err)
    return fig, html_P, html_table

@app.callback(
    Output(component_id='map', component_property='figure'),
    Input(component_id='dropdown', component_property='value')
)
def update_map_callback1(column):
    return update_map(column)

# Run app and display result inline in the notebook
app.run_server(debug=True,port=8660)
