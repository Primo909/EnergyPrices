from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
from datetime import date
from plotly.subplots import make_subplots

from plots import * 

template_main='simple_white'
def update_graph(fig):
    return fig.update_layout(
                        template=template_main,
                        plot_bgcolor= 'rgba(0, 0, 0, 0)',
                        paper_bgcolor= 'rgba(0, 0, 0, 0)',
                    )


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
def drawText(text,size="big"):
    if size=="big":
        obj = html.Div([
          dbc.Card(
              dbc.CardBody([
                  html.Div([
                          html.H4(text),
                  ], style={'textAlign': 'center'}) 
              ])
          ),
      ])
    elif size=="p":
        obj = html.Div([
          dbc.Card(
              dbc.CardBody([
                  html.Div([
                          html.P(text),
                  ], style={'textAlign': 'center'}) 
              ])
          ),
      ])
    return obj

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
                    value=30,
                    id='test_size'
                    ),
                ],
                )
            ])
        ),
    ])

def drawDatePicker():
    return html.Div([
        dbc.Card(
            dbc.CardBody([
                html.Div([
                html.Label("Choose range in which to average data "),
                dcc.DatePickerRange(
                        id='date_picker',
                        min_date_allowed=date(2022, 2, 1),
                        max_date_allowed=date(2022, 11, 26),
                        start_date=date(2022, 3, 1),
                        end_date=date(2022, 4, 1),
                    ),
                ],
                )
            ])
        ),
    ])

def drawDayChooser():
    return html.Div([
        dbc.Card(
            dbc.CardBody([
                html.Div([
                html.Label("Choose start date for prediction"),
                dcc.DatePickerSingle(
                    id='day_chooser',
                    min_date_allowed=date(2022, 2, 1),
                    max_date_allowed=date(2022, 11, 25),
                    date=date(2022, 6, 21)
                )
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
                html.Label("Choose model"),
                dcc.RadioItems(id='model_type',
                    options=[
                    {'label': 'Linear Regression', 'value': 'lr'},
                    {'label': 'Random Forest', 'value': 'rf'},
                    {'label': 'Neural Network', 'value': 'nn'},
                    ],
                    value='lr', inline=True),
                html.Label("Exclude features"),
                dcc.Checklist(id="exclude_features",
                   options=[
                       {'label': 'Price -1h', 'value': 'Price-1'},
                       {'label': 'Price -2h', 'value': 'Price-2'},
                       {'label': 'Price -24h', 'value': 'Price-24'},
                   ],inline=True,
                   value=['Price-1',"Price-2"]
                ),
                html.Label("Feature list"),
                html.P(id="used_features")
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
                        options=[{'label': i, 'value': i} for i in ['LoadActual', 'DayAheadPrice',"GenTotal","ImportTotal","GenFossilTotal"]],
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
app = Dash(__name__,external_stylesheets=[dbc.themes.FLATLY])
app.title = "Balkan Energy Price"

tabs = html.Div(
    [
        dbc.Tabs(
            [
                dbc.Tab(label="Data by Country", 
                    tab_id="tab-1",
                    tabClassName="flex-grow-1 text-center",),
                dbc.Tab(label="Overview of the Balkan Region",
                    tab_id="tab-2",
                    tabClassName="flex-grow-1 text-center",),
                dbc.Tab(label="Price Prediction for one Day",
                    tab_id="tab-3",
                    tabClassName="flex-grow-1 text-center",),
                dbc.Tab(label="Contributions",
                    tab_id="tab-4",
                    tabClassName="flex-grow-1 text-center",)
            ],
            id="tabs",
            active_tab="tab-4",
        ),
        html.Div(id="content"),
    ]
)



app.layout = html.Div([
    html.Br(),
    html.H1("Energy Prices Dashboard of the Balkan Region"),
    html.Br(),
    dbc.Card(dbc.CardBody(dbc.Row([dbc.Col([tabs],width=12),])))
    ])

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
                dbc.Col([
                    drawText("Note: Bulgaria has the entire month of October 2022 missing and some data before and after.","p")
                ], width=6),
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
                ], width=6),
                #
                dbc.Col([
                    dbc.Card(dbc.CardBody([dcc.Graph(id="power_price_pred")]))
                    
                ], width=6),
            ], align='center'),      
        ]), 
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
                dbc.Col([
                    drawDatePicker()
                ], width=6),
            ], align='center'), 
            html.Br(),
            dbc.Row([
                dbc.Col([
                    dbc.Card(dbc.CardBody([dcc.Graph(id="map")]))
                ], width=6),
                dbc.Col([
                    dbc.Card(dbc.CardBody([dcc.Graph(id="map_hist")]))
                ], width=6),
            ], align='center'),      
            ]), 
    )
])


tab3_content=html.Div([
    dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    drawText("Prediction of DayAheadPrice for the next Day")
                ], width=6),
                dbc.Col([
                    drawDayChooser()
                ], width=6),
            ], align='center'), 
            html.Br(),
            dbc.Row([
                dbc.Col([
                    dbc.Card(dbc.CardBody([dcc.Graph(id="graph_one_day")]))
                ], width=12),
                dbc.Col([
                    dbc.Card(dbc.CardBody([html.Div(id='one_day_table')])),
                ], width=12),
                dbc.Col([
                    drawText("In the latter half of the year, the results will be better, because there will be more training data availeable. The countries displayed are not always the same, since some countries have missing values for the DayAheadPrice or rather the entire day is missing from the dataset. Used features are selected based on mutual information and optimized for each country. Importantly the only feature that is time dependent is Price-24. Republic of Serbia will always be missing, since the dataset of this country is problematic when doing this prediction.","p")
                ], width=12),
            ], align='center'),      
            ]), 
    )
])

tab4_content=html.Div([
    dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    drawText("Contributions in the Project")
                ], width=12),
            ], align='center'), 
            html.Br(),
            dbc.Row([
                dbc.Col([
                    html.P("Jaka Godec collected the raw data from the API, did the data preparation, data Fusion. He developed most of the code for the machine learning models, as well as some code of the geodata visualization."),
                        html.Br(),
                    html.P("Kevin Steiner designed and implemented the dashboard, dashboard interactivity and extracted new information from the data. He modified the models to fit different prediction usecases and explored different sets of hyperparameters.")
                ], width=12),
            ], align='center',),      
            ]), 
    )
])



@app.callback(Output("content", "children"), [Input("tabs", "active_tab")])
def switch_tab(at):
    if at == "tab-1":
        return tab1_content
    elif at == "tab-2":
        return tab2_content
    elif at == "tab-3":
        return tab3_content
    elif at == "tab-4":
        return tab4_content
    return html.P("This shouldn't ever be displayed...")

@app.callback(
    Output(component_id='power_gen_stacked', component_property='figure'),
    Input(component_id='country-select', component_property='value')
)
def update_plotGeneration(code, roll_window=24):
    return update_graph(plotGeneration(code, roll_window=24))

@app.callback(
    Output(component_id='power_price_raw', component_property='figure'),
    Input(component_id='country-select', component_property='value')
)
def update_plotPrice(code, roll_window=24):
    return update_graph(plotPrice(code, roll_window=24))


@app.callback(
    [Output(component_id='power_price_pred', component_property='figure'),
        Output(component_id="used_features", component_property="children"),
        Output('metrics_table', 'children'),
        ],
    [Input(component_id='country-select', component_property='value'),
        Input(component_id="n_features", component_property="value"),
        Input(component_id="model_type", component_property="value"),
    Input(component_id='test_size', component_property='value'),
    Input("exclude_features","value")])
def update_plotPricePrediction(code,n_features,model_type, test_size, exclude_features, roll_window=24):
    test_size=1-test_size/100
    print(exclude_features)
    fig, selected_features, err= plotPricePrediction(code, n_features, model_type, test_size, exclude_features, roll_window)
    html_P = ', '.join(selected_features)
    html_table = generate_table(err)

    return update_graph(fig), html_P, html_table

@app.callback(
    [Output(component_id='map', component_property='figure'),
    Output(component_id='map_hist', component_property='figure'),],
    [Input(component_id='dropdown', component_property='value'),
        Input("date_picker","start_date"),
        Input("date_picker","end_date"),]
)
def update_map_callback1(column,start,end):
    f1,f2 = update_map(column,start,end)
    return update_graph(f1), update_graph(f2)
# Run app and display result inline in the notebook

@app.callback(
    [Output(component_id='graph_one_day', component_property='figure'),
        Output('one_day_table', 'children'),],
    Input(component_id='day_chooser', component_property='date'),
)
def update_plotDayPrediction(date,n_features=12,model_type='rf',exclude_features=["Price-1","Price-2"]):
    fig,a,table = plotDayPrediction(date,n_features,model_type='rf',exclude_features=["Price-1","Price-2"])
    print(table)
    table = generate_dataframe_table(table) 
    print(table)
    return update_graph(fig).update_layout(height=1200, width=1200),table


app.run_server(debug=True,port=8660)
