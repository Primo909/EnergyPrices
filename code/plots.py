import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

import dash
from dash import html,Input,Output,dcc
from dash.dependencies import Input, Output, State
from dash import Dash, dash_table
import geopandas as gpd


path = './'

def load_augumented(code):
    file = path + 'data/augumented/' + code + '.pkl'
    with open(file, 'rb') as f:
        df = pickle.load(f)
    return df.drop(columns=["Price-1","Price-2","Load-24"])
countries = ["BG", "GR", "HR", "RO", "RS", "SI"]
country_dict = {
"BG":"Bulgaria",
"GR":"Greece",
"HR":"Croatia",
"RO":"Romania",
"RS":"Republic of Serbia",
"SI":"Slovenia",
}

colors = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#000000']

# %%
def plotGeneration(code, roll_window=24):
        df = load_augumented(code)
        df = df.rolling(window=roll_window).mean()
        gen_cols = []
        imp_cols = []
        fossil_cols = []
        for col in df.columns:
            if col[:3] == 'Gen' and col[-5:] != "Total":
                gen_cols.append(col)
            if col[:6] == 'Import':
                imp_cols.append(col)
            if col[:9] == 'GenFossil':
                fossil_cols.append(col)
        
        fig = go.Figure()
        for i,col in enumerate(gen_cols):
                color = colors[i]
                visibility='legendonly'
                visibility=True
                fig.add_trace(go.Scatter(x=df.index, 
                                        y=df[col], 
                                        name=col[3:],
                                        mode='lines',
                                        line_width=0,
                                        fillcolor=colors[i],
                                        stackgroup='one',
                                        visible=visibility))
                                        #groupnorm='percent'))
        fig.add_trace(go.Scatter(x=df.index, 
                                y=df["GenTotal"], 
                                name="GenTotal",
                                mode="lines",
                                line_color="black"))  

        fig.update_layout(
                #title={'text':'Rolling Average of Power Generation in %s with window of %.i hours' % (country_dict[code], roll_window),
                       #'x':0.5, 
                       #'xanchor':'center'},
                yaxis_title='Power Generation (MW)',
                xaxis_title='Date',
                showlegend=True)
        fig.update_layout(legend=dict(
            orientation="h",
            entrywidth=90,
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ))
        return fig

# %%
def plotPrice(code, roll_window=24):
        df = load_augumented(code)
        roll_df = df.rolling(window=roll_window).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, 
                                y=df["DayAheadPrice"], 
                                name="GenTotal",
                                mode="lines",
                                line_color="black"))  
        fig.add_trace(go.Scatter(x=roll_df.index, 
                                y=roll_df["DayAheadPrice"], 
                                name="RA  %.i h" % (roll_window),
                                mode="lines",
                                line_color="red")) 
        fig.update_layout(
                #title={'text':'Raw Data of Energy Prices in %s' % (country_dict[code]),
                       #'x':0.5, 'xanchor':'center'},
                yaxis_title='Price (EUR/MWh)',
                xaxis_title='Date',
                showlegend=True)
        fig.update_layout(legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ))

        return fig

# %%
from functions import *

# %%


# %%
def plotPricePrediction(code,n_features,model_type='rf',test_size=0.75,roll_window=24):
        df = load_augumented(code)
        model, params, selected_features, err, predictions = train_model_country( df, n_features,model_type, test_size=test_size)

        def train(dataframe):
                return dataframe[:int((1-test_size)*dataframe.shape[0])]
        def test(dataframe):
                return dataframe[int((1-test_size)*dataframe.shape[0]):]
        def rolling(dataframe):
                return dataframe.rolling(window=roll_window).mean()
        
        
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=df.index, 
                                y=df["DayAheadPrice"], 
                                name="Real price",
                                mode="lines",
                                line_color="blue"))  
        fig.add_trace(go.Scatter(x=df.index, 
                                y=rolling(df)["DayAheadPrice"], 
                                name="Real price (RA  %.i h)" % (roll_window),
                                mode="lines",
                                line_color="midnightblue"))
          
        fig.add_trace(go.Scatter(x=predictions.index, 
                                y=predictions, 
                                name="Prediction",
                                mode="lines",
                                line_color="orange"))  #
        fig.add_trace(go.Scatter(x=predictions.index, 
                                y=rolling(predictions), 
                                name="Pred price (RA  %.i h)" % (roll_window),
                                mode="lines",
                                line_color="red"))  

        fig.update_layout(
                #title={'text':'Prediction of Energy Price in %s' % (country_dict[code]),'x':0.5, 'xanchor':'center'},
                yaxis_title='Price (EUR/MWh)',
                xaxis_title='Date',
                showlegend=True)
        fig.update_layout(legend=dict(
            orientation="h",
            entrywidth=120,
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ))
        return fig, selected_features, err

def createMapValues():
    N = len(countries)
    geoplot_columns = ["DayAheadPrice",'LoadActual']
    n = len(geoplot_columns)
    values = {"ISO":countries}
    for i in range(n):
        values[geoplot_columns[i]]=np.zeros(N)
        
    for i,code in enumerate(countries):
        df = load_augumented(code)
        copy = df.copy()
        for j,column in enumerate(geoplot_columns):
            values[column][i] = copy[column].mean()
    
    indicators = pd.DataFrame.from_dict(values,orient='index').transpose()
    indicators[geoplot_columns] = indicators[geoplot_columns].astype(float)
    with open('data/clean/balkan.pkl', 'rb') as f:
        geo = pickle.load(f)
    merged = gpd.GeoDataFrame(geo.merge(indicators, on='ISO', how='left'))
    return merged
    
# %%
value1 = 'LoadActual'
value2 = "DayAheadPrice"
def update_map(column):
    # Filter data by selection
    merged = createMapValues()
    filtered_df = merged[[column, 'geometry', 'COUNTRY']]
    # Create the choropleth map
    fig = px.choropleth(
        filtered_df, 
        geojson=filtered_df.geometry, 
        locations=filtered_df.index, 
        scope="europe",
        color=column, 
        color_continuous_scale='reds', 
        #range_color=(240, 270), 
        projection='natural earth',
        #labels={'Energy Consumption':'Energy Consumption (kWh)'}
    )

    fig.update_geos(
        center=dict(lon=21, lat=43), 
        lataxis_range=[30, 45], 
        lonaxis_range=[9, 30]
    )    
    
    fig.update_layout(coloraxis_showscale=True,
                      margin={"r":0,"t":0,"l":0,"b":0})
    
    return fig
def generate_table(series, max_rows=10):
    dataframe = pd.DataFrame({'Metric':series.index, 'Value':series.values})
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])
# %%



