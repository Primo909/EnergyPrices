import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    return df.drop(columns=["Load-1","Load-24"])
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
                yaxis_title='RA 24h Power Generation (MW)',
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
                                name="DayAheadPrice",
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
def plotPricePrediction(code,n_features,model_type='rf',test_size=0.75,exclude_features=["Price-1"],roll_window=24):
        df = load_augumented(code).drop(columns=exclude_features)
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

def createMapValues(start=0,end=0):
    N = len(countries)
    geoplot_columns = ['LoadActual', 'DayAheadPrice',"GenTotal","ImportTotal","GenFossilTotal"]
    n = len(geoplot_columns)
    values = {"ISO":countries}
    for i in range(n):
        values[geoplot_columns[i]]=np.zeros(N)
        
    for i,code in enumerate(countries):
        df = load_augumented(code)
        copy = df.copy()
        for j,column in enumerate(geoplot_columns):
            if start==0 and end==0:
                values[column][i] = copy[column].mean()
            else: 
                values[column][i] = copy.loc[start:end][column].mean()
    
    indicators = pd.DataFrame.from_dict(values,orient='index').transpose()
    indicators[geoplot_columns] = indicators[geoplot_columns].astype(float)
    with open('data/clean/balkan.pkl', 'rb') as f:
        geo = pickle.load(f)
    merged = gpd.GeoDataFrame(geo.merge(indicators, on='ISO', how='left'))
    return merged
    
# %%
value1 = 'LoadActual'
value2 = "DayAheadPrice"
def update_map(column,start=0,end=0):
    # Filter data by selection
    merged = createMapValues(start,end)
    filtered_df = merged[[column, 'geometry', 'COUNTRY']]
    # Create the choropleth map
    fig = px.choropleth(
        filtered_df, 
        geojson=filtered_df.geometry, 
        locations=filtered_df.index, 
        scope="europe",
        color=column, 
        color_continuous_scale='plasma', 
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
    fig2 = px.bar(merged.sort_values(by=column), 
            x='COUNTRY', y=column,
            color=column, color_continuous_scale='plasma',
            labels={'pop':'population of Canada'}, height=400)
    mini =merged[column].min()
    maxi =merged[column].max()
    diff = mini*(maxi-mini)/maxi
    fig2.update_yaxes(range=[mini-diff,maxi+diff])
    
    return fig, fig2
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

def plotDayPrediction(day,n_features,model_type='rf',exclude_features=["Price-1","Price-2"]):
        print("start")
        countries = ["BG", "HR", "RO","SI"]
        countries_clean=[]
        for i,code in enumerate(countries):
                print(i,code)
                try: 
                        test=load_augumented(code).loc[day]
                        print(test.isnull().values.any())
                except: print("Date for this day does not exist") 
                else: 
                        if test.shape[0]==24: countries_clean.append(code)
        print(countries_clean)
        n = len(countries_clean)
        print(n)
        clist = []
        [clist.append(country_dict[x]) for x in countries_clean]
        fig = make_subplots(rows=int(n/2), cols=2,
                vertical_spacing = 0.25,
                subplot_titles=clist)

        def predOneDayOneCode(code,row,col):
                df = load_augumented(code).drop(columns=exclude_features)
                print(df.isnull().values.any())
                model, params, selected_features, err, predictions = train_model_country_day(df, day, 12, model_type="rf")
                def rolling(dataframe):
                        return dataframe.rolling(window=roll_window).mean()
                print(selected_features)
                
                
                df = df.loc[day]

                fig.add_trace(go.Scatter(x=df.index, 
                                        y=df["DayAheadPrice"], 
                                        name="Real price",
                                        mode="lines",
                                        line_color="blue",),
                                        row=row,col=col)  
                
                fig.add_trace(go.Scatter(x=predictions.index, 
                                        y=predictions, 
                                        name="Prediction",
                                        mode="lines",
                                        line_color="orange",),
                                        row=row,col=col)  #

                fig.update_layout(
                        #title={'text':'Prediction of Energy Price in %s' % (country_dict[code]),'x':0.5, 'xanchor':'center'},
                        showlegend=False)
                
                #return fig, selected_features, err
                return 0
        #figa=[None]*n
        #sfa=[None]*n
        #erra=[None]*n
        j=1
        k=1
        for i,code in enumerate(countries_clean):
                predOneDayOneCode(code,k,j)
                if k==j and j==1: j=j+1
                elif k==j and j==2: j=j-1
                elif j>k: k=k+1
        #code = "SI"
        #figa, sfa, erra=predOneDayOneCode(code,1,1)
        #print("aaskdjflöaksjdlfkjaskldjfkajdfölaj")
        fig.update_layout(legend=dict(
                orientation="h",
                entrywidth=120,
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
                ))
        print("finished")
        return fig,0,0
# %%



