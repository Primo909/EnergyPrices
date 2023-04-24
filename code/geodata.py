import pandas as pd
import geopandas as gpd
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pickle


# Load data
with open('data/clean/gis_demo.pkl', 'rb') as f:
    indicators = pickle.load(f)

with open('data/clean/balkan.pkl', 'rb') as f:
    geo = pickle.load(f)

# Merge data and shapefile
merged = gpd.GeoDataFrame(geo.merge(indicators, on='ISO', how='left'))

# Create the app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Define the layout
app.layout = html.Div([
    html.H1("GIS Dashboard demo"),
    dcc.Dropdown(
        id='dropdown',
        options=[{'label': i, 'value': i} for i in merged[['LoadActual', 'DayAheadPrice']].columns],
        value='DayAheadPrice'
    ),
    html.Br(),
    dcc.Graph(id='map', figure={}),
    
])

# Define the callback
@app.callback(
    Output(component_id='map', component_property='figure'),
    Input(component_id='dropdown', component_property='value')
)
def update_map(column):
    # Filter data by selection
    filtered_df = merged[[column, 'geometry', 'COUNTRY']]
    
    # Create the choropleth map
    fig = px.choropleth(
        filtered_df, 
        geojson=filtered_df.geometry, 
        locations=filtered_df.index, 
        color=column, 
        color_continuous_scale='reds', 
        range_color=(0, 5000), 
        projection='natural earth',
        #labels={'Energy Consumption':'Energy Consumption (kWh)'}
    )
    fig.update_geos(
        center=dict(lon=21, lat=43), 
        lataxis_range=[30, 45], 
        lonaxis_range=[9, 30]
    )    
    
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
