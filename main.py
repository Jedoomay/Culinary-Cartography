import pandas as pd#panda data manipulation
import ast #used to turn string list into actual python list
import dash# dash data visualization
from dash import dcc, html# used for app desing, radio buttons, dropdown etc.
from dash.dependencies import Input, Output#used to manage radio buttons , dropdowns
import plotly.graph_objects as go#for layout of the graphs
import dash_bootstrap_components as dbc #for the design of the app
from sklearn.preprocessing import MultiLabelBinarizer# to turn into a matrix, for machine learning
from sklearn.model_selection import train_test_split#to parse test and train
from scipy.sparse import csr_matrix# to turn sparse matrix into csr matrix
from sklearn.linear_model import LogisticRegression#used for machine learning implemtation
from sklearn.metrics import classification_report, confusion_matrix#used for the purpose of visualziing machine learningn stats
import networkx as nx##for the network graph
import statsmodels.api as sm#for fit line, linear regression

# Load the recipes dataset
file_path = 'recipes_with_region.csv'  

recipesdf = pd.read_csv(file_path)

# used exception handling , if ingredient_str is python literal evalute it as true
def parse_ingredients(ingredient_str):
        return ast.literal_eval(ingredient_str)

recipesdf['ingredients'] = recipesdf['ingredients'].apply(parse_ingredients)

# To fix Middle Eastern with extra space, could've been dealth manually
recipesdf = recipesdf[recipesdf['region'] != 'Middle Eastern ']
all_regions=sorted(recipesdf['region'].unique())#sort the regions

# used to turn it into a ingredients matrix, all ingredients is a vector, if in a recipe ingerdients is occuring it is labeld as 1 o.w 0
mlb = MultiLabelBinarizer(sparse_output=True)
ingredients_encoded_sparse = mlb.fit_transform(recipesdf['ingredients'])#converting into a binary format 

# used for convertin sparse matrix into csr matrix for efficieny issues, o.w it takes to much time 
ingredients_sparse_csr = csr_matrix(ingredients_encoded_sparse)

#convert into numpy array correspongind region values
region_labels = recipesdf['region'].values
# Split the data into training and testing sets for %20 for test %80 for trainingn , also add a random state for reproductibality
X_train, X_test, y_train, y_test = train_test_split(
    ingredients_sparse_csr, 
    region_labels, 
    test_size=0.2, ## can be changed to to increase test data or train data
    stratify=region_labels,
    random_state=41  #might be deleted to achieve complete randomness
)

#train a logistic regression model
model1 = LogisticRegression(max_iter=1000, solver='liblinear')
model1.fit(X_train, y_train)

#tahmint
y_pred = model1.predict(X_test)
#reports to be used in dta visulaization of the model
report = classification_report(y_test, y_pred, target_names=all_regions, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred, labels=all_regions)


# add all data sets and turn them into a dataframe
scores_data = pd.read_csv("health_scores_of_regions.csv")
recip_data = pd.read_csv('recipe.csv')
salaries_data = pd.read_csv("recipes_with_region.csv").drop_duplicates().drop(columns="ID")
country_stats_data = pd.read_csv("combined_aspects.csv")
country_stats_df = pd.DataFrame(country_stats_data)
correlation_data = pd.read_csv("averages_data.csv")
correlation_data = correlation_data.dropna()

# Process the salaries dataset to create region_ingredient_counts and region_dish_counts
region_dish_counts = {}
region_ingredient_counts = {}
#count every occurence of ingredient for a certain region then divide it into a number of dished to evaluate the frequency of a certian ingredient
for index, row in salaries_data.iterrows():
    region = row['region']
    ingredients = eval(row['ingredients'])  # conver the string list into actual lsit
    if region not in region_ingredient_counts:
        region_ingredient_counts[region] = {}
        region_dish_counts[region] = 1
    for ingredient in ingredients:
        if ingredient not in region_ingredient_counts[region]:
            region_ingredient_counts[region][ingredient] = 0
        region_ingredient_counts[region][ingredient] += 1
    region_dish_counts[region] += 1

#create  af frequency ingredient, region dataframe
data = []
for region, ingredients in region_ingredient_counts.items():
    for ingredient, count in ingredients.items():
        data.append({'Region': region, 'Ingredient': ingredient, 'Count': count})
freq_data = pd.DataFrame(data)
total_ingredient_counts = freq_data.groupby('Ingredient')['Count'].sum()

# Filter ingredients that appear frequently
frequent_ingredients = total_ingredient_counts[total_ingredient_counts >= 20].index
freq_data = freq_data[freq_data['Ingredient'].isin(frequent_ingredients)]
freq_data["Frequency"] = freq_data.apply(lambda row: row["Count"] / region_dish_counts[row["Region"]] * 100, axis=1)

#  bar chart for health score of the regions
def create_bar_chart(dataframe):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dataframe['region'],
        y=dataframe['health_score'],
        marker=dict(color='rgb(26, 118, 255)')
    ))
    fig.update_layout(
        title='Average Health Scores by Region',
        xaxis=dict(title='Region'),
        yaxis=dict(title='Average Health Score'),
        margin=dict(l=40, r=0, t=40, b=40),
        height=550,# set a fixed height of 550 for every graph
        width=1000,  
        autosize=True  
    )
    return fig

# Initialize the Dash app with Bootstrap CSS for better app design 
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# Extract unique generic names for the dropdown menu to be used in the network graph
generic_names = recip_data['generic_name'].dropna().unique()
generic_options = [{'label': name, 'value': name} for name in generic_names]

# Define dropdown options for ingredients
ingredient_options = [{'label': ingredient, 'value': ingredient} for ingredient in freq_data['Ingredient'].unique()]

# main layout of the app
app.layout = html.Div([
    html.Div([
        html.H2("Culinary Cartography", style={"color": "white"}),
        html.Nav([
            dcc.Link('Home', href='/', className='navbar-link'),
            dcc.Link('Ingredient Network Graph', href='/network', className='navbar-link'),
            dcc.Link('Ingredient Occurrence Frequency Heatmap', href='/heatmap', className='navbar-link'),
            dcc.Link('Country Statistics Map', href='/country_map', className='navbar-link'),
            dcc.Link('Correlation Graphs', href='/correlation', className='navbar-link'),
            dcc.Link('Health Scores by Region', href='/bar_chart', className='navbar-link'),  
            dcc.Link('Model Performance', href='/model_performance', className='navbar-link')  
        ], className='navbar')
    ]),
    
    html.Div([
        dcc.Location(id='url', refresh=False),
        html.Div(id='page-content', className='main-content container')
    ])
])

# home page introdutcion , who did the project info about the project etc.
home_page = html.Div([
    html.H1("Culinary Cartography", className="header-title"),
    html.P("Welcome to the Culinary Cartography Dashboard. Here you can explore various graphs and visualizations related to culinary data.", className="header-description"),
    html.Div(" This project has done by Emirhan Oğuz and Ahmet Ergin for PROJ 201 course. Aim of this project is to provide some " \
             "aspects regarding the possible relations between recipes and ingredients in terms of correlations " \
                "such as IQ, Calorie Supply, Life expectancy, GDP per capita, healht score of the recipes of the region. Also a Machine Learning model is trained and " \
                    "used to estimate the origin of the recipe based on ingredients.", className="project-info"),
])
#network graph with dropdown option of generic names 
network_page = html.Div([
    html.Label("Select Generic Name:", className='label'),
    dcc.Dropdown(
        id='generic-name-dropdown',
        options=generic_options,
        value=generic_names[0],
        className='dash-dropdown'
    ),
    dcc.Graph(id='network-graph', className='graph-container')
])
#heatmap for ingredient frequncy for a certain regionm
heatmap_page = html.Div([
    html.Label("Select Ingredients:", className='label'),
    dcc.Dropdown(
        id='ingredient-dropdown',
        options=ingredient_options,
        value=['soy sauce'],
        multi=True,
        className='dash-dropdown'
    ),
    dcc.Graph(id='heatmap', className='graph-container')
])
#country statistci graphs , for gdp vs calorie, iq vs caloire, life expectancy vs calorie, didn't get what we expected :)
country_map_page = html.Div([
    html.Label("Select Metric:", className='label'),
    dbc.RadioItems(
        id='metric-radio',
        options=[
            {'label': 'GDP per Capita', 'value': 'gdp'},
            {'label': 'Average IQ', 'value': 'average_iq'},
            {'label': 'Life Expectancy', 'value': 'life_expectancy'}
        ],
        value='gdp',
        inline=True,
        className='custom-radio-items'
    ),
    dcc.Graph(id='country-map', className='graph-container')
])
#bar chart layout for health scores
bar_chart_page = html.Div([
    html.H3('Average Health Scores by Region'),
    dcc.Graph(
        id='bar-chart',
        figure=create_bar_chart(scores_data),
        className='graph-container'
    )
])
#layout for correlation graphs
correlation_page = html.Div([
    html.Label("Select Correlation Type:", className='label'),
    dbc.RadioItems(
        id='correlation-radio',
        options=[
            {'label': 'IQ vs. Calorie Supply', 'value': 'iq_calorie'},
            {'label': 'GDP vs. Calorie Supply', 'value': 'gdp_calorie'},
            {'label': 'Life Expectancy vs. Calorie Supply', 'value': 'life_calorie'}
        ],
        value='iq_calorie',
        inline=True,
        className='custom-radio-items'
    ),
    dcc.Graph(id='correlation-graph', className='graph-container')
])
#layout for model performance page, i.e machine learning model classifcatoin bar chart and confusion matrix
model_performance_page = html.Div([
    html.H1("Recipe Region Classification Performance"),
    
    html.Label("Select Graph:"),
    dbc.RadioItems(
        id='model-radio',
        options=[
            {'label': 'Confusion Matrix', 'value': 'confusion_matrix'},
            {'label': 'Classification Report', 'value': 'classification_report'}
        ],
        value='confusion_matrix',
        inline=True,
        className='custom-radio-items'
    ),
    
    dcc.Graph(id='model-graph')
])


# Callback to update the page content based on the url
@app.callback(Output('page-content', 'children'), Input('url', 'pathname'))
def display_page(pathname):
  
    if pathname == '/network':  
        return network_page  
    elif pathname == '/heatmap': 
        return heatmap_page  
    elif pathname == '/country_map':  
        return country_map_page  
    elif pathname == '/correlation':  
        return correlation_page  
    elif pathname == '/bar_chart':  
        return bar_chart_page  
    elif pathname == '/model_performance':  
        return model_performance_page 
    else:



        return home_page  

# Callback to update the network graph based on the selected generic name
@app.callback(
    Output('network-graph', 'figure'),  #
    Input('generic-name-dropdown', 'value')  
)
def update_network_graph(selected_generic_name):
    # Filter the recipe data based on the selected generic name
    filtered_data = recip_data[recip_data['generic_name'] == selected_generic_name]
    G = nx.Graph()  # new graph
    G.add_node(selected_generic_name, size=20, color='red')  

    # Add nodes and edges for each ingredient related to the selected generic name
    for _, row in filtered_data.iterrows():
        G.add_node(row['ingredient'], size=10, color='blue')  
        G.add_edge(selected_generic_name, row['ingredient'])  
    pos = nx.spring_layout(G)  # Compute the layout of the grph
    # Prepare the edge coordinates for visualizaton
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    # Create a trace for the edges
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
    node_x, node_y, node_size, node_color, node_text = [], [], [], [], []##sd
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_size.append(G.nodes[node]['size'])
        node_color.append(G.nodes[node]['color'])
        node_text.append(node)


    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text', text=node_text, hoverinfo='text',
        marker=dict(showscale=False, size=node_size, color=node_color, line_width=2), textposition="top center"
    )

    # Create the figure with the edge and node traces
    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(title='Ingredient Network Graph', titlefont_size=16,  
        showlegend=False,  # Do not show the legend
        hovermode='closest',  margin=dict(b=20, l=5, r=5, t=40),  #.
        annotations=[dict(text="Source: Recipe Dataset",  showarrow=False, xref="paper",   yref="paper",  x=0.005,   y=-0.002  
        )],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),  yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), height=550,  #fixed height size imp
        width=1000, autosize=True   
    ))
    return fig 

# Ccllback to update the heatmap based on the selected ingredients
@app.callback(
    Output('heatmap', 'figure'), 
    Input('ingredient-dropdown', 'value')  
)
def update_heatmap(selected_ingredients):
    # filter the frequency data based on the selected ingredients
    filtered_df = freq_data[freq_data['Ingredient'].isin(selected_ingredients)]
    # create a heatmap trace
    heatmap_data = go.Heatmap(x=filtered_df["Region"],  y=filtered_df["Ingredient"],  z=filtered_df["Frequency"],  colorscale="Rainbow",  showscale=True  
    )
    layout = go.Layout(
        title="Ingredient Frequency For Each Region",  
        xaxis=dict(
            title='Region',  tickmode='array',  tickvals=filtered_df["Region"].unique(),  ticktext=filtered_df["Region"].unique(),tickangle=90, tickfont=dict(size=10) ),yaxis=dict(
            title='Ingredient',  
            tickfont=dict(size=10), 
            automargin=True  
        ),
        margin=dict(l=150, r=0, t=40, b=200),  height=600,  width=1000,  autosize=True 
    )
    return {'data': [heatmap_data], 'layout': layout} 

# Callback to update the country map based on the selected metric
@app.callback(
    Output('country-map', 'figure'),  
    Input('metric-radio', 'value') 
)
def update_country_map(selected_metric):
    fig = go.Figure(data=go.Choropleth(locations=country_stats_df['country'], locationmode='country names', z=country_stats_df[selected_metric],  colorscale=[[0, 'yellow'], [0.5, 'orange'], [1, "red"]],   colorbar_title=selected_metric.replace('_', ' ').title(),  
    ))
    # Define the layout for the country map
    fig.update_layout(
        title_text=f'{selected_metric.replace("_", " ").title()} by Country',  
        geo=dict(
            showframe=False,  showcoastlines=False, projection_type='equirectangular'  
        ),
        margin=dict(l=0, r=0, t=40, b=0), 
        height=550, 
        width=1000,  
        autosize=True   
    )
    return fig 

# Callback to update the correlation graph based on the selected correlation type
@app.callback(
    Output('correlation-graph', 'figure'), 
    Input('correlation-radio', 'value')
)

#   NOT: EKSIK  BOLGELER TAMAMLA
def update_correlation_graph(selected_correlation):
    # Extended color map for different regions
    color_map = {
        "Rest Africa": 'blue',
        "South American": 'green',
        "Australian": 'red',
        "Deutschland": 'purple',
        "Indian Subcontinent": 'orange',
        "Eastern European": 'brown',
        "Belgian": 'pink',
        "Canadian": 'gray',
        "Chinese and Mongolian": 'magenta',
        "Central American": 'yellow',
        "Caribbean": 'lightblue',
        "Scandinavian": 'lightgreen',
        "Middle Eastern": 'lightcoral',
        "French": 'navy',
        "Greek": 'olive',
        "Southeast Asian": 'teal',
        "Irish": 'lime',
        "Italian": 'maroon',
        "Japanese": 'black',
        "Northern Africa": 'salmon',
        "Mexican": 'orchid',
        "Spanish and Portuguese": 'sienna',
        "Thai": 'violet',
        "US": 'turquoise',
        "UK": 'plum',
        "Korean": 'azure'
    }

    #R^2 value importante
    def create_correlation_plot(x, y, x_title, y_title, title):
        X = sm.add_constant(x)  
        model = sm.OLS(y, X).fit()  
        predictions = model.predict(X)  
        r_squared = model.rsquared 

        fig = go.Figure()  

        # Adding scatter points colored by region, 
        for region, color in color_map.items():  
            region_data = correlation_data[correlation_data['region123'] == region]  
            fig.add_trace(go.Scatter(
                x=region_data[x.name],  
                y=region_data[y.name],  
                mode='markers',   
                name=region,  
                text=region_data['country'], 
                hoverinfo='text'  
            ))

        # Adding the regression line, i.e fit line aka r^2, important
        fig.add_trace(go.Scatter(
            x=x, y=predictions, mode='lines', name=f'Fit Line (R^2={r_squared:.2f})', line=dict(color='black')  # Add the regression line
        ))

        # Define the layout for the correlation plot
        fig.update_layout(
            title=title,  xaxis_title=x_title,  yaxis_title=y_title,  margin=dict(l=40, r=0, t=40, b=0), height=550, 
            width=1000, 
            autosize=True,  legend=dict(title="Region")  
        )
        return fig  


    if selected_correlation == 'iq_calorie':  # IQ vs Calorie Supply
        fig = create_correlation_plot(
            correlation_data['avg_calorie'], correlation_data['average_iq_region'],
            'Average Calorie Supply', 'Average IQ', 'Correlation between Average IQ and Calorie Supply'
        )
    elif selected_correlation == 'gdp_calorie':  # GDP vs Calorie supply
        fig = create_correlation_plot(
            correlation_data['avg_calorie'], correlation_data['average_gdp_region'],
            'Average Calorie Supply', 'GDP', 'Correlation between GDP and Calorie Supply'
        )
    else:  #  Life Expectancy vs Calorie Supply
        fig = create_correlation_plot(
            correlation_data['avg_calorie'], correlation_data['average_life_expectancy_region'],
            'Average Calorie Supply', 'Life Expectancy', 'Correlation between Life Expectancy and Calorie Supply'
        )
    return fig
# Callback to update the model performance graph based on the selected graph type
@app.callback(
    Output('model-graph', 'figure'),  
    [Input('model-radio', 'value')]  
)
def update_model_performance_graph(selected_graph):
    if selected_graph == 'confusion_matrix':  # Confusion Matrix, heatmao actually
        
        conf_matrix_fig = go.Figure(data=go.Heatmap(z=conf_matrix,  x=all_regions,  y=all_regions, 
            colorscale='Rainbow'  
        ))
        conf_matrix_fig.update_layout(
            title='Confusion Matrix', xaxis_title='Predicted Label', yaxis_title='True Label',  xaxis=dict(tickmode='array', tickvals=list(range(len(all_regions))), ticktext=all_regions), 
            yaxis=dict(tickmode='array', tickvals=list(range(len(all_regions))), ticktext=all_regions), height=550, width=1000, 
            autosize=True 
        )
        return conf_matrix_fig  
    elif selected_graph == 'classification_report': 
        # Prepare data for the classification report
        precision = [report[label]['precision'] for label in all_regions]  
        recall = [report[label]['recall'] for label in all_regions]  
        f1_score = [report[label]['f1-score'] for label in all_regions]  
        # Create a bar chart for the classification report, for f1 score recall score precision scoe
        classification_report_fig = go.Figure()
        classification_report_fig.add_trace(go.Bar(x=all_regions, y=precision, name='Precision'))  
        classification_report_fig.add_trace(go.Bar(x=all_regions, y=recall, name='Recall'))  
        classification_report_fig.add_trace(go.Bar(x=all_regions, y=f1_score, name='F1-Score'))  

        # Define the layout for the classification report
        classification_report_fig.update_layout(
            title='Classification Report',  barmode='group',  xaxis_title='Region',   yaxis_title='Score',  height=550,  
            width=1000,  
            autosize=True  
        )
        return classification_report_fig 
#son
if __name__ == '__main__': 
    app.run_server(debug=True)  # Run the Dash app 
