import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
import altair as alt
import requests
import streamlit.components.v1 as components
import time
import plotly.graph_objs as go
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from plotly.subplots import make_subplots
import PIL.Image
from PIL import Image
import seaborn as sns

#Import bases de données
meteor_nasa= pd.read_csv("https://raw.githubusercontent.com/MaximeNICASTRO/Projet-Hackathon/main/meteor_nasa")
all_cr = pd.read_csv("https://raw.githubusercontent.com/MaximeNICASTRO/Projet-Hackathon/main/all_cr")
dfh = pd.read_csv("https://raw.githubusercontent.com/MaximeNICASTRO/Projet-Hackathon/main/nasa.csv")
orbits = pd.read_csv("https://raw.githubusercontent.com/MaximeNICASTRO/Projet-Hackathon/main/orbits-for-near-earth-asteroids-neas.csv", sep=';')

st.sidebar.title("")
st.sidebar.write("Que souhaitez vous faire aujourd'hui?")
choice = st.sidebar.selectbox("", ('Introduction','Visualisations','Prédiction', 'Conclusion', 'Sources et outils'))

if choice == 'Introduction':
    st.markdown("""  <style> .reportview-container { background:
    url("https://static.actu.fr/uploads/2021/03/asteroid-4369511-1920.jpg")}
    </style> """, unsafe_allow_html=True)

    components.html("<body style='color:white;font-family:Andale Mono;; font-size:60px; text-align: center'><b>Sommes-nous en danger?</b></body>")
    
    st.markdown("***")
    components.html("<body style='color:white;font-family:Andale Mono;; font-size:25px; text-align: center; padding: 1px'><b>66 Millions d'années</b></body>")
    components.html("<body style='color:white;font-family:Andale Mono;; font-size:25px; text-align: center; padding: 1px'><b>1'700'000 Astéroïdes</b></body>")
    components.html("<body style='color:white;font-family:Andale Mono;; font-size:25px; text-align: center; padding: 1px'><b>44 Tonnes</b></body>")

elif choice == 'Visualisations':
    st.markdown("""  <style> .reportview-container { background:
    url("https://static.actu.fr/uploads/2021/03/asteroid-4369511-1920.jpg")}
    </style> """, unsafe_allow_html=True)
    
    components.html("<body style='color:white;font-family:Andale Mono;; font-size:30px; text-align: center; padding: 20px'><b>De l'espace à la terre...</b></body>")

    years = []
    for i in range(2013, 1650,-1):
        years.append(i)
    year_to_filter = st.select_slider('Choisissez la période à étudier:',options=years)
    filtered_data = meteor_nasa[meteor_nasa['Année'] >= year_to_filter]

    #Premier graph
    st.subheader('Où tombent les météorites?')
    fig = px.scatter_mapbox(filtered_data, lat="reclat", lon="reclong",
                        #hover_name="GeoLocation", hover_data=["Nom", "Poids (gr)"],
                        #color = "Poids (gr)", 
                        color_discrete_sequence=["#bc3441"],
                        zoom=1, width=800, height=400,
                        #size = "Poids (gr)",
                        #size_max =100,
                        center=go.layout.mapbox.Center(lat=27,lon=0))
    fig.update_layout( mapbox_style="white-bg",mapbox_layers=[{"below": 'traces',"sourcetype": "raster",
            "sourceattribution": "United States Geological Survey",
            "source": ["https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"]}])
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.write(fig)

    #Deuxieme graph
    st.subheader('La composition des météorites')
    fig2 = px.scatter(data_frame = filtered_data, x= 'Année', y="Poids (gr)", color="Catégories", 
                #size = "Poids (gr)",
                hover_name="Nom",
                color_discrete_sequence=["red", "yellow", "blue"]
                )
    fig2.update_yaxes(range = [0,4000000])
    fig2.update_layout({'plot_bgcolor': 'rgba(0,0,0,0.2)',
                       'paper_bgcolor': 'rgba(0,0,0,0)', })
    st.write(fig2)

    def convert_column(x):
        if x=="Roche":
              x=1
        elif x=="Roche-Fer":
              x=2
        else:
            return 3
        return x

    meteor_nasa['Composition'] = meteor_nasa.Catégories.apply(convert_column)

    meteor_nasa['Composition_en_pourcentage'] = (meteor_nasa['Composition'] / meteor_nasa['Composition'].sum()) * 100

    Composition_Catégorie = pd.pivot_table(meteor_nasa, index = 'Catégories', values = 'Composition_en_pourcentage', aggfunc='sum')
    
    Poids_moyen_Catégorie = pd.pivot_table(meteor_nasa, index = 'Catégories', values = 'Poids (gr)', aggfunc='mean')
    #Poids_moyen_Catégorie
    Composition_Catégorie =np.round(pd.pivot_table(meteor_nasa, index = 'Catégories', values = 'Composition_en_pourcentage', aggfunc='sum'),2)


    col1, col2, col3, col4 = st.beta_columns(4)
    with col1:
        st.write('.')
        st.write("Répartition")

    with col2:
        st.write('Météorites en fer:')
        st.write(str(Composition_Catégorie['Composition_en_pourcentage'].iloc[0]),"%")
        

    with col3:
        st.write('Météorites en roche:')
        st.write(str(Composition_Catégorie['Composition_en_pourcentage'].iloc[1]),"%")

    with col4:
        st.write('Météorites mixtes:')
        st.write(str(Composition_Catégorie['Composition_en_pourcentage'].iloc[2]),"%")

    Poids_moyen_Catégorie = np.round((pd.pivot_table(meteor_nasa, index = 'Catégories', values = 'Poids (gr)', aggfunc='mean')/1000),2)

    col1, col2, col3, col4= st.beta_columns(4)
    with col1:
        #st.write('Météorites en fer:')
        st.write('Poids moyen')

    with col2:
        #st.write('Météorites en fer:')
        st.write(str(Poids_moyen_Catégorie['Poids (gr)'].iloc[0]),"kg")
    
    with col3:
        #st.write('Météorites en roche:')
        st.write(str(Poids_moyen_Catégorie['Poids (gr)'].iloc[1]),"kg")

    with col4:
        #st.write('Météorites mixtes:')
        st.write(str(Poids_moyen_Catégorie['Poids (gr)'].iloc[2]),"kg")

    #Troisième graph
    st.subheader("Le rescencement des plus gros cratères")
    fig3 = px.scatter_mapbox(all_cr, lat="Latitude",lon="Longitude", 
                        hover_name="Crater Name",hover_data=["Age (Ma)"], 
                        color = "Diameter (km)",
                        #color_continuous_scale = 'viridis',
                        #color_discrete_sequence=["fuchsia"],
                        color_continuous_scale=px.colors.sequential.Viridis,
                        zoom=1, width=800, height=400,
                        size = 'Diameter (km)',
                        size_max =60,center=go.layout.mapbox.Center(lat=27,lon=0)
                       )
    fig3.update_layout( mapbox_style="white-bg",mapbox_layers=[{"below": 'traces',"sourcetype": "raster",
            "sourceattribution": "United States Geological Survey",
            "source": ["https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"]}])
    
    #fig3.update_layout(mapbox_style="open-street-map")
    fig3.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.write(fig3)

elif choice == "Prédiction":
    st.markdown("""  <style> .reportview-container { background:
    url("https://static.actu.fr/uploads/2021/03/asteroid-4369511-1920.jpg")}
    </style> """, unsafe_allow_html=True)

    components.html("<body style='color:white;font-family:Andale Mono; font-size:30px; text-align: center; padding: 1px'><b>Comment prédire la dangerosité d'un astéroïde</b></body>")

    st.subheader("*It’s 100 percent certain we’ll be hit [by a devastating asteroid], but we’re not 100 percent certain when!* B612 Foundation")
    st.markdown("***")


    orbits['Last observation'] = pd.to_datetime(orbits['Last observation'])
    fig8 = px.histogram(orbits, x = 'Last observation', nbins = 42, color_discrete_sequence=["#86eeba"])

    fig8.update_layout({'plot_bgcolor': 'rgba(0,0,0,0.2)',
                   'paper_bgcolor': 'rgba(0,0,0,0)', })
    fig8.update_layout(
              title={'text' : "Observations des astéroïdes les plus proches de la Terre",'x':0.5,})
    st.write(fig8)
   
    fig7 = px.histogram(dfh, x = 'Type', color='Type')
    
    fig7.update_layout({'plot_bgcolor': 'rgba(0,0,0,0.2)',
                       'paper_bgcolor': 'rgba(0,0,0,0)', })
    fig7.update_layout(
                  title={
                        'text' : "Typologie des astéroïdes observés jusqu'en 2017",
                        'x':0.5,
                        }

                 )
    st.write(fig7)
    



    st.markdown("***")
    components.html("<body style='color:white;font-family:Andale Mono; font-size:25px; text-align: center; padding: 1px'><b>Présentation du modèle de Decision Tree</b></body>")

    st.subheader("Les ceintures d'astéroïdes")
    im = Image.open(requests.get(
        "https://cdn.zmescience.com/wp-content/uploads/2017/09/asteroid-belt.jpg", stream=True).raw)
    st.image(im, use_column_width=True)
    st.subheader("Le modèle")
    im = Image.open(requests.get(
        "https://cdn.xoriant.com/cdn/ff/weqpbrtpXGjLpVQ_X-gWqsFlvjAxpv5Wv3xNW0A4vuQ/1602007254/public/2020-10/a-decisionTreesforClassification-AMachineLearningAlgorithm.jpg", stream=True).raw)
    st.image(im, use_column_width=True)

elif choice == 'Conclusion':
    st.markdown("""  <style> .reportview-container { background:
    url("https://static.actu.fr/uploads/2021/03/asteroid-4369511-1920.jpg")}
    </style> """, unsafe_allow_html=True)

    components.html("<body style='color:white;font-family:Andale Mono; font-size:30px; text-align: center; padding: 1px'><b>La Planetary Defense Conférence veille sur nous!</b></body>")

    im = Image.open(requests.get(
        "https://cdn.radiofrance.fr/s3/cruiser-production/2017/01/e5014e5b-c0d4-4d97-84d0-cdd7b456e3e0/838_asteroide.jpg", stream=True).raw)
    st.image(im, use_column_width=True)

    components.html("<body style='color:white;font-family:Andale Mono; font-size:25px; text-align: center; padding: 1px'><b>Impressionnant, n’est-ce pas?</b></body>")
    components.html("<body style='color:white;font-family:Andale Mono; font-size:25px; text-align: center; padding: 1px'><b>Dorénavant, vous regarderez le ciel différemment...</b></body>")
    st.markdown("***")
    components.html("<body style='color:white;font-family:Andale Mono; font-size:30px; text-align: center; padding: 1px'><b>Merci de votre attention</b></body>")

elif choice == 'Sources et outils':
    st.markdown("""  <style> .reportview-container { background:
    url("https://static.actu.fr/uploads/2021/03/asteroid-4369511-1920.jpg")}
    </style> """, unsafe_allow_html=True)
    st.markdown("***")
    st.header('Notre équipe de Data Analyst')
    col1, col2, col3, col4= st.beta_columns(4)
    with col1:
        st.write('Violaine')

    with col2:
        st.write('Ahlem')

    with col3:
        st.write('Thomas')

    with col4:
        st.write('Maxime')
    st.markdown("***")
    st.header('Sources et bases de données:')
    link = '[DataSet NASA : Météorites](https://data.nasa.gov/api/views/gh4g-9sfh/rows.csv?accessType=DOWNLOAD)'
    st.markdown(link, unsafe_allow_html=True)
    link = '[DataSet NASA : Classification des astéroïdes](https://www.kaggle.com/shrutimehta/nasa-asteroids-classification)'
    st.markdown(link, unsafe_allow_html=True)
    link = '[Portail de données Datastro](https://www.datastro.eu/)'
    st.markdown(link, unsafe_allow_html=True)
    link = '[Center For Near Earth Object Studies](https://cneos.jpl.nasa.gov/)'
    st.markdown(link, unsafe_allow_html=True)
    link = '[Database EarthImpact](http://passc.net/EarthImpactDatabase)'
    st.markdown(link, unsafe_allow_html=True)
    link = '[The Meteoritical Society](https://www.lpi.usra.edu/meteor/)'
    st.markdown(link, unsafe_allow_html=True)
    st.markdown("***")
    st.header('Outils:')

    col1, col2, col3, col4, col5= st.beta_columns(5)
    with col1:
        st.write('*','Python')
        im = Image.open(requests.get(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0a/Python.svg/600px-Python.svg.png", stream=True).raw)
        st.image(im, use_column_width=True, width=200)

    with col2:
        st.write('*','VSCode')
        im = Image.open(requests.get(
        "https://img1.daumcdn.net/thumb/R800x0/?scode=mtistory2&fname=https%3A%2F%2Ft1.daumcdn.net%2Fcfile%2Ftistory%2F9920003359A8E33517", stream=True).raw)
        st.image(im, use_column_width=True, width=200)

    with col3:
        st.write('*','Plotly')
        im = Image.open(requests.get(
        "https://avatars.githubusercontent.com/u/5997976?s=200&v=4", stream=True).raw)
        st.image(im, use_column_width=True, width=200)

    with col4:
        st.write('*','Scikit Learn')
        im = Image.open(requests.get(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/1200px-Scikit_learn_logo_small.svg.png", stream=True).raw)
        st.image(im, use_column_width=True, width=200)

    with col5:
        st.write('*','Streamlit')
        im = Image.open(requests.get(
        "https://camo.githubusercontent.com/07618dd26f0bf8936cc444cfbe6f7ddcd1dc4a78196a51de0d4122693b7f1274/68747470733a2f2f6173736574732e776562736974652d66696c65732e636f6d2f3564633362343764646336633063326131616637346164302f3565313831383238626139663965393262366562633665375f5247425f4c6f676f6d61726b5f436f6c6f725f4c696768745f42672e706e67", stream=True).raw)
        st.image(im, use_column_width=True, width=200)


