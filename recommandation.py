import streamlit as st

# Librairies
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#from tensorflow.keras.models import Sequential, load_model
#from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
#from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.callbacks import ModelCheckpoint

#from tensorflow.keras.models import load_model


##############
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from wordcloud import WordCloud
from nltk.corpus import stopwords

import nltk
from nltk.tokenize import word_tokenize
from unidecode import unidecode
import unicodedata
import re
import math

# Initialiser la variable des mots vides
stop_words = list(stopwords.words('french'))

import spacy
from spacy.lang.fr.stop_words import STOP_WORDS

stop_words_fr = list(spacy.lang.fr.stop_words.STOP_WORDS)

wc = WordCloud(background_color="white", max_words=100, stopwords=stop_words, max_font_size=50, random_state=42)

stop_words.extend('nbsp'.split())
stop_words_fr.extend('nbsp'.split())

import time

import selenium.webdriver
from bs4 import BeautifulSoup as bs
import requests


import pickle

###############
###############
###############
###############
@st.cache_data
def df_slim():
    df_slim = pd.read_csv('data/df_slim.csv')

    return df_slim

@st.cache_data
def blog_post_fr():
    blog_post_fr = pd.read_json('data/blog_post_fr.json')

    return blog_post_fr[['article_id', 'article_title', 'article_has_offers', 'article_content', 'article_tags', 'article_url']]


@st.cache_data
def job_posting():
    post = pd.read_json('data/job_posting_cleaned.json')
    post.columns = ['post_id','title_offre', 'description_offre', 'published_at_offre', 'competences_offre', 'description_offre_cleaned']

    return post





def freework_url_check(url):
    url_prefixe = 'https://www.free-work.com'
        
    res = requests.head(url, allow_redirects=False)

    if res.status_code == 200:
 
        driver = selenium.webdriver.Chrome()
        # driver = selenium.webdriver.Firefox() => a choisir selon son navigateur Web
    
        driver.get(url)
        time.sleep(1) 

        soup = bs(driver.page_source, "lxml")

        div_job_posting = soup.find("div", class_= 'divide-y')

        job_posting_order = []
        job_posting_title = []
        job_posting_url = []
        job_posting_description = []


        if div_job_posting is not None:

            for index, div in enumerate(div_job_posting):
                div_item = div.find("a")

                if div_item is not None :
                    job_posting_order.append(index + 1)
                    job_posting_title.append(div_item.text.strip())
                    job_posting_url.append(div_item['href'])

                    description = []
                    page = requests.get(url_prefixe + div_item['href'])
                    soup = bs(page.content, "lxml")
 
                    divs = soup.find_all('div', class_='prose-content')

                    for div in divs:
                        description.append(div.text)

                    job_posting_description.append(' '.join(description))




        df = pd.DataFrame(list(zip(job_posting_order, job_posting_title, job_posting_description, job_posting_url)),
                       columns =['order_offre', 'title_offre', 'description_offre','url_offre'])

         
    else:
        print('Impossible de trouver l\'Url sur www.free-work.com')
        
        df = pd.DataFrame()
        
        
    
    return df



blog = blog_post_fr()
post = job_posting()
df_slim = df_slim()





###############
###############
###############
###############
count_vectorizer = CountVectorizer(stop_words=stop_words_fr, strip_accents='ascii')







st.title("Freework")
st.sidebar.title("Sommaire")
pages=["Classification ML", "Recommandation"]
page=st.sidebar.radio("Aller vers", pages)



####################################################################
####################################################################
####################################################################
if page == pages[0] : 

    lr_model_loaded = pickle.load(open("models/best_lr_model.pkl", 'rb'))
    xgb_model_loaded = pickle.load(open("models/best_xgb_model.pkl", 'rb'))

    def prediction(classifier):
        if classifier == 'Logistic Regression':
            clf = lr_model_loaded
        elif classifier == 'XGBoost':
            clf = xgb_model_loaded
        
        return clf

    def scores(clf, choice):
        if choice == 'Accuracy':
            return clf.score(X_test, y_test)
        elif choice == 'Confusion matrix':
            return pd.crosstab(y_test, clf.predict(X_test), rownames=['True'], colnames=['Predict'])

    st.write("### Classification Machine Learning")

    st.dataframe(df_slim)

    text_cols = ['article_tags_word_preproc', 'article_cont_word_preproc', 'article_title_word_preproc', 'title_word_preproc', 'description_offre_stem', 'competences']
    numeric_cols = ['article_title_nb_car', 'article_title_nb_mot', 'article_title_lg_mot_moyen', 'article_title_word_preproc_nb_mot']
    target_col = 'target'

    # Combiner les colonnes textuelles en une seule
    X_text = df_slim[text_cols].apply(lambda x: ' '.join(x.dropna()), axis=1)
    y = df_slim[target_col]

    # Fixer max_features à 200
    max_features = 200

    # Vectorisation des données textuelles avec la valeur fixée de max_features
    tfidf = TfidfVectorizer(max_features=max_features)
    text_features = tfidf.fit_transform(X_text).toarray()

    # Extraire les colonnes numériques
    numeric_features = df_slim[numeric_cols].values

    # Standardiser les colonnes numériques
    scaler = StandardScaler()
    numeric_features_standardized = scaler.fit_transform(numeric_features)

    # Concaténer les caractéristiques textuelles et numériques
    X_combined = np.hstack((text_features, numeric_features_standardized))

    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)


    
    choix = ['Logistic Regression', 'XGBoost']
    option = st.selectbox('Choix du modèle', choix)
    st.write('Le modèle choisi est :', option)


    clf = prediction(option)
    display = st.radio('Que souhaitez-vous montrer ?', ('Accuracy', 'Confusion matrix'))
    if display == 'Accuracy':
        st.write(scores(clf, display))
    elif display == 'Confusion matrix':
        st.dataframe(scores(clf, display))
    








####################################################################
####################################################################
####################################################################
if page == pages[1] : 
    st.write("### Recommandation")

    event = st.dataframe(
        blog,
        on_select= 'rerun',  # activate selection
        selection_mode=['single-row']
    )

    #####
    # Selectbox
    #####
    choix = ['Description','Titre']
    select = st.selectbox('Quelle feature de l\'offre pour le calcul de la distance de la similarité Cosinus : ', choix)

    selected_info = event['selection']
    #st.text(selected_info)

    
    df_selected = blog.iloc[selected_info['rows']]
    #st.write(df_selected)


    try:
        

        article_titre = df_selected['article_title'].values[0]
        article_content = df_selected['article_content'].values[0]
        article_url = df_selected['article_url'].values[0]

        st.header('**Résultat**')


        #####
        # Nuage de mot
        #####
        fig = plt.figure()
        wc.generate(article_content)           # "Calcul" du wordcloud
        plt.imshow(wc) # Affichage
        #plt.title(f'Nuage de mots de l\'article \n"{article_titre}"')
        plt.axis('off')
        st.pyplot(fig)



        

        #####
        # CountVectorizer
        #####
        article_countvec  = count_vectorizer.fit_transform([article_content])

        if select == 'Description':
            description_offre_countvec = count_vectorizer.transform(post['description_offre_cleaned'])
            cs_description_offre_countvec = np.array(cosine_similarity(article_countvec, description_offre_countvec)).flatten()
            post['cs_desc_countvec'] = cs_description_offre_countvec
            select_name = 'desc'
        elif select == 'Titre':       
            title_offre_countvec = count_vectorizer.transform(post['title_offre'])
            cs_title_offre_countvec = np.array(cosine_similarity(article_countvec, title_offre_countvec)).flatten()
            post['cs_title_countvec'] = cs_title_offre_countvec
            select_name = 'title'

        post['order_offre'] = post['cs_'+select_name+'_countvec'].rank(method="first", ascending=False)

        #####
        # Recherche du blog sur www.free-work.com
        #####
        df_freework = freework_url_check(article_url)

        
        if not df_freework.empty:
            if select == 'Description':
                description_freework_countvec = count_vectorizer.transform(df_freework['description_offre'])
                cs_description_freework_countvec = np.array(cosine_similarity(article_countvec, description_freework_countvec)).flatten()
                df_freework['cs_desc_countvec'] = cs_description_freework_countvec
            elif select == 'Titre':       
                title_freework_countvec = count_vectorizer.transform(df_freework['title_offre'])
                cs_title_freework_countvec = np.array(cosine_similarity(article_countvec, title_freework_countvec)).flatten()
                df_freework['cs_title_countvec'] = cs_title_freework_countvec
            
            
            
            
            ####
            # Graphiques de comparaison entre les offres de freework et le système de recommnandations
            ####
            df1 = df_freework[['order_offre', 'title_offre', 'cs_'+select_name+'_countvec']]
            df1['origine'] = 'freework'

            df2 = post[['title_offre', 'cs_'+select_name+'_countvec']].sort_values(by='cs_'+select_name+'_countvec', ascending=False).head(df_freework.shape[0])
            df2['order_offre'] = df2['cs_'+select_name+'_countvec'].rank(method="first", ascending=False)
            df2['origine'] = 'recommandation'

            df_graph = pd.concat([df1,df2])
            df_melt = pd.melt(df_graph, id_vars= ['order_offre', 'origine'], value_vars= ['cs_'+select_name+'_countvec'], value_name='score')
            df_melt = df_melt.drop('variable', axis = 1)
            
            
            # Bar side by side version Seaborn
            fig = plt.figure()
            sns.barplot(x='order_offre', y='score', hue='origine', data = df_melt)
            plt.xlabel('Offre n°')
            #st.pyplot(fig)

            # Bar side by side version Plotly Express
            fig = px.bar(x='order_offre', y='score', color = 'origine', data_frame = df_melt, barmode= "group")
            st.plotly_chart(fig)

        #####
        # DataFrames CountVectorizer
        #####

        st.markdown('**Ranking Freework**')
        st.dataframe(df_freework)

        st.markdown('**Ranking Système de recommandation**')
        st.dataframe(post[post['cs_'+select_name+'_countvec'] > 0.1][['order_offre', 'title_offre', 'description_offre_cleaned','cs_'+select_name+'_countvec']].sort_values(by='order_offre').reset_index(drop = True))

    except:
        st.text("Sélectionner un article")

    
    
    
    







    if len(selected_info['rows']) and len(selected_info['columns']):
        st.markdown('**Cell value at intersection**')
        cell_value = blog.loc[selected_info['rows'][0], selected_info['columns'][0]]
        st.text(cell_value)









