
# cd C:\Users\User\Downloads\app
# call: python -m streamlit run LEPAn.py

# https://share.streamlit.io/streamlit/example-app-interactive-table/main
# https://streamlit.io/gallery


# imports
import streamlit as st
from PIL import Image
from collections import Counter
import pandas as pd
pd.set_option("max_colwidth", 200)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
plt.style.use("seaborn-talk")

import plotly.express as px
import plotly
import plotly.graph_objects as go
import wordcloud
from wordcloud import WordCloud, STOPWORDS


# **************************************************



# functions

def user_rhetoric_v2(dataframe, source_column = 'Source', ethos_col = 'ethos_name',
                  pathos_col = 'pathos_name', logos_col = 'logos_name'):
  '''
  '''
  import warnings
  from pandas.core.common import SettingWithCopyWarning
  warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

  sources_list = dataframe[dataframe[source_column] != 'nan'][source_column].unique()
  metric_value = []
  users_list = []

  map_ethos_weight = {'attack':-1, 'neutral':0, 'support':1}
  map_pathos_weight = {'negative':-1, 'neutral':0, 'positive':1}
  map_logos_weight = {'attack':-0.5, 'neutral':0, 'support':0.5}

  for u in sources_list:
    users_list.append(str(u))
    df_user = dataframe[dataframe[source_column] == u]

    ethos_pathos_logos_user = 0

    df_user_rhetoric = df_user.groupby([str(pathos_col), str(logos_col), str(ethos_col)], as_index=False).size()

    # map weights
    df_user_rhetoric[pathos_col] = df_user_rhetoric[pathos_col].map(map_pathos_weight)
    df_user_rhetoric[ethos_col] = df_user_rhetoric[ethos_col].map(map_ethos_weight)
    df_user_rhetoric[logos_col] = df_user_rhetoric[logos_col].map(map_logos_weight)

    ethos_pathos_logos_sum_ids = []

    for id in df_user_rhetoric.index:
      ethos_pathos_val = np.sum(df_user_rhetoric.loc[id, str(pathos_col):str(ethos_col)].to_numpy())
      ethos_pathos_val = ethos_pathos_val * df_user_rhetoric.loc[id, 'size']
      ethos_pathos_logos_sum_ids.append(ethos_pathos_val)

    ethos_pathos_logos_user = np.sum(ethos_pathos_logos_sum_ids)
    metric_value.append(int(ethos_pathos_logos_user))

  df = pd.DataFrame({'user': users_list, 'rhetoric_metric': metric_value})
  return df


def make_word_cloud(comment_words, width, height, colour = "black", colormap = "brg"):
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(max_words=250, colormap=colormap, width = width, height = height,
                background_color ='white',
                min_font_size = 10).generate(comment_words) # , stopwords = stopwords

    fig_x = width / 100
    fig_y = height / 100
    fig, ax = plt.subplots(figsize = (fig_x, fig_y), facecolor = colour)
    #plt.figure(figsize = (fig_x, fig_y), facecolor = colour)
    ax.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.show()
    return fig


def prepare_cloud_lexeme_data(data_neutral, data_support, data_attack):
  #import spacy
  #nlp = spacy.load("en_core_web_lg")
  #stops = nlp.Defaults.stop_words
  #stops = [s for s in stops if len(s) < 6]
  #print(f"The stop-words list comprises {len(stops)} words.")

  # neutral df
  neu_text = " ".join(data_neutral['clean_Text_lemmatized'].apply(lambda x: " ".join(t for t in set(x.split()))).to_numpy())
  count_dict_df_neu_text = Counter(neu_text.split(" "))
  df_neu_text = pd.DataFrame( {"word": list(count_dict_df_neu_text.keys()),
                              'neutral #': list(count_dict_df_neu_text.values())} )
  df_neu_text.sort_values(by = 'neutral #', inplace=True, ascending=False)
  df_neu_text.reset_index(inplace=True, drop=True)
  #df_neu_text = df_neu_text[~(df_neu_text.word.isin(stops))]

  # support df
  supp_text = " ".join(data_support['clean_Text_lemmatized'].apply(lambda x: " ".join(t for t in set(x.split()))).to_numpy())
  count_dict_df_supp_text = Counter(supp_text.split(" "))
  df_supp_text = pd.DataFrame( {"word": list(count_dict_df_supp_text.keys()),
                              'support #': list(count_dict_df_supp_text.values())} )

  df_supp_text.sort_values(by = 'support #', inplace=True, ascending=False)
  df_supp_text.reset_index(inplace=True, drop=True)
  #df_supp_text = df_supp_text[~(df_supp_text.word.isin(stops))]

  merg = pd.merge(df_supp_text, df_neu_text, on = 'word', how = 'outer')

  #attack df
  att_text = " ".join(data_attack['clean_Text_lemmatized'].apply(lambda x: " ".join(t for t in set(x.split()))).to_numpy())
  count_dict_df_att_text = Counter(att_text.split(" "))
  df_att_text = pd.DataFrame( {"word": list(count_dict_df_att_text.keys()),
                              'attack #': list(count_dict_df_att_text.values())} )

  df_att_text.sort_values(by = 'attack #', inplace=True, ascending=False)
  df_att_text.reset_index(inplace=True, drop=True)
  #df_att_text = df_att_text[~(df_att_text.word.isin(stops))]

  df2 = pd.merge(merg, df_att_text, on = 'word', how = 'outer')
  df2.fillna(0, inplace=True)
  df2['general #'] = df2['support #'] + df2['attack #'] + df2['neutral #']
  df2['word'] = df2['word'].str.replace("'", "_").replace("”", "_").replace("’", "_")
  return df2



def wordcloud_lexeme(dataframe, lexeme_threshold = 90, analysis_for = 'support', cmap_wordcloud = 'crest'):
  '''
  analysis_for:
  'support',
  'attack',
  'both' (both support and attack)

  cmap_wordcloud: best to choose from:
  gist_heat, flare_r, crest, viridis

  '''
  if analysis_for == 'attack':
    #print(f'Analysis for: {analysis_for} ')
    cmap_wordcloud = 'gist_heat'
    dataframe['% lexeme'] = (round(dataframe['attack #'] / dataframe['general #'], 3) * 100).apply(float) # att
  elif analysis_for == 'both':
    #print(f'Analysis for: {analysis_for} ')
    cmap_wordcloud = 'viridis'
    dataframe['% lexeme'] = (round((dataframe['support #'] + dataframe['attack #']) / dataframe['general #'], 3) * 100).apply(float) # both supp & att
  else:
    #print(f'Analysis for: {analysis_for} ')
    dataframe['% lexeme'] = (round(dataframe['support #'] / dataframe['general #'], 3) * 100).apply(float) # supp

  dfcloud = dataframe[(dataframe['% lexeme'] >= int(lexeme_threshold)) & (dataframe['general #'] > 1) & (dataframe.word.map(len)>3)]
  #print(f'There are {len(dfcloud)} words for the analysis of language {analysis_for} with % lexeme threshold equal to {lexeme_threshold}.')

  text = []
  for i in dfcloud.index:
    w = dfcloud.loc[i, 'word']
    w = str(w).strip()
    if analysis_for == 'both':
      n = int(dfcloud.loc[i, 'support #'] + dfcloud.loc[i, 'attack #'])
    else:
      n = int(dfcloud.loc[i, str(analysis_for)+' #']) #  + dfcloud.loc[i, 'attack #']   dfcloud.loc[i, 'support #']+  general
    l = np.repeat(w, n)
    text.extend(l)

  import random
  random.shuffle(text)

  figure_cloud = make_word_cloud(" ".join(text), 1300, 750, '#1E1E1E', str(cmap_wordcloud)) #gist_heat / flare_r crest viridis
  return figure_cloud


# app version
def user_stats_app(dataframe, source_column = 'Source', logos_column = 'logos_name',
               ethos_column = 'ethos_name', pathos_column = 'pathos_name'):

  sources_list = dataframe[dataframe[source_column] != 'nan'][source_column].unique()
  df = pd.DataFrame(columns = ['user', 'text_n',
                               'ethos_n', 'ethos_support_n', 'ethos_attack_n',
                               'pathos_n', 'pathos_negative_n', 'pathos_positive_n',
                               'logos_n', 'logos_support_n', 'logos_attack_n',
                             'ethos_percent', 'ethos_support_percent', 'ethos_attack_percent',
                             'pathos_percent', 'pathos_negative_percent', 'pathos_positive_percent',
                             'logos_percent', 'logos_support_percent', 'logos_attack_percent'])
  users_list = []

  for i, u in enumerate(sources_list):
    users_list.append(str(u))
    df_user = dataframe[dataframe[source_column] == u]
    N_user = int(len(df_user))

    df_user_logos = df_user.groupby(logos_column, as_index = False)["Text"].size()
    try:
      N_ra = int(df_user_logos[df_user_logos[logos_column] == 'support']['size'].iloc[0])
      N_ra = int(N_ra)
    except:
      N_ra = 0

    df_user_ca = df_user.groupby(logos_column, as_index = False)["Text"].size()
    try:
      N_ca = int(df_user_ca[df_user_ca[logos_column] == 'attack']['size'].iloc[0])
      N_ca = int(N_ca)
    except:
      N_ca = 0

    df_user_ethos = df_user.groupby(ethos_column, as_index = False)["Text"].size()
    try:
      N_support = int(df_user_ethos[df_user_ethos[ethos_column] == 'support']['size'].iloc[0])
    except:
      N_support = 0

    try:
      N_attack = int(df_user_ethos[df_user_ethos[ethos_column] == 'attack']['size'].iloc[0])
    except:
      N_attack=0

    df_user_pathos = df_user.groupby(pathos_column, as_index = False)["Text"].size()
    try:
      N_neg = int(df_user_pathos[df_user_pathos[pathos_column] == 'negative']['size'].iloc[0])
    except:
      N_neg = 0

    try:
      N_pos = int(df_user_pathos[df_user_pathos[pathos_column] == 'positive']['size'].iloc[0])
    except:
      N_pos = 0

    counts_list = [N_support+N_attack, N_support, N_attack, N_neg+N_pos, N_neg, N_pos, N_ra+N_ca, N_ra, N_ca]
    percent_list = list((np.array(counts_list) / N_user).round(3) * 100)
    df.loc[i] = [u] + [N_user] + counts_list + percent_list
  return df




def add_spacelines(number=2):
    for i in range(number):
        st.write("\n")



# page config
st.set_page_config(
    page_title="LEP Analytics", layout="centered"
)


st.title("LEPAn: Logos - Ethos - Pathos Analytics")
add_spacelines(2)


st.write("#### Aristotle's rhetoric")
with st.expander("Rhetoric definitions"):
    add_spacelines(1)
    st.write("**Logos** is ...")
    add_spacelines(1)
    st.write("**Ethos** is ...")
    add_spacelines(1)
    st.write("**Pathos** is ...")

add_spacelines(1)

st.write("### Trust Analytics in Digital Rhetoric")
with st.expander("Read abstract"):
    add_spacelines(1)
    st.write("""
    Trust plays a critical role in establishing intellectual humility and interpersonal civility in
argumentation and discourse: without it, credibility is doomed, reputation is endangered,
cooperation is compromised. The major threats associated with digitalisation – hate speech and
fake news – are violations of the basic condition for trusting and being trustworthy which are key
for constructive, reasonable and responsible communication as well as for the collaborative and
ethical organisation of societies. This calls for a reliable and rich model which allows us to
recognise, both manually and automatically, how trust is established, supported, attacked and
destroyed in natural argumentation.

The aim of this paper is to recognise references to (dis)trust using Artificial Intelligence with a
linguistics, computational and analytics perspective to understand the specific language that is
used in politics and conspiracy theories, when describing the trusted and distrusted entities, such
as politicians and organisations. Building upon the previous work in argument analytics (Lawrence
et al 206; 2017) and theoretical and computational language models for ethos mining (Budzynska
and Duthie 2018; Pereira-Farina at al. 2022), the paper will create language resources and an
annotation scheme which will allow the curation of a large corpus of references to trust in Reddit -
specifically the subreddit dedicated to the US 2016 presidential debates and to conspiracy
theories. Natural Language Processing techniques will be utilised to produce a computational
model of references to trust with the ability to precisely classify unseen text as containing trust,
distrust or neither.

This will allow us to infer from structured data statistical patterns such as: frequencies of using
appeals to trust expressing hate to specific persons, e.g., to Trump or Clinton; frequencies of using
different authorities such as (pseudo-)science or law to increase the “credibility” of fake news; and,
investigate how user’s opinions of these entities swing from positive to negative over time. These
insights will reveal which trends are common in social media. The long-term ambition of this work
is to contribute to the recently announced priority of the EC of Europe fit for the Digital Age.
""")
add_spacelines(2)
#st.write(" ******************************************************** ")


# sidebar
# https://newethos.org/
with st.sidebar:
    image = Image.open(r'tne_logo.png')
    imageuam = Image.open(r'uam_logo_black.png')
    #st.subheader("Trust Analytics in Digital Rhetoric")
    #st.write(" ************************** ")
    st.write("#### Project developed by:")
    st.write("""\n
    Katarzyna Budzynska\n
    Ewelina Gajewska\n
    Barbara Konat\n
    Marcin Koszowy\n
    Marie-Amelie Paquin\n
    He Zhang""")

    st.write(" ************************** ")
    st.write("Paper related to the project: ")
    st.write("**Budzynska, K et al. (2022). Trust Analytics in Digital Rhetoric. 4th European Conference on Argumentation.**")
    st.write(" ************************** ")

    st.subheader("Laboratory of The New Ethos")
    st.write("""#### Faculty of Administration and Social Sciences, Warsaw University of Technology""")
    #st.subheader("Warsaw University of Technology")
    st.image(image, caption='TNE logo')
    st.write("See: [Laboratory of The New Ethos](https://newethos.org/)")
    add_spacelines(2)
    st.write("""#### Faculty of Psychology and Cognitive Sciences, Adam Mickiewicz University in Poznan""")
    #st.subheader("Adam Mickiewicz University in Poznan")
    st.image(imageuam, caption='AMU logo')
    st.write("See: [Faculty of Psychology and Cognitive Sciences](https://psychologia.amu.edu.pl/)")
    st.write(" ************************** ")


selected_dataset = st.selectbox(
     "Choose a dataset you would like to analyse",
     ("US2016 Reddit", "Conspiracy Theories Reddit"))

#selected_dataset_caption = f'<p style="color:#076B96; font-size: 19px; font-weight: bold;">\t{selected_dataset}</p>'
#st.write('**Selected dataset:**')
#st.markdown(selected_dataset_caption, unsafe_allow_html=True)

st.info(f'Selected dataset: **{selected_dataset}**')

add_spacelines(3)
st.header(f" \t Analytics for {selected_dataset} data")
add_spacelines(2)


@st.cache
def load_dataset(dataset):
    if dataset == "US2016 Reddit":
        data = pd.read_excel(r"app_US2016.xlsx", index_col = 0)
    elif dataset ==  "Conspiracy Theories Reddit":
        data = pd.read_excel(r"app_conspiracy.xlsx", index_col = 0)
    return data

df = load_dataset(selected_dataset)

st.subheader("Text-Level Analysis")
rhetoric_dims = ['logos', 'ethos', 'pathos']

#st.markdown(".stMultiSelect > label {font-size:16px; font-weight:bold; color:#076B96;} ", unsafe_allow_html=True)


var_to_plot = st.multiselect("Choose rhetoric dimensions you would like to visualize", rhetoric_dims, rhetoric_dims[1])
var_to_plot = [str(x).replace("ethos", "ethos_name").replace("logos", "logos_name").replace("pathos", "pathos_name") for x in var_to_plot]

pathos_cols = ['No_pathos', 'Contains_pathos',
       'positive_valence', 'negative_valence', 'happiness', 'anger', 'sadness',
       'fear', 'disgust', 'surprise', 'trust', 'anticipation']


def plot_pathos_emo(data):
    df_melt_emo_pathos = data[pathos_cols[4:]].fillna(0).melt(var_name='emotion', value_name = 'value')
    df_melt_emo_pathos_perc = df_melt_emo_pathos.groupby('emotion')['value'].mean().reset_index()
    df_melt_emo_pathos_perc['value'] = df_melt_emo_pathos_perc['value'].round(3) * 100

    color_map_pathos_emo = {'anger': '#CB6600','anticipation': '#FF0783','disgust': '#6A009B',
     'fear':'#000000','happiness': '#8DF903','sadness': '#010598','surprise': '#00C8CB','trust': '#FFFB07'}

    fig_emo = sns.catplot(data = df_melt_emo_pathos_perc.sort_values(by = 'value', ascending=False),
                x = 'value', y = 'emotion', hue = 'emotion', kind = 'bar',
                aspect=1.8, dodge=False, palette = color_map_pathos_emo, height = 6, legend=False)
    plt.xlabel('\npercentage', fontsize=16)
    plt.ylabel(' ')
    plt.yticks(fontsize=16)
    plt.xticks(np.arange(0, df_melt_emo_pathos_perc['value'].max()+3, 2), fontsize=15)
    plt.xlim(0, df_melt_emo_pathos_perc['value'].max()+3)
    plt.title('Pathos - emotions analytics\n', fontsize=23)
    plt.show()
    return fig_emo

def plot_rhetoric_basic_stats1(var_multiselect):
    num_vars = len(var_multiselect)
    if num_vars == 1:
        var_name1 = var_multiselect[0]
        df_prop = pd.DataFrame(df[str(var_name1)].value_counts(normalize=True).round(3)*100)
        df_prop.columns = ['percentage']
        df_prop.reset_index(inplace=True)
        df_prop.columns = ['label', 'percentage']
        #df_prop1['label'] = df_prop1['label'].str.replace('negative', ' negative')
        df_prop = df_prop.sort_values(by = 'label')

        title_str = str(var_name1).replace('_name', '').capitalize()
        figlog, axlog = plt.subplots(figsize=(10, 6))
        axlog.bar(df_prop['label'], df_prop['percentage'], color = ['r', 'b', 'g'])
        plt.ylabel('percentage %\n', fontsize=16)
        plt.yticks(np.arange(0, df_prop['percentage'].max()+16, 10), fontsize=15)
        plt.xticks(fontsize=17)
        plt.title(f"{title_str} analytics\n", fontsize=23)
        vals = df_prop['percentage'].values.round(1)
        for i, v in enumerate(vals):
            plt.text(x=i , y = v+1 , s=f"{v}%" , fontdict=dict(fontsize=17, ha='center'))
        plt.show()
    return figlog

def plot_rhetoric_basic_stats2(var_multiselect):
    num_vars = len(var_multiselect)
    if num_vars == 2:
        #plot1
        var_name1 = var_multiselect[0]
        df_prop1 = pd.DataFrame(df[str(var_name1)].value_counts(normalize=True).round(3)*100)
        df_prop1.columns = ['percentage']
        df_prop1.reset_index(inplace=True)
        df_prop1.columns = ['label', 'percentage']
        #df_prop1['label'] = df_prop1['label'].str.replace('negative', ' negative')
        df_prop1 = df_prop1.sort_values(by = 'label')

        title_str1 = str(var_name1).replace('_name', '').capitalize()
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.bar(df_prop1['label'], df_prop1['percentage'], color = ['r', 'b', 'g'])
        plt.ylabel('percentage %\n', fontsize=16)
        plt.yticks(np.arange(0, df_prop1['percentage'].max()+16, 10), fontsize=15)
        plt.xticks(fontsize=17)
        plt.title(f"{title_str1} analytics\n", fontsize=23)
        vals1 = df_prop1['percentage'].values.round(1)
        for i, v in enumerate(vals1):
            plt.text(x=i , y = v+1 , s=f"{v}%" , fontdict=dict(fontsize=17, ha='center'))
        plt.show()

        #plot2
        var_name2 = var_multiselect[1]
        df_prop2 = pd.DataFrame(df[str(var_name2)].value_counts(normalize=True).round(3)*100)
        df_prop2.columns = ['percentage']
        df_prop2.reset_index(inplace=True)
        df_prop2.columns = ['label', 'percentage']
        #df_prop2['label'] = df_prop2['label'].str.replace('negative', ' negative')
        df_prop2 = df_prop2.sort_values(by = 'label')

        title_str2 = str(var_name2).replace('_name', '').capitalize()
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.bar(df_prop2['label'], df_prop2['percentage'], color = ['r', 'b', 'g'])
        plt.ylabel('percentage %\n', fontsize=16)
        plt.yticks(np.arange(0, df_prop2['percentage'].max()+16, 10), fontsize=15)
        plt.xticks(fontsize=17)
        plt.title(f"{title_str2} analytics\n", fontsize=23)
        vals2 = df_prop2['percentage'].values.round(1)
        for i, v in enumerate(vals2):
            plt.text(x=i , y = v+1 , s=f"{v}%" , fontdict=dict(fontsize=17, ha='center'))
        plt.show()
    return fig1, fig2


def plot_rhetoric_basic_stats3(var_multiselect):
    num_vars = len(var_multiselect)
    if num_vars == 3:
        #plot1
        var_name1 = var_multiselect[0]
        df_prop1 = pd.DataFrame(df[str(var_name1)].value_counts(normalize=True).round(3)*100)
        df_prop1.columns = ['percentage']
        df_prop1.reset_index(inplace=True)
        df_prop1.columns = ['label', 'percentage']
        #df_prop1['label'] = df_prop1['label'].str.replace('negative', ' negative')
        df_prop1 = df_prop1.sort_values(by = 'label')

        title_str1 = str(var_name1).replace('_name', '').capitalize()
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.bar(df_prop1['label'], df_prop1['percentage'], color = ['r', 'b', 'g'])
        plt.ylabel('percentage %\n', fontsize=16)
        plt.yticks(np.arange(0, df_prop1['percentage'].max()+16, 10), fontsize=15)
        plt.xticks(fontsize=17)
        plt.title(f"{title_str1} analytics\n", fontsize=23)
        vals1 = df_prop1['percentage'].values.round(1)
        for i, v in enumerate(vals1):
            plt.text(x=i , y = v+1 , s=f"{v}%" , fontdict=dict(fontsize=17, ha='center'))
        plt.show()

        #plot2
        var_name2 = var_multiselect[1]
        df_prop2 = pd.DataFrame(df[str(var_name2)].value_counts(normalize=True).round(3)*100)
        df_prop2.columns = ['percentage']
        df_prop2.reset_index(inplace=True)
        df_prop2.columns = ['label', 'percentage']
        #df_prop2['label'] = df_prop2['label'].str.replace('negative', ' negative')
        df_prop2 = df_prop2.sort_values(by = 'label')

        title_str2 = str(var_name2).replace('_name', '').capitalize()
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.bar(df_prop2['label'], df_prop2['percentage'], color = ['r', 'b', 'g'])
        plt.ylabel('percentage %\n', fontsize=16)
        plt.yticks(np.arange(0, df_prop2['percentage'].max()+16, 10), fontsize=15)
        plt.xticks(fontsize=17)
        plt.title(f"{title_str2} analytics\n", fontsize=23)
        vals2 = df_prop2['percentage'].values.round(1)
        for i, v in enumerate(vals2):
            plt.text(x=i , y = v+1 , s=f"{v}%" , fontdict=dict(fontsize=17, ha='center'))
        plt.show()

        #plot3
        var_name3 = var_multiselect[2]
        df_prop3 = pd.DataFrame(df[str(var_name3)].value_counts(normalize=True).round(3)*100)
        df_prop3.columns = ['percentage']
        df_prop3.reset_index(inplace=True)
        df_prop3.columns = ['label', 'percentage']
        #df_prop3['label'] = df_prop3['label'].str.replace('negative', ' negative')
        df_prop3 = df_prop3.sort_values(by = 'label')

        title_str3 = str(var_name3).replace('_name', '').capitalize()
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.bar(df_prop3['label'], df_prop3['percentage'], color = ['r', 'b', 'g'])
        plt.ylabel('percentage %\n', fontsize=16)
        plt.yticks(np.arange(0, df_prop3['percentage'].max()+16, 10), fontsize=15)
        plt.xticks(fontsize=17)
        plt.title(f"{title_str3} analytics\n", fontsize=23)
        vals3 = df_prop3['percentage'].values.round(1)
        for i, v in enumerate(vals3):
            plt.text(x=i , y = v+1 , s=f"{v}%" , fontdict=dict(fontsize=17, ha='center'))
        plt.show()
    return fig1, fig2, fig3


if len(var_to_plot) == 1:
    fig = plot_rhetoric_basic_stats1(var_to_plot)
    st.pyplot(fig)
    add_spacelines(1)
    if "pathos_name" in var_to_plot:
        add_spacelines(1)
        fig_emo_pat = plot_pathos_emo(df)
        st.pyplot(fig_emo_pat)
        add_spacelines(1)
elif len(var_to_plot) == 2:
    fig1, fig2 = plot_rhetoric_basic_stats2(var_to_plot)
    st.pyplot(fig1)
    add_spacelines(1)
    st.pyplot(fig2)
    if "pathos_name" in var_to_plot:
        add_spacelines(1)
        fig_emo_pat = plot_pathos_emo(df)
        st.pyplot(fig_emo_pat)
        add_spacelines(1)

elif len(var_to_plot) == 3:
    fig1, fig2, fig3 = plot_rhetoric_basic_stats3(var_to_plot)
    st.pyplot(fig1)
    add_spacelines(1)
    st.pyplot(fig2)
    add_spacelines(1)
    st.pyplot(fig3)
    if "pathos_name" in var_to_plot:
        add_spacelines(1)
        fig_emo_pat = plot_pathos_emo(df)
        st.pyplot(fig_emo_pat)
        add_spacelines(1)


st.write(" ******************************************************** ")
add_spacelines(2)


# wordclouds
st.subheader("High Precision Words - WordCloud")
add_spacelines(1)
# ['logos_name', 'ethos_name', 'pathos_name']
selected_rhet_dim = st.selectbox(
     "Choose a rhetoric dimension for a WordCloud",
     rhetoric_dims, index=1)
add_spacelines(1)
label_cloud = st.radio(
     "Choose a label for a WordCloud",
     ('attack / negative', 'support / positive', 'both'))

selected_rhet_dim = selected_rhet_dim.replace("ethos", "ethos_name").replace("logos", "logos_name").replace("pathos", "pathos_name")
label_cloud = label_cloud.replace("attack / negative", "attack").replace("support / positive", "support")
add_spacelines(1)
threshold_cloud = st.slider('Select a precision value (threshold) for a WordCloud', 0, 100, 90)
#st.write("Selected precision: " + str(threshold_cloud))
st.info(f'Selected precision: **{threshold_cloud}**')

add_spacelines(1)

if st.button('Generate a WordCloud'):
    if (selected_rhet_dim == 'ethos_name') or (selected_rhet_dim == 'logos_name'):
         df_for_wordcloud = prepare_cloud_lexeme_data(df[df[str(selected_rhet_dim)] == 'neutral'],
         df[df[str(selected_rhet_dim)] == 'support'],
         df[df[str(selected_rhet_dim)] == 'attack'])
    else:
        df_for_wordcloud = prepare_cloud_lexeme_data(df[df[str(selected_rhet_dim)] == 'neutral'],
        df[df[str(selected_rhet_dim)] == 'positive'],
        df[df[str(selected_rhet_dim)] == 'negative'])

    fig_cloud = wordcloud_lexeme(df_for_wordcloud, lexeme_threshold = 80, analysis_for = str(label_cloud))
    st.pyplot(fig_cloud)

else:
    st.write("**You need to click the button in order to generate a WordCloud**")


add_spacelines(2)
st.write(" ******************************************************** ")
add_spacelines(2)


st.subheader("Target Entity Analysis")

list_targets = df["Target"].unique()
list_targets = [x for x in list_targets if str(x) != "nan"]
#list_targets = df[df.Target != "nan"]["Target"].unique()

selected_target = st.selectbox("Choose a target entity you would like to analyse", list_targets)

# all df targets
df_target_all = pd.DataFrame(df[df.ethos_name != 'neutral']['ethos_name'].value_counts(normalize = True).round(2)*100)
df_target_all.columns = ['percentage']
df_target_all.reset_index(inplace=True)
df_target_all.columns = ['label', 'percentage']
df_target_all = df_target_all.sort_values(by = 'label')

df_target_all_att = df_target_all[df_target_all.label == 'attack']['percentage'].iloc[0]
df_target_all_sup = df_target_all[df_target_all.label == 'support']['percentage'].iloc[0]

# chosen target df
df_target = pd.DataFrame(df[df.Target == str(selected_target)]['ethos_name'].value_counts(normalize = True).round(2)*100)
df_target.columns = ['percentage']
df_target.reset_index(inplace=True)
df_target.columns = ['label', 'percentage']

hero_labels = {'attack', 'support'}
if len(df_target) == 1:
  added_label = list(hero_labels - set(df_target.label.unique()))
  df_target.loc[len(df_target)] = [str(added_label[0]), 0]

df_target = df_target.sort_values(by = 'label')
df_target_att = df_target[df_target.label == 'attack']['percentage'].iloc[0]
df_target_sup = df_target[df_target.label == 'support']['percentage'].iloc[0]


with st.container():
    #selected_target_caption = f'<p style="color:#076B96; font-size: 19px; font-weight: bold;">\t{selected_target}</p>'
    #st.write('**Selected entity:**')
    #st.markdown(selected_target_caption, unsafe_allow_html=True)
    st.info(f'Selected entity: **{selected_target}**')
    add_spacelines(1)
    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Hero score")
        col1.metric(str(selected_target), str(df_target_sup)+ str('%'), str(round((df_target_sup - df_target_all_sup),  1))+ str(' p.p.')) # round(((df_target_sup / df_target_all_sup) * 100) - 100, 1)

    with col2:
        st.subheader("Anti-hero score")
        col2.metric(str(selected_target), str(df_target_att)+ str('%'), str(round((df_target_att - df_target_all_att),  1))+ str(' p.p.'), delta_color="inverse") # ((df_target_att / df_target_all_att) * 100) - 100, 1)

    df_tar_emo_exp = df[df.Target == selected_target]
    df_tar_emo_exp_senti = df_tar_emo_exp.groupby(['expressed_sentiment'], as_index=False).size()
    df_tar_emo_exp_senti.sort_values(by = 'expressed_sentiment')
    df_tar_emo_exp_senti['size'] = round(df_tar_emo_exp_senti['size'] / len(df_tar_emo_exp), 3) * 100
    df_tar_emo_exp_senti['expressed_sentiment'] = df_tar_emo_exp_senti['expressed_sentiment'].str.lower()

    user_exp_labs = df_tar_emo_exp_senti['expressed_sentiment'].unique()
    if not ('negative' in user_exp_labs):
        df_tar_emo_exp_senti.loc[len(df_tar_emo_exp_senti)] = ['negative', 0]
    if not ('neutral' in user_exp_labs):
        df_tar_emo_exp_senti.loc[len(df_tar_emo_exp_senti)] = ['neutral', 0]
    if not ('positive' in user_exp_labs):
        df_tar_emo_exp_senti.loc[len(df_tar_emo_exp_senti)] = ['positive', 0]

    add_spacelines(2)
    figsenti_user, axsenti = plt.subplots(figsize=(8, 5))
    axsenti.bar(df_tar_emo_exp_senti['expressed_sentiment'], df_tar_emo_exp_senti['size'], color = ['r', 'b', 'g'])
    plt.ylabel('percentage %\n', fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(np.arange(0, 105, 10), fontsize=12)
    plt.title(f"Sentiment expressed towards {selected_target}\n", fontsize=15)
    vals_senti = df_tar_emo_exp_senti['size'].values.round(1)
    for index_senti, v in enumerate(vals_senti):
        plt.text(x=index_senti , y = v+1 , s=f"{v}%" , fontdict=dict(fontsize=12, ha='center'))
    plt.show()
    st.pyplot(figsenti_user)

add_spacelines(1)
st.write(" ******************************************************** ")
add_spacelines(2)


st.subheader("User-Level Analysis")
st.write(f"#### **Users' rhetoric strategies in {selected_dataset}**")

user_stats_df = user_stats_app(df)
user_stats_df.fillna(0, inplace=True)
for c in ['text_n', 'ethos_n', 'ethos_support_n', 'ethos_attack_n',
          'pathos_n', 'pathos_negative_n', 'pathos_positive_n',
          'logos_n', 'logos_support_n', 'logos_attack_n']:
       user_stats_df[c] = user_stats_df[c].apply(int)

user_stats_df_desc = user_stats_df.describe().round(3)

cols_strat = ['ethos_support_percent', 'ethos_attack_percent',
              'pathos_positive_percent',  'pathos_negative_percent',
              'logos_support_percent', 'logos_attack_percent']

def plot_strateies(data):
    i = 0
    for c in range(3):
        print(cols_strat[c+i], cols_strat[c+i+1])
        fig_stats, axs = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
        axs[0].hist(data[cols_strat[c+i]], color='#009C6F')
        title_str0 = " ".join(cols_strat[c+i].split("_")[:-1]).capitalize()
        axs[0].set_title(title_str0, fontsize=20)
        axs[0].set_ylabel('number of users\n', fontsize=15)
        axs[0].set_xlabel('\npercentage of texts %', fontsize=15)
        axs[0].set_xticks(np.arange(0, 101, 10), fontsize=14)

        axs[1].hist(data[cols_strat[c+i+1]], color='#9F0155')
        title_str1 = " ".join(cols_strat[c+i+1].split("_")[:-1]).capitalize()
        axs[1].set_xlabel('\npercentage of texts %', fontsize=15)
        axs[1].yaxis.set_tick_params(labelbottom=True)
        axs[1].set_title(title_str1, fontsize=20)
        axs[1].set_xticks(np.arange(0, 101, 10), fontsize=14)
        plt.show()
        i+=1
        st.pyplot(fig_stats)
        add_spacelines(1)

plot_strateies(data = user_stats_df)

ethos_strat = user_stats_df[(user_stats_df.ethos_percent > user_stats_df.ethos_percent.std()+user_stats_df.ethos_percent.mean()) & \
        (user_stats_df.pathos_percent < user_stats_df.pathos_percent.std()+user_stats_df.pathos_percent.mean()) & \
        (user_stats_df.logos_percent < user_stats_df.logos_percent.std()+user_stats_df.logos_percent.mean())]

pathos_strat = user_stats_df[(user_stats_df.ethos_percent < user_stats_df.ethos_percent.std()+user_stats_df.ethos_percent.mean()) & \
        (user_stats_df.pathos_percent > user_stats_df.pathos_percent.std()+user_stats_df.pathos_percent.mean()) & \
        (user_stats_df.logos_percent < user_stats_df.logos_percent.std()+user_stats_df.logos_percent.mean())]

logos_strat = user_stats_df[(user_stats_df.ethos_percent < user_stats_df.ethos_percent.std()+user_stats_df.ethos_percent.mean()) & \
        (user_stats_df.pathos_percent < user_stats_df.pathos_percent.std()+user_stats_df.pathos_percent.mean()) & \
        (user_stats_df.logos_percent > user_stats_df.logos_percent.std()+user_stats_df.logos_percent.mean())]

with st.container():
    col1, col2, col3 = st.columns([10, 9, 9])
    with col1:
        st.write(f"**Dominant logos strategy**")
        col1.metric(str(logos_strat.shape[0]) + " users", str(round(logos_strat.shape[0] / len(user_stats_df) * 100, 1)) + "%")

    with col2:
        st.write(f"**Dominant ethos strategy**")
        col2.metric(str(ethos_strat.shape[0]) + " users", str(round(ethos_strat.shape[0] / len(user_stats_df) * 100, 1)) + "%")

    with col3:
        st.write(f"**Dominant pathos strategy**")
        col3.metric(str(pathos_strat.shape[0]) + " users", str(round(pathos_strat.shape[0] / len(user_stats_df) * 100, 1)) + "%")

    add_spacelines(1)
    dominant_percent_strategy = round(pathos_strat.shape[0] / len(user_stats_df) * 100, 1) + round(ethos_strat.shape[0] / len(user_stats_df) * 100, 1) + round(logos_strat.shape[0] / len(user_stats_df) * 100, 1)
    st.write(f"##### **{round(dominant_percent_strategy, 1)}%** of users employ one dominant rhetoric strategy in {selected_dataset} data.")

add_spacelines(3)

st.write(f"#### **Users' rhetoric metric in {selected_dataset}**")
data_rh = user_rhetoric_v2(df)
data_rh = data_rh[ ~(data_rh.user.isin(['[deleted]', 'deleted', 'nan']))]

color = sns.color_palette("Reds", data_rh[data_rh.rhetoric_metric < 0]['rhetoric_metric'].nunique()+15)[::-1][:data_rh[data_rh.rhetoric_metric < 0]['rhetoric_metric'].nunique()] +sns.color_palette("Blues", 3)[2:] + sns.color_palette("Greens", data_rh[data_rh.rhetoric_metric > 0]['rhetoric_metric'].nunique()+20)[data_rh[data_rh.rhetoric_metric < 0]['rhetoric_metric'].nunique()*-1:] # + sns.color_palette("Greens", 15)[4:]

fig_rh_raw = sns.catplot(kind = 'count', data = data_rh, x = 'rhetoric_metric',
            aspect = 2, palette = color, height = 7)
for ax in fig_rh_raw.axes.ravel():
  for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2.,
        p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), fontsize = 14.5,
        textcoords = 'offset points')
plt.yticks(np.arange(0, data_rh.rhetoric_metric.value_counts().iloc[0]+26, 30), fontsize=16)
plt.ylabel('number of users\n', fontsize = 18)
plt.title("User rhetoric metric\n", fontsize = 23)
plt.xticks(fontsize = 16)
plt.xlabel('\nscore', fontsize = 18)
plt.show()
st.pyplot(fig_rh_raw)


# change raw scores to percentages
counts = data_rh.groupby('rhetoric_metric')['rhetoric_metric'].size().values
ids = data_rh.groupby('rhetoric_metric')['rhetoric_metric'].size().index
perc = (counts / len(data_rh)) * 100

data_rh2 = pd.DataFrame({'rhetoric_metric': ids, 'percent':perc})
data_rh2['percent'] = data_rh2['percent'].apply(lambda x: round(x, 1))

fig_rh_percent = sns.catplot(kind = 'bar', data = data_rh2, x = 'rhetoric_metric',
                 y = 'percent',
            aspect = 2, palette = color, height = 7, ci = None)
for ax in fig_rh_percent.axes.ravel():
  for p in ax.patches:
    ax.annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2.,
        p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), fontsize = 14.5,
        textcoords = 'offset points')
    plt.yticks(np.arange(0, data_rh2.percent.max()+6, 5), fontsize = 16)
plt.ylabel('percentage of users %\n', fontsize = 18)
plt.xticks(fontsize = 16)
plt.title("User rhetoric metric\n", fontsize = 23)
plt.xlabel('\nscore', fontsize = 18)
plt.show()
st.pyplot(fig_rh_percent)


st.write(" ******************************************************** ")
add_spacelines(1)



def standardize(data):
  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  data0 = data.copy()
  scaled_values = scaler.fit_transform(data0)
  data0.loc[:, :] = scaled_values
  return data0

data_rh['standardized_scores'] = standardize(data_rh[['rhetoric_metric']])
most_neg_users = data_rh.nsmallest(8, 'rhetoric_metric')
most_pos_users = data_rh.nlargest(8, 'rhetoric_metric')

most_neg_users_names = most_neg_users.user.tolist()
most_pos_users_names = most_pos_users.user.tolist()

users_rhet_cols = ['Text', 'Target', 'pathos_name',
                   'ethos_name','logos_name'] # 'expressed_sentiment', 'T5_emotion', 'anger', 'sadness', 'fear', 'disgust', 'surprise', 'trust', 'anticipation', 'happiness'

with st.container():
    #st.subheader("**Most negative users**")
    head_neg_users = f'<p style="color:#D10000; font-size: 26px; font-weight: bold;">Most negative users</p>'
    st.markdown(head_neg_users, unsafe_allow_html=True)
    col111, col222, col333, col444 = st.columns(4)
    with col111:
        st.write(f"**{most_neg_users_names[0]}**")
        col111.metric('rhetoric score', most_neg_users['rhetoric_metric'].iloc[0], str(round(most_neg_users['standardized_scores'].iloc[0], 1))+ str(' SD'))

    with col222:
        st.write(f"**{most_neg_users_names[1]}**")
        col222.metric('rhetoric score', most_neg_users['rhetoric_metric'].iloc[1], str(round(most_neg_users['standardized_scores'].iloc[1], 1))+ str(' SD'))

    with col333:
        st.write(f"**{most_neg_users_names[2]}**")
        col333.metric('rhetoric score', most_neg_users['rhetoric_metric'].iloc[2], str(round(most_neg_users['standardized_scores'].iloc[2], 1))+ str(' SD'))

    with col444:
        st.write(f"**{most_neg_users_names[3]}**")
        col444.metric('rhetoric score', most_neg_users['rhetoric_metric'].iloc[3], str(round(most_neg_users['standardized_scores'].iloc[3], 1))+ str(' SD'))

add_spacelines(2)

with st.container():
    col111, col222, col333, col444 = st.columns(4)
    with col111:
        st.write(f"**{most_neg_users_names[4]}**")
        col111.metric('rhetoric score', most_neg_users['rhetoric_metric'].iloc[4], str(round(most_neg_users['standardized_scores'].iloc[4], 1))+ str(' SD'))

    with col222:
        st.write(f"**{most_neg_users_names[5]}**")
        col222.metric('rhetoric score', most_neg_users['rhetoric_metric'].iloc[5], str(round(most_neg_users['standardized_scores'].iloc[5], 1))+ str(' SD'))

    with col333:
        st.write(f"**{most_neg_users_names[6]}**")
        col333.metric('rhetoric score', most_neg_users['rhetoric_metric'].iloc[6], str(round(most_neg_users['standardized_scores'].iloc[6], 1))+ str(' SD'))

    with col444:
        st.write(f"**{most_neg_users_names[7]}**")
        col444.metric('rhetoric score', most_neg_users['rhetoric_metric'].iloc[7], str(round(most_neg_users['standardized_scores'].iloc[7], 1))+ str(' SD'))

    add_spacelines(2)
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    st.write("##### Choose username to see details about the user")
    neg_users_to_df = st.radio(" ", most_neg_users_names)
    add_spacelines(2)
    st.write(f"Texts posted by: **{neg_users_to_df}** ")
    st.dataframe(df[df.Source == str(neg_users_to_df)].set_index("Source")[users_rhet_cols])
    add_spacelines(1)

user_stats_df_user1 = user_stats_df[user_stats_df['user'] == str(neg_users_to_df)]

with st.container():
    st.write(f"##### {neg_users_to_df}'s rhetoric strategy")
    col111, col222, col333 = st.columns(3)
    with col111:
        st.write(f"**Logos strategy**")
        col111.metric(f'{neg_users_to_df}', round(((user_stats_df_user1['logos_attack_n'] / user_stats_df_user1['text_n']) * 100).iloc[0], 1), str(round(((user_stats_df_user1['logos_attack_n'] / user_stats_df_user1['text_n']) * 100).iloc[0] - (user_stats_df_desc.loc['mean', 'logos_attack_n'] / user_stats_df_desc.loc['mean', 'text_n']) * 100, 1))+ str(' p.p.'), delta_color="inverse")

    with col222:
        st.write(f"**Ethos strategy**")
        col222.metric(f'{neg_users_to_df}', round(((user_stats_df_user1['ethos_attack_n'] / user_stats_df_user1['text_n']) * 100).iloc[0], 1), str(round(((user_stats_df_user1['ethos_attack_n'] / user_stats_df_user1['text_n']) * 100).iloc[0] - (user_stats_df_desc.loc['mean', 'ethos_attack_n'] / user_stats_df_desc.loc['mean', 'text_n']) * 100, 1))+ str(' p.p.'), delta_color="inverse")

    with col333:
        st.write(f"**Pathos strategy**")
        col333.metric(f'{neg_users_to_df}', round(((user_stats_df_user1['pathos_negative_n'] / user_stats_df_user1['text_n']) * 100).iloc[0], 1), str(round(((user_stats_df_user1['pathos_negative_n'] / user_stats_df_user1['text_n']) * 100).iloc[0] - (user_stats_df_desc.loc['mean', 'pathos_negative_n'] / user_stats_df_desc.loc['mean', 'text_n']) * 100, 1))+ str(' p.p.'), delta_color="inverse")

    strat_user_val_neg = [round(((user_stats_df_user1['logos_attack_n'] / user_stats_df_user1['text_n']) * 100).iloc[0], 1),
                      round(((user_stats_df_user1['ethos_attack_n'] / user_stats_df_user1['text_n']) * 100).iloc[0], 1),
                      round(((user_stats_df_user1['pathos_negative_n'] / user_stats_df_user1['text_n']) * 100).iloc[0], 1)]
    strat_user_val_neg_max = np.max(strat_user_val_neg)
    add_spacelines(1)
    if strat_user_val_neg[0] == strat_user_val_neg_max:
        st.error(f"**{neg_users_to_df}**'s negativity comes mostly from **logos**")
    elif strat_user_val_neg[1] == strat_user_val_neg_max:
        st.error(f"**{neg_users_to_df}**'s negativity comes mostly from **ethos**")
    elif strat_user_val_neg[2] == strat_user_val_neg_max:
        st.error(f"**{neg_users_to_df}**'s negativity comes mostly from **pathos**")

add_spacelines(1)
st.write(" ******************************************************** ")
add_spacelines(1)

with st.container():
    #st.subheader("**Most positive users**")
    head_pos_users = f'<p style="color:#00A90D; font-size: 26px; font-weight: bold;">Most positive users</p>'
    st.markdown(head_pos_users, unsafe_allow_html=True)
    col11, col22, col33, col44 = st.columns(4)

    with col11:
        st.write(f"**{most_pos_users_names[0]}**")
        col11.metric('rhetoric score', most_pos_users['rhetoric_metric'].iloc[0], str(round(most_pos_users['standardized_scores'].iloc[0], 1))+ str(' SD'))

    with col22:
        st.write(f"**{most_pos_users_names[1]}**")
        col22.metric('rhetoric score', most_pos_users['rhetoric_metric'].iloc[1], str(round(most_pos_users['standardized_scores'].iloc[1], 1))+ str(' SD'))

    with col33:
        st.write(f"**{most_pos_users_names[2]}**")
        col33.metric('rhetoric score', most_pos_users['rhetoric_metric'].iloc[2], str(round(most_pos_users['standardized_scores'].iloc[2], 1))+ str(' SD'))

    with col44:
        st.write(f"**{most_pos_users_names[3]}**")
        col44.metric('rhetoric score', most_pos_users['rhetoric_metric'].iloc[3], str(round(most_pos_users['standardized_scores'].iloc[3], 1))+ str(' SD'))

add_spacelines(2)

with st.container():
    col11, col22, col33, col44 = st.columns(4)
    with col11:
        st.write(f"**{most_pos_users_names[4]}**")
        col11.metric('rhetoric score', most_pos_users['rhetoric_metric'].iloc[4], str(round(most_pos_users['standardized_scores'].iloc[4], 1))+ str(' SD'))

    with col22:
        st.write(f"**{most_pos_users_names[5]}**")
        col22.metric('rhetoric score', most_pos_users['rhetoric_metric'].iloc[5], str(round(most_pos_users['standardized_scores'].iloc[5], 1))+ str(' SD'))

    with col33:
        st.write(f"**{most_pos_users_names[6]}**")
        col33.metric('rhetoric score', most_pos_users['rhetoric_metric'].iloc[6], str(round(most_pos_users['standardized_scores'].iloc[6], 1))+ str(' SD'))

    with col44:
        st.write(f"**{most_pos_users_names[7]}**")
        col44.metric('rhetoric score', most_pos_users['rhetoric_metric'].iloc[7], str(round(most_pos_users['standardized_scores'].iloc[7], 1))+ str(' SD'))

    add_spacelines(2)
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    st.write("##### Choose username to see the details about the user")
    pos_users_to_df = st.radio("", most_pos_users_names)
    add_spacelines(2)
    st.write(f"Texts posted by: **{pos_users_to_df}** ")
    st.dataframe(df[df.Source == str(pos_users_to_df)].set_index("Source")[users_rhet_cols])
    add_spacelines(1)

user_stats_df_user2 = user_stats_df[user_stats_df['user'] == str(pos_users_to_df)]

with st.container():
    st.write(f"##### {pos_users_to_df}'s rhetoric strategy")
    col111, col222, col333 = st.columns(3)
    with col111:
        st.write(f"**Logos strategy**")
        col111.metric(f'{pos_users_to_df}', round(((user_stats_df_user2['logos_support_n'] / user_stats_df_user2['text_n']) * 100).iloc[0], 1), str(round(((user_stats_df_user2['logos_support_n'] / user_stats_df_user2['text_n']) * 100).iloc[0] - (user_stats_df_desc.loc['mean', 'logos_support_n'] / user_stats_df_desc.loc['mean', 'text_n']) * 100, 1))+ str(' p.p.'))

    with col222:
        st.write(f"**Ethos strategy**")
        col222.metric(f'{pos_users_to_df}', round(((user_stats_df_user2['ethos_support_n'] / user_stats_df_user2['text_n']) * 100).iloc[0], 1), str(round(((user_stats_df_user2['ethos_support_n'] / user_stats_df_user2['text_n']) * 100).iloc[0] - (user_stats_df_desc.loc['mean', 'ethos_support_n'] / user_stats_df_desc.loc['mean', 'text_n']) * 100, 1))+ str(' p.p.'))

    with col333:
        st.write(f"**Pathos strategy**")
        col333.metric(f'{pos_users_to_df}', round(((user_stats_df_user2['pathos_positive_n'] / user_stats_df_user2['text_n']) * 100).iloc[0], 1), str(round(((user_stats_df_user2['pathos_positive_n'] / user_stats_df_user2['text_n']) * 100).iloc[0] - (user_stats_df_desc.loc['mean', 'pathos_positive_n'] / user_stats_df_desc.loc['mean', 'text_n']) * 100, 1))+ str(' p.p.'))

    strat_user_val_pos = [round(((user_stats_df_user2['logos_support_n'] / user_stats_df_user2['text_n']) * 100).iloc[0], 1),
                      round(((user_stats_df_user2['ethos_support_n'] / user_stats_df_user2['text_n']) * 100).iloc[0], 1),
                      round(((user_stats_df_user2['pathos_positive_n'] / user_stats_df_user2['text_n']) * 100).iloc[0], 1)]
    strat_user_val_pos_max = np.max(strat_user_val_pos)
    add_spacelines(1)
    if strat_user_val_pos[0] == strat_user_val_pos_max:
        st.success(f"**{pos_users_to_df}**'s positivity comes mostly from **logos**")
    elif strat_user_val_pos[1] == strat_user_val_pos_max:
        st.success(f"**{pos_users_to_df}**'s positivity comes mostly from **ethos**")
    elif strat_user_val_pos[2] == strat_user_val_pos_max:
        st.success(f"**{pos_users_to_df}**'s positivity comes mostly from **pathos**")

add_spacelines(1)
st.write(" ******************************************************** ")
