
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


def make_word_cloud(comment_words, width = 1100, height = 650, colour = "black", colormap = "brg"):
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(max_words=250, colormap=colormap, width = width, height = height,
                background_color ='white',
                min_font_size = 14, stopwords = stopwords).generate(comment_words) # , stopwords = stopwords

    fig, ax = plt.subplots(figsize = (width/ 100, height/100), facecolor = colour)
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

  figure_cloud = make_word_cloud(" ".join(text), 1000, 620, '#1E1E1E', str(cmap_wordcloud)) #gist_heat / flare_r crest viridis
  return figure_cloud


def standardize(data):
  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  data0 = data.copy()
  scaled_values = scaler.fit_transform(data0)
  data0.loc[:, :] = scaled_values
  return data0


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




def plot_pathos_emo(data):
    df_melt_emo_pathos = data[pathos_cols[4:]].fillna(0).melt(var_name='emotion', value_name = 'value')
    df_melt_emo_pathos_perc = df_melt_emo_pathos.groupby('emotion')['value'].mean().reset_index()
    df_melt_emo_pathos_perc['value'] = df_melt_emo_pathos_perc['value'].round(3) * 100

    color_map_pathos_emo = {'anger': '#BB0000','anticipation': '#D87D00','disgust': '#BB0000',
     'fear':'#BB0000','happiness': '#026F00','sadness': '#BB0000','surprise': '#D87D00','trust': '#026F00'}

    fig_emo = sns.catplot(data = df_melt_emo_pathos_perc.sort_values(by = 'value', ascending=False),
                x = 'value', y = 'emotion', hue = 'emotion', kind = 'bar',
                aspect=1.8, dodge=False, palette = color_map_pathos_emo, height = 6, legend=False)
    plt.xlabel('\npercentage %', fontsize=18)
    plt.ylabel(' ')
    plt.yticks(fontsize=16)
    plt.xticks(np.arange(0, df_melt_emo_pathos_perc['value'].max()+3, 2), fontsize=15)
    plt.xlim(0, df_melt_emo_pathos_perc['value'].max()+3)
    plt.title('Pathos - emotions analytics\n', fontsize=23)
    import matplotlib.patches as mpatches
    red_patch = mpatches.Patch(color='#BB0000', label='negative emotion')
    green_patch = mpatches.Patch(color='#026F00', label='positive emotion')
    gold_patch = mpatches.Patch(color='#D87D00', label='ambivalent emotion')
    plt.legend(handles=[red_patch, green_patch, gold_patch], fontsize=15)
    plt.show()
    return fig_emo

def plot_pathos_emo_counts(data):
    df_melt_emo_pathos = data[pathos_cols[4:]].fillna(0).melt(var_name='emotion', value_name = 'value')
    df_melt_emo_pathos_perc = df_melt_emo_pathos.groupby(['emotion', 'value'], as_index=False).size()
    df_melt_emo_pathos_perc = df_melt_emo_pathos_perc[df_melt_emo_pathos_perc.value == 1]

    color_map_pathos_emo = {'anger': '#BB0000','anticipation': '#D87D00','disgust': '#BB0000',
     'fear':'#BB0000','happiness': '#026F00','sadness': '#BB0000','surprise': '#D87D00','trust': '#026F00'}

    fig_emo = sns.catplot(data = df_melt_emo_pathos_perc.sort_values(by = 'size', ascending=False),
                x = 'size', y = 'emotion', hue = 'emotion', kind = 'bar',
                aspect=1.8, dodge=False, palette = color_map_pathos_emo, height = 6, legend=False)
    plt.xlabel('\ncount', fontsize=18)
    plt.ylabel(' ')
    plt.yticks(fontsize=16)
    plt.xticks(np.arange(0, df_melt_emo_pathos_perc['size'].max()+25, 50), fontsize=15)
    plt.xlim(0, df_melt_emo_pathos_perc['size'].max()+25)
    plt.title('Pathos - emotions analytics\n', fontsize=23)
    import matplotlib.patches as mpatches
    red_patch = mpatches.Patch(color='#BB0000', label='negative emotion')
    green_patch = mpatches.Patch(color='#026F00', label='positive emotion')
    gold_patch = mpatches.Patch(color='#D87D00', label='ambivalent emotion')
    plt.legend(handles=[red_patch, green_patch, gold_patch], fontsize=15)
    plt.show()
    return fig_emo


def plot_rhetoric_basic_stats1(var_multiselect, val_type = "percentages"):
    num_vars = len(var_multiselect)
    if val_type == "counts":
        var_name1 = var_multiselect[0]
        df_prop1 = pd.DataFrame(df[str(var_name1)].value_counts())
        df_prop1.columns = ['count']
        df_prop1.reset_index(inplace=True)
        df_prop1.columns = ['label', 'count']
        #df_prop1['label'] = df_prop1['label'].str.replace('negative', ' negative')
        df_prop1 = df_prop1.sort_values(by = 'label')

        title_str1 = str(var_name1).replace('_name', '').capitalize()
        fig1, ax1 = plt.subplots(figsize=(9, 6))
        ax1.bar(df_prop1['label'], df_prop1['count'], color = ['#BB0000', '#022D96', '#026F00'])
        plt.ylabel('count\n', fontsize=16)
        plt.yticks(np.arange(0, df_prop1['count'].max()+206, 500), fontsize=15)
        plt.xticks(fontsize=17)
        plt.title(f"{title_str1} analytics\n", fontsize=23)
        vals1 = df_prop1['count'].values
        for i, v in enumerate(vals1):
            plt.text(x=i , y = v+1 , s=f"{v}" , fontdict=dict(fontsize=17, ha='center'))
        plt.show()

    else:
        var_name1 = var_multiselect[0]
        df_prop = pd.DataFrame(df[str(var_name1)].value_counts(normalize=True).round(3)*100)
        df_prop.columns = ['percentage']
        df_prop.reset_index(inplace=True)
        df_prop.columns = ['label', 'percentage']
        df_prop = df_prop.sort_values(by = 'label')

        title_str = str(var_name1).replace('_name', '').capitalize()
        #fig1, axlog = plt.subplots(figsize=(9, 6))
        #axlog.bar(df_prop['label'], df_prop['percentage'], color = ['#BB0000', '#022D96', '#026F00'])
        #plt.ylabel('percentage %\n', fontsize=16)
        #plt.yticks(np.arange(0, df_prop['percentage'].max()+16, 10), fontsize=15)
        #plt.xticks(fontsize=17)
        #plt.title(f"{title_str} analytics\n", fontsize=23)
        #vals = df_prop['percentage'].values.round(1)
        #for i, v in enumerate(vals):
    #        plt.text(x=i , y = v+1 , s=f"{v}%" , fontdict=dict(fontsize=17, ha='center'))

        fig1, ax1 = plt.subplots(2, 1, figsize=(10, 12))
        ax1[0].bar(df_prop['label'], df_prop['percentage'], color = ['#BB0000', '#022D96', '#026F00'])
        ax1[0].set_ylabel('percentage\n', fontsize=16)
        ax1[0].set_yticks(np.arange(0, df_prop['percentage'].max()+16, 10), fontsize=15)
        ax1[0].tick_params(axis='x', labelsize=17)
        ax1[0].set_title(f"{title_str} analytics\n", fontsize=23)
        vals0 = df_prop['percentage'].values
        for i, v in enumerate(vals0):
            ax1[0].text(x=i , y = v+1 , s=f"{round(v, 1)}%" , fontdict=dict(fontsize=17, ha='center'))

        df_prop11 = pd.DataFrame(df[df[str(var_name1)] != 'neutral'][str(var_name1)].value_counts(normalize=True).round(3)*100)
        df_prop11.columns = ['percentage']
        df_prop11.reset_index(inplace=True)
        df_prop11.columns = ['label', 'percentage']
        df_prop11 = df_prop11.sort_values(by = 'label')
        ax1[1].bar(df_prop11['label'], df_prop11['percentage'], color = ['#BB0000', '#026F00'])
        ax1[1].set_ylabel('percentage\n', fontsize=16)
        ax1[1].set_yticks(np.arange(0, df_prop11['percentage'].max()+11, 10), fontsize=15)
        ax1[1].tick_params(axis='x', labelsize=17)
        ax1[1].set_title(f"\n", fontsize=23)
        vals1 = df_prop11['percentage'].values
        for i, v in enumerate(vals1):
            ax1[1].text(x=i , y = v+1 , s=f"{round(v, 0)}%" , fontdict=dict(fontsize=17, ha='center'))
        plt.tight_layout()
        plt.show()

    return fig1


def plot_rhetoric_basic_stats2(var_multiselect, val_type = "percentages"):
    num_vars = len(var_multiselect)
    if val_type == "counts":
        #plot1
        var_name1 = var_multiselect[0]
        df_prop1 = pd.DataFrame(df[str(var_name1)].value_counts())
        df_prop1.columns = ['count']
        df_prop1.reset_index(inplace=True)
        df_prop1.columns = ['label', 'count']
        #df_prop1['label'] = df_prop1['label'].str.replace('negative', ' negative')
        df_prop1 = df_prop1.sort_values(by = 'label')

        title_str1 = str(var_name1).replace('_name', '').capitalize()
        fig1, ax1 = plt.subplots(figsize=(9, 6))
        ax1.bar(df_prop1['label'], df_prop1['count'], color = ['#BB0000', '#022D96', '#026F00'])
        plt.ylabel('count\n', fontsize=16)
        plt.yticks(np.arange(0, df_prop1['count'].max()+206, 500), fontsize=15)
        plt.xticks(fontsize=17)
        plt.title(f"{title_str1} analytics\n", fontsize=23)
        vals1 = df_prop1['count'].values
        for i, v in enumerate(vals1):
            plt.text(x=i , y = v+1 , s=f"{v}" , fontdict=dict(fontsize=17, ha='center'))
        plt.show()

        #plot2
        var_name2 = var_multiselect[1]
        df_prop2 = pd.DataFrame(df[str(var_name2)].value_counts())
        df_prop2.columns = ['count']
        df_prop2.reset_index(inplace=True)
        df_prop2.columns = ['label', 'count']
        #df_prop2['label'] = df_prop2['label'].str.replace('negative', ' negative')
        df_prop2 = df_prop2.sort_values(by = 'label')

        title_str2 = str(var_name2).replace('_name', '').capitalize()
        fig2, ax2 = plt.subplots(figsize=(9, 6))
        ax2.bar(df_prop2['label'], df_prop2['count'], color = ['#BB0000', '#022D96', '#026F00'])
        plt.ylabel('count\n', fontsize=16)
        plt.yticks(np.arange(0, df_prop2['count'].max()+206, 500), fontsize=15)
        plt.xticks(fontsize=17)
        plt.title(f"{title_str2} analytics\n", fontsize=23)
        vals2 = df_prop2['count'].values
        for i, v in enumerate(vals2):
            plt.text(x=i , y = v+1 , s=f"{v}" , fontdict=dict(fontsize=17, ha='center'))
        plt.show()

    else:
        #plot1
        var_name1 = var_multiselect[0]
        df_prop1 = pd.DataFrame(df[str(var_name1)].value_counts(normalize=True).round(3)*100)
        df_prop1.columns = ['percentage']
        df_prop1.reset_index(inplace=True)
        df_prop1.columns = ['label', 'percentage']
        #df_prop1['label'] = df_prop1['label'].str.replace('negative', ' negative')
        df_prop1 = df_prop1.sort_values(by = 'label')

        title_str1 = str(var_name1).replace('_name', '').capitalize()
        fig1, ax1 = plt.subplots(figsize=(9, 6))
        ax1.bar(df_prop1['label'], df_prop1['percentage'], color = ['#BB0000', '#022D96', '#026F00'])
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
        fig2, ax2 = plt.subplots(figsize=(9, 6))
        ax2.bar(df_prop2['label'], df_prop2['percentage'], color = ['#BB0000', '#022D96', '#026F00'])
        plt.ylabel('percentage %\n', fontsize=16)
        plt.yticks(np.arange(0, df_prop2['percentage'].max()+16, 10), fontsize=15)
        plt.xticks(fontsize=17)
        plt.title(f"{title_str2} analytics\n", fontsize=23)
        vals2 = df_prop2['percentage'].values.round(1)
        for i, v in enumerate(vals2):
            plt.text(x=i , y = v+1 , s=f"{v}%" , fontdict=dict(fontsize=17, ha='center'))
        plt.show()

    return fig1, fig2


def plot_rhetoric_basic_stats3(var_multiselect, val_type = "percentages"):
    num_vars = len(var_multiselect)
    if val_type == "counts":
#plot1
        var_name1 = var_multiselect[0]
        df_prop1 = pd.DataFrame(df[str(var_name1)].value_counts())
        df_prop1.columns = ['count']
        df_prop1.reset_index(inplace=True)
        df_prop1.columns = ['label', 'count']
        #df_prop1['label'] = df_prop1['label'].str.replace('negative', ' negative')
        df_prop1 = df_prop1.sort_values(by = 'label')

        title_str1 = str(var_name1).replace('_name', '').capitalize()
        fig1, ax1 = plt.subplots(figsize=(9, 6))
        ax1.bar(df_prop1['label'], df_prop1['count'], color = ['#BB0000', '#022D96', '#026F00'])
        plt.ylabel('count\n', fontsize=16)
        plt.yticks(np.arange(0, df_prop1['count'].max()+206, 500), fontsize=15)
        plt.xticks(fontsize=17)
        plt.title(f"{title_str1} analytics\n", fontsize=23)
        vals1 = df_prop1['count'].values
        for i, v in enumerate(vals1):
            plt.text(x=i , y = v+1 , s=f"{v}" , fontdict=dict(fontsize=17, ha='center'))
        plt.show()

        #plot2
        var_name2 = var_multiselect[1]
        df_prop2 = pd.DataFrame(df[str(var_name2)].value_counts())
        df_prop2.columns = ['count']
        df_prop2.reset_index(inplace=True)
        df_prop2.columns = ['label', 'count']
        #df_prop2['label'] = df_prop2['label'].str.replace('negative', ' negative')
        df_prop2 = df_prop2.sort_values(by = 'label')

        title_str2 = str(var_name2).replace('_name', '').capitalize()
        fig2, ax2 = plt.subplots(figsize=(9, 6))
        ax2.bar(df_prop2['label'], df_prop2['count'], color = ['#BB0000', '#022D96', '#026F00'])
        plt.ylabel('count\n', fontsize=16)
        plt.yticks(np.arange(0, df_prop2['count'].max()+206, 500), fontsize=15)
        plt.xticks(fontsize=17)
        plt.title(f"{title_str2} analytics\n", fontsize=23)
        vals2 = df_prop2['count'].values
        for i, v in enumerate(vals2):
            plt.text(x=i , y = v+1 , s=f"{v}" , fontdict=dict(fontsize=17, ha='center'))
        plt.show()

        #plot3
        var_name3 = var_multiselect[2]
        df_prop3 = pd.DataFrame(df[str(var_name3)].value_counts())
        df_prop3.columns = ['count']
        df_prop3.reset_index(inplace=True)
        df_prop3.columns = ['label', 'count']
        #df_prop3['label'] = df_prop3['label'].str.replace('negative', ' negative')
        df_prop3 = df_prop3.sort_values(by = 'label')

        title_str3 = str(var_name3).replace('_name', '').capitalize()
        fig3, ax3 = plt.subplots(figsize=(9, 6))
        ax3.bar(df_prop3['label'], df_prop3['count'], color = ['#BB0000', '#022D96', '#026F00'])
        plt.ylabel('count\n', fontsize=16)
        plt.yticks(np.arange(0, df_prop3['count'].max()+206, 500), fontsize=15)
        plt.xticks(fontsize=17)
        plt.title(f"{title_str3} analytics\n", fontsize=23)
        vals3 = df_prop3['count'].values
        for i, v in enumerate(vals3):
            plt.text(x=i , y = v+1 , s=f"{v}" , fontdict=dict(fontsize=17, ha='center'))
        plt.show()

    else:
        #plot1
        var_name1 = var_multiselect[0]
        df_prop1 = pd.DataFrame(df[str(var_name1)].value_counts(normalize=True).round(3)*100)
        df_prop1.columns = ['percentage']
        df_prop1.reset_index(inplace=True)
        df_prop1.columns = ['label', 'percentage']
        #df_prop1['label'] = df_prop1['label'].str.replace('negative', ' negative')
        df_prop1 = df_prop1.sort_values(by = 'label')

        title_str1 = str(var_name1).replace('_name', '').capitalize()
        fig1, ax1 = plt.subplots(figsize=(9, 6))
        ax1.bar(df_prop1['label'], df_prop1['percentage'], color = ['#BB0000', '#022D96', '#026F00'])
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
        fig2, ax2 = plt.subplots(figsize=(9, 6))
        ax2.bar(df_prop2['label'], df_prop2['percentage'], color = ['#BB0000', '#022D96', '#026F00'])
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
        fig3, ax3 = plt.subplots(figsize=(9, 6))
        ax3.bar(df_prop3['label'], df_prop3['percentage'], color = ['#BB0000', '#022D96', '#026F00'])
        plt.ylabel('percentage %\n', fontsize=16)
        plt.yticks(np.arange(0, df_prop3['percentage'].max()+16, 10), fontsize=15)
        plt.xticks(fontsize=17)
        plt.title(f"{title_str3} analytics\n", fontsize=23)
        vals3 = df_prop3['percentage'].values.round(1)
        for i, v in enumerate(vals3):
            plt.text(x=i , y = v+1 , s=f"{v}%" , fontdict=dict(fontsize=17, ha='center'))
        plt.show()

    return fig1, fig2, fig3


def add_spacelines(number=2):
    for i in range(number):
        st.write("\n")


@st.cache
def load_dataset(dataset):
    if dataset == "US2016 Presidential Debate Reddit":
        data = pd.read_excel(r"app_US2016.xlsx", index_col = 0)
    elif dataset ==  "Conspiracy Theories Reddit":
        data = pd.read_excel(r"app_conspiracy.xlsx", index_col = 0)
    return data



pathos_cols = ['No_pathos', 'Contains_pathos',
       'positive_valence', 'negative_valence', 'happiness', 'anger', 'sadness',
       'fear', 'disgust', 'surprise', 'trust', 'anticipation']

rhetoric_dims = ['logos', 'ethos', 'pathos']


# page config
st.set_page_config(
    page_title="LEP Analytics", layout="centered"
)



def style_css(file):
    with open(file) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


# multi pages functions

def MainPage():
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

    add_spacelines(2)

    st.write("#### Trust Analytics in Digital Rhetoric")
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



    with st.container():
        image = Image.open(r'tne_logo.png')
        imageuam = Image.open(r'uam_logo_blue.jfif') # uam_logo_black.jpg

        add_spacelines(2)
        st.write(" **************************** ")
        st.write(" Project developed by:")
        col11, col22, col33 = st.columns([4, 4, 3])
        with col11:
            st.write("**Katarzyna Budzynska**")
        with col22:
            st.write("**Ewelina Gajewska**")
        with col33:
            st.write("**Konrad Kiljan**")

        col111, col222, col333, col7 = st.columns([4, 4, 4, 3])
        with col111:
            st.write("**Barbara Konat**")      
        with col222:
            st.write("**Marcin Koszowy**")
        with col333:
            st.write("**Marie-Amelie Paquin**")
        with col7:
            st.write("**He Zhang**")


        st.write(" ************************** ")
        st.write("Paper related to the project: ")
        st.write("**Budzynska, K et al. (2022). Trust Analytics in Digital Rhetoric. 4th European Conference on Argumentation.**")
        st.write(" ************************** ")

        col1, col2, col3 = st.columns([3, 1, 3])
        with col1:
            st.write("**Laboratory of The New Ethos**")
            st.write(""" Faculty of Administration and Social Sciences, Warsaw University of Technology""")
        #st.subheader("Warsaw University of Technology")
            add_spacelines(2)
            st.image(image, caption='TNE logo')
            add_spacelines(1)
            st.write("**See: [Laboratory of The New Ethos](https://newethos.org/)**")
            add_spacelines(1)
        with col2:
            st.write("")
            add_spacelines(1)
        with col3:
            add_spacelines(2)
            st.write(""" Faculty of Psychology and Cognitive Sciences, Adam Mickiewicz University in Poznan""")
            #st.subheader("Adam Mickiewicz University in Poznan")
            st.image(imageuam, caption='AMU logo')
            st.write("**See: [Faculty of Psychology and Cognitive Sciences](https://psychologia.amu.edu.pl/)**")
    st.write('<style>div.row-widget.stRadio > div{flex-direction:column;font-size=18px;}</style>', unsafe_allow_html=True)

pathos_cols = ['No_pathos', 'Contains_pathos',
       'positive_valence', 'negative_valence', 'happiness', 'anger', 'sadness',
       'fear', 'disgust', 'surprise', 'trust', 'anticipation']

def basicLEPAn():
    st.subheader(f" Text-Level Analysis ")
    add_spacelines(2)

    rhetoric_dims = ['logos', 'ethos', 'pathos']
    pathos_cols = ['No_pathos', 'Contains_pathos',
           'positive_valence', 'negative_valence', 'happiness', 'anger', 'sadness',
           'fear', 'disgust', 'surprise', 'trust', 'anticipation']


    var_to_plot_raw = st.multiselect("Choose rhetoric dimensions you would like to visualize", rhetoric_dims, rhetoric_dims[1])
    var_to_plot = [str(x).replace("ethos", "ethos_name").replace("logos", "logos_name").replace("pathos", "pathos_name") for x in var_to_plot_raw]

    #add_spacelines(1)
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;font-size=18px;}</style>', unsafe_allow_html=True)
    check_rhet_dim = st.radio("", ("percentages", "counts"))
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;font-size=18px;}</style>', unsafe_allow_html=True)

    if len(var_to_plot) == 1:
        if check_rhet_dim == "counts":
            fig1 = plot_rhetoric_basic_stats1(var_to_plot, val_type = "counts")
        else:
            fig1 = plot_rhetoric_basic_stats1(var_to_plot)
        st.pyplot(fig1)
        if "pathos_name" in var_to_plot:
            add_spacelines(1)
            if check_rhet_dim == "percentages":
                fig_emo_pat = plot_pathos_emo(df)
                st.pyplot(fig_emo_pat)
            if check_rhet_dim == "counts":
                fig_emo_pat = plot_pathos_emo_counts(df)
                st.pyplot(fig_emo_pat)


    elif len(var_to_plot) == 2:
        tab1, tab2 = st.tabs([x.upper() for x in var_to_plot_raw])
        if check_rhet_dim == "counts":
            fig1, fig2 = plot_rhetoric_basic_stats2(var_to_plot, val_type = "counts")
        else:
            fig1, fig2 = plot_rhetoric_basic_stats2(var_to_plot)
        #st.pyplot(fig1)
        #add_spacelines(1)
        #st.pyplot(fig2)
        if "pathos_name" in var_to_plot:
            tab_id = list(var_to_plot_raw).index('pathos')
            #add_spacelines(1)
            if check_rhet_dim == "percentages":
                fig_emo_pat = plot_pathos_emo(df)
                #st.pyplot(fig_emo_pat)
                #add_spacelines(1)
            elif check_rhet_dim == "counts":
                fig_emo_pat = plot_pathos_emo_counts(df)
                #st.pyplot(fig_emo_pat)
                #add_spacelines(1)
        with tab1:
            st.pyplot(fig1)
        with tab2:
            st.pyplot(fig2)

        if "pathos_name" in var_to_plot:
            if tab_id == 0:
                with tab1:
                    add_spacelines(2)
                    st.pyplot(fig_emo_pat)
            elif tab_id == 1:
                with tab2:
                    add_spacelines(2)
                    st.pyplot(fig_emo_pat)

    elif len(var_to_plot) == 3:
        tab1, tab2, tab3 = st.tabs([x.upper() for x in var_to_plot_raw])
        if check_rhet_dim == "counts":
            fig1, fig2, fig3 = plot_rhetoric_basic_stats3(var_to_plot, val_type = "counts")
        else:
            fig1, fig2, fig3 = plot_rhetoric_basic_stats3(var_to_plot)
        #st.pyplot(fig1)
        #add_spacelines(1)
        #st.pyplot(fig2)
        #add_spacelines(1)
        #st.pyplot(fig3)
        if "pathos_name" in var_to_plot:
            tab_id = list(var_to_plot_raw).index('pathos')
            #add_spacelines(1)
            if check_rhet_dim == "percentages":
                fig_emo_pat = plot_pathos_emo(df)
                #st.pyplot(fig_emo_pat)
                #add_spacelines(1)
            elif check_rhet_dim == "counts":
                fig_emo_pat = plot_pathos_emo_counts(df)
                #st.pyplot(fig_emo_pat)
                #add_spacelines(1)
        with tab1:
            st.pyplot(fig1)

        with tab2:
            st.pyplot(fig2)

        with tab3:
            st.pyplot(fig3)

        if "pathos_name" in var_to_plot:
            if tab_id == 0:
                with tab1:
                    add_spacelines(2)
                    st.pyplot(fig_emo_pat)
            elif tab_id == 1:
                with tab2:
                    add_spacelines(2)
                    st.pyplot(fig_emo_pat)
            elif tab_id == 2:
                with tab3:
                    add_spacelines(2)
                    st.pyplot(fig_emo_pat)

    else:
        st.write("")
    st.write('<style>div.row-widget.stRadio > div{flex-direction:column;font-size=18px;}</style>', unsafe_allow_html=True)


def generateWordCloud():
    #st.header(f" Text-Level Analytics ")
    st.subheader("High Precision Words - WordCloud")
    add_spacelines(2)

    rhetoric_dims = ['logos', 'ethos', 'pathos']
    pathos_cols = ['No_pathos', 'Contains_pathos',
           'positive_valence', 'negative_valence', 'happiness', 'anger', 'sadness',
           'fear', 'disgust', 'surprise', 'trust', 'anticipation']


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
        st.warning("**You need to click the button in order to generate a WordCloud**")

    add_spacelines(4)
    with st.expander("High Precision Words"):
        add_spacelines(1)
        st.write("How accurate we are with finding a text belonging to the chosen category when a particular word is present in the text.")
    st.write('<style>div.row-widget.stRadio > div{flex-direction:column;font-size=18px;}</style>', unsafe_allow_html=True)


def TargetHeroScores():
    st.subheader(f"(Anti)Heroes - Target Entity Analytics ")
    add_spacelines(2)

    rhetoric_dims = ['logos', 'ethos', 'pathos']
    pathos_cols = ['No_pathos', 'Contains_pathos',
           'positive_valence', 'negative_valence', 'happiness', 'anger', 'sadness',
           'fear', 'disgust', 'surprise', 'trust', 'anticipation']


    list_targets = df["Target"].unique()
    list_targets = [x for x in list_targets if str(x) != "nan"]

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
        st.info(f'Selected entity: **{selected_target}**')
        add_spacelines(1)
        col1, col2 = st.columns([3, 2])
        with col1:
            st.subheader("Hero score")
            col1.metric(str(selected_target), str(df_target_sup)+ str('%'), str(round((df_target_sup - df_target_all_sup),  1))+ str(' p.p.')) # round(((df_target_sup / df_target_all_sup) * 100) - 100, 1)

        with col2:
            st.subheader("Anti-hero score")
            col2.metric(str(selected_target), str(df_target_att)+ str('%'), str(round((df_target_att - df_target_all_att),  1))+ str(' p.p.'), delta_color="inverse") # ((df_target_att / df_target_all_att) * 100) - 100, 1)

        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;font-size=18px;}</style>', unsafe_allow_html=True)
        radio_senti_target = st.radio("", ("percentage", "count"))
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;font-size=18px;}</style>', unsafe_allow_html=True)

        df_tar_emo_exp = df[df.Target == selected_target]
        df_tar_emo_exp_senti = df_tar_emo_exp.groupby(['expressed_sentiment'], as_index=False).size()
        df_tar_emo_exp_senti.sort_values(by = 'expressed_sentiment')
        if radio_senti_target == "percentage":
            df_tar_emo_exp_senti['size'] = round(df_tar_emo_exp_senti['size'] / len(df_tar_emo_exp), 3) * 100
        df_tar_emo_exp_senti['expressed_sentiment'] = df_tar_emo_exp_senti['expressed_sentiment'].str.lower()

        user_exp_labs = df_tar_emo_exp_senti['expressed_sentiment'].unique()
        if not ('negative' in user_exp_labs):
            df_tar_emo_exp_senti.loc[len(df_tar_emo_exp_senti)] = ['negative', 0]
        if not ('neutral' in user_exp_labs):
            df_tar_emo_exp_senti.loc[len(df_tar_emo_exp_senti)] = ['neutral', 0]
        if not ('positive' in user_exp_labs):
            df_tar_emo_exp_senti.loc[len(df_tar_emo_exp_senti)] = ['positive', 0]

        figsenti_user, axsenti = plt.subplots(figsize=(8, 5))
        axsenti.bar(df_tar_emo_exp_senti['expressed_sentiment'], df_tar_emo_exp_senti['size'], color = ['#BB0000', '#022D96', '#026F00'])
        plt.xticks(fontsize=13)
        plt.title(f"Sentiment expressed towards {selected_target}\n", fontsize=15)
        vals_senti = df_tar_emo_exp_senti['size'].values.round(1)
        if radio_senti_target == "percentage":
            plt.yticks(np.arange(0, 105, 10), fontsize=12)
            plt.ylabel('percentage %\n', fontsize=13)
            for index_senti, v in enumerate(vals_senti):
                plt.text(x=index_senti , y = v+1 , s=f"{v}%" , fontdict=dict(fontsize=12, ha='center'))
        else:
            if len(df_tar_emo_exp) > 120:
                plt.yticks(np.arange(0, df_tar_emo_exp_senti['size'].max()+16, 20), fontsize=12)
            elif len(df_tar_emo_exp) > 40 and len(df_tar_emo_exp) < 120:
                plt.yticks(np.arange(0, df_tar_emo_exp_senti['size'].max()+6, 5), fontsize=12)
            else:
                plt.yticks(np.arange(0, df_tar_emo_exp_senti['size'].max()+3, 2), fontsize=12)
            plt.ylabel('count\n', fontsize=13)
            for index_senti, v in enumerate(vals_senti):
                plt.text(x=index_senti , y = v , s=f"{v}" , fontdict=dict(fontsize=12, ha='center'))
        plt.show()
        st.pyplot(figsenti_user)

    add_spacelines(4)
    with st.expander("(Anti)Hero scores"):
        add_spacelines(1)
        st.write("""
        Hero and Anti-hero scores are calculated based on the ethos annotation. \n

        Values indicate what proportion of users support and attack, respectively, a given entity. Higher Hero score means that users support the target entity more often, and when Anti-hero score is higher, users tend to attack rather than support the entity.
        """)
    st.write('<style>div.row-widget.stRadio > div{flex-direction:column;font-size=18px;}</style>', unsafe_allow_html=True)


def UserRhetStrategy():
    #st.header(f" User-Level Analytics ")
    st.subheader(f"Rhetoric Strategies")
    add_spacelines(2)

    rhetoric_dims = ['logos', 'ethos', 'pathos']
    pathos_cols = ['No_pathos', 'Contains_pathos',
           'positive_valence', 'negative_valence', 'happiness', 'anger', 'sadness',
           'fear', 'disgust', 'surprise', 'trust', 'anticipation']

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

    def plot_strategies(data):
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

    plot_strategies(data = user_stats_df)

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
        add_spacelines(2)
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

        if datasets_singles_conspiracy and datasets_singles_us2016:
            st.write(f"###### **{round(dominant_percent_strategy, 1)}%** of users have one dominant rhetoric strategy")
        elif datasets_singles_us2016:
            st.write(f"###### **{round(dominant_percent_strategy, 1)}%** of users have one dominant rhetoric strategy in $US2016 Presidential Debate Reddit$ data.")
        elif datasets_singles_conspiracy:
            st.write(f"###### **{round(dominant_percent_strategy, 1)}%** of users have one dominant rhetoric strategy in $Conspiracy Theories Reddit$ data.")
        #st.write(f"##### **{round(dominant_percent_strategy, 1)}%** of users have one dominant rhetoric strategy in {dataset_name} data.")

    add_spacelines(4)
    with st.expander("Rhetoric strategy"):
        add_spacelines(1)
        st.write("""
        **User rhetoric strategy**:
        Calculates the proportion of posts generated by a given user that belong to the categories of ethos, pathos, and logos.\n

        **Logos strategy**:
        What percentage of texts generated by a given user attack/support other posts in terms of logos.\n

        **Ethos  strategy**:
        What percentage of texts posted by a given user attack/support other users' or third parties' character.\n\n

        **Pathos strategy**:
        What percentage of texts posted by a given user elicit negative/positive pathos.\n\n

        **Dominant strategy**:
        When a proportion of a user's texts belonging to logos, ethos or pathos is above one standard deviation and a proportion of texts belonging to the other two rhetoric categories is below one standard deviation. \n

        """)

    st.write('<style>div.row-widget.stRadio > div{flex-direction:column;font-size=18px;}</style>', unsafe_allow_html=True)

def UserRhetMetric():
    #st.header(f" User-Level Analytics ")
    st.subheader(f" Rhetoric Metric")
    add_spacelines(2)

    rhetoric_dims = ['logos', 'ethos', 'pathos']
    pathos_cols = ['No_pathos', 'Contains_pathos',
           'positive_valence', 'negative_valence', 'happiness', 'anger', 'sadness',
           'fear', 'disgust', 'surprise', 'trust', 'anticipation']

    data_rh = user_rhetoric_v2(df)
    data_rh = data_rh[ ~(data_rh.user.isin(['[deleted]', 'deleted', 'nan']))]

    user_stats_df = user_stats_app(df)
    user_stats_df.fillna(0, inplace=True)
    for c in ['text_n', 'ethos_n', 'ethos_support_n', 'ethos_attack_n',
              'pathos_n', 'pathos_negative_n', 'pathos_positive_n',
              'logos_n', 'logos_support_n', 'logos_attack_n']:
           user_stats_df[c] = user_stats_df[c].apply(int)

    user_stats_df_desc = user_stats_df.describe().round(3)

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

    add_spacelines(2)

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

    add_spacelines(4)
    with st.expander("Rhetoric metric"):
        add_spacelines(1)
        st.write("""
        Scores are calculated based on the number of positive/support and negative/attack posts (in terms of logos, ethos, and pathos) generated by a given user.
        """)
    st.write('<style>div.row-widget.stRadio > div{flex-direction:column;font-size=18px;}</style>', unsafe_allow_html=True)

def UsersExtreme():
    st.subheader("Users Profiles")
    add_spacelines(2)

    rhetoric_dims = ['logos', 'ethos', 'pathos']
    pathos_cols = ['No_pathos', 'Contains_pathos',
           'positive_valence', 'negative_valence', 'happiness', 'anger', 'sadness',
           'fear', 'disgust', 'surprise', 'trust', 'anticipation']

    data_rh = user_rhetoric_v2(df)
    data_rh = data_rh[ ~(data_rh.user.isin(['[deleted]', 'deleted', 'nan']))]

    user_stats_df = user_stats_app(df)
    user_stats_df.fillna(0, inplace=True)
    for c in ['text_n', 'ethos_n', 'ethos_support_n', 'ethos_attack_n',
              'pathos_n', 'pathos_negative_n', 'pathos_positive_n',
              'logos_n', 'logos_support_n', 'logos_attack_n']:
           user_stats_df[c] = user_stats_df[c].apply(int)

    user_stats_df_desc = user_stats_df.describe().round(3)

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
        #st.write("##### Choose username to see details about the user")
        neg_users_to_df = st.radio("Choose username to see details about the user \n", most_neg_users_names)
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
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
    st.write(" **************************************************************************** ")
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
        #st.write("##### Choose username to see the details about the user")
        pos_users_to_df = st.radio("Choose username to see details about the user \n", most_pos_users_names)
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
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

    add_spacelines(4)
    with st.expander("Users Analytics"):
        add_spacelines(1)
        st.write("""
        **Negative and positive users**:
        Negative and positive users are chosen based on rhetoric metric values. Scores are calculated based on the number of positive/support and negative/attack posts (in terms of logos, ethos, and pathos) generated by a given user.

        Additionally, we convert rhetoric metric scores into standard deviations for the ease of interpretation.\n
        \n

        **User rhetoric strategy**:
        Calculates the proportion of posts generated by a given user that belong to the categories of ethos, pathos, and logos.
        """)
    #st.write('<style>div.row-widget.stRadio > div{flex-direction:column;font-size=18px;}</style>', unsafe_allow_html=True)



def plotRhetoricCompare3(data1, data2, data3, rhet_dim_to_plot):
    rhet_dim = str(rhet_dim_to_plot)
    rhet_dim_var = rhet_dim.replace("ethos", "ethos_name").replace("logos", "logos_name").replace("pathos", "pathos_name")

    df_prop1 = pd.DataFrame(data1[rhet_dim_var].value_counts(normalize=True).round(3)*100)
    df_prop1.columns = ['percentage']
    df_prop1.reset_index(inplace=True)
    df_prop1.columns = ['label', 'percentage']
    df_prop1 = df_prop1.sort_values(by = 'label')

    df_prop2 = pd.DataFrame(data2[rhet_dim_var].value_counts(normalize=True).round(3)*100)
    df_prop2.columns = ['percentage']
    df_prop2.reset_index(inplace=True)
    df_prop2.columns = ['label', 'percentage']
    df_prop2 = df_prop2.sort_values(by = 'label')

    df_prop3 = pd.DataFrame(data3[rhet_dim_var].value_counts(normalize=True).round(3)*100)
    df_prop3.columns = ['percentage']
    df_prop3.reset_index(inplace=True)
    df_prop3.columns = ['label', 'percentage']
    df_prop3 = df_prop3.sort_values(by = 'label')

    fig_stats, axs = plt.subplots(1, 3, figsize=(26, 6), sharey=True, constrained_layout=True)
    axs[0].bar(df_prop1['label'], df_prop1['percentage'], color = ['#BB0000', '#022D96', '#026F00'])
    title_str0 = str(rhet_dim).capitalize() + " in " + str(data1['Dataset'].iloc[0])
    axs[0].set_title(title_str0, fontsize=20)
    axs[0].set_ylabel('percentage %\n', fontsize=16)
    axs[0].set_ylim(0, 101)
    axs[0].tick_params(axis='x', labelsize=17)
    axs[0].tick_params(axis='y', labelsize=15)
    vals0 = df_prop1['percentage'].values.round(1)
    for i, v in enumerate(vals0):
        axs[0].text(x=i , y = v+1 , s=f"{v}%" , fontdict=dict(fontsize=14, ha='center', weight='bold'))

    axs[1].bar(df_prop2['label'], df_prop2['percentage'], color = ['#BB0000', '#022D96', '#026F00'])
    title_str1 = str(rhet_dim).capitalize() + " in " + str(data2['Dataset'].iloc[0])
    axs[1].set_title(title_str1, fontsize=20)
    #axs[1].set_ylabel('percentage %\n', fontsize=16)
    axs[1].yaxis.set_tick_params(labelbottom=True)
    axs[1].tick_params(axis='x', labelsize=17)
    axs[1].tick_params(axis='y', labelsize=15)
    vals1 = df_prop2['percentage'].values.round(1)
    for i, v in enumerate(vals1):
        axs[1].text(x=i , y = v+1 , s=f"{v}%" , fontdict=dict(fontsize=14, ha='center', weight='bold'))

    axs[2].bar(df_prop3['label'], df_prop3['percentage'], color = ['#BB0000', '#022D96', '#026F00'])
    title_str3 = str(rhet_dim).capitalize() + " in " + str(data3['Dataset'].iloc[0])
    axs[2].set_title(title_str3, fontsize=20)
    #axs[1].set_ylabel('percentage %\n', fontsize=16)
    axs[2].yaxis.set_tick_params(labelbottom=True)
    axs[2].tick_params(axis='x', labelsize=17)
    axs[2].tick_params(axis='y', labelsize=15)
    vals3 = df_prop3['percentage'].values.round(1)
    for i, v in enumerate(vals3):
        axs[2].text(x=i , y = v+1 , s=f"{v}%" , fontdict=dict(fontsize=14, ha='center', weight='bold'))

    fig_stats.subplots_adjust(hspace=.3)
    plt.show()
    st.pyplot(fig_stats)

def plotRhetoricCompare3_finegrained(data1, data2, data3, rhet_dim_to_plot):
    rhet_dim = str(rhet_dim_to_plot)
    rhet_dim_var = rhet_dim.replace("ethos", "ethos_name").replace("logos", "logos_name").replace("pathos", "pathos_name")

    data1 = data1[data1[rhet_dim_var] != 'neutral']
    df_prop1 = pd.DataFrame(data1[rhet_dim_var].value_counts(normalize=True).round(3)*100)
    df_prop1.columns = ['percentage']
    df_prop1.reset_index(inplace=True)
    df_prop1.columns = ['label', 'percentage']
    df_prop1 = df_prop1.sort_values(by = 'label')

    data2 = data2[data2[rhet_dim_var] != 'neutral']
    df_prop2 = pd.DataFrame(data2[rhet_dim_var].value_counts(normalize=True).round(3)*100)
    df_prop2.columns = ['percentage']
    df_prop2.reset_index(inplace=True)
    df_prop2.columns = ['label', 'percentage']
    df_prop2 = df_prop2.sort_values(by = 'label')

    data3 = data3[data3[rhet_dim_var] != 'neutral']
    df_prop3 = pd.DataFrame(data3[rhet_dim_var].value_counts(normalize=True).round(3)*100)
    df_prop3.columns = ['percentage']
    df_prop3.reset_index(inplace=True)
    df_prop3.columns = ['label', 'percentage']
    df_prop3 = df_prop3.sort_values(by = 'label')

    fig_stats2, axs = plt.subplots(1, 3, figsize=(26, 6), sharey=True, constrained_layout=True)
    axs[0].bar(df_prop1['label'], df_prop1['percentage'], color = ['#BB0000', '#026F00'])
    title_str0 = str(rhet_dim).capitalize() + " in " + str(data1['Dataset'].iloc[0])
    axs[0].set_title(title_str0, fontsize=20)
    axs[0].set_ylabel('percentage %\n', fontsize=16)
    axs[0].set_ylim(0, 101)
    axs[0].tick_params(axis='x', labelsize=17)
    axs[0].tick_params(axis='y', labelsize=15)
    vals0 = df_prop1['percentage'].values.round(0)
    for i, v in enumerate(vals0):
        axs[0].text(x=i , y = v+1 , s=f"{v}%" , fontdict=dict(fontsize=14, ha='center', weight='bold'))

    axs[1].bar(df_prop2['label'], df_prop2['percentage'], color = ['#BB0000', '#026F00'])
    title_str1 = str(rhet_dim).capitalize() + " in " + str(data2['Dataset'].iloc[0])
    axs[1].set_title(title_str1, fontsize=20)
    axs[1].yaxis.set_tick_params(labelbottom=True)
    axs[1].tick_params(axis='x', labelsize=17)
    axs[1].tick_params(axis='y', labelsize=15)
    vals1 = df_prop2['percentage'].values.round(0)
    for i, v in enumerate(vals1):
        axs[1].text(x=i , y = v+1 , s=f"{v}%" , fontdict=dict(fontsize=14, ha='center', weight='bold'))

    axs[2].bar(df_prop3['label'], df_prop3['percentage'], color = ['#BB0000', '#026F00'])
    title_str3 = str(rhet_dim).capitalize() + " in " + str(data3['Dataset'].iloc[0])
    axs[2].set_title(title_str3, fontsize=20)
    axs[2].yaxis.set_tick_params(labelbottom=True)
    axs[2].tick_params(axis='x', labelsize=17)
    axs[2].tick_params(axis='y', labelsize=15)
    vals3 = df_prop3['percentage'].values.round(0)
    for i, v in enumerate(vals3):
        axs[2].text(x=i , y = v+1 , s=f"{v}%" , fontdict=dict(fontsize=14, ha='center', weight='bold'))

    fig_stats2.subplots_adjust(hspace=.3)
    plt.show()
    st.pyplot(fig_stats2)


def CompareDatasetsText():
    st.subheader("Compare Multiple Datasets on Rhetoric")
    add_spacelines(2)

    if hansard_box:
        compare_rhet_dim = st.selectbox("Choose a rhetoric dimension", rhetoric_dims[:-1], index=0)
    else:
        compare_rhet_dim = st.selectbox("Choose a rhetoric dimension", rhetoric_dims[::-1], index=0)

    add_spacelines(2)


    def plotRhetoricCompare2(data1, data2, rhet_dim_to_plot):
        rhet_dim = str(rhet_dim_to_plot)
        rhet_dim_var = rhet_dim.replace("ethos", "ethos_name").replace("logos", "logos_name").replace("pathos", "pathos_name")

        df_prop1 = pd.DataFrame(data1[rhet_dim_var].value_counts(normalize=True).round(3)*100)
        df_prop1.columns = ['percentage']
        df_prop1.reset_index(inplace=True)
        df_prop1.columns = ['label', 'percentage']
        df_prop1 = df_prop1.sort_values(by = 'label')

        df_prop2 = pd.DataFrame(data2[rhet_dim_var].value_counts(normalize=True).round(3)*100)
        df_prop2.columns = ['percentage']
        df_prop2.reset_index(inplace=True)
        df_prop2.columns = ['label', 'percentage']
        df_prop2 = df_prop2.sort_values(by = 'label')

        fig_stats, axs = plt.subplots(1, 2, figsize=(18, 7), sharey=True, constrained_layout=True)
        axs[0].bar(df_prop1['label'], df_prop1['percentage'], color = ['#BB0000', '#022D96', '#026F00'])
        title_str0 = str(rhet_dim).capitalize() + " in " + str(data1['Dataset'].iloc[0])
        axs[0].set_title(title_str0, fontsize=20)
        axs[0].set_ylabel('percentage %\n', fontsize=16)
        axs[0].set_ylim(0, 101)
        axs[0].tick_params(axis='x', labelsize=17)
        axs[0].tick_params(axis='y', labelsize=15)
        vals0 = df_prop1['percentage'].values.round(1)
        for i, v in enumerate(vals0):
            axs[0].text(x=i , y = v+1 , s=f"{v}%" , fontdict=dict(fontsize=14, ha='center', weight='bold'))

        axs[1].bar(df_prop2['label'], df_prop2['percentage'], color = ['#BB0000', '#022D96', '#026F00'])
        title_str1 = str(rhet_dim).capitalize() + " in " + str(data2['Dataset'].iloc[0])
        axs[1].set_title(title_str1, fontsize=20)
        #axs[1].set_ylabel('percentage %\n', fontsize=16)
        axs[1].yaxis.set_tick_params(labelbottom=True)
        axs[1].tick_params(axis='x', labelsize=17)
        axs[1].tick_params(axis='y', labelsize=15)
        vals1 = df_prop2['percentage'].values.round(1)
        for i, v in enumerate(vals1):
            axs[1].text(x=i , y = v+1 , s=f"{v}%" , fontdict=dict(fontsize=14, ha='center', weight='bold'))
        fig_stats.subplots_adjust(hspace=.3)
        plt.show()
        st.pyplot(fig_stats)

    def plotRhetoricCompare2_finegrained(data1, data2, rhet_dim_to_plot):
        rhet_dim = str(rhet_dim_to_plot)
        rhet_dim_var = rhet_dim.replace("ethos", "ethos_name").replace("logos", "logos_name").replace("pathos", "pathos_name")

        data1 = data1[data1[rhet_dim_var] != 'neutral']
        df_prop1 = pd.DataFrame(data1[rhet_dim_var].value_counts(normalize=True).round(3)*100)
        df_prop1.columns = ['percentage']
        df_prop1.reset_index(inplace=True)
        df_prop1.columns = ['label', 'percentage']
        df_prop1 = df_prop1.sort_values(by = 'label')

        data2 = data2[data2[rhet_dim_var] != 'neutral']
        df_prop2 = pd.DataFrame(data2[rhet_dim_var].value_counts(normalize=True).round(3)*100)
        df_prop2.columns = ['percentage']
        df_prop2.reset_index(inplace=True)
        df_prop2.columns = ['label', 'percentage']
        df_prop2 = df_prop2.sort_values(by = 'label')

        fig_stats2, axs = plt.subplots(1, 2, figsize=(18, 7), sharey=True, constrained_layout=True)
        axs[0].bar(df_prop1['label'], df_prop1['percentage'], color = ['#BB0000', '#026F00'])
        title_str0 = str(rhet_dim).capitalize() + " in " + str(data1['Dataset'].iloc[0])
        axs[0].set_title(title_str0, fontsize=20)
        axs[0].set_ylabel('percentage %\n', fontsize=16)
        axs[0].set_ylim(0, 101)
        axs[0].tick_params(axis='x', labelsize=17)
        axs[0].tick_params(axis='y', labelsize=15)
        vals0 = df_prop1['percentage'].values.round(0)
        for i, v in enumerate(vals0):
            axs[0].text(x=i , y = v+1 , s=f"{v}%" , fontdict=dict(fontsize=14, ha='center', weight='bold'))

        axs[1].bar(df_prop2['label'], df_prop2['percentage'], color = ['#BB0000', '#026F00'])
        title_str1 = str(rhet_dim).capitalize() + " in " + str(data2['Dataset'].iloc[0])
        axs[1].set_title(title_str1, fontsize=20)
        axs[1].yaxis.set_tick_params(labelbottom=True)
        axs[1].tick_params(axis='x', labelsize=17)
        axs[1].tick_params(axis='y', labelsize=15)
        vals1 = df_prop2['percentage'].values.round(0)
        for i, v in enumerate(vals1):
            axs[1].text(x=i , y = v+1 , s=f"{v}%" , fontdict=dict(fontsize=14, ha='center', weight='bold'))
        fig_stats2.subplots_adjust(hspace=.3)
        plt.show()
        st.pyplot(fig_stats2)

    if hansard_box and compare_rhet_dim == 'logos':
        hansard_df = hansard_df_logos[['logos_name', 'Dataset']]
    elif hansard_box and compare_rhet_dim == 'ethos':
        hansard_df = hansard_df_ethos[['ethos_name', 'Dataset']]

    if (us2016_box and conspiracy_box and hansard_box):
        plotRhetoricCompare3(data1 = us2016_df, data2 = conspiracy_df, data3 = hansard_df, rhet_dim_to_plot = compare_rhet_dim)
        add_spacelines(2)
        plotRhetoricCompare3_finegrained(data1 = us2016_df, data2 = conspiracy_df, data3 = hansard_df, rhet_dim_to_plot = compare_rhet_dim)

    elif (us2016_box and conspiracy_box and not hansard_box):
        plotRhetoricCompare2(data1 = us2016_df, data2 = conspiracy_df, rhet_dim_to_plot = compare_rhet_dim)
        add_spacelines(2)
        plotRhetoricCompare2_finegrained(data1 = us2016_df, data2 = conspiracy_df, rhet_dim_to_plot = compare_rhet_dim)

    elif (hansard_box and conspiracy_box and not us2016_box):
        plotRhetoricCompare2(data1 = hansard_df, data2 = conspiracy_df, rhet_dim_to_plot = compare_rhet_dim)
        add_spacelines(2)
        plotRhetoricCompare2_finegrained(data1 = hansard_df, data2 = conspiracy_df, rhet_dim_to_plot = compare_rhet_dim)

    elif (hansard_box and us2016_box and not conspiracy_box):
        plotRhetoricCompare2(data1 = hansard_df, data2 = us2016_df, rhet_dim_to_plot = compare_rhet_dim)
        add_spacelines(2)
        plotRhetoricCompare2_finegrained(data1 = hansard_df, data2 = us2016_df, rhet_dim_to_plot = compare_rhet_dim)

def CompareDatasetsHeroes():
    st.subheader("Compare Datasets on (Anti)heroes")
    add_spacelines(2)
    if us2016_box and conspiracy_box and not hansard_box:
        targets_all_conspiracy = ['the vaccinated', 'BBC', 'Reuters', 'Talkradio', 'CNN', 'they',
       'government', 'medics', 'USA', 'Fauci','Pro-vaccinators', 'leftists', 'media', 'big pharma',
       'Reddit users', 'the unvaccinated','The experts', 'Russians', 'the public', 'Americans',
       'the elites', 'social media', 'right wingers', 'Trump',
       'obese people', 'people','the elderly', 'scientists', 'drug addicts',
       'hospital patients', 'medical schools', 'hospitals',
       'pro-restriction', 'the rich', 'politicians', 'antivaxxers',
       'insurance companies', 'healthcare system', 'conspiracy theory believers']

        targets_all_us2016 = ['Webb', 'Sanders', 'Clinton', 'Trump', 'Paul',
       'Romney', 'Obama', 'Cooper', 'Democrats', 'Russia',
       'NATO', 'Republicans', 'Government','O Malley',
       'BLM', 'Chafee', 'CNN', 'Bush', 'Snowden', 'the Times', 'USA', 'Huckabee',
       'Christie', 'Joe Rogan', 'John Oliver', 'Sarah Palin',
       'Fox News', 'Kasich', 'Perry', 'Rubio', 'Cruz', 'Kelly','Carson', 'Facebook',
       'Walker', 'Left', 'Conservatives', 'Trudeau','Bill Clinton', 'Holt', 'McCain',
       'Supporters Trump', 'Biden','GOP', 'Kaplan', 'Media', 'Occupy']

        data1 = us2016_df[us2016_df.Target.isin(targets_all_us2016)]
        data2 = conspiracy_df[conspiracy_df.Target.isin(targets_all_conspiracy)]
        title1 = "US2016 Presidential Debate Reddit"
        title2 = "Conspiracy Theories Reddit"

    elif conspiracy_box and hansard_box and not us2016_box:
        concat_cols = ['Text', 'Target', 'Ethos_Label', 'ethos_name']

        targets_all_conspiracy = ['the vaccinated', 'BBC', 'Reuters', 'Talkradio', 'CNN', 'they',
       'government', 'medics', 'USA', 'Fauci','Pro-vaccinators', 'leftists', 'media', 'big pharma',
       'Reddit users', 'the unvaccinated','The experts', 'Russians', 'the public', 'Americans',
       'the elites', 'social media', 'right wingers', 'Trump',
       'obese people', 'people','the elderly', 'scientists', 'drug addicts',
       'hospital patients', 'medical schools', 'hospitals',
       'pro-restriction', 'the rich', 'politicians', 'antivaxxers',
       'insurance companies', 'healthcare system', 'conspiracy theory believers']

        data1 = conspiracy_df[conspiracy_df.Target.isin(targets_all_conspiracy)]

        len_targets = hansard_df.groupby("Target", as_index = False).size()
        len_targets = len_targets[len_targets['size'] > 2]
        data2 = hansard_df[hansard_df.Target.isin(len_targets.Target.unique())]

        title1 = "Conspiracy Theories Reddit"
        title2 = "Hansard"

    elif us2016_box and hansard_box and not conspiracy_box:
        concat_cols = ['Text', 'Target', 'Ethos_Label', 'ethos_name']

        targets_all_us2016 = ['Webb', 'Sanders', 'Clinton', 'Trump', 'Paul',
       'Romney', 'Obama', 'Cooper', 'Democrats', 'Russia',
       'NATO', 'Republicans', 'Government','O Malley',
       'BLM', 'Chafee', 'CNN', 'Bush', 'Snowden', 'the Times', 'USA', 'Huckabee',
       'Christie', 'Joe Rogan', 'John Oliver', 'Sarah Palin',
       'Fox News', 'Kasich', 'Perry', 'Rubio', 'Cruz', 'Kelly','Carson', 'Facebook',
       'Walker', 'Left', 'Conservatives', 'Trudeau','Bill Clinton', 'Holt', 'McCain',
       'Supporters Trump', 'Biden','GOP', 'Kaplan', 'Media', 'Occupy']

        data1 = us2016_df[us2016_df.Target.isin(targets_all_us2016)]

        len_targets = hansard_df.groupby("Target", as_index = False).size()
        len_targets = len_targets[len_targets['size'] > 2]
        data2 = hansard_df[hansard_df.Target.isin(len_targets.Target.unique())]

        title1 = "US2016 Presidential Debate Reddit"
        title2 = "Hansard"

    dd = pd.DataFrame(data1.groupby(['Target'])['Ethos_Label'].value_counts(normalize=True))
    dd.columns = ['normalized_value']
    dd = dd.reset_index()
    dd = dd[dd.Ethos_Label != 0]
    dd_hero = dd[dd.Ethos_Label == 1]
    dd_antihero = dd[dd.Ethos_Label == 2]
    dd2 = pd.DataFrame({'Target': dd.Target.unique()})
    dd2['score'] = np.nan
    for t in dd.Target.unique():
        try:
            h = dd_hero[dd_hero.Target == t]['normalized_value'].iloc[0]
        except:
            h = 0
        try:
            ah = dd_antihero[dd_antihero.Target == t]['normalized_value'].iloc[0]
        except:
            ah = 0
        i = dd2[dd2.Target == t].index
        dd2.loc[i, 'score'] = h - ah
    dd2['Ethos_Label'] = np.where(dd2.score < 0, 'anti-hero', 'neutral')
    dd2['Ethos_Label'] = np.where(dd2.score > 0, 'hero', dd2['Ethos_Label'])
    dd2 = dd2.sort_values(by = ['Ethos_Label', 'Target'])
    dd2['score'] = dd2['score'] * 100
    dd2['score'] = np.where(dd2.score == 0, 2, dd2['score'])

    dd_conspiracy = pd.DataFrame(data2.groupby(['Target'])['Ethos_Label'].value_counts(normalize=True))
    dd_conspiracy.columns = ['normalized_value']
    dd_conspiracy = dd_conspiracy.reset_index()
    dd_conspiracy = dd_conspiracy[dd_conspiracy.Ethos_Label != 0]
    dd_conspiracy_hero = dd_conspiracy[dd_conspiracy.Ethos_Label == 1]
    dd_conspiracy_antihero = dd_conspiracy[dd_conspiracy.Ethos_Label == 2]

    dd_conspiracy2 = pd.DataFrame({'Target': dd_conspiracy.Target.unique()})
    dd_conspiracy2['score'] = np.nan
    for t in dd_conspiracy.Target.unique():
        try:
            h = dd_conspiracy_hero[dd_conspiracy_hero.Target == t]['normalized_value'].iloc[0]
        except:
            h = 0
        try:
            ah = dd_conspiracy_antihero[dd_conspiracy_antihero.Target == t]['normalized_value'].iloc[0]
        except:
            ah = 0
        i = dd_conspiracy2[dd_conspiracy2.Target == t].index
        dd_conspiracy2.loc[i, 'score'] = h - ah

    dd_conspiracy2['Ethos_Label'] = np.where(dd_conspiracy2.score < 0, 'anti-hero', 'neutral')
    dd_conspiracy2['Ethos_Label'] = np.where(dd_conspiracy2.score > 0, 'hero', dd_conspiracy2['Ethos_Label'])
    dd_conspiracy2 = dd_conspiracy2.sort_values(by = ['Ethos_Label', 'Target'])
    dd_conspiracy2['score'] = dd_conspiracy2['score'] * 100
    dd_conspiracy2['score'] = np.where(dd_conspiracy2.score == 0, 2, dd_conspiracy2['score'])

    dd2['Dataset'] = str(title1)
    dd_conspiracy2['Dataset'] = str(title2)

    dd2 = pd.concat([dd2, dd_conspiracy2], axis = 0)

    color = sns.color_palette("Reds", 5)[-1:]  + sns.color_palette("Greens", 5)[::-1][:1] +  sns.color_palette("Blues", 5)[::-1][:1]
    sns.set(font_scale=4)
    f = sns.catplot(kind = 'bar', data = dd2, y = 'Target', x = 'score',
                   hue = 'Ethos_Label', palette = color, dodge = False, sharey=False,
                   aspect = 1, height = 40, alpha = 1, legend = False, col = "Dataset")
    #plt.xticks(np.arange(-100, 101, 20), fontsize=16)
    #plt.yticks(fontsize=16)
    #plt.xlabel("\nscore", fontsize=18)
    plt.ylabel("")
    f.set_axis_labels('\nscore', '')
    plt.legend(fontsize=45, title = '', bbox_to_anchor=(0.1, 1.12), ncol = 3)
    plt.tight_layout()
    sns.set(font_scale=4)
    plt.show()
    st.pyplot(f)





style_css(r"multi_style.css")


# sidebar
with st.sidebar:
    st.write('<style>div[class="css-1siy2j7 e1fqkh3o3"] > div{background-color: #ECEBEB;}</style>', unsafe_allow_html=True)
    st.write('<style>div.row-widget.stRadio > div{flex-direction:column;}</style>', unsafe_allow_html=True)
    st.title("Contents")
    contents_radio = st.radio("",
    ("Main Page", "Single Dataset Analytics", "Compare Datasets"))
    #add_spacelines(1)


if contents_radio == 'Compare Datasets':
    with st.sidebar:
        st.write("Choose datasets")
        us2016_box = st.checkbox("US2016 Presidential Debate Reddit", value=True)
        conspiracy_box = st.checkbox("Conspiracy Theories Reddit", value=True)
        hansard_box = st.checkbox("Hansard")
        add_spacelines(2)

        if us2016_box:
            us2016_df = pd.read_excel(r"app_US2016.xlsx", index_col = 0)
            us2016_df['Dataset'] = "US2016 Presidential Debate Reddit"

        if conspiracy_box:
            conspiracy_df = pd.read_excel(r"app_conspiracy.xlsx", index_col = 0)
            conspiracy_df['Dataset'] = "Conspiracy Theories Reddit"

        if hansard_box:
            hansard_df_ethos = pd.read_excel(r"app_hansard_ethos.xlsx", index_col = 0)
            hansard_df_logos = pd.read_excel(r"app_hansard_logos.xlsx", index_col = 0)
            hansard_df_ethos['Dataset'] = "Hansard"
            hansard_df_logos['Dataset'] = "Hansard"

        if np.sum([int(us2016_box), int(conspiracy_box), int(hansard_box)]) < 2:
            add_spacelines(1)
            st.warning("**You need to select at least 2 dataset.**")
            add_spacelines(1)

        contents_radio3 = st.radio("Analytics",
            ("Text-Level Analysis", '(Anti)Heroes'))
        add_spacelines(4)

elif contents_radio == "Main Page":
    with st.sidebar:
        add_spacelines(24)

elif contents_radio == "Single Dataset Analytics":
    # sidebar
    with st.sidebar:
        add_spacelines(1)
        st.write("Choose datasets")
        datasets_singles_us2016 = st.checkbox("US2016 Presidential Debate Reddit", value=True)
        datasets_singles_conspiracy = st.checkbox("Conspiracy Theories Reddit")
        #datasets_singles_hansard = st.checkbox("Hansard")
        add_spacelines(1)

        add_spacelines(1)
        if datasets_singles_conspiracy and datasets_singles_us2016:
            concat_cols = ['map_ID', 'Text', 'Source', 'Target', 'No_Ethos', 'Contains_ethos',
                       'Support', 'Attack', 'Ethos_Label', 'No_pathos', 'Contains_pathos',
                       'positive_valence', 'negative_valence', 'happiness', 'anger', 'sadness',
                        'fear', 'disgust', 'surprise', 'trust', 'anticipation', 'RA_logos',
                        'RA_relation', 'CA_logos', 'CA_relation', 'T5_emotion', 'expressed_sentiment',
                        'pathos_name', 'ethos_name', 'logos_name', 'clean_Text', 'clean_Text_lemmatized']
            df01 = load_dataset("US2016 Presidential Debate Reddit")
            df02 = load_dataset("Conspiracy Theories Reddit")
            df = pd.concat([df01[concat_cols], df02[concat_cols]], axis = 0)
        elif datasets_singles_conspiracy:
            df = load_dataset("Conspiracy Theories Reddit")
        elif datasets_singles_us2016:
            df = load_dataset("US2016 Presidential Debate Reddit")

        #st.header("Analytics")
        contents_radio2 = st.radio("Analytics",
            ("Text-Level Analysis", "High Precision Words - WordCloud", '(Anti)Heroes',
             " Rhetoric Strategies", ' Rhetoric Metric', 'Users Profiles'))
        add_spacelines(3)


if contents_radio == "Main Page":
    MainPage()

elif contents_radio == "Single Dataset Analytics" and contents_radio2 == "Text-Level Analysis":
    basicLEPAn()

elif contents_radio == "Single Dataset Analytics" and contents_radio2 == "High Precision Words - WordCloud":
    generateWordCloud()

elif contents_radio == "Single Dataset Analytics" and contents_radio2 == '(Anti)Heroes':
    TargetHeroScores()

elif contents_radio == "Single Dataset Analytics" and contents_radio2 == " Rhetoric Strategies":
    UserRhetStrategy()

elif contents_radio == "Single Dataset Analytics" and contents_radio2 == ' Rhetoric Metric':
    UserRhetMetric()

elif contents_radio == "Single Dataset Analytics" and contents_radio2 == 'Users Profiles':
    UsersExtreme()

elif contents_radio == 'Compare Datasets' and contents_radio3 == "Text-Level Analysis":
    CompareDatasetsText()

elif contents_radio == 'Compare Datasets' and contents_radio3 == "(Anti)Heroes":
    if hansard_box:
        hansard_df = hansard_df_ethos
    if np.sum([int(us2016_box), int(conspiracy_box), int(hansard_box)]) > 1:
        CompareDatasetsHeroes()
    else:
        st.subheader("Compare Datasets on (Anti)heroes")
        add_spacelines(5)
        st.warning("**You need to select at least 2 dataset.**")

else:
    MainPage()

