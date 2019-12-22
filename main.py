import pandas as pd
import numpy as np
import re
import collections
import nltk
import matplotlib.pyplot as plt
import warnings
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from helpers.dataHelpers import fillColumnList
from helpers.dataHelpers import maxList
from helpers.dataHelpers import uniformList
from helpers.dataHelpers import moveGames
from helpers.dataHelpers import subGenreDataFrame
from helpers.dataHelpers import uniqueWords
from helpers.dataHelpers import VectorizeWordsDataFrame
from helpers.dataHelpers import hasSubGenres
from helpers.dataHelpers import hasConnection
from helpers.dataHelpers import wordListToFreqDict
from helpers.dataHelpers import sortFreqDict
from helpers.dataHelpers import splitInApp
from helpers.dataHelpers import castInApp
from helpers.dataHelpers import explode
from helpers.dataHelpers import removeMoreNoise
from nltk.corpus import stopwords
from ripser import ripser
from persim import plot_diagrams 
stop = stopwords.words('english')
warnings.filterwarnings('ignore')

def main():
    df1 = pd.read_csv('data/appstore_games.csv')
    df_games = df1[df1['Genres'].apply(lambda x: 'Games' in x.split(',')[0])]
    df_games['GenresList'] = df_games['Genres'].apply(lambda x: sorted(x.lower().replace('&', '').replace(' ', '').split(',')))
    df_games['GenresList'] = df_games.apply(lambda x: uniformList(x, 'GenresList'), axis=1)
    df_games['AppType'] = df_games.apply(lambda x: moveGames(x, 'GenresList'), axis=1)
    df_games = subGenreDataFrame(df=df_games)
    df_games['GenresList'] = df_games.apply(lambda x: tuple(uniformList(x, 'GenresList')), axis=1)
    df_games['GenresSplit'] = df_games['GenresList'].apply(lambda x: ','.join(x))
    df_games['Current Version Release Date'] = pd.to_datetime(df_games['Current Version Release Date'])
    df_games['Original Release Date'] = pd.to_datetime(df_games['Original Release Date'])

    # remove stop words
    df_games['Description'] = df_games['Description'].apply(lambda x: x.lower().split('\n'))
    df_games['Description'] = df_games['Description'].apply(lambda x: [y.replace('\\n', '') for y in x])
    df_games['Description'] = df_games['Description'].apply(lambda x: [y.replace('\\u', '') for y in x])
    df_games['Description'] = df_games['Description'].apply(lambda x: tuple([y.replace('2022', '') for y in x]))
    df_games['stopWordsRemoved'] = df_games['Description'].apply(lambda x: [item for item in x[0].split() if item not in stop])
    df_games['stopWordsRemoved'] = df_games['stopWordsRemoved'].apply(lambda x: [re.sub(r"[^a-zA-Z0-9]+", ' ', k) for k in x])
    df_games['stopWordsRemoved'] = df_games['stopWordsRemoved'].apply(lambda x: [k.replace('u2022', '') for k in x])
    df_games['stopWordsRemoved'] = df_games['stopWordsRemoved'].apply(lambda x: [k.replace(' ', '') for k in x])
    df_games['stopWordsRemoved'] = df_games['stopWordsRemoved'].apply(lambda x: [item for item in x if item not in stop])
    df_games['stopWordsRemoved'] = df_games['stopWordsRemoved'].apply(lambda x: list(filter(None, x)))
    df_games['stopWordsRemoved'] = df_games['stopWordsRemoved'].apply(lambda x: sortFreqDict(wordListToFreqDict(x)))
    df_games['In-app Purchases'] = df_games['In-app Purchases'].apply(lambda x: splitInApp(x))
    df_games['In-app Purchases'] = df_games['In-app Purchases'].apply(lambda x: castInApp(x))
    df_games = df_games.fillna(0)

    df_games['stopWordsRemoved'] = df_games['stopWordsRemoved'].apply(lambda x: list(x))
    df_games['stopWordsRemoved'] = df_games['stopWordsRemoved'].apply(lambda x: x[0:4])
    df_games['categories'] = df_games['stopWordsRemoved'].apply(lambda x: [i[1] for i in x])
    df_games['categoryCount'] = df_games['stopWordsRemoved'].apply(lambda x: [i[0] for i in x])
    df_games_exploded = explode(df_games, lst_cols=['categories', 'categoryCount'])
    df = df_games_exploded[['Name', 'Subtitle', 'Icon URL', 'Average User Rating',
                            'User Rating Count', 'Price', 'In-app Purchases','Age Rating', 
                            'Original Release Date', 'Current Version Release Date',
                            'AppType', 'SubGenre_0', 'SubGenre_1', 'SubGenre_2',
                            'SubGenre_3', 'SubGenre_4', 
                            'categories', 'categoryCount']]
    df_groupby = df.groupby(['SubGenre_0', 'Name', 'categories'])
    df_mean = df_groupby.mean().reset_index() 
    df_mean['ratio_name_assn'] = df_mean.Name.transform(lambda x: len(x)/ len(df))
    df_mean['ration_name_recv'] = df_mean['ratio_name_assn'].transform(lambda x: x/df.categoryCount.sum())
    df_mean['ratio_sub_assn'] = df_mean.SubGenre_0.transform(lambda x: len(x)/ len(df))
    df_mean['ration_sub_recv'] = df_mean['ratio_sub_assn'].transform(lambda x: x/df.categoryCount.sum())
    df_mean['ratio_categories_assn'] = df_mean.categories.transform(lambda x: len(x)/ len(df))
    df_mean['ration_categories_recv'] = df_mean['ratio_categories_assn'].transform(lambda x: x/df.categoryCount.sum())
    data = df_mean[['Average User Rating',
                    'User Rating Count', 'Price', 'In-app Purchases', 'categoryCount',
                    'ratio_name_assn', 'ration_name_recv', 'ratio_sub_assn',
                    'ration_sub_recv', 'ratio_categories_assn', 'ration_categories_recv']].head(5000).values
    return data, df_mean, df_games, df


if __name__ == "__main__":
    print(main())
