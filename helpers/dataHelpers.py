import pandas as pd
import numpy as np
def fillColumnList(x, index):
    """populates a record for a series 
       overrides index and populates None if out of range."""
    try:
        return x[index]
    except Exception as e:
        raise e

def maxList(df, index):
    """returns the max number of a index series."""
    return df[index].max()

def uniformList(x, index):  
    """creates a list of the same lenght"""
    return x[index] + ['nosubgenre'] * (6 - len(x['GenresList']))

def moveGames(x, index):
    """deletes games in list and cast literal in new series."""
    del x[index][x[index].index('games')]
    return 'games'

def subGenreDataFrame(df):
    """create SubGenre columns."""
    for index in range(5):
        df[f'SubGenre_{index}'] = df['GenresList'].apply(lambda x: fillColumnList(x, index))
    return df

def uniqueWords(df, index):
    """just gives me a set of uniqueWords."""
    uniqueWords = list()
    for array in df[index].to_list():
        for word in array:
            uniqueWords.append(word)
    return set(uniqueWords)


def VectorizeWordsDataFrame(df, index):
    """vectorize data frame."""
    newList = list()
    # create one list for the genres list
    for array in df[index].to_list():
        for subGenre in array:
            newList.append(subGenre)
    count = CountVectorizer()
    bag_of_words = count.fit_transform(newList)
    # Get feature names
    feature_names = count.get_feature_names()
    # Create data frame
    df_vector = pd.DataFrame(bag_of_words.toarray(), columns=feature_names)
    return df_vector


def hasSubGenres(df, index):
    """make a sub genre index"""
    return df[df[f'SubGenre_{index}'] != 'nosubgenre']
    

def hasConnection(df, index):
    """find a unique connection by sub genre."""
    npUnique = df[f'SubGenre_{index}'].unique()
    d = dict()
    for value in npUnique:
        d[value] = set()
        for element in df[df[f'SubGenre_{index}'] == value][f'SubGenre_{index + 1}']:
            d[value].add(element)
    return d


def wordListToFreqDict(wordlist):
    """compose a woord to frequency"""
    wordfreq = [wordlist.count(p) for p in wordlist]
    return dict(list(zip(wordlist,wordfreq)))

def sortFreqDict(freqdict):
    """sort the word"""
    aux = [(freqdict[key], key) for key in freqdict]
    aux.sort()
    aux.reverse()
    return tuple(aux)

def splitInApp(app):
    """split the app words"""
    if isinstance(app, str):
        return app.replace(' ', '').split(',')
    else:
        return app
    
def castInApp(app):
    """cast data type float"""
    if isinstance(app, list):
        newApp = []
        for a in app:
            newApp.append(float(a))
        return sum(newApp) 
    else:
        return app
    
def explode(df, lst_cols, fill_value=''):
    """explode the words"""
    if lst_cols and not isinstance(lst_cols, list):
        lst_cols = [lst_cols]
    idx_cols = df.columns.difference(lst_cols)
    lens = df[lst_cols[0]].str.len()

    if (lens > 0).all():
        return pd.DataFrame({
            col:np.repeat(df[col].values, df[lst_cols[0]].str.len())
            for col in idx_cols
        }).assign(**{col:np.concatenate(df[col].values) for col in lst_cols}) \
          .loc[:, df.columns]
    else:
        return pd.DataFrame({
            col:np.repeat(df[col].values, df[lst_cols[0]].str.len())
            for col in idx_cols
        }).assign(**{col:np.concatenate(df[col].values) for col in lst_cols}) \
          .append(df.loc[lens==0, idx_cols]).fillna(fill_value) \
          .loc[:, df.columns]
    
def removeMoreNoise(df):
    """remove the noise"""
    l = []
    for inputString in df.columns:
        if any(char.isdigit() for char in inputString):
            continue
        else:
            l.append(inputString)
    return sorted(list(set(l)))
        