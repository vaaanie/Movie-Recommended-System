import numpy as np 
import pandas as pd 
import seaborn as sn 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity 
from difflib import get_close_matches 
 
df = pd.read_csv(r'/content/Top_10000_Movies.csv', lineterminator='\n') 
 
df[df['vote_average'] > 8.7] 
 
def rep(what, s): 
    for item in what: 
        s = s.replace(item, '') 
    return s 
 
df.fillna('', inplace=True) 
df['genre'] = df['genre'].apply(lambda x: rep([',', '[', ']', "'"], x)) 
df['new'] = df['original_title'] + ' ' + df['genre'] + ' ' + df['overview'] + 
' ' + df['tagline'] 
 
vectorizer = TfidfVectorizer() 
transformed_data = vectorizer.fit_transform(df['new'].values) 
 
sim = cosine_similarity(transformed_data) 
 
movie_name = input() 
closest_name = get_close_matches(movie_name, df.original_title.values, 1)[0] 
print(closest_name) 
movie = df[df['original_title'] == closest_name] 
 
 
movie_index = movie.index.values[0] 
ls = sorted(list(enumerate(sim[movie_index])), key=lambda x: x[1], 
reverse=True) 
for i in ls[1:9]: 
    print(df.loc[i[0]].original_title)