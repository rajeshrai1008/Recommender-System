#Item Based Collaborative Filtering

#Read Data
import pandas as pd

r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('D:\ITM\Udemy Data Science\DataScience\DataScience\ml-100k\u.data', sep='\t', names=r_cols, usecols=range(3))

m_cols = ['movie_id', 'title']
movies = pd.read_csv('D:\ITM\Udemy Data Science\DataScience\DataScience\ml-100k\u.item', sep='|', names=m_cols, usecols=range(2))

ratings = pd.merge(movies, ratings)

ratings.head()

#Out[25]: 
#   movie_id             title  user_id  rating
#0         1  Toy Story (1995)      308       4
#1         1  Toy Story (1995)      287       5
#2         1  Toy Story (1995)      148       4
#3         1  Toy Story (1995)      280       4
#4         1  Toy Story (1995)       66       3


#Now create user/movie rating matrix
movieRatings = ratings.pivot_table(index=['user_id'],columns=['title'],values='rating')
movieRatings.head()

#Partial Output
#title    'Til There Was You (1997)  1-900 (1994)  101 Dalmatians (1996)  \
#user_id                                                                   
#0                              NaN           NaN                    NaN   
#1                              NaN           NaN                    2.0   
#2                              NaN           NaN                    NaN   
#3                              NaN           NaN                    NaN   
#4                              NaN           NaN                    NaN   

toyStoryRatings = movieRatings['Toy Story (1995)']
toyStoryRatings.head()

#Find correlation between the toy story's vector of user rating with the other movie.

similarMovies = movieRatings.corrwith(toyStoryRatings)
similarMovies = similarMovies.dropna()
df = pd.DataFrame(similarMovies)
df.head(10)

#                                           0
#title                                        
#'Til There Was You (1997)            0.534522
#101 Dalmatians (1996)                0.232118
#12 Angry Men (1957)                  0.334943
#187 (1997)                           0.651857
#2 Days in the Valley (1996)          0.162728
#20,000 Leagues Under the Sea (1954)  0.328472
#2001: A Space Odyssey (1968)        -0.069060
#39 Steps, The (1935)                 0.150055
#8 1/2 (1963)                        -0.117259
#8 Heads in a Duffel Bag (1997)       0.500000

similarMovies.sort_values(ascending=False)

#After running above command you may notice that the results are messed up. This can be because of some movies being watched by only a handfull of people.
# We will eliminate movies that are watched less than 100 times

import numpy as np
movieStats = ratings.groupby('title').agg({'rating': [np.size, np.mean]})
movieStats.head()

popularMovies = movieStats['rating']['size'] >= 100
movieStats[popularMovies].sort_values([('rating', 'mean')], ascending=False)[:15]

df = movieStats[popularMovies].join(pd.DataFrame(similarMovies, columns=['similarity']))

df.head()

df.sort_values(['similarity'], ascending=False)[:15]

#                               (rating, size)  (rating, mean)  similarity
#title                                                                    
#Toy Story (1995)                          452        3.878319    1.000000
#Craft, The (1996)                         104        3.115385    0.549100
#Down Periscope (1996)                     101        2.702970    0.457995
#Miracle on 34th Street (1994)             101        3.722772    0.456291
#G.I. Jane (1997)                          175        3.360000    0.454756
#Amistad (1997)                            124        3.854839    0.449915
#Beauty and the Beast (1991)               202        3.792079    0.442960
#Mask, The (1994)                          129        3.193798    0.432855
#Cinderella (1950)                         129        3.581395    0.428372
#That Thing You Do! (1996)                 176        3.465909    0.427936
#Lion King, The (1994)                     220        3.781818    0.426778
#Aladdin (1992)                            219        3.812785    0.411731
#Great Escape, The (1963)                  124        4.104839    0.401238
#African Queen, The (1951)                 152        4.184211    0.397874
#Dumbo (1941)                              123        3.495935    0.387716
