import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import requests

st.header('Movie Recommender System')
movies = pickle.load(open('movie_list.pkl', 'rb'))


cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vector)

movie_list = movies['title'].values
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)


def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    for i in distances[1:6]:
        # fetch the movie poster
        recommended_movie_names.append(movies.iloc[i[0]].title)

    return recommended_movie_names


if st.button('Show Recommendation'):
    recommended_movie_names = recommend(selected_movie)
    for movie in recommended_movie_names:
        st.write(movie)
