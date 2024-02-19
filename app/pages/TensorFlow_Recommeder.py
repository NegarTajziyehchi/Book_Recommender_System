import tensorflow_recommenders as tfrs
import tensorflow as tf
import numpy as np
import pandas as pd
import streamlit as st
import sys
print(sys.path)
sys.path.append("/home/ubuntu/recommender_system/")  # Adjust path as necessary
from scripts.TFRSModel import build_model
from scripts.TFRSModel import return_book_titles  # Ensure these functions are correctly defined

book_titles = return_book_titles()

# Load the model (adjust caching as needed for your application)
@st.cache_resource
def load_model():
    model = build_model()
    model.load_weights("/home/ubuntu/recommender_system/model/model_weights/")  # Update path accordingly
    return model

model = load_model()

emoji_1, emoji_2 = "\U0001F4DA", "\U0001F31F"

# Streamlit UI
st.title(f"Book Recommender System {emoji_1} {emoji_2}")

with st.form("recommendation_form"):
    input_user = st.text_input("Enter your User-ID:", "")
    input_author = st.text_input("Enter an Author name:", "")
    input_age = st.number_input("Enter your Age:", min_value=0, max_value=100, value=25, step=1)
    top_k = st.number_input("Number of recommendations:", min_value=1, max_value=20, value=5, step=1)
    
    submit_button = st.form_submit_button("Get Recommendations")

    if submit_button:
        if input_user and input_author and input_age:
            
            # Setup for recommendations
            index = tfrs.layers.factorized_top_k.BruteForce(model.user_model, k=top_k)
            index.index_from_dataset(
                tf.data.Dataset.zip((book_titles.batch(1000), book_titles.batch(1000).map(model.title_model)))
            )
            
            raw_input = {
                'Age': np.array([input_age]),
                'Book-Author': np.array([input_author]),
                'user': np.array([input_user])
            }
            
            input_dict = {key: tf.constant(value) for key, value in raw_input.items()}
            
            _, titles = index(input_dict)
            
            test_rating = {}
            for book in titles.numpy()[0]:
                raw_input['Book-Title'] = book.decode("utf-8")  # Assuming book titles are decoded properly
                
                input_dict = {key: tf.constant(np.array([value])) for key, value in raw_input.items()}
                
                trained_book_embeddings, trained_user_embeddings, predicted_rating = model(input_dict)
                test_rating[book] = predicted_rating.numpy()[0][0]
            
            sorted_dict = sorted(test_rating.items(), key=lambda x: x[1], reverse=True)
            
            # Display recommendations
            st.write(f"Top {top_k} recommendations for User: {input_user}")
            for book, score in sorted_dict:
                st.write(f"- {book.decode('utf-8')} (Score: {score:.2f})")  # Ensure proper decoding and formatting
        else:
            st.error("Please fill out all fields.")
