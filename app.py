import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the saved pickle files
with open('user_encoder.pkl', 'rb') as f:
    user_encoder = pickle.load(f)

with open('user_similarity.pkl', 'rb') as f:
    user_similarity = pickle.load(f)

with open('user_item_matrix.pkl', 'rb') as f:
    user_item_matrix = pickle.load(f)

# Function to recommend products
def recommend_products(user_id, user_similarity, user_item_matrix):
    user_index = user_encoder[user_id]  # Use the manually encoded UserId
    similarity_scores = user_similarity[user_index]

    # Get top-k similar users
    k = 3
    similar_users = np.argsort(similarity_scores)[-k:][::-1]
    
    # Get the products rated by similar users but not rated by the current user
    unrated_products = user_item_matrix.loc[user_id, :].isna()
    unrated_products = unrated_products.index.values

    # Get similar users' ratings for unrated products
    similar_user_ratings = user_item_matrix.iloc[similar_users, :]

    # Calculate the weighted average of ratings from similar users
    recommended_ratings = similar_user_ratings.loc[:, unrated_products].mean(axis=0)
    
    # Return the top-k recommended products
    recommended_ratings_sorted = recommended_ratings.sort_values(ascending=False)
    return recommended_ratings_sorted.index[:k]

# Streamlit UI setup
st.title('Best Detergent Product Recommendation System')

# Instructions for the user
st.write("Welcome to the product recommendation system! This tool recommends the best detergent products for you based on your past ratings and similar users' preferences.")

# Input for UserId selection
user_id = st.text_input('Enter UserId for recommendations', 'User1')

# Check if UserId is in the encoder
if user_id not in user_encoder:
    st.error(f"User ID '{user_id}' not found. Please try another UserId.")

# Display recommendations button
if st.button('Recommend Products') and user_id in user_encoder:
    recommended_products = recommend_products(user_id, user_similarity, user_item_matrix)
    
    # Display the recommended products
    st.write(f"Top recommended products for **{user_id}**:")
    
    # Create columns to display each recommended product in a neat format
    cols = st.columns(3)
    
    for i, product in enumerate(recommended_products):
        with cols[i % 3]:  # Distribute products across 3 columns
            st.subheader(f"Product {i + 1}")
            st.write(product)  # Display product name (or any additional info)
            # You can replace the line above with more information like product description if available

    st.success("Recommendations fetched successfully!")

# Footer with a message
st.markdown("""
---
**Note:** The product recommendations are based on user similarity and past ratings. For the best experience, make sure to input a valid UserId from the system.
""")
