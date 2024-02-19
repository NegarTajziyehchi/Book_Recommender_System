import streamlit as st

def main():
    emoji_1, emoji_2 = "\U0001F4DA", "\U0001F31F"
    st.title(f"Book Recommender System {emoji_1} {emoji_2}")  
    st.markdown(
    """
    This Streamlit app leverages the power of **TensorFlow Recommenders** to offer you personalized book suggestions. Dive into a world of literature curated just for you, and discover your next favorite read with precision and ease!

    """
    )

    st.image("/home/ubuntu/recommender_system/app/image.png")

if __name__ == "__main__":
    main()
 