import streamlit as st
from transformers import pipeline

# Load the model and tokenizer from Hugging Face
@st.cache(allow_output_mutation=True)
def load_model():
    return pipeline("text-generation", model="gpt2")

model = load_model()

st.title("LLM-Based Book Recommender")

# Step 1: User asks for top 100 books in a genre
genre = st.text_input("Enter a genre to find top 100 books:")
if genre:
    st.write(f"Finding top 100 books in the genre: {genre}")

    # Dummy list of top 100 books
    top_100_books = [f"Book {i}" for i in range(1, 101)]
    st.write(top_100_books)

    # Step 2: Agent finds top 10 books from the 100
    top_10_books = top_100_books[:10]
    st.write("Top 10 books in the genre:", top_10_books)

    # Step 3: Agent finds 1 book for the user
    user_book = st.selectbox("Select a book from the top 10 for a detailed recommendation:", top_10_books)
    if user_book:
        st.write(f"Recommending the book: {user_book}")

        # Generate a personalized message using the LLM
        prompt = f"Why should you read {user_book}? This book is great because"
        recommendation = model(prompt, max_length=50)[0]['generated_text']
        st.write(recommendation)

        # Step 4: Conclude the workflow with a thank you message
        st.write("Thank you for using the LLM-Based Book Recommender!")
