from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

model = pipeline("text-generation", model="gpt2")

class GenreRequest(BaseModel):
    genre: str

@app.post("/recommend")
def recommend_book(request: GenreRequest):
    genre = request.genre
    top_100_books = [f"Book {i}" for i in range(1, 101)]
    top_10_books = top_100_books[:10]
    user_book = top_10_books[0]  # For simplicity, just pick the first book

    prompt = f"Why should you read {user_book}? This book is great because"
    recommendation = model(prompt, max_length=50)[0]['generated_text']
    return {"recommended_book": user_book, "recommendation": recommendation, "message": "Thank you for using the LLM-Based Book Recommender!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
