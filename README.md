﻿# Design Document for LLM-Based Book Recommender

## Introduction
This document outlines the design and implementation of a simple LLM-based autonomous agent to recommend books in various genres. The project uses Streamlit for the user interface and Hugging Face's GPT-2 model for generating personalized book recommendations.

## Approach
We chose Streamlit for its simplicity in creating interactive web applications. Hugging Face's GPT-2 model was selected for its strong natural language generation capabilities.

## Workflow
1. The user inputs a genre.
2. The agent provides a list of the top 100 books in that genre.
3. The agent narrows down the list to the top 10 books.
4. The user selects a book from the top 10 for a detailed recommendation.
5. The agent generates a personalized message about the selected book.
6. The workflow concludes with a thank you message.

## Reasoning
The approach ensures a user-friendly interface and leverages powerful language models to provide meaningful recommendations. Streamlit's caching mechanism helps in efficient model loading and interaction.

## Future Enhancements
1. Integration with real book databases for dynamic data.
2. Advanced filtering options based on user preferences.
3. Expanding the recommendation system to include more genres and sub-genres.

4. [![Watch the video](https://img.youtube.com/vi/I-62OOJoSF8/maxresdefault.jpg)](https://youtu.be/I-62OOJoSF8)

