# AI Flight Support Assistant

A beginner-friendly AI customer support prototype built with Streamlit and OpenRouter.

## What it does
- Answers airline support questions
- Uses policy text as context
- Demonstrates a simple Retrieval-Augmented Generation (RAG) workflow

## Tech stack
- Python
- Streamlit
- OpenRouter
- OpenAI Python SDK
- scikit-learn

## Example questions
- Can I get compensation for a 4 hour delay?
- How many cabin bags can I bring?
- Can I get a refund if my flight is delayed for 6 hours?

## Architecture
User question
-> Streamlit app
-> Policy search
-> Relevant policy chunks
-> OpenRouter free model
-> Final answer