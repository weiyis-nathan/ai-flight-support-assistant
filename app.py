import streamlit as st
from openai import OpenAI
from knowledge_base import build_knowledge_base, search_knowledge_base

OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

st.set_page_config(page_title="AI Flight Support Assistant", page_icon="✈️")
st.title("✈️ AI Flight Support Assistant")
st.write("Ask about delays, refunds, baggage, or airline policies.")

try:
    policy_chunks = build_knowledge_base("airline_policy.txt")
except Exception as e:
    st.error(f"Error loading knowledge base: {e}")
    st.stop()

question = st.text_input("Ask a question about your flight")

if question:
    try:
        relevant_chunks = search_knowledge_base(question, policy_chunks, top_k=3)

        if relevant_chunks:
            context = "\n\n".join(relevant_chunks)
        else:
            context = "No relevant policy text found."

        system_prompt = """
You are a helpful airline customer support assistant.

Answer the user's question using ONLY the provided policy context.
If the answer is not clearly stated in the context, say:
"I’m not sure based on the current policy information provided."
Keep the answer clear, concise, and helpful.
"""

        user_prompt = f"""
Policy context:
{context}

User question:
{question}
"""

        response = client.chat.completions.create(
            model="openrouter/free",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

        answer = response.choices[0].message.content

        st.subheader("Answer")
        st.write(answer)

        with st.expander("See retrieved policy context"):
            for i, chunk in enumerate(relevant_chunks, start=1):
                st.markdown(f"**Chunk {i}:**")
                st.write(chunk)

    except Exception as e:
        st.error(f"Error: {e}")