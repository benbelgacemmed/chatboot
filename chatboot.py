import streamlit as st

import os 
from langchain_core.prompts import ChatPromptTemplate , HumanMessagePromptTemplate , SystemMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv('GROQ_API_KEY') 

st.title("Chat Boot")

#Creation of LLM's
prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template('you are a AI assistant at first , but in otherwise i will give you some information about a woman and you will saved it and when you ask for this information you aswer ok here is the information {information} '),
        HumanMessagePromptTemplate.from_template('{user_question}')
    ]
)

model_one = ChatGroq(model = 'llama-3.1-8b-instant' , api_key=api_key ) 
model_two = ChatGroq(model = 'qwen/qwen3-32b' , api_key=api_key ) 
model_thre = ChatGroq(model = 'openai/gpt-oss-safeguard-20b' , api_key=api_key ) 
model_four = ChatGroq(model = 'openai/gpt-oss-20b' , api_key=api_key ) 
llm = model_one.with_fallbacks([
    model_two,
    model_thre,
    model_four
])
parser = StrOutputParser()

chain = prompt_template | llm | parser



# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#user input 
user_input = st.chat_input('Ask what you want ?')


if user_input:
    #Show user message
    st.chat_message('user').markdown(user_input)
    #Save user message
    st.session_state.messages.append({"role" : "user" , "content" : user_input})

    #Show streamed response
    with st.chat_message("ai"):
        stremmed_text = st.write_stream(chain.stream({"user_question" : user_input }))

    #Save AI reply 
    st.session_state.messages.append({"role" : "ai" , "content" : stremmed_text})
