from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import streamlit as st
from streamlit_chat import message
from utils import *
import pinecone
import openai



hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

headers = {
     "auhorization": st.secrets["OPENAI_API_KEY"],
     "auhorization": st.secrets["PINECONE_API_KEY"],
     "auhorization": st.secrets["openai.api_key"],
     "content-type": "application/json"
}


st.subheader("Ask ChanakyaGPT")

OPENAI_API_KEY = ("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")
PINECONE_API_KEY = ("8c2079b0-03bf-40e3-aacd-d41e5bd4c589")
openai.api_key = "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"

####################################################

pinecone.init(api_key=PINECONE_API_KEY,environment='us-west1-gcp-free' )
index = pinecone.Index('chann')
from langchain.vectorstores import Chroma, Pinecone

from langchain.embeddings.openai import OpenAIEmbeddings
model_name = 'text-embedding-ada-002'
text_field = "text"
embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=OPENAI_API_KEY
)
vectorstore = Pinecone(
    index, embed.embed_query, text_field
)

def find_match(input):
    sum = vectorstore.similarity_search(
    query,  # our search query
    k=1  # return 3 most relevant docs
    )
    return sum

def query_refiner(conversation, query):

    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response['choices'][0]['text']


def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string



if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY, temperature=1.4,streaming=True)

if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)


system_msg_template = SystemMessagePromptTemplate.from_template(template="""You are Chanakya. Answer the question as chanakya would answer, using the provided context of chanakya's work as a reference. 
                                                                The context provided is timeless, but the questions asked might be related to modern problems of man. Try to answer with respect to question and sound like chanakya.
                                                                Answer in first person as chanakya.
                                                                if the answer is not contained within the text below, say 'Ask me something relevant'""")


human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)




# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()


with textcontainer:
    query1 = st.text_input("Query: ", key="input")
    query = ("Hey Chanakya,") + query1
    if query1:
        with st.spinner("typing..."):
            conversation_string = get_conversation_string()
            st.code(conversation_string)
            refined_query = query_refiner(conversation_string, query)
            #st.subheader("Refined Query:")
            #st.write(refined_query)
            context = find_match(refined_query)
            #print(context)  
            response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
        st.session_state.requests.append(query)
        st.session_state.responses.append(response)
with response_container:
    if st.session_state['responses']:

        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')
