from dotenv import load_dotenv, dotenv_values
import streamlit as st
import torch
import os
from pathlib import Path
from huggingface_hub import login
load_dotenv()
login(os.environ['HF_TOKEN'])

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts.prompts import SimpleInputPrompt
from transformers import BitsAndBytesConfig
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from langchain_community.embeddings import HuggingFaceEmbeddings


model_name = "meta-llama/Llama-2-7b-chat-hf"

# Create a directory to save uploaded file
directory_name = "data"

# Check if the directory exists
if not os.path.exists(directory_name):
    # Create the directory if it doesn't exist
    os.mkdir(directory_name)

# UI
st.set_page_config(page_title="RAG Application using Llama and LlamaIndex", page_icon=":robot:")
st.header("Hey, what can I retrieve for you?")

# Query system prompt
system_prompt="""
You are a Q&A assistant. Your goal is to answer questions as
accurately as possible based on the instructions and context provided.
"""
# Default format supportable by LLama2
query_wrapper_prompt=SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")

# download model
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
    model_name="meta-llama/Llama-2-7b-chat-hf",
    device_map="auto",
    # uncomment this if using CUDA to reduce memory usage
    model_kwargs={"torch_dtype": torch.float16 , "quantization_config":quantization_config}
)

# download embedding model
embed_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

file = st.file_uploader("Upload file", type="pdf")
submit = st.button("Submit", type="primary")
if file and submit:
    bar = st.progress(50)
    time.sleep(3)
    bar.progress(100)
    st.markdown("**The file is sucessfully Uploaded.**")

    # Save uploaded file to 'F:/tmp' folder.
    save_folder = './data'
    save_path = Path(save_folder, file.name)
    with open(save_path, mode='wb') as w:
        w.write(file.getvalue())

    documents = SimpleDirectoryReader("./data").load_data()

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
    
    
    index=VectorStoreIndex.from_documents(documents)
    
    query_engine=index.as_query_engine()

    user_input = st.text_input('What can I search for you in the uploaded document?')
    if st.button("Search"):
        response=query_engine.query(user_input)
    
        with st.spinner(text='In progress'):
            time.sleep(3)
    
        st.write(response)