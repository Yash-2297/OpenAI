# -*- coding: utf-8 -*-
"""
Created on Sun May 21 11:42:41 2023

@author: yp229
"""

import streamlit as st
import os
import pickle
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS                
from langchain.llms import OpenAI                      
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.chains import RetrievalQA

#OPENAI API
os.environ['OPENAI_API_KEY'] = "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"

#Sidebar contetns

# Sidebar contents 
with st.sidebar:
    st.title('MY PDF CHAT')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
 
    ''')
    add_vertical_space(5)
    
 
def main():
    st.header("Chat with PDF")
    
    #upload a PDF file
    pdf = st.file_uploader("Upload your PDF file", type ='pdf')
    #st.write(pdf.name)
    #store_name = ""
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        st.write(pdf_reader)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
            
        # st.write(text)
        
        text_splitter = RecursiveCharacterTextSplitter(
                             chunk_size = 1000,
                             chunk_overlap = 200,
                             length_function = len
                             )
        chunks = text_splitter.split_text(text = text)
        
        # st.write(file_path)
        
    #     #embedding
        if pdf is None:
            store_name = ""
        else:
            store_name = pdf.name[:-4]
        file_name = f"{store_name}.pkl"
        file_path = f"D:\OpenAI\Langchain\{store_name}.pkl"#+file_name
        #st.write(file_path)

        if os.path.exists(file_path):
            with open(file_path,"rb") as f:
               VectorStore = pickle.load(f)
            # st.write('Embeddings Loaded from the disk')
        else:
            with open(file_path,"wb") as f:
                embedding = OpenAIEmbeddings()
                VectorStore = FAISS.from_texts(chunks,embedding=embedding)
                pickle.dump(VectorStore,f)
            # st.write('Embeddings Loaded first time and  saved in disk')

             
        #Accept user questions and query
        query = st.text_input(f"Ask questions about your pdf file: {store_name}.pdf")
        # st.write(query)     

        if query:
            #docs = VectorStore.similarity_search(query=query,k=5)  
            retriever = VectorStore.as_retriever(search_type="similarity", search_kwargs={"k":4})

            llm = OpenAI()#model_name = 'gpt-3.5-turbo')
            chain = RetrievalQA.from_chain_type(llm=llm, 
                                                chain_type="stuff", 
                                                retriever=retriever, 
                                                return_source_documents=True)
            with get_openai_callback() as cb:
                response = chain(query)['result']
                print(cb)
            st.write(response)

           
           
           
if __name__== '__main__':
    main() 