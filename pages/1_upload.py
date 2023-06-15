from typing import List
import streamlit as st
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.vectorstores import FAISS
from datetime import datetime

# InstructorEmbedding 
from langchain.embeddings import HuggingFaceInstructEmbeddings

GS_PATH = "global_state.pkl"
class_names: List[str]
add_class_name: str

st.set_page_config(page_title="Upload files/audio", page_icon="📈")
st.markdown("# Upload files/audio")

# # add class
# add_class_name = st.text_input('Add new class/subject', key='c_name', value='')
# class_names = st.session_state['gs']['classes']

def add_class():
    if add_class_name != '' and add_class_name not in class_names:
        class_names.append(add_class_name)
        st.session_state['gs']['classes'] = class_names
        with open(GS_PATH, "wb") as f:
            pickle.dump(st.session_state['gs'], f)

# # add pdf
# if pdf and selected_class_name != '':
#     pdf_reader = PdfReader(pdf)
#     text = ""
#     for page in pdf_reader.pages:
#         text += page.extract_text()
    
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
#     chunks: list[str] = text_splitter.split_text(text=text)
    
#     doc_name = pdf.name[:-4]

#     if os.path.exists(class_name_file):
#         st.write('Retrieved vector store')
#         with open(class_name_file, "rb") as f:
#             data = pickle.load(f)
#             files = data['files']
#             st.session_state['VectorStore'] = data['store']        # all the embeddings
#     else:
#         embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base", 
#                     model_kwargs={"device": "cpu"})
#         st.write(datetime.now().time())
#         VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

#         data = {'files': [doc_name], 'store': VectorStore}
#         with open(class_name_file, "wb") as f:
#             pickle.dump(data, f)
#         st.write('Done embedding pdf file')
#         st.write(datetime.now().time())
    
# query = st.text_input("Ask question: ")
# if selected_class_name != '' and query:
#     if os.path.exists(class_name_file):
#         st.write('Retrieved vector store')
#         with open(class_name_file, "rb") as f:
#             data = pickle.load(f)
#             files = data['files']
#             st.session_state['VectorStore'] = data['store']        # all the embeddings
#         similar_chunks = st.session_state['VectorStore'].similarity_search(query=query, k=3)
#         st.write(similar_chunks)

def main():
    global class_names
    global add_class_name
    
    class_names = st.session_state['gs']['classes']

    # add new class
    add_class_name = st.text_input('Add new class/subject', key='c_name', value='')
    st.button('Add class', on_click=add_class)

    # Select class to upload file to
    selected_class_name: str = st.selectbox(
        'Class to upload files',
        st.session_state['gs']['classes'])

    # add file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    if pdf == None or selected_class_name ==  '': return

    # Read text
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    # Divide text into chunks (fit to context size)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50, length_function=len)
    chunks: List[str] = text_splitter.split_text(text=text)


    doc_name: str = pdf.name[:-4]
    directory:str = f'./VectorStores/{selected_class_name}'
    file_path: str = f'./VectorStores/{selected_class_name}/{doc_name}.pkl'
    
    # Save VectorStore for this document
    if not os.path.exists(directory):
        os.makedirs(directory)
    if os.path.exists(file_path):
        st.write('Retrieved vector store')
        with open(file_path, "rb") as f:
            VectorStore = pickle.load(f)
    else:
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base", 
                    model_kwargs={"device": "cpu"})
        st.write(datetime.now().time())
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)            # Convert each chunk to embedding

        with open(file_path, "wb") as f:
            pickle.dump(VectorStore, f)
        st.write('Done embedding pdf file')
        st.write(datetime.now().time())
    
if __name__ == "__main__":
    main()

        


