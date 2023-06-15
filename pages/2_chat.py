from typing import List
import streamlit as st
import os
import pickle

st.set_page_config(page_title="Chat", page_icon="📈")
st.markdown("# Chat")
class_names = st.session_state['gs']['classes']



def main():
    # Select class, and input question
    selected_class_name: str = st.selectbox(
        'Class',
        st.session_state['gs']['classes'])
    class_name_directory_path = f'./VectorStores/{selected_class_name}'
    query: str = st.text_input("Ask question: ")
    
    if selected_class_name == '' or query == '': return

    # Embedding similarity match
    if os.path.exists(class_name_directory_path):
        files = [ os.path.join(class_name_directory_path, file_name) for file_name in os.listdir(class_name_directory_path)]
        # top3: List[tuple[str, float]]  = []

        for file_path in files:
            with open(file_path, "rb") as f:
                VectorStore = pickle.load(f)
                chunks_and_scores: List[tuple[str, float]] = VectorStore.similarity_search_with_score(query=query, k=3)
                st.write(chunks_and_scores)
                # top3.extend(chunks_and_scores)
        # top3.sort(key=lambda c_s: -c_s[1])
        # st.write(top3)
        # st.write(top3[:3])

if __name__ == "__main__":
    main()
