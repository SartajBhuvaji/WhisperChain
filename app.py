import streamlit as st





def main():
    st.set_page_config(page_title='WhisperChain', layout='wide', page_icon='🔗')
    st.header('WhisperChain')

    st.text_input('Ask your document here')
    
    with st.sidebar:
        st.subheader("Your documents")
        st.file_uploader("Upload your PDF files", type=['pdf'])
        st.button('Process')

    

if __name__ == '__main__':
    main()