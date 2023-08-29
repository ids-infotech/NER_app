import spacy
import spacy_streamlit
import streamlit as st
from streamlit_option_menu import option_menu



med7 = spacy.load("en_core_med7_lg")
nlp = spacy.load('en_core_web_sm')


selected = option_menu(
    menu_title=None,
    options=["NER", "Medical NER"],
    icons=["house", "upload"],
    default_index=0,
    orientation="horizontal",
)


if selected =='NER':
    def main():
        raw_data = st.text_area("Enter Input Text here")
        docx = nlp(raw_data)
        if st.button("Show Entities"):
            if raw_data == '':
                st.warning("Sorry, Please input your data to access this functionality!!")
            else:
                spacy_streamlit.visualize_ner(docx, labels = nlp.get_pipe('ner').labels)
    main()

if selected =='Medical NER':
    col_dict = {}
    seven_colours = ['#e6194B', '#3cb44b', '#ffe119', '#ffd8b1', '#f58231', '#f032e6', '#42d4f4']
    for label, colour in zip(med7.pipe_labels['ner'], seven_colours):
        col_dict[label] = colour
        
    options = {'ents': med7.pipe_labels['ner'], 'colors':col_dict}
    text = st.text_area('Enter Input text here')
    if st.button('Show Entities'):
        if text =='':
            st.warning('Sorry, Please input your data to access this functionality!!')
        
        else:
            doc = med7(text)
            html=spacy.displacy.render(doc, style='ent', options=options)
            st.write(html,unsafe_allow_html=True)




hide_style = """
    <style>
    footer {visibility:hidden;}
    </style>
    """
st.markdown(hide_style, unsafe_allow_html=True)




