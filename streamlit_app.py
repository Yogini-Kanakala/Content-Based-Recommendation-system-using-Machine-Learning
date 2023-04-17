import json
import pickle

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

base_url = "https://www.mastgeneralstore.com"

mappings = json.load(open('mappings.json'))

index_to_product_id = mappings["index_to_product_id"]
product_id_to_index = mappings["product_id_to_index"]
del mappings
# print(product_id_to_index)
vectors = pickle.load(open('feature_store/vectors.pkl', 'rb'))
print("Vectors Shape: ", vectors.shape)

knn_model = NearestNeighbors(n_neighbors=30, metric='cosine')
_ = knn_model.fit(vectors)

@st.cache_data
def load_data(url):
    data = pd.read_csv(url)
    return data

data = load_data('mast_product_level_information.csv')
print("Data Shape: ",data.shape)
st.title("Content Based Rec - Demo")

product_id = st.text_input("Enter product ID")
product_id = product_id.strip()

if st.button("Search"):
    index = product_id_to_index[product_id]
    thumbnail_url = data.iloc[index].Thumbnail_URL
    name = data.iloc[index].Name
    st.markdown("###  Product")
    
    st.image(base_url+thumbnail_url, width=300, caption=name)

    vector = vectors[index]
    _, indices = knn_model.kneighbors(vector, 13)

    res=data.iloc[indices[0][1:]].reset_index(drop=True)
    urls = res.Thumbnail_URL.tolist()
    urls = [base_url+url for url in urls]
    names = res.Name.tolist()
    ids = res.Product_ID.tolist()

    st.markdown("### Recommendations..")
    
    col0, col1, col2 = st.columns(3)
    col3, col4, col5 = st.columns(3)
    col6, col7, col8 = st.columns(3)
    col9, col10, col11 = st.columns(3)

    with col0:
        st.write("ID: ",str(ids[0]))
        st.image(urls[0], caption=names[0])

    with col1:
        st.write("ID: ",str(ids[1]))
        st.image(urls[1], caption=names[1])

    with col2:
        st.write("ID: ",str(ids[2]))
        st.image(urls[2], caption=names[2])

    with col3:
        st.write("ID: ",str(ids[3]))
        st.image(urls[3], caption=names[3])

    with col4:
        st.write("ID: ",str(ids[4]))
        st.image(urls[4], caption=names[4])

    with col5:
        st.write("ID: ",str(ids[5]))
        st.image(urls[5],caption=names[5])

    with col6:
        st.write("ID: ",str(ids[6]))
        st.image(urls[6], caption=names[6])

    with col7:
        st.write("ID: ",str(ids[7]))
        st.image(urls[7], caption=names[7])

    with col8:
        st.write("ID: ",str(ids[8]))
        st.image(urls[8], caption=names[8])

    with col9:
        st.write("ID: ",str(ids[9]))
        st.image(urls[9], caption=names[9])

    with col10:
        st.write("ID: ",str(ids[10]))
        st.image(urls[10], caption=names[10])

    with col11:
        st.write("ID: ",str(ids[11]))
        st.image(urls[11], caption=names[11])
    
    # print(urls)





    