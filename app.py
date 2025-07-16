# app.py

import streamlit as st
from PIL import Image
from milvus_engine import FeatureExtractor, setup_milvus, insert_images, search_image

st.title("📷 Image Similarity Search with Milvus")

if "milvus_client" not in st.session_state:
    st.session_state["extractor"] = FeatureExtractor()
    st.session_state["milvus_client"] = setup_milvus()
    st.session_state["indexed"] = False

if not st.session_state["indexed"]:
    with st.spinner("Indexing images..."):
        insert_images(st.session_state["milvus_client"], st.session_state["extractor"])
        st.session_state["indexed"] = True
    st.success("Image dataset indexed into Milvus!")

query_image_path = st.text_input("Enter query image path:", "./test/Afghan_hound/n02088094_4261.JPEG")

if st.button("Search"):
    with st.spinner("Searching similar images..."):
        results = search_image(
            st.session_state["milvus_client"],
            st.session_state["extractor"],
            query_image_path
        )

        st.subheader("Query Image")
        st.image(Image.open(query_image_path).resize((150,150)))

        st.subheader("Top Results")
        cols = st.columns(5)
        for i, path in enumerate(results):
            with cols[i % 5]:
                st.image(Image.open(path).resize((150,150)))
                st.caption(path)
