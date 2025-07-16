import os
import torch
import numpy as np
from PIL import Image
from sklearn.preprocessing import normalize
from pymilvus import MilvusClient
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from dotenv import load_dotenv

load_dotenv()
ZILLIZ_URI = os.getenv("ZILLIZ_URI")
ZILLIZ_API_KEY = os.getenv("ZILLIZ_API_KEY")

class FeatureExtractor:
    def __init__(self, modelname="resnet34"):
        self.model = timm.create_model(modelname, pretrained=True, num_classes=0, global_pool="avg")
        self.model.eval()
        config = resolve_data_config({}, model=modelname)
        self.preprocess = create_transform(**config)

    def __call__(self, imagepath):
        image = Image.open(imagepath).convert("RGB")
        tensor = self.preprocess(image).unsqueeze(0)
        with torch.no_grad():
            features = self.model(tensor).squeeze().numpy()
        return normalize(features.reshape(1, -1), norm="l2").flatten()

def setup_milvus():
    client = MilvusClient(
        uri=ZILLIZ_URI,
        token = ZILLIZ_API_KEY
        )
    if client.has_collection("image_embeddings"):
        client.drop_collection("image_embeddings")
    client.create_collection(
        collection_name="image_embeddings",
        vector_field_name="vector",
        dimension=512,
        auto_id=True,
        enable_dynamic_field=True,
        metric_type="COSINE"
    )
    return client

def insert_images(client, extractor, root="./train"):
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(".JPEG"):
                filepath = os.path.join(dirpath, filename)
                embedding = extractor(filepath)
                client.insert("image_embeddings", {"vector": embedding, "filename": filepath})

def search_image(client, extractor, query_image, top_k=10):
    query_vector = extractor(query_image)
    results = client.search(
        "image_embeddings",
        data=[query_vector],
        output_fields=["filename"],
        search_params={"metric_type": "COSINE"}
    )
    return [hit["entity"]["filename"] for hit in results[0][:top_k]]
