import numpy as np
import openai as ai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import BertTokenizer, AutoModel
from pymilvus import connections, Collection
from pydantic import BaseModel
from umap import UMAP
from uuid import uuid4
from sklearn.cluster import KMeans
import torch
import uvicorn
import os

ai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")


def initialize_milvus():
    connections.connect(
        alias="default",
        # host="localhost",
        host="standalone",
        port="19530"
    )

    collection = Collection("papers")

    collection.load()

    return collection


collection = initialize_milvus()


def fetch_data_from_milvus():
    limit = 10000

    results = collection.query(
        expr="",
        output_fields=[
            "id",
            "title",
            "authors",
            "categories",
            "license",
            "abstract",
            "update_date",
            "merged_embeddings",
            "dimension_X",
            "dimension_Y",
            "cluster",
            "distance_to_centroid",
            "normalized_distance"
        ],
        limit=limit
    )

    return results


milvus_data = fetch_data_from_milvus()

search_params = {
    'metric_type': "L2",
    'offset': 0,
    'ignore_growing': False,
    'params': {'nprobe': 50}
}


def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.last_hidden_state.mean(dim=1).numpy().astype(np.float32)


def process_string(s):
    if isinstance(s, list):
        return s

    inner_str = s.strip("[]")
    float_list = [float(value) for value in inner_str.split()]

    return float_list


def fetch_cluster_distances(cluster):
    query_results = collection.query(f"cluster == {cluster}", output_fields=["distance_to_centroid"])

    return [item["distance_to_centroid"] for item in query_results]
    pass


def calculate_normalized_distance(cluster, distance_to_centroid):
    cluster_distances = fetch_cluster_distances(cluster)
    if not cluster_distances:
        return 0

    min_dist, max_dist = min(cluster_distances), max(cluster_distances)
    normalized_distance = (distance_to_centroid - min_dist) / (max_dist - min_dist) if max_dist > min_dist else 0

    return normalized_distance


embeddings = np.array([process_string(item["merged_embeddings"]) for item in milvus_data]).astype(np.float32)

umap_model = UMAP(n_neighbors=15, min_dist=0.5, n_components=2, random_state=42)

umap_result = umap_model.fit_transform(embeddings)

num_clusters = 6
kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto")
cluster_labels = kmeans.fit_predict(embeddings)


@app.get("/")
async def get_data():
    limit = 10000

    results = collection.query(
        expr="",
        output_fields=[
            "id",
            "title",
            "authors",
            "categories",
            "license",
            "abstract",
            "update_date",
            "merged_embeddings",
            "dimension_X",
            "dimension_Y",
            "cluster",
            "distance_to_centroid",
            "normalized_distance"
        ],
        limit=limit
    )

    processed_results = []

    for item in results:
        processed_item = {
            "id": item["id"],
            "title": item["title"],
            "authors": item["authors"],
            "categories": item["categories"],
            "abstract": item["abstract"],
            "date": item["update_date"],
            "dimension_X": float(item["dimension_X"]) if item["dimension_X"] is not None else None,
            "dimension_Y": float(item["dimension_Y"]) if item["dimension_Y"] is not None else None,
            "cluster": item["cluster"],
            "normalized_distance": float(item.get("normalized_distance")) if item.get("normalized_distance") is not
                                                                             None else None
        }

        processed_results.append(processed_item)

    return processed_results


class SearchQuery(BaseModel):
    query: str


@app.post("/search")
async def search(query: SearchQuery):
    if not query.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    embedding = embed_text(query.query)

    search_results = collection.search(
        data=embedding.tolist(),
        anns_field="merged_embeddings",
        param=search_params,
        limit=10000,
        expr=None,
        output_fields=[
            "id",
            "title",
            "authors",
            "categories",
            "abstract",
            "update_date",
            "dimension_X",
            "dimension_Y",
            "cluster"
        ],
    )

    processed_results = [{
        'id': hit.get("id"),
        "title": hit.entity.get("title"),
        'authors': hit.get("authors"),
        "categories": hit.entity.get("categories"),
        'abstract': hit.get("abstract"),
        'date': hit.get("update_date"),
        "dimension_X": hit.entity.get("dimension_X"),
        "dimension_Y": hit.entity.get("dimension_Y"),
        "cluster": hit.entity.get("cluster")
    } for hit in search_results[0]]

    return processed_results


def query_and_process_results(expr):
    res = collection.query(
        expr=expr,
        offset=0,
        limit=10000,
        output_fields=[
            "id",
            "title",
            "authors",
            "categories",
            "abstract",
            "update_date",
            "dimension_X",
            "dimension_Y",
            "cluster"
        ]
    )

    filtered_results = []

    for item in res:
        processed_item = {
            'id': item.get("id"),
            'title': item.get("title"),
            'authors': item.get("authors"),
            'categories': item.get("categories"),
            'abstract': item.get("abstract"),
            'date': item.get("update_date"),
            'dimension_X': float(item.get("dimension_X")) if item.get("dimension_X") is not None else None,
            'dimension_Y': float(item.get("dimension_Y")) if item.get("dimension_Y") is not None else None,
            'cluster': item.get("cluster"),
            'normalized_distance': float(item.get("normalized_distance")) if item.get(
                "normalized_distance") is not None else None
        }

        filtered_results.append(processed_item)

    return filtered_results


class CategoryFilterQuery(BaseModel):
    category: str


@app.post("/filter-category")
async def filter_papers_by_category(filter_query: CategoryFilterQuery):
    subcategories = {
        "Physics": [
            "astro-ph",
            "cond-mat",
            "gr-qc",
            "hep-ex",
            "hep-lat",
            "hep-ph",
            "hep-th",
            "math-ph",
            "nlin",
            "nucl-ex",
            "nucl-th",
            "physics",
            "quant-ph",
        ],
        "Mathematics": ["math.AG", "math.AT", "math.AP", "math.CT", "math.CA", "math.CO", "math.AC", "math.CV",
                        "math.DG", "math.DS", "math.FA", "math.GM", "math.GN", "math.GT", "math.GR", "math.HO",
                        "math.IT", "math.KT", "math.LO", "math.MP", "math.MG", "math.NT", "math.NA", "math.OA",
                        "math.OC", "math.PR", "math.QA", "math.RT", "math.RA", "math.SP", "math.ST", "math.SG"],
        "Computer Science": ["cs.AI", "cs.AR", "cs.CC", "cs.CE", "cs.CG", "cs.CL", "cs.CR", "cs.CV", "cs.CY",
                             "cs.DB", "cs.DC", "cs.DL", "cs.DM", "cs.DS", "cs.ET", "cs.FL", "cs.GL", "cs.GR",
                             "cs.GT", "cs.HC", "cs.IR", "cs.IT", "cs.LG", "cs.LO", "cs.MA", "cs.MM", "cs.MS",
                             "cs.NA", "cs.NE", "cs.NI", "cs.OH", "cs.OS", "cs.PF", "cs.PL", "cs.RO", "cs.SC",
                             "cs.SD", "cs.SE", "cs.SI", "cs.SY"],
        "Quantitative Biology": ["q-bio.BM", "q-bio.CB", "q-bio.GN", "q-bio.MN", "q-bio.NC", "q-bio.OT", "q-bio.PE",
                                 "q-bio.QM", "q-bio.SC", "q-bio.TO"],
        "Quantitative Finance": ["q-fin.CP", "q-fin.EC", "q-fin.GN", "q-fin.MF", "q-fin.PM", "q-fin.PR", "q-fin.RM",
                                 "q-fin.ST", "q-fin.TR"],
        "Statistics": ["stat.AP", "stat.CO", "stat.ML", "stat.ME", "stat.OT", "stat.TH"],
        "Electrical Engineering and Systems Science": ["eess.AS", "eess.IV", "eess.SP", "eess.SY"],
        "Economics": ["econ.EM", "econ.GN", "econ.TH"]
    }

    selected_subcategories = subcategories.get(filter_query.category, [])

    expr = " || ".join([f"categories like '{subcat}'" for subcat in selected_subcategories])

    filtered_results = query_and_process_results(expr)

    return filtered_results


class DateFilterQuery(BaseModel):
    dateGroup: str


@app.post("/filter-date")
async def filter_papers_by_date(filter_query: DateFilterQuery):
    date_conditions = {
        "Since 2023": "update_date >= '2023-01-01'",
        "Since 2022": "update_date >= '2022-01-01'",
        "Since 2019": "update_date >= '2019-01-01'"
    }

    expr = date_conditions.get(filter_query.dateGroup, "")

    filtered_results = query_and_process_results(expr)

    return filtered_results


class LicenseFilterQuery(BaseModel):
    onlyLicensed: bool = False


@app.post("/filter-license")
async def filter_papers(filter_query: LicenseFilterQuery):
    expr = "license != 'nan' && license != ''" if filter_query.onlyLicensed else ""

    filtered_results = query_and_process_results(expr)

    return filtered_results


class PaperSubmission(BaseModel):
    title: str
    authors: str
    abstract: str


@app.post("/submit-paper")
async def submit_paper(submission: PaperSubmission):
    submission_id = str(uuid4())

    embeddings_2d = embed_text(submission.abstract)
    embeddings_2d = embeddings_2d.astype(np.float32).reshape(1, -1)

    predicted_cluster = kmeans.predict(embeddings_2d)[0]

    centroid = kmeans.cluster_centers_[predicted_cluster]
    distance_to_centroid = np.linalg.norm(embeddings_2d - centroid)

    normalized_distance = calculate_normalized_distance(predicted_cluster, distance_to_centroid)

    new_point_umap = umap_model.transform(embeddings_2d)

    data_dict = {
        'id': submission_id,
        'title': submission.title,
        'authors': submission.authors,
        'categories': "",
        'license': "",
        'abstract': submission.abstract,
        'update_date': "",
        "merged": "",
        'merged_embeddings': embeddings_2d.flatten().tolist(),
        'dimension_X': new_point_umap[0, 0],
        'dimension_Y': new_point_umap[0, 1],
        'cluster': predicted_cluster,
        'distance_to_centroid': distance_to_centroid,
        'normalized_distance': normalized_distance
    }

    collection.insert([data_dict])

    search_results = collection.search(
        data=embeddings_2d.tolist(),
        anns_field="merged_embeddings",
        param=search_params,
        limit=5,
        expr=None,
        output_fields=["id", "title", "authors", "abstract", "update_date"]
    )

    similar_papers = [{
        "id": hit.entity.get("id"),
        "title": hit.entity.get("title"),
        "authors": hit.entity.get("authors"),
        "abstract": hit.entity.get("abstract"),
        "date": hit.entity.get("update_date")
    } for hit in search_results[0]]

    new_data = {
        "title": submission.title,
        "categories": "",
        "abstract": submission.abstract,
        "dimension_X": float(new_point_umap[0, 0]),
        "dimension_Y": float(new_point_umap[0, 1]),
        "cluster": int(predicted_cluster),
        "normalized_distance": float(normalized_distance),
        "isNew": True,
        "similar_papers": similar_papers
    }

    return new_data


class GenerateAbstractRequest(BaseModel):
    original_abstract: str
    similar_abstracts: list[str]


@app.post("/generate-abstract")
async def generate_abstract(request: GenerateAbstractRequest):
    prompt = ("Based on the following abstracts, generate a comprehensive abstract that encompasses the key themes and "
              "ideas:\n\n")
    prompt += f"1. Abstract of Uploaded Paper: {request.original_abstract}\n\n"
    for i, abstract in enumerate(request.similar_abstracts, start=2):
        prompt += f"{i}. Abstract of Similar Paper {i - 1}: {abstract}\n\n"
    prompt += "Generated Abstract:"

    try:
        response = ai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=500
        )

        generated_abstract = response.choices[0].text.strip()
    except Exception as e:
        print(f"Error in generating abstract: {e}")
        return {"error": "Abstract generation failed."}

    return {"generated_abstract": generated_abstract}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5002, reload=True)
