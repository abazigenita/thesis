import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import BertTokenizer, AutoModel
from pymilvus import connections, Collection
from pydantic import BaseModel
from umap import UMAP
from uuid import uuid4
import torch
import uvicorn
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

app = FastAPI()

# Configure CORS
origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load SciBERT tokenizer and model
model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")


# Define the text embedding function
def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    # Convert the output to float32
    return outputs.last_hidden_state.mean(dim=1).numpy().astype(np.float32)


# Define the Milvus connection and collection initialization here
def initialize_milvus():
    connections.connect(
        alias="default",
        user="username",
        password="password",
        host="localhost",
        port="19530"
    )

    # Retrieve the "arXiv" collection
    milvus_collection = Collection("arXiv")

    return milvus_collection


# Initialize the Milvus collection
collection = initialize_milvus()

# expr = "authors == 'Genita Abazi'"
# query_results = collection.query(expr, output_fields=["id"])
#
# primary_keys_to_delete = [item["id"] for item in query_results]
#
# if primary_keys_to_delete:
#     delete_expr = "id in " + str(primary_keys_to_delete)
#     collection.delete(delete_expr)
# else:
#     print("No records found for the given author.")


# Create an index for each embedding in the collection
index_params = {
    'metric_type': "L2",
    'index_type': "IVF_FLAT",
    'params': {"nlist": 1024},
}

collection.create_index("merged_embeddings", index_params)

# Define search parameters
search_params = {
    'metric_type': "L2",
    'offset': 0,
    'ignore_growing': False,
    'params': {'nprobe': 50}
}

# Import arXiv data
data = pd.read_csv("arxiv_research_papers.csv")


# Define a function to process the string within the column "merged_embeddings"'s series values
def process_string(s):
    inner_str = s.strip("[]")
    float_list = [float(value) for value in inner_str.split()]
    return float_list


# Apply the function to the series
data["merged_embeddings"] = data["merged_embeddings"].apply(process_string)

# Convert the embeddings into a 2D NumPy array
embeddings = np.array(data["merged_embeddings"].tolist()).astype(np.float32)

# Create a UMAP model
umap_model = UMAP(n_neighbors=15, min_dist=0.5, n_components=2, random_state=42)

# Fit and transform the embeddings
umap_result = umap_model.fit_transform(embeddings)

# Perform k-means clustering
num_clusters = 6
kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto")
cluster_labels = kmeans.fit_predict(embeddings)
data["cluster"] = cluster_labels

# Calculate distances to centroids and normalize
centroids = kmeans.cluster_centers_
distances = euclidean_distances(embeddings, centroids)
closest_centroid_distances = distances[np.arange(len(distances)), cluster_labels]
data['distance_to_centroid'] = closest_centroid_distances
data['normalized_distance'] = data.groupby('cluster')['distance_to_centroid'].transform(
    lambda x: (x - x.min()) / (x.max() - x.min()))


# Define a Pydantic model for query data
class SearchQuery(BaseModel):
    query: str


@app.post("/search")
async def search(query: SearchQuery):
    if not query.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Generate embedding for the search query
    embedding = embed_text(query.query)

    # Perform search in Milvus
    search_results = collection.search(
        data=embedding.tolist(),
        anns_field="merged_embeddings",
        param=search_params,
        limit=10000,
        expr=None,
        output_fields=["title", "categories", "dimension_X", "dimension_Y", "cluster"]
    )

    # Process and return the search results
    processed_results = [{
        "title": hit.entity.get("title"),
        "categories": hit.entity.get("categories"),
        "dimension_X": hit.entity.get("dimension_X"),
        "dimension_Y": hit.entity.get("dimension_Y"),
        "cluster": hit.entity.get("cluster")
    } for hit in search_results[0]]

    return processed_results


# Define a Pydantic model for incoming data
class PaperSubmission(BaseModel):
    title: str
    authors: str
    abstract: str


@app.post("/submit-paper")
async def submit_paper(submission: PaperSubmission):
    # Generate an ID for the submission
    submission_id = str(uuid4())

    # Generate embeddings for the submitted abstract
    embeddings_2d = embed_text(submission.abstract)

    # Ensure embeddings are in float32 format
    embeddings_2d = embeddings_2d.astype(np.float32)

    # Reshape embeddings to 2D for single sample
    embeddings_2d = embeddings_2d.reshape(1, -1)

    # Predict the cluster for the new submission
    predicted_cluster = kmeans.predict(embeddings_2d)[0]

    # For the UMAP transformation, use the 2D embeddings directly
    new_point_umap = umap_model.transform(embeddings_2d)

    # Flatten the 2D array to a 1D array for Milvus insertion
    embeddings_flattened = embeddings_2d.flatten().tolist()

    # Prepare the data for insertion
    data_dict = {
        'id': submission_id,
        'title': submission.title,
        'authors': submission.authors,
        'categories': "",  # Update as needed
        'license': "",
        'abstract': submission.abstract,
        'update_date': "",
        'merged': "",
        'merged_embeddings': embeddings_flattened,
        'dimension_X': new_point_umap[0, 0],
        'dimension_Y': new_point_umap[0, 1],
        'cluster': predicted_cluster
    }

    # Insert the data into the Milvus collection
    collection.insert([data_dict])

    # Check if embeddings_2d is a list of lists where each inner list is a vector
    if isinstance(embeddings_2d, np.ndarray):
        search_data = embeddings_2d.tolist()
    else:
        search_data = [embeddings_2d]

    # Perform the search
    search_results = collection.search(
        data=search_data,
        anns_field="merged_embeddings",
        param=search_params,
        limit=5,  # Retrieve top 5 similar embeddings
        expr=None,
        output_fields=["id", "title", "authors", "abstract"]
    )

    # Extract details of similar papers
    similar_papers = [{
        "id": hit.entity.get("id"),
        "title": hit.entity.get("title"),
        "authors": hit.entity.get("authors"),
        "abstract": hit.entity.get("abstract")
    } for hit in search_results[0]]

    # Prepare the response data
    new_data = {
        "dimension_X": float(new_point_umap[0, 0]),
        "dimension_Y": float(new_point_umap[0, 1]),
        "cluster": int(predicted_cluster),
        "title": submission.title,
        "categories": "",
        "isNew": True,
        "similar_papers": similar_papers
    }

    return new_data


@app.get("/get-data")
async def get_data():
    limit = 10000
    results = collection.query(expr="", output_fields=["dimension_X", "dimension_Y", "cluster", "title", "categories"],
                               limit=limit)
    processed_results = []
    for item in results:
        processed_item = {
            "dimension_X": float(item["dimension_X"]) if item["dimension_X"] is not None else None,
            "dimension_Y": float(item["dimension_Y"]) if item["dimension_Y"] is not None else None,
            "cluster": item["cluster"],
            "title": item["title"],
            "categories": item["categories"],
            "normalized_distance": float(data.loc[data['title'] == item["title"], 'normalized_distance'].iloc[0]) if
            data['title'].isin([item["title"]]).any() else None
        }
        processed_results.append(processed_item)
    return processed_results


class CategoryFilterQuery(BaseModel):
    category: str


@app.post("/filter-category")
async def filter_papers_by_category(filter_query: CategoryFilterQuery):
    # Mapping of categories to their subcategories
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

    # Find the subcategories for the selected category
    selected_subcategories = subcategories.get(filter_query.category, [])

    # Construct the query expression
    category_expr = " || ".join([f"categories like '{subcat}'" for subcat in selected_subcategories])

    # Perform the query in Milvus
    res = collection.query(
        expr=category_expr,
        limit=10000,
        output_fields=["title", "categories", "dimension_X", "dimension_Y", "cluster"]
    )

    # Process the results
    filtered_results = [{
        'title': hit.get("title"),
        'categories': hit.get("categories"),
        'dimension_X': float(hit.get("dimension_X")) if hit.get("dimension_X") is not None else None,
        'dimension_Y': float(hit.get("dimension_Y")) if hit.get("dimension_Y") is not None else None,
        'cluster': int(hit.get("cluster")) if hit.get("cluster") is not None else None
    } for hit in res]

    return filtered_results


class LicenseFilterQuery(BaseModel):
    onlyLicensed: bool = False


@app.post("/filter-license")
async def filter_papers(filter_query: LicenseFilterQuery):
    expr = "license != 'nan' && license != ''" if filter_query.onlyLicensed else ""

    # Perform the query in Milvus
    res = collection.query(
        expr=expr,
        offset=0,
        limit=10000,
        output_fields=["title", "categories", "dimension_X", "dimension_Y", "cluster"]
    )

    # Extract the results
    filtered_results = []
    for item in res:
        processed_item = {
            "dimension_X": float(item["dimension_X"]) if item["dimension_X"] is not None else None,
            "dimension_Y": float(item["dimension_Y"]) if item["dimension_Y"] is not None else None,
            "cluster": item["cluster"],
            "title": item["title"],
            "categories": item["categories"],
            "normalized_distance": float(data.loc[data['title'] == item["title"], 'normalized_distance'].iloc[0]) if
            data['title'].isin([item["title"]]).any() else None
        }
        filtered_results.append(processed_item)
    return filtered_results


class DateFilterQuery(BaseModel):
    dateGroup: str


@app.post("/filter-date")
async def filter_papers_by_date(filter_query: DateFilterQuery):
    # Define date group conditions
    date_conditions = {
        "Since 2023": "update_date >= '2023-01-01'",
        "Since 2022": "update_date >= '2022-01-01'",
        "Since 2019": "update_date >= '2019-01-01'"
    }

    # Get the date condition or None if "Any time" is selected
    expr = date_conditions.get(filter_query.dateGroup, "")

    # Perform the query in Milvus
    res = collection.query(
        expr=expr,
        offset=0,
        limit=10000,
        output_fields=["title", "categories", "dimension_X", "dimension_Y", "cluster"]
    )

    filtered_results = []
    # Extract and return the results
    for item in res:
        processed_item = {
            "dimension_X": float(item["dimension_X"]) if item["dimension_X"] is not None else None,
            "dimension_Y": float(item["dimension_Y"]) if item["dimension_Y"] is not None else None,
            "cluster": item["cluster"],
            "title": item["title"],
            "categories": item["categories"],
            "normalized_distance": float(data.loc[data['title'] == item["title"], 'normalized_distance'].iloc[0]) if
            data['title'].isin([item["title"]]).any() else None
        }
        filtered_results.append(processed_item)
    return filtered_results


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5002, reload=True)
