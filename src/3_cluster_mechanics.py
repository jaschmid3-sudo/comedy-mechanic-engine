import os
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

INPUT_FILE = "data/structured_jokes.parquet"
OUTPUT_DIR = "outputs"
MODEL_DIR = "models/bertopic_comedy"


def cluster_jokes():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    df = pd.read_parquet(INPUT_FILE)
    docs = df["embedding_text"].tolist()

    print("Generating dense structural embeddings...")
    embedding_model = SentenceTransformer("all-mpnet-base-v2")
    embeddings = embedding_model.encode(docs, show_progress_bar=True)

    # Configure UMAP for local semantic neighborhoods
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    # Configure HDBSCAN to isolate distinct formulas and reject unfunny/messy outliers (-1)
    hdbscan_model = HDBSCAN(
        min_cluster_size=12,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        verbose=True,
    )

    topics, probs = topic_model.fit_transform(docs, embeddings)
    df["cluster_id"] = topics
    df["cluster_prob"] = probs

    # Rank clusters by median engagement score (ignoring noise cluster -1)
    valid_clusters = df[df["cluster_id"] != -1]
    cluster_stats = (
        valid_clusters.groupby("cluster_id")
        .agg(
            joke_count=("id", "count"),
            median_score=("score", "median"),
            max_score=("score", "max"),
            dominant_device=("comedic_device", lambda x: x.mode()[0]),
            sample_formula=("structural_formula", "first"),
        )
        .sort_values(by="median_score", ascending=False)
        .reset_index()
    )

    df.to_parquet(os.path.join(OUTPUT_DIR, "clustered_jokes.parquet"), index=False)
    cluster_stats.to_csv(os.path.join(OUTPUT_DIR, "top_formulas.csv"), index=False)
    topic_model.save(MODEL_DIR, serialization="safetensors", save_ctfidf=True)

    print("\nTop 5 Highest-Performing Comedic Clusters:")
    print(cluster_stats.head(5).to_string(index=False))


if __name__ == "__main__":
    cluster_jokes()
