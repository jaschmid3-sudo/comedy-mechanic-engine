import os
import json
import requests
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/taivop/joke-dataset/master/reddit_jokes.json"
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "top_jokes.parquet")


def fetch_and_filter_jokes(min_score: int = 500, sample_limit: int = 2500):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Downloading dataset from {DATA_URL}...")
    response = requests.get(DATA_URL, timeout=60)
    response.raise_for_status()

    raw_data = response.json()
    df = pd.DataFrame(raw_data)

    # Combine title and body cleanly
    df["title"] = df["title"].fillna("").str.strip()
    df["body"] = df["body"].fillna("").str.strip()
    df["full_text"] = (df["title"] + "\n" + df["body"]).str.strip()

    # Filter out removed/deleted posts and ultra-long wall-of-text stories
    df = df[~df["body"].isin(["[removed]", "[deleted]", ""])]
    df = df[df["full_text"].str.len().between(25, 600)]

    # Isolate high-engagement outliers
    df = df[df["score"] >= min_score].sort_values(by="score", ascending=False)
    df = df.head(sample_limit).reset_index(drop=True)

    df[["id", "score", "full_text"]].to_parquet(OUTPUT_FILE, index=False)
    print(f"Saved {len(df)} high-scoring jokes to {OUTPUT_FILE}")


if __name__ == "__main__":
    fetch_and_filter_jokes(min_score=500, sample_limit=2000)
