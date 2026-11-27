import pandas as pd
import ollama

CLUSTERED_FILE = "outputs/clustered_jokes.parquet"
STATS_FILE = "outputs/top_formulas.csv"
MODEL_NAME = "qwen2.5:3b"


def generate_from_top_cluster(target_topic: str, cluster_rank: int = 0, num_jokes: int = 3):
    df = pd.read_parquet(CLUSTERED_FILE)
    stats = pd.read_csv(STATS_FILE)

    best_cluster_id = stats.iloc[cluster_rank]["cluster_id"]
    formula = stats.iloc[cluster_rank]["sample_formula"]
    device = stats.iloc[cluster_rank]["dominant_device"]

    exemplars = (
        df[df["cluster_id"] == best_cluster_id]
        .sort_values(by="score", ascending=False)
        .head(4)
    )

    examples_block = "\n\n".join(
        [
            f"Example (Score: {row['score']}):\n{row['full_text']}\n[Logic: {row['incongruity_pivot']}]"
            for _, row in exemplars.iterrows()
        ]
    )

    prompt = f"""You are a master comedy writer. Write {num_jokes} original, punchy jokes about the topic: "{target_topic}".

You MUST strictly follow the proven comedic mechanism extracted from this top-performing cluster:
- Comedic Device: {device}
- Structural Formula: {formula}

Study how these top-scoring examples execute the pivot, then replicate that exact rhythm and logical subversion for "{target_topic}" without copying their premises:

{examples_block}
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.85},
    )

    print(f"\n=== Generated Jokes (Using Cluster #{best_cluster_id}: {device}) ===\n")
    print(response.message.content)


if __name__ == "__main__":
    generate_from_top_cluster(target_topic="remote work and video calls", cluster_rank=0)
