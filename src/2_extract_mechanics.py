import os
import pandas as pd
from pydantic import BaseModel, Field
import ollama
from tqdm import tqdm

INPUT_FILE = "data/top_jokes.parquet"
OUTPUT_FILE = "data/structured_jokes.parquet"
MODEL_NAME = "qwen2.5:3b"


class ComedicBlueprint(BaseModel):
    comedic_device: str = Field(
        description="Primary mechanism: e.g., Misdirection, Double Entendre, Rule of Three, Anti-Humor, Dark Twist, Literal Interpretation, False Analogy."
    )
    abstract_setup: str = Field(
        description="Topic-agnostic summary of the expectation established in the setup (avoid proper nouns)."
    )
    incongruity_pivot: str = Field(
        description="Topic-agnostic explanation of how the punchline subverts the setup."
    )
    structural_formula: str = Field(
        description="A reusable 1-sentence abstract formula describing the joke's logic."
    )


SYSTEM_PROMPT = """You are an expert computational humor analyst. 
Analyze the provided joke and strip away its surface topic (e.g., bars, doctors, animals) to extract its underlying comedic engineering.
Focus strictly on the logical pivot that creates surprise."""


def main(max_jokes: int = 300):
    df = pd.read_parquet(INPUT_FILE).head(max_jokes)
    results = []

    for row in tqdm(df.to_dict(orient="records"), desc="Extracting mechanics (Local LLM)"):
        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": row["full_text"]},
                ],
                format=ComedicBlueprint.model_json_schema(),
                options={"temperature": 0.1},
            )
            parsed = ComedicBlueprint.model_validate_json(response.message.content)
            results.append(
                {
                    **row,
                    "comedic_device": parsed.comedic_device,
                    "abstract_setup": parsed.abstract_setup,
                    "incongruity_pivot": parsed.incongruity_pivot,
                    "structural_formula": parsed.structural_formula,
                    "embedding_text": f"[{parsed.comedic_device}] Setup: {parsed.abstract_setup} | Pivot: {parsed.incongruity_pivot} | Formula: {parsed.structural_formula}",
                }
            )
        except Exception as e:
            print(f"Failed on joke {row['id']}: {e}")

    out_df = pd.DataFrame(results)
    out_df.to_parquet(OUTPUT_FILE, index=False)
    print(f"Saved {len(out_df)} structured blueprints to {OUTPUT_FILE}")


if __name__ == "__main__":
    main(max_jokes=300)
