import os
import asyncio
import pandas as pd
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

load_dotenv()
client = AsyncOpenAI()

INPUT_FILE = "data/top_jokes.parquet"
OUTPUT_FILE = "data/structured_jokes.parquet"


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


async def analyze_joke(sem: asyncio.Semaphore, row: dict) -> dict:
    async with sem:
        try:
            response = await client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": row["full_text"]},
                ],
                response_format=ComedicBlueprint,
                temperature=0.1,
            )
            parsed = response.choices[0].message.parsed
            return {
                **row,
                "comedic_device": parsed.comedic_device,
                "abstract_setup": parsed.abstract_setup,
                "incongruity_pivot": parsed.incongruity_pivot,
                "structural_formula": parsed.structural_formula,
                # Combined text specifically designed for structural embedding
                "embedding_text": f"[{parsed.comedic_device}] Setup: {parsed.abstract_setup} | Pivot: {parsed.incongruity_pivot} | Formula: {parsed.structural_formula}",
            }
        except Exception as e:
            print(f"Failed on joke {row['id']}: {e}")
            return None


async def main(concurrency: int = 20):
    df = pd.read_parquet(INPUT_FILE)
    records = df.to_dict(orient="records")
    sem = asyncio.Semaphore(concurrency)

    tasks = [analyze_joke(sem, row) for row in records]
    results = await tqdm_asyncio.gather(*tasks, desc="Extracting comedic mechanics")

    clean_results = [r for r in results if r is not None]
    out_df = pd.DataFrame(clean_results)
    out_df.to_parquet(OUTPUT_FILE, index=False)
    print(f"Saved {len(out_df)} structured blueprints to {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
