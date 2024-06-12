# Create Alpaca ft jsonl dataset
import json
import pandas as pd


def prepare_alpaca_data(df):
    with open("ft_data/alpaca_data.jsonl", "w", encoding="utf-8") as file_alpaca:
        for row in df.itertuples():
            alpaca_data = json.dumps(
                {
                    "instruction": "You are an assistant that takes a piece of text that has been corrupted during OCR digitisation, and produce a corrected version of the same text.",
                    "input": row.corrupt_text,
                    "output": row.text,
                }
            )
            file_alpaca.write(f"{alpaca_data}\n")


if __name__ == "__main__":
    from datasets import load_dataset

    # Load dataset from Hugging Face repo
    dataset = load_dataset("pbevan11/synthetic-ocr-correction-gpt4o")
    df = pd.DataFrame(dataset["train"])
    prepare_alpaca_data(df)
