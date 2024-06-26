import asyncio
import atexit
from datasets import load_dataset
import json
import os
from openai import OpenAI
import pandas as pd
import random
import signal
import time
import re

from api_request_parallel_processor import process_api_requests_from_file
from prepare_hf_dataset import prepare_alpaca_data


def get_chat_request(text, index, model="gpt-4o"):
    text = text.replace("\\", "")

    messages = [
        {
            "role": "system",
            "content": "You are an expert at creating examples of text that has been corrupted during OCR processing of historical documents. You respond only with the corrupted text and nothing else.",
        },
        {
            "role": "user",
            "content": f"Here are three examples of corrupted text and their corrections:\n\noriginal: THE OMAHA DAILY BEE, TUESDAY, JUNE 24, 1890 NEWS ABOUT THE BLUFFS Comparatively Little Damage Done by Sunday Night's Storm, SOME EXCEEDINGLY NARROW ESCAPES An Odyssean Memorial Communication Department at H.E. An Unfounded Storm Notice.\n\ncorrupted: THHJ C M A 14 A1 HAM p 0 _ _ THE OMAHA DAILY BEE , TUEBPAY , JUNE 24 , 1890 , _ _ NEWS ABOUT THE BLUFFS Comparatively Little Damage Done b , Sunday Night's Storm , I , SOME EXCEEDINGLY NARROW ESCAPES An OdiirolloxvH * . Memorial CoiniiionccniKiit I'ro rniiunc nt Ht. AuniliMtiy An Un- fondcd Htiinof NotcH.\n\n---\n\noriginal: THE OMAHA DAILY BEE.\nTWENTIETH YEAR. OMAHA, WEDNESDAY MORNING. JUNE 25, 1890. NUMBER 7.\nLICKED UP BY THE FLAMES,\nAn Incendiary Wreaks His Vengeance on Blue Hill, Nebraska.\nNEARLY TWENTY STORES BLOTTED OUT,\nThe Amount of Damage Done Is Estimated at Over Fifty Thousand Dollars, With Comparatively Little Insurance.\nBLUE HILL, Neb., June 24. (Special Telegraph to THE BEE.) At 2:30 this morning a fire broke out simultaneously in two places on the north side of Main street in Blue Hill. The one at the opera house, at almost the extreme east end of the street, was extinguished by the efforts of O. C. J. Longman, Mrs. B. H. Munson and the girl help at the Munson House.\n\ncorrupted: THE OMAHA ! DAILY BEE.\nTWENTIETH YEAR. OMAHA. WEDNESDAY JMjgNING. ( ! JUNE 25. 1890. NUMBER 7.\nLICKED UP BY THE FLAMES , An Incendiary Wreaks His Vengeance o Blue Hill , Nebraska. NEARLY TWENTY STORES BLOTTED OUT , Tlio Amount of lnmnc Done Iloimlily Kutlmnted .nt Over Fifty Thousand DollurH , With Comparatively Little Insurance.\nBLUB HIM , Neb. , Juno 24. ( Special Tele-pram to TUB BBK. ) At 2M : this morning a.flro broke out simultaneously In two places on the north sldo of Main street in Blue Hill. The ono at the opera house , nt almost the ex treme cast end of the street , was extinguished by the efforts of O. C. 1C. Lolgman , Mrs. B. II. Munson and the girl help at the Muuson Louse. I\n\n---\n\noriginal: BOARD OF EDUCATION IN TROUBLE With a Contractor, MORE TEACHERS INCREASED Efforts to Secure the Best Teachers of the State, Nebraska. May 2. [Special to THE BEAR.] The Board of Education is very anxious to interview one J. M. Anderson of Des Moines, the contractor who built the Clinton School house on North Twenty-eighth Street. The building was faultily constructed, and living been a source of considerable expense for repairs, the board has not paid Anderson in full for his work.\n\ncorrupted: Board of Education In Trouble with a.Ooutractor , MARIES OF TEACHERS INCREASED . VFm Nrcrm ijIn Onlrr to Secure the 8 T lrr * 'if tli llott t'lfKn of Kiln * ciuorn New .Scute ' Com- | > rlitntlun , Neb . May 2. [ Special to TUB EB.J The Hoard of Education Is very nnx- mi to Interview ono .1. M. Anderson of DCS lolnes. the contractor who built the Clinton Miool house on North Twenty-eighth street , ho building was faultily constructed , and living been a source of consldrablc expense or repairs , the board lias not paid Anderson n full for his work.\n\n---\n\nGiven the original text below, provide a corrupted version, using the examples as a guide on how the text may become corrupted:\n\noriginal: {text}\n\ncorrupted:",
        },
    ]

    request = {
        "model": model,
        "messages": messages,
        "temperature": 1,
        "max_tokens": 1000,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "metadata": {"index": index},
    }

    return json.dumps(request)


def get_openai_rate_limits(model="gpt-4o"):
    client = OpenAI()
    openai_headers = client.chat.completions.with_raw_response.create(
        messages=[{"role": "user", "content": "Respond with a full stop only (.):"}],
        model=model,
        stop=["."],
        max_tokens=1,
    ).headers

    tpm = int(openai_headers.get("x-ratelimit-limit-tokens"))
    rpm = int(openai_headers.get("x-ratelimit-limit-requests"))

    return tpm, rpm


def clean_up_requests(
    signum=None,
    frame=None,
    requests_file_path="requests.jsonl",
    responses_file_path="output.jsonl",
):
    if os.path.exists(requests_file_path):
        os.remove(requests_file_path)
        print("Cached requests temp file deleted.")
    if os.path.exists(responses_file_path):
        os.remove(responses_file_path)
        print("Cached responses temp file deleted.")
    if signum is not None:
        exit(0)


def get_openai_chat_responses(dataset):
    atexit.register(clean_up_requests)
    signal.signal(signal.SIGINT, clean_up_requests)
    signal.signal(signal.SIGTERM, clean_up_requests)
    requests_file_path = "requests.jsonl"
    responses_file_path = "output.jsonl"
    with open(requests_file_path, "w", encoding="utf-8") as file_requests:
        for row in dataset:
            file_requests.write(f"{get_chat_request(row['text'], row['index'])}\n")

    with open(responses_file_path, "w", encoding="utf-8") as file_responses:
        df = process_requests_from_json(file_responses)
        df = df.merge(dataset.to_pandas(), on="index")
        df.to_csv("ft_data/corrupt_text.csv", index=False)

    # Create Alpaca ft jsonl dataset
    prepare_alpaca_data(df)


def process_requests_from_json(file_responses):
    requests_file_path = "requests.jsonl"
    responses_file_path = "output.jsonl"

    tpm, rpm = get_openai_rate_limits()

    start = time.time()
    asyncio.run(
        process_api_requests_from_file(
            requests_filepath=str(requests_file_path),
            save_filepath=str(responses_file_path),
            request_url=str("https://api.openai.com/v1/chat/completions"),
            api_key=str(os.environ["OPENAI_API_KEY"]),
            max_requests_per_minute=float(5000 * 0.85),  # *0.85 to give headroom
            max_tokens_per_minute=float(tpm * 0.85),
            token_encoding_name="cl100k_base",
            max_attempts=int(10),
            logging_level=int(30),
        )
    )

    with open(responses_file_path, "r", encoding="utf-8") as file_responses:
        lines = file_responses.readlines()

    data = []
    total_tokens = 0
    for line in lines:
        entry = json.loads(line)

        corrupt_text = entry[0]["choices"][0]["message"]["content"]
        index = entry[1]["index"]
        total_tokens += entry[0]["usage"]["total_tokens"]

        data.append({"index": index, "corrupt_text": corrupt_text})

    end = time.time()
    time_taken = end - start

    df = pd.DataFrame(data)

    print(
        f"Time taken to process {total_tokens} tokens: {round(time_taken,2)} seconds ({round(total_tokens/time_taken, 2)}tok/s)"
    )

    return df


def filter_text(entry):
    text = entry["text"]
    # Regular expression to detect common HTML entities and css
    html_entity_pattern = re.compile(r"&[a-zA-Z]+;|#[0-9a-fA-F]{3,6}")
    css_pattern = re.compile(r"(font|color|size|face)=[\'\"]?.+[\'\"]?", re.IGNORECASE)

    has_html_entities = bool(html_entity_pattern.search(text))
    has_css_patterns = bool(css_pattern.search(text))
    is_valid_length = 250 < len(text) < 2500
    contains_http = "http" in text

    return (
        not (has_html_entities or has_css_patterns)
        and is_valid_length
        and not contains_http
    )


if __name__ == "__main__":
    dataset = load_dataset("fancyzhx/ag_news", split="train")
    dataset = dataset.map(lambda x, idx: {"index": idx}, with_indices=True)
    # Remove entries with HTML entities, CSS patterns, invalid length, or HTTP links
    filtered_dataset = dataset.filter(filter_text)
    random.seed(42)
    sampled_dataset = filtered_dataset.shuffle(seed=42).select(range(10000))

    get_openai_chat_responses(sampled_dataset)
