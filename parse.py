import os
import json

from dotenv import load_dotenv

from datasets import load_dataset

from openai import OpenAI

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


load_dotenv()

anyscale_api_key = os.environ['ANYSCALE_API_KEY']

llm_client = OpenAI(
    base_url="https://api.endpoints.anyscale.com/v1",
    api_key=anyscale_api_key
)

mongo_uri = os.environ["MONGO_URI"]

mongo_client = MongoClient(mongo_uri, server_api=ServerApi('1'))
db_name = "product_information"
product_db = mongo_client[db_name]

collection_name = "Amazon"
amazon_collection = product_db[collection_name]

examples_data = load_dataset("pgurazada1/amazon_india_products", split="train")

examples_df = examples_data.to_pandas().sample(16).loc[:, ['Product Title', 'Brand', 'Category']]

llm = "meta-llama/Meta-Llama-3-8B-Instruct"

system_message = """
You are an expert at extracting structured information out of product information.
You will be presented with a product title and you are tasked to extract the following fields from the input in a JSON format.

{{
"Brand": "Brand mentioned in the title",
"Category": "The category to which the product belongs to"
}}

The category field can only be one of: Skin Care, Bath & Shower, Hair Care, Fragrance or Grocery & Gourmet Foods.
Ensure that you ONLY output a JSON. Do not mention anything before or after your JSON output.
"""

few_shot_prompt = [
    {
        "role":"system",
        "content":system_message
    }
]

for index, example_row in examples_df.iterrows():

    example_title = example_row.iloc[0]
    example_brand = example_row.iloc[1]
    example_category = example_row.iloc[2]

    example_response = {"Brand": example_brand, "Category": example_category}

    few_shot_prompt.append(
        {
            'role': "user",
            "content": example_title
        }
    )
    
    few_shot_prompt.append(
        {
            'role': "assistant",
            "content": str(example_response)
        }
    )


def extract_product_information(few_shot_prompt: list, input: str):

    response = llm_client.chat.completions.create(
        messages=few_shot_prompt + [{"role": "user", "content": input}],
        model=llm,
        temperature=0,
        response_format={"type": "json_object"}
    )
    
    return response.choices[0].message.content

# New data to index to MongoDB

new_data_to_index = examples_data.to_pandas().sample(54).loc[:, ['Product Title', 'Brand', 'Category']]
new_records = []

for index, row in new_data_to_index.iterrows():
    _title = example_row.iloc[0]

    extracted_fields = extract_product_information(few_shot_prompt, _title)
    extracted_fields_dict = json.loads(extracted_fields.replace("'", '"'))
    extracted_fields_dict['Product_Title'] = _title.strip()

    try:
        new_records.append(extracted_fields_dict)
    except Exception as e:
        print(e)

amazon_collection.insert_many(new_records)