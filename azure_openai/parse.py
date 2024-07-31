import os
import json

from dotenv import load_dotenv

from datasets import load_dataset

from openai import AzureOpenAI

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# Set up
load_dotenv()

azure_api_key = os.environ["AZURE_OPENAI_KEY"]

llm_client = AzureOpenAI(
  azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"],
  api_key=azure_api_key,
  api_version="2024-02-01"
)

# Unstructed Data Source

# As an illustration we are using the same data source to build examples and 
# as new data to index. In practise, there will be a handful of examples and 
# several records where ground truth is not available
 
examples_data = load_dataset("pgurazada1/amazon_india_products", split="train")

new_data_to_index = (
    examples_data.to_pandas()
                 .sample(54)
                 .loc[:, ['Product Title', 'Product Description', 'Price']]
)

examples_df = (
    examples_data.to_pandas()
                 .sample(16)
                 .loc[:, ['Product Title', 'Brand', 'Category']]
)

llm = os.environ["AZURE_DEPLOYMENT_NAME"]

# Mongo DB Initialization

mongo_uri = os.environ["MONGO_URI"]

mongo_client = MongoClient(mongo_uri, server_api=ServerApi('1'))
db_name = "product_information"
product_db = mongo_client[db_name]

collection_name = "Amazon"
amazon_collection = product_db[collection_name]

# Prompt for Structured Data Extraction

system_message = """
You are an expert at extracting structured information out of product information.
You will be presented with a product title and you are tasked to extract the following fields from the input in a JSON format.

{{
"Brand": "Brand mentioned in the title",
"Category": "The category to which the product belongs"
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
        temperature=0
    )
    
    return response.choices[0].message.content

# New data to index to MongoDB

new_records = []

for index, row in new_data_to_index.iterrows():
    _title = row.iloc[0]
    _description = row.iloc[1]
    _price = row.iloc[2]

    try:
        extracted_fields = extract_product_information(few_shot_prompt, _title)
        extracted_fields_dict = json.loads(extracted_fields.replace("'", '"'))
        extracted_fields_dict['Product_Title'] = _title.strip()
        extracted_fields_dict['Product_Description'] = _description.strip()
        extracted_fields_dict['Price'] = _price
        new_records.append(extracted_fields_dict)
    except Exception as e:
        print(e)
        continue

# Once the extraction is correctly done, we insert these as records into MongoDB
# Here we are only checking for basic failures. More robustness checks such as field
# sanity checks should be conducted before database writes
assert len(new_records) == 54

amazon_collection.insert_many(new_records)