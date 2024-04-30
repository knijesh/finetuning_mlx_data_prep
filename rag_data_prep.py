"""_summary_

   Run this File only after prep_data_chunkwise.py
   This file uses the queries from the results file of "prep_data_chunkwise.py"  and generates answers using RAG approach.
   
   Output Train file name : "trainable_records_ragwise.txt"
"""

import json
import os
import re
from typing import List

from dotenv import load_dotenv

load_dotenv()
import warnings

import pandas as pd

from prep_data_chunkwise import Datagenerator

warnings.filterwarnings("ignore")
from rag import *


def get_queries(query_file):
    total_list = []
    dicts = []
    with open(query_file) as f:
        data = f.readlines()
        import ast

        for each in data:
            try:
                each = ast.literal_eval(each)
                # print(type(each))
                df = pd.DataFrame.from_dict(each, orient="index").T
                rec = df.to_records("test")
                for each in rec:
                    for i in each:
                        if type(i) == dict:
                            i["Answer"] = i.pop("Response")
                            dicts.append(i)
            except Exception as e:
                pass

    result = pd.DataFrame.from_records(dicts)
    query_list = result["Query"].values.tolist()
    with open("queries.txt", "w") as f:
        for each in query_list:
            f.write(each)
            f.write("\n")
    return query_list


def get_response(query_list, filename=os.getenv("FILEPATH")):
    total_dict = {}
    rag = RAGwatsonx(
        apikey=os.getenv("GA_API_KEY"),
        filename="queries.txt",
        project_id=os.getenv("PROJECT_ID"),
    )

    texts = rag.textloader()

    docsearch = rag.load_embeddings(texts)

    watsonx_granite = rag.load_model()

    for i, query in enumerate(query_list):
        response = rag.query_qa(query, watsonx_granite, docsearch)
        total_dict["query"] = response
        result = (
            json.dumps({"text": f"<s>[INST] {query}[/INST] {response['result']}</s>"})
            + "\n"
        )
        with open("trainable_records_ragwise.txt", "a+") as file:
            file.write(result)


query_list = get_queries("total_chunkwise_raw_results.txt")
get_response(query_list)
