"""
First File to Run.

Uses ChunkWise Approach 

"""

import json
import os
import re
from dataclasses import dataclass
from typing import List

import pandas as pd
import PyPDF2
import requests
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Datagenerator:
    filenames: List
    prompt: str
    apikey: str
    filepath: str = "total_chunkwise_raw_results.txt"

    def load_docs_pdf_MAS(self) -> tuple:
        """Load Docs from Data Directory

        Args:
            filenames (list[str]): List of Filenames

        Returns:
            List[str]: Text and metadat
        """
        texts = []
        metadatas = []
        i = 0
        for filename in self.filenames:
            with open(filename, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    i = i + 1
                    text = page.extract_text()
                    texts.append(text)
                    metadatas.append({"file": filename, "page": i})
        return texts, metadatas

    def clean_text(self, texts: List) -> List:
        """
        :rtype: List
        :type texts: List
        """
        texts = [each.replace("\n", " ") for each in texts]
        total = []
        for each in texts:
            reads = each.replace('"', "")
            reads = reads.replace("*** IBM Company Confidential ***", "")
            reads = reads.replace("DemographicsMetric", "Demographics Metric")
            regex = re.compile("Page ([0-9]*)")
            # re.sub("\(.+?\)", lambda x: x.group(0).replace(" ", ""), " ".join(reads))
            result = re.sub(regex, "", reads)
            total.append(result)

        cleaned = "".join(total)
        return cleaned

    def get_qa_pair(self, snippet: object) -> object:
        """

        :param snippet: Text Passage
        :param prompt: Prompt passed to the LLM.
        :return: Json response of queries and responses.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.apikey}",
        }

        params = {
            "version": "2024-02-15",
        }

        json_data = {
            "model_id": "mistralai/mixtral-8x7b-instruct-v01",
            "parameters": {
                "decoding_method": "greedy",
                "min_new_tokens": 1,
                "max_new_tokens": 7000,
            },
            "moderations": {
                "hap": {
                    "threshold": 0.75,
                    "input": True,
                    "output": True,
                },
                "stigma": {
                    "threshold": 0.75,
                    "input": True,
                    "output": True,
                },
            },
            "prompt_id": "prompt_builder",
            "data": {
                "input": snippet,
                "instruction": self.prompt,
                "input_prefix": "Input:",
                "output_prefix": "Output:",
                "examples": [],
            },
        }

        response = requests.post(
            "https://bam-api.res.ibm.com/v2/text/generation",
            params=params,
            headers=headers,
            json=json_data,
            stream=True,
        )
        return response

    def response_gen(self):
        total = []
        texts, metadata = self.load_docs_pdf_MAS()
        # texts = self.clean_text(texts)
        texts = [each.replace("\n", " ") for each in texts]
        # print(texts)
        try:
            for i, item in enumerate(texts):
                print(f" Item No :{i + 1}")
                results = self.get_qa_pair(item)
                final = (
                    results.json()["results"][0]["generated_text"]
                    .replace("\n", " ")
                    .replace("```", "")
                )
                total.append(final)

            for each in total:
                if os.path.exists(self.filepath):
                    with open(self.filepath, "a") as f:
                        f.write(each)
                        f.write("\n")
                else:
                    with open(self.filepath, "w") as f:
                        f.write(each)
                        f.write("\n")
        except Exception as e:
            print(e)

    def driver(self, response_file):
        self.response_gen()
        total_list = []
        dicts = []
        with open(self.filepath) as f:
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
        result["text"] = (
            "\nQuestion:\n\n" + result["Query"] + "\n\nAnswer:\n" + result["Answer"]
        )
        df_new = result.drop(columns=["Query", "Answer"], axis=1)
        df_new.to_csv(response_file, index=False)


if __name__ == "__main__":
    prompt = "You are an Al Assistant, tasked with generating a comparative query and complex queries from the below paragraph. \
    Kindly abide by the following instructions:\
    1. Read the entire document.\
    2. Check to see if there are two entities or phrases for which \
    similarities and differences can be assessed according to certain criteria or characteristics. \
    3.If there are such entities or phrases, generate a list of comparative queries and provide responses.\
    4. All the queries must be followed with a response from the document only. Don't make up answers.\
    5. The Queries and Answers must be formatted in a JSON format.\
    6. Don't add opinions to your answers.\
    7. Don't create categories . Put all complex and comparitive queries together."

    generator = Datagenerator(
        apikey=os.getenv("BAM_API_KEY"),
        filenames=[os.getenv("FILEPATH")],
        prompt=prompt,
    )
    generator.driver(response_file="trainable_records_chunkwise.csv")
