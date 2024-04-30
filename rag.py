"""
RAG Implementation

"""

import os

from dotenv import load_dotenv

load_dotenv()

from dataclasses import dataclass

from ibm_watsonx_ai.foundation_models.utils import get_embedding_model_specs
from ibm_watsonx_ai.foundation_models.utils.enums import (
    DecodingMethods,
    EmbeddingTypes,
    ModelTypes,
)
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_ibm import WatsonxEmbeddings, WatsonxLLM


@dataclass
class RAGwatsonx:
    apikey: str
    filename: str
    project_id: str
    url: str = "https://us-south.ml.cloud.ibm.com"

    def textloader(self):
        loader = TextLoader(self.filename)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        return texts

    def load_embeddings(self, texts):
        # embeddings = WatsonxEmbeddings(
        #     model_id=EmbeddingTypes.IBM_SLATE_30M_ENG.value,
        #     url=self.url,
        #     apikey=self.apikey,
        #     project_id=self.project_id,
        # )
        embeddings = HuggingFaceEmbeddings()
        docsearch = Chroma.from_documents(texts, embeddings)
        return docsearch
        # get_embedding_model_specs(credentials.get("url"))

    def load_model(self):
        model_id = ModelTypes.LLAMA_2_70B_CHAT
        parameters = {
            GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
            GenParams.MIN_NEW_TOKENS: 1,
            GenParams.MAX_NEW_TOKENS: 800,
            # GenParams.STOP_SEQUENCES: ["<|endoftext|>"],
        }
        watsonx_granite = WatsonxLLM(
            model_id=model_id.value,
            url=self.url,
            apikey=self.apikey,
            project_id=self.project_id,
            params=parameters,
        )
        return watsonx_granite

    def query_qa(self, query, watsonx_granite, docsearch):
        qa = RetrievalQA.from_chain_type(
            llm=watsonx_granite, chain_type="stuff", retriever=docsearch.as_retriever()
        )
        return qa.invoke(query)
