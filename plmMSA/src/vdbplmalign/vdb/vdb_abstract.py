from abc import ABC, abstractmethod

from pydantic import BaseModel


class QueryParams(BaseModel):
    input_sequence: str
    collection_name: str
    nprobe: int = 100
    limit: int = 100


class VDBClient(ABC):

    @abstractmethod
    def search(self, collection_name: str, query, query_params: QueryParams):
        pass

    # @abstractmethod
    # def get_sequences(self, collection_name: str, ids: list):
    #     pass

    @staticmethod
    @abstractmethod
    def prepare_results(search_result, output_fields: list):
        pass
