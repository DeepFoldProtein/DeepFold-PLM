from typing import Any, Dict, List

import requests
from vdbplmalign.vdb.vdb_abstract import QueryParams, VDBClient


class FaissClient(VDBClient):
    def __init__(self, api_url: str):
        self.api_url = api_url

    def search(
        self, collection_name: str, query: Any, query_params: QueryParams
    ) -> Dict[str, Any]:
        payload = self._build_payload(collection_name, query, query_params)
        response = requests.post(self.api_url, json=payload)
        self._check_response(response)
        return response.json()

    def _build_payload(
        self, collection_name: str, query: Any, query_params: QueryParams
    ) -> Dict[str, Any]:
        return {
            "vectors": [query.tolist()],
            "k": query_params.limit,
            "nprobe": query_params.nprobe,
            "collection_name": collection_name,
        }

    def _check_response(self, response: requests.Response) -> None:
        if response.status_code != 200:
            raise Exception(
                f"Request failed with status code {response.status_code}: {response.text}"
            )

    @staticmethod
    def prepare_results(search_result: List[Dict[str, Any]]) -> Dict[str, Any]:
        prepared_results = {"query_embedding": {}, "results": {}}
        for result in search_result[0]:
            filtered_result = {}
            filtered_result["sequence"] = result["sequence"]
            filtered_result["score"] = result["distance"]
            prepared_results["results"][result["id"]] = filtered_result
        return prepared_results
