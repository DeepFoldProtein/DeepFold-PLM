from typing import Any, Dict, List


def combine_results(model_results: Dict[str, Dict]) -> Dict[str, Any]:
    """Combine search results from different models into a single result."""
    combined_results = {"query_embedding": {}, "results": {}}

    for result in model_results.values():
        combined_results["query_embedding"].update(result["query_embedding"])
        for key, value in result["results"].items():
            if key in combined_results["results"]:
                combined_results["results"][key]["sequence"] = value.get(
                    "sequence", combined_results["results"][key].get("sequence")
                )
                combined_results["results"][key]["score"] = max(
                    combined_results["results"][key]["score"], value["score"]
                )
            else:
                combined_results["results"][key] = value

    return combined_results


def json_to_fasta(
    input_sequence: str, results: Dict[str, Dict[str, Any]], fasta_file: str
):
    """Write the input sequence and results to a FASTA file."""
    with open(fasta_file, "w") as file:
        file.write(f"{input_sequence}\n")
        file.write(f"{input_sequence}\n")
        for protein_id, attributes in results.items():
            file.write(f">{protein_id}\n{attributes['sequence']}\n")


def make_fasta(input_sequence: str, result_json: Dict[str, Any], fasta_file: str):
    """Generate a FASTA file from the result JSON."""
    results = result_json.get("results", {})
    json_to_fasta(input_sequence, results, fasta_file)


def read_fasta(input_fasta: str) -> str:
    """Read the input FASTA file and return the sequence."""
    with open(input_fasta, "r") as file:
        return file.read().strip()


def combine_and_generate_fasta(
    input_fasta: str, output_fasta: str, model_results: Dict[str, Dict]
):
    """Combine search results from different models and generate a FASTA file."""
    input_sequence = read_fasta(input_fasta)  # Read input FASTA file
    combined_results = combine_results(model_results)  # Combine the results
    make_fasta(
        input_sequence, combined_results, output_fasta
    )  # Generate the FASTA file
    print(f"FASTA file {output_fasta} has been successfully created.")
