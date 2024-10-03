"""
Script: read
"""

import argparse
import logging
import sys
from pathlib import Path

from gguf.gguf_reader import GGUFReader

logger = logging.getLogger("reader")


def read_gguf_file(gguf_file_path, verbose):
    """
    Reads and prints key-value pairs and tensor information from a GGUF file in an improved format.

    Parameters:
    - gguf_file_path: Path to the GGUF file.
    - verbose: Whether to enable detailed logging.
    """

    if verbose:
        logger.setLevel(logging.DEBUG)
        logging.basicConfig(level=logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
        logging.basicConfig(level=logging.INFO)

    reader = GGUFReader(gguf_file_path)

    # List all key-value pairs in a columnized format
    print("Key-Value Pairs:")  # noqa: NP100
    max_key_length = max(len(key) for key in reader.fields.keys())
    for key, field in reader.fields.items():
        value = field.parts[field.data[0]]
        print(f"{key:{max_key_length}} : {value}")  # noqa: NP100
    print("----")  # noqa: NP100

    # List all tensors
    print("Tensors:")  # noqa: NP100
    tensor_info_format = "{:<30} | Shape: {:<15} | Size: {:<12} | Quantization: {}"
    print(tensor_info_format.format("Tensor Name", "Shape", "Size", "Quantization"))  # noqa: NP100
    print("-" * 80)  # noqa: NP100
    for tensor in reader.tensors:
        shape_str = "x".join(map(str, tensor.shape))
        size_str = str(tensor.n_elements)
        quantization_str = tensor.tensor_type.name
        print(
            tensor_info_format.format(tensor.name, shape_str, size_str, quantization_str)
        )  # noqa: NP100


def main():
    parser = argparse.ArgumentParser(description="Read and inspect GGUF model files.")
    parser.add_argument("model_path", help="Path to the GGUF model file.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output.")
    args = parser.parse_args()

    # Validate the model path
    model_path = Path(args.model_path)
    if not model_path.is_file():
        logger.error(f"The specified model path {model_path} does not exist or is not a file.")
        sys.exit(1)

    # Read and inspect the GGUF file
    read_gguf_file(model_path, args.verbose)


if __name__ == "__main__":
    main()
