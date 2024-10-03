"""
Script: read
"""

import logging
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

from gguf.gguf_reader import GGUFReader

logger = logging.getLogger("reader")


def list_key_value_pairs(reader: GGUFReader) -> None:
    # List all key-value pairs in a columnized format
    print("Key-Value Pairs:")  # noqa: NP100
    max_key_length = max(len(key) for key in reader.fields.keys())
    for key, field in reader.fields.items():
        value = field.parts[field.data[0]]
        print(f"{key:{max_key_length}} : {value}")  # noqa: NP100
    print("----")  # noqa: NP100


def list_tensors(reader: GGUFReader) -> None:
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


def set_logger(verbose: bool):
    if verbose:
        logger.setLevel(logging.DEBUG)
        logging.basicConfig(level=logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
        logging.basicConfig(level=logging.INFO)


def get_arguments() -> Namespace:
    parser = ArgumentParser(description="Read and inspect GGUF model files.")
    parser.add_argument("model_path", help="Path to the GGUF model file.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output.")
    return parser.parse_args()


def main():
    """
    Reads and prints key-value pairs and tensor information from a GGUF file in an improved format.

    Parameters:
    - gguf_file_path: Path to the GGUF file.
    - verbose: Whether to enable detailed logging.
    """

    args = get_arguments()
    set_logger(args.verbose)

    # Validate the model path
    model_path = Path(args.model_path)
    if not model_path.is_file():
        logger.error(f"The specified model path {model_path} does not exist or is not a file.")
        sys.exit(1)

    # Read and inspect the GGUF file

    reader = GGUFReader(gguf_file_path)

    list_key_value_pairs(reader)
    list_tensors(reader)

    read_gguf_file(model_path, args.verbose)


if __name__ == "__main__":
    main()
