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
    """List all key-value pairs in a columnized format."""
    print("Key-Value Pairs:")
    max_key_length = max(len(key) for key in reader.fields.keys())
    for key, field in reader.fields.items():
        value = field.parts[field.data[0]]
        if key.startswith("general."):
            print(f"{key:{max_key_length}} : {''.join(chr(v) for v in value)}")
        else:
            print(f"{key:{max_key_length}} : {value}")
    print("---")


def list_tensors(reader: GGUFReader) -> None:
    """List all tensors with shape, size, and quantization info."""
    tensor_info_format = "{:<30} | Shape: {:<15} | Size: {:<12} | Quantization: {}"
    print(tensor_info_format.format("Tensor Name", "Shape", "Size", "Quantization"))
    print("---")
    for tensor in reader.tensors:
        shape_str = "x".join(map(str, tensor.shape))
        size_str = str(tensor.n_elements)
        quantization_str = tensor.tensor_type.name
        print(tensor_info_format.format(tensor.name, shape_str, size_str, quantization_str))


def set_logger(verbose: bool) -> None:
    """Set the logger level based on the verbosity flag."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level)
    logger.setLevel(log_level)


def get_arguments() -> Namespace:
    """Parse and return command-line arguments."""
    parser = ArgumentParser(description="Read and inspect GGUF model files.")
    parser.add_argument("model_path", help="Path to the GGUF model file.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output.")
    return parser.parse_args()


def main() -> None:
    """Main function to read and print information from a GGUF file."""
    args = get_arguments()
    set_logger(args.verbose)

    # Validate the model path
    model_path = Path(args.model_path)
    if not model_path.is_file():
        logger.error(f"The specified model path {model_path} does not exist or is not a file.")
        sys.exit(1)

    # Read and inspect the GGUF file
    reader = GGUFReader(model_path)

    list_key_value_pairs(reader)
    list_tensors(reader)


if __name__ == "__main__":
    main()
