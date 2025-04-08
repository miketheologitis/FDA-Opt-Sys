"""fdaopt: A Flower / HuggingFace app."""

import warnings
import transformers
from datasets.utils.logging import disable_progress_bar
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset, load_from_disk

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
disable_progress_bar()
transformers.logging.set_verbosity_error()


fds = None  # Cache FederatedDataset


def tokenize_function(ds_path, ds_name, tokenizer):
    """
    Return a tokenization function based on the dataset path and name.

    This function returns the appropriate tokenization function for the specified dataset.
    Currently, it supports the GLUE MRPC dataset.

    Args:
        ds_path (str): The path or identifier of the dataset.
        ds_name (str): The name of the dataset.
        tokenizer (AutoTokenizer): The tokenizer to be used.

    Returns:
        function: A tokenization function.
    """

    if ds_path == "glue" and ds_name == "mrpc":
        return lambda example: tokenizer(example["sentence1"], example["sentence2"], truncation=True)
    if ds_path == "glue" and ds_name == "rte":
        return lambda example: tokenizer(example["sentence1"], example["sentence2"], truncation=True)
    if ds_path == "glue" and ds_name == "stsb":
        return lambda example: tokenizer(example["sentence1"], example["sentence2"], truncation=True)
    if ds_path == "glue" and ds_name == "cola":
        return lambda example: tokenizer(example["sentence"], truncation=True)
    if ds_path == "glue" and ds_name == "sst2":
        return lambda example: tokenizer(example["sentence"], truncation=True)
    if ds_path == "glue" and ds_name == "qnli":
        return lambda example: tokenizer(example["question"], example["sentence"], truncation=True)

    return None


def get_test_ds(model_checkpoint, ds_path, ds_name):
    """ Prepare test dataset and create corresponding DataLoaders. """

    # Load the raw dataset
    raw_datasets = load_dataset(path=ds_path, name=ds_name)

    # Test dataset
    raw_test_dataset = raw_datasets['validation']

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # Create the tokenization function
    tokenize_fn = tokenize_function(ds_path, ds_name, tokenizer)

    # Create DataLoaders for each client dataset
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Preprocess the test dataset
    test_ds = preprocess_test_dataset(raw_test_dataset, tokenize_fn, data_collator, 8)

    return test_ds


def tokenize_client_dataset(client_dataset, tokenize_fn):
    """
    Tokenize and preprocess a client dataset.

    This function tokenizes and preprocesses each dataset in the provided list of client datasets.
    It applies the specified tokenization function, renames the "label" column to "labels",
    removes unnecessary columns, and sets the format to PyTorch tensors.

    Args:
        client_dataset (datasets.arrow_dataset.Dataset): A client dataset to be tokenized and preprocessed.
        tokenize_fn (function): A function that takes an example and returns its tokenized form.

    Returns:
        datasets.arrow_dataset.Dataset: A tokenized and preprocessed client datasets.
    """

    # Define the expected columns
    expected_columns = ['labels', 'input_ids', 'token_type_ids', 'attention_mask']

    tok_client_dataset = client_dataset.map(tokenize_fn, batched=True)

    tok_client_dataset = tok_client_dataset.rename_column("label", "labels")

    # Identify columns to remove
    columns_to_remove = [
        column for column in tok_client_dataset.column_names
        if column not in expected_columns
    ]

    # Remove unnecessary columns
    tok_client_dataset = tok_client_dataset.remove_columns(columns_to_remove)

    # Set the format to PyTorch tensors
    tok_client_dataset.set_format("torch")

    return tok_client_dataset


def load_data(partition_id, num_partitions, model_checkpoint, ds_path, ds_name, dirichlet_alpha, data_path=""):
    """Load data (training) """
    
    if data_path:
        client_partition = load_from_disk(data_path)
    
    else:
        global fds  # Only initialize `FederatedDataset` once

        if fds is None:
            partitioner = DirichletPartitioner(
                num_partitions=num_partitions, alpha=dirichlet_alpha, partition_by="label"
            )
            fds = FederatedDataset(
                dataset=ds_path,
                subset=ds_name,
                partitioners={"train": partitioner},
            )
    
        # Create client dataset
        client_partition = fds.load_partition(partition_id, "train")
        
    # ---- 1. Handle training data for client

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    # Create the tokenization function
    tokenize_fn = tokenize_function(ds_path, ds_name, tokenizer)
    
    # Tokenize the client dataset
    tok_client_dataset = tokenize_client_dataset(client_partition, tokenize_fn)
    
    # Create DataLoader for the client dataset
    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)
    client_dataloader = DataLoader(
        tok_client_dataset, shuffle=True, batch_size=8, collate_fn=collate_fn
    )

    return client_dataloader
            
            
def preprocess_test_dataset(raw_test_dataset, tokenize_fn, data_collator, batch_size):

    # Define the expected columns
    expected_columns = ['labels', 'input_ids', 'token_type_ids', 'attention_mask']

    tok_test_dataset = raw_test_dataset.map(tokenize_fn, batched=True)

    tok_test_dataset = tok_test_dataset.rename_column("label", "labels")

    # Identify columns to remove
    columns_to_remove = [
        column for column in tok_test_dataset.column_names
        if column not in expected_columns
    ]

    # Remove unnecessary columns
    tok_test_dataset = tok_test_dataset.remove_columns(columns_to_remove)

    # Set the format to PyTorch tensors
    tok_test_dataset.set_format("torch")

    # Create a DataLoader for the test dataset
    test_ds = DataLoader(
        tok_test_dataset, batch_size=batch_size, collate_fn=data_collator
    )

    return test_ds
