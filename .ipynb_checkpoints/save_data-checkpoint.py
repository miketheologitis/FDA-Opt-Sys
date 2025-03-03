from flwr_datasets.partitioner import DirichletPartitioner
from flwr_datasets import FederatedDataset

total_clients = 10
dirichlet_alpha = 1.
ds_path = "glue"
ds_name = "mrpc"

"""Load data (training) """

partitioner = DirichletPartitioner(
    num_partitions=total_clients, alpha=dirichlet_alpha, partition_by="label"
)
fds = FederatedDataset(
    dataset=ds_path, 
    subset=ds_name,
    partitioners={"train": partitioner},
)

for i in range(total_clients):
    data = fds.load_partition(i, "train")
    
    data.save_to_disk(f"data/glue/mrpc/c{total_clients}_{i}.pth")
    