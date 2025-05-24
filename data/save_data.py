import argparse
from flwr_datasets.partitioner import DirichletPartitioner
from flwr_datasets import FederatedDataset

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--total_clients", type=int, default=10)
    parser.add_argument("--dirichlet_alpha", type=float, default=1.)
    parser.add_argument("--ds_path", type=str, default='glue')
    parser.add_argument("--ds_name", type=str, default='mrpc')
    args = parser.parse_args()

    partitioner = DirichletPartitioner(
        num_partitions=args.total_clients, alpha=args.dirichlet_alpha, partition_by="label"
    )
    fds = FederatedDataset(
        dataset=f"{args.ds_path}/{args.ds_name}",
        partitioners={"train": partitioner},
    )

    for i in range(args.total_clients):
        data = fds.load_partition(i, "train")
        data.save_to_disk(f"{args.ds_path}/{args.ds_name}/c{args.total_clients}_{i}.pth")

    