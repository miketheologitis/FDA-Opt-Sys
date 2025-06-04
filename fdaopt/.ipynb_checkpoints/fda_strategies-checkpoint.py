from typing import Callable, Optional, Union

import numpy as np

from flwr.common import (
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.strategy.aggregate import aggregate, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAdam, FedAvgM, FedAdagrad

import torch


import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')


def vectorize_ndarrays(parameters):
    """Convert a list of ndarrays to a single vector (torch tensor)."""
    con = []
    for param in parameters:
        param = torch.from_numpy(param)
        con.append(param.reshape(-1))

    return torch.cat(con)



class FdaAdam(FedAdam):

    """FDA-Adam strategy for federated learning with variance monitoring."""
    def __init__(self, local_epochs, threshold, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize local epochs and threshold for variance monitoring
        self.local_epochs = local_epochs
        self.threshold = threshold
        
        
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:

        
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        
        client_vecs = [
            vectorize_ndarrays(client_params)
            for client_params, _ in weights_results
        ]
        
        # aggregate(weight_results) is list of ndarrays
        aggregated_vec = vectorize_ndarrays(aggregate(weights_results))
        
        drifts = [
            client_vec - aggregated_vec
            for client_vec in client_vecs
        ]

        # Calculate actual variance of clients
        variance = sum([
            torch.dot(drift, drift)
            for drift in drifts
        ]).item() / len(drifts)
        
        logging.info(f"[FdaAdam] Actual Variance: {variance}!")

        # We have given the aggregation function which extracts `epochs_completed` from the metrics
        # (returned by clients).
        fedavg_parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round=server_round, results=results, failures=failures
        )

        # Extract the number of epochs completed from the aggregated metrics
        epochs_completed = metrics_aggregated['epochs_completed']

        # Update the threshold accordingly:
        self.threshold.value = ((self.local_epochs / 2) / epochs_completed) * variance
        
        logging.info(f"[FdaAdam] epochs completed: {epochs_completed}  ,  new threshold: {self.threshold.value}!")
        
        return fedavg_parameters_aggregated, metrics_aggregated


class FdaAvgM(FedAvgM):
    """FDA-Adam strategy for federated learning with variance monitoring."""

    def __init__(self, local_epochs, threshold, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize local epochs and threshold for variance monitoring
        self.local_epochs = local_epochs
        self.threshold = threshold

    def aggregate_fit(
            self,
            server_round: int,
            results: list[tuple[ClientProxy, FitRes]],
            failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        client_vecs = [
            vectorize_ndarrays(client_params)
            for client_params, _ in weights_results
        ]

        # aggregate(weight_results) is list of ndarrays
        aggregated_vec = vectorize_ndarrays(aggregate(weights_results))

        drifts = [
            client_vec - aggregated_vec
            for client_vec in client_vecs
        ]

        # Calculate actual variance of clients
        variance = sum([
            torch.dot(drift, drift)
            for drift in drifts
        ]).item() / len(drifts)

        logging.info(f"[FdaAvgM] Actual Variance: {variance}!")

        # We have given the aggregation function which extracts `epochs_completed` from the metrics
        # (returned by clients).
        fedavg_parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round=server_round, results=results, failures=failures
        )

        # Extract the number of epochs completed from the aggregated metrics
        epochs_completed = metrics_aggregated['epochs_completed']

        # Update the threshold accordingly:
        self.threshold.value = ((self.local_epochs / 2) / epochs_completed) * variance

        logging.info(f"[FdaAvgM] epochs completed: {epochs_completed}  ,  new threshold: {self.threshold.value}!")

        return fedavg_parameters_aggregated, metrics_aggregated


class FdaAdagrad(FedAdagrad):
    """FDA-Adam strategy for federated learning with variance monitoring."""

    def __init__(self, local_epochs, threshold, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize local epochs and threshold for variance monitoring
        self.local_epochs = local_epochs
        self.threshold = threshold

    def aggregate_fit(
            self,
            server_round: int,
            results: list[tuple[ClientProxy, FitRes]],
            failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        client_vecs = [
            vectorize_ndarrays(client_params)
            for client_params, _ in weights_results
        ]

        # aggregate(weight_results) is list of ndarrays
        aggregated_vec = vectorize_ndarrays(aggregate(weights_results))

        drifts = [
            client_vec - aggregated_vec
            for client_vec in client_vecs
        ]

        # Calculate actual variance of clients
        variance = sum([
            torch.dot(drift, drift)
            for drift in drifts
        ]).item() / len(drifts)

        logging.info(f"[FdaAdagrad] Actual Variance: {variance}!")

        # We have given the aggregation function which extracts `epochs_completed` from the metrics
        # (returned by clients).
        fedavg_parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round=server_round, results=results, failures=failures
        )

        # Extract the number of epochs completed from the aggregated metrics
        epochs_completed = metrics_aggregated['epochs_completed']

        # Update the threshold accordingly:
        self.threshold.value = ((self.local_epochs / 2) / epochs_completed) * variance

        logging.info(f"[FdaAdagrad] epochs completed: {epochs_completed}  ,  new threshold: {self.threshold.value}!")

        return fedavg_parameters_aggregated, metrics_aggregated