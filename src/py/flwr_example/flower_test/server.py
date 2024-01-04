from typing import Dict, List, Tuple, Optional, Union
from functools import reduce
from collections import OrderedDict
import argparse
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from flwr.common import Metrics, Scalar, EvaluateRes, FitRes, parameters_to_ndarrays, ndarrays_to_parameters, NDArray, NDArrays
from flwr.server.client_proxy import ClientProxy
from utils import H5Dataset

import numpy as np
import copy

import flwr as fl
import torch

import utils
import math
import warnings

import our_detect_model_poisoning

warnings.filterwarnings("ignore")

def get_on_fit_config_fn():

    def fit_config(server_round: int):
        """Return training configuration dict for each round.

        Keep batch size fixed at 32, perform two rounds of training with one
        local epoch, increase to two local epochs afterwards.
        """
        new_list = ""
        if selectedDataset == "fedemnist":
            id_list = np.random.choice(3383, math.floor(3383*.01), replace=False)
            #print("ID list:")
            #print(id_list)
            #The ID list has to be converted to a string because Flower won't except a list as a config option
            new_list = ""
            for item in id_list:
                new_list += " " + str(item)
        #new_list is used by the clients to select a dataslice during fedemnist test...it is not needed for cifar 
        else:
            new_list == ""
        #print(new_list)
        config = {
            "batch_size": 64 if selectedDataset == "fedemnist" else 256,
            "current_round": server_round,
            "local_epochs": 10 if selectedDataset == "fedemnist" else 2,
            "id_list": new_list,
        }
        return config
    
    return fit_config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 1 #if server_round < 4 else 10
    return {"val_steps": val_steps}


def get_evaluate_fn(model: torch.nn.Module, toy: bool, data):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    trainset, testset, _ = utils.load_data(data)

    n_train = len(trainset)
    if toy:
        # use only 10 samples as validation set
        valset = torch.utils.data.Subset(trainset, range(n_train - 10, n_train))
        idxs = (testset.targets == 5).nonzero().flatten().tolist()
        poisoned_valset = utils.DatasetSplit(copy.deepcopy(testset), idxs)
        utils.poison_dataset(poisoned_valset.dataset, "cifar10", idxs, poison_all=True)
    else:
        # Use the last 5k training examples as a validation set
        #valset = torch.utils.data.Subset(trainset, range(n_train - 5000, n_train))
        idxs = (testset.targets == 5).nonzero().flatten().tolist()
        poisoned_valset = utils.DatasetSplit(copy.deepcopy(testset), idxs)
        #utils.poison_dataset(poisoned_valset.dataset, "cifar10", idxs, poison_all=True)

    if(selectedDataset == "cifar10"):
        valLoader = DataLoader(testset, batch_size=256, shuffle=False)
        poisoned_val_loader = DataLoader(poisoned_valset, 256, shuffle=False)
    else:
        valLoader = DataLoader(testset, batch_size=64, shuffle=False)
        poisoned_val_loader = DataLoader(poisoned_valset, 64, shuffle=False)

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        # Update model with the latest parameters
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=False)

        loss, accuracy, per_class_accuracy = utils.test(model, valLoader)
        poison_loss, poison_accuracy, poison_per_class_accuracy = utils.test(model, poisoned_val_loader)
        if(loss == "nan"):
            print("LOSS IS NAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAN")
        return loss, {"accuracy": accuracy, "per_class_accuracy": per_class_accuracy, "poison_accuracy": poison_accuracy}

    return evaluate

class AggregateCustomMetricStrategy(fl.server.strategy.FedAvgM):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}
        
        #for client in results:
        #   print("Client: " + str(client[1].metrics))
        #    for metric in client[1].metrics:
        #        print(metric)

        # Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        #print("Aggregated parameters")
        #print(parameters_to_ndarrays(aggregated_parameters))

        #new custom aggregation (delta value implementation)
        _, clientExample = results[0]
        #n_params = len(parameters_to_ndarrays(clientExample.parameters))

        #number of cifar model parameters
        if selectedDataset == "cifar10":
            n_params = 537610
        #number of fedemnist model parameters 
        else:
            n_params = 1199882

        #lr_vector = torch.Tensor([self.server_learning_rate]*n_params)
        lr_vector = np.array([self.server_learning_rate]*n_params)
        # Convert results (creates tuples of the client updates and their number of training examples for weighting purposes)
        #for _, fit_res in results:
            #print("FIT RES")
            #print(len(fit_res.parameters))
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        #print("WEIGHT TEST")
        #print(weights_results[1])
        #print(len(weights_results))
        #print(len(weights_results[0][0]))

        #testing to see if I can get the params from the model in the format I want 
        #model = utils.Net()
        #params_dict = zip(model.state_dict().keys(), weights_results[0][0])
        #state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        #model.load_state_dict(state_dict, strict=False)
        #UTD_test = parameters_to_vector(model.parameters()).detach()
        #print("UTD test")
        #print(UTD_test)
        #print(len(UTD_test))

        update_dict = {}
        #Construct the UTD update dict for detection purposes
        for _, r in results:
            #print("THE CLIENTS ID IS: ")
            #print(r.metrics["clientID"])
            #print(r.parameters)

            #select the correct model for the selected dataset
            if selectedDataset == "cifar10":
                model = utils.Net()
            else:
                model = utils.CNN_MNIST()

            params_dict = zip(model.state_dict().keys(), parameters_to_ndarrays(r.parameters))
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=False)
            UTD_test = parameters_to_vector(model.parameters()).detach()
            update_dict[r.metrics["clientID"]] = UTD_test

        #Checking if the update dict is being created correctly 
        #print("UPDATE DICT")
        #print(update_dict)
            
        #convert data into a dataframe for our detection code
        df = pd.DataFrame(update_dict)
        #print(df)
        K = len(df.columns)
        detection_slice = df.tail(10).reset_index(drop=True)
        for column in detection_slice.columns:
            #print(column)
            detection_slice.rename({column: "Client_" + str(column)}, axis=1, inplace=True)
        #print(detection_slice)
        #call our detection code
        our_detect_model_poisoning.detect_malicious(selectedDataset, detection_slice, K, "lof")
        
        
        #print("LR vector before detect check")
        print(lr_vector)
        #This line runs the detection code...without this line, the LR vector won't do anything
        if detect:
            print("RUNNING DETECTION")
            lr_vector = compute_robustLR(update_dict)
        
        #Testing to see if the LR vector is being created correctly 
        #print("LR vector AFTER detect check")
        print(lr_vector)
        print(len(lr_vector))

        #vectorTest = lr_vector * update_dict[1]
        #print("LR vector multiplication test")
        #print(vectorTest)
        
        #interpretation of the aggregate.py flower code
        num_examples_total = sum([num_examples for _, num_examples in weights_results])
        weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in weights_results
        ]
        #print("WEIGHTED WEIGHTS")
        #print(len(weighted_weights))
        #total_data = 0
        #for layer in weighted_weights:
        #    total_data += len(layer)
        #compute average weights of each layer
        weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
        ]

        #Multiply LR vector with the prime weights (do the final detection step)
        if selectedDataset == "cifar10":
            model = utils.Net()
        else:
            model = utils.CNN_MNIST()

        params_dict = zip(model.state_dict().keys(), weights_prime)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=False)
        primeParams = parameters_to_vector(model.parameters()).detach()
        finalParams = primeParams * lr_vector
        #print("FINAL PARAMS")
        #print(finalParams)
        vector_to_parameters(finalParams, model.parameters())
        #print("WEIGHT PRIME")
        #print(weights_prime)
        weights_prime = utils.get_model_params(model)
        #print("AFTER DETECTION")
        #print(weights_prime)

        cur_global_params = parameters_to_ndarrays(self.initial_parameters)
        params_old = self.initial_parameters
        #print(lr_vector.shape)
        #print(lr_vector)
        #print(weights_prime)
        #total_data = 0
        #for layer in weights_prime:
        #    total_data += len(layer)
        #print("TOTAL DATA: " + str(total_data))
        #test1 = lr_vector * torch.from_numpy(parameters_to_ndarrays(ndarrays_to_parameters(weights_prime)))
        #test2 = cur_global_params + test1

        #metric stuff
        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["train_accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]
        poisonAccuracies = [r.metrics["poison_accuracy"] * r.num_examples for _, r in results]

        # Aggregate and print custom metric
        aggregated_accuracy = sum(accuracies) / sum(examples)
        print(f"Round {server_round} accuracy aggregated from client fit results: {aggregated_accuracy}")

        aggregated_poison_accuracy = sum(poisonAccuracies) / sum(examples)
        print(f"Round {server_round} poison accuracy aggregated from client fit results: {aggregated_poison_accuracy}")

        # Return aggregated model paramters and other metrics (i.e., aggregated accuracy)
        return ndarrays_to_parameters(weights_prime), {"accuracy": aggregated_accuracy}
        #return model.parameters(), {"accuracy": aggregated_accuracy}
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation accuracy using weighted average."""

        if not results:
            return None, {}
        
        #for client in results:
        #    print("Client results: " + str(client[1].metrics))

        # Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]
        #poisonAccuracies = [r.metrics["poison_accuracy"] * r.num_examples for _, r in results]

        # Aggregate and print custom metric
        aggregated_accuracy = sum(accuracies) / sum(examples)
        print(f"Round {server_round} accuracy aggregated from client eval results: {aggregated_accuracy}")

        #aggregated_poison_accuracy = sum(poisonAccuracies) / sum(examples)
        #print(f"Round {server_round} poison accuracy aggregated from client fit results: {aggregated_poison_accuracy}")

        # Return aggregated loss and metrics (i.e., aggregated accuracy)
        #print("AHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
        return aggregated_loss, {"accuracy": aggregated_accuracy}

def compute_robustLR(agent_updates_dict):
        agent_updates_sign = [torch.sign(update) for update in agent_updates_dict.values()]  
        sm_of_signs = torch.abs(sum(agent_updates_sign))
        
        sm_of_signs[sm_of_signs < 8] = -1
        sm_of_signs[sm_of_signs >= 8] = 1                                           
        return sm_of_signs

def main():
    """Load model for
    1. server-side parameter initialization
    2. server-side parameter evaluation
    """

    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--toy",
        type=bool,
        default=False,
        required=False,
        help="Set to true to use only 10 datasamples for validation. \
            Useful for testing purposes. Default: False",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="cifar10",
        required=False,
        help="Used to select the dataset to train on"
    )
    parser.add_argument(
        "--detect",
        type=bool,
        default=False,
        required=False,
        help="Toggle to enable or disable poisoning detection/mitigation"
    )

    args = parser.parse_args()

    global selectedDataset 
    global detect
    detect = args.detect
    selectedDataset = args.data
    
    if(args.data == "cifar10"):
        model = utils.Net()
    else:
        model = utils.CNN_MNIST()

    print("NUMBER OF PARAMETERS: " + str(len(parameters_to_vector(model.parameters()))))

    model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]

    # Create strategy
    num_agents = 33 if selectedDataset == "fedemnist" else 40
    #strategy = fl.server.strategy.FedAvg(
    strategy = AggregateCustomMetricStrategy(
        min_fit_clients=num_agents,
        min_evaluate_clients=num_agents,
        #fraction_fit=.01,
        #fraction_evaluate=.01,
        min_available_clients=num_agents,
        evaluate_fn=get_evaluate_fn(model, args.toy, args.data),
        on_fit_config_fn=get_on_fit_config_fn(), #fit_config,
        on_evaluate_config_fn=evaluate_config,
        server_learning_rate = 1.0,
        server_momentum = 0,
    #    evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
        
    )

    # Start Flower server for four rounds of federated learning
    fl.server.start_server(
        server_address="10.100.116.10:8080",
        config=fl.server.ServerConfig(num_rounds=500 if selectedDataset == "fedemnist" else 200),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()