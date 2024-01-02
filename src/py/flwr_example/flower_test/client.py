import utils
from torch.utils.data import DataLoader
import torchvision.datasets
import torch
import flwr as fl
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import argparse
from collections import OrderedDict
import warnings
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
from pytorch_lightning import seed_everything
from utils import H5Dataset

warnings.filterwarnings("ignore")


class CifarClient(fl.client.NumPyClient):
    def __init__(
        self,
        trainset: torchvision.datasets,
        testset: torchvision.datasets,
        device: str,
        validation_split: int = 0.1,
    ):
        self.device = device
        self.trainset = trainset
        self.testset = testset
        self.validation_split = validation_split

    def set_parameters(self, parameters):
        """Loads a CNN model and replaces it parameters with the ones
        given."""
        #print("Params: " + str(parameters))
        if(selectedDataset == 'cifar10'):
            model = utils.Net()
        else:
            model = utils.CNN_MNIST()
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=False)
        return model

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""
        print("\nCurrent round: " + str(config['current_round']))
        print("Batch size: " + str(config['batch_size']))
        print("Local epochs: " + str(config['local_epochs']))
        # Update local model parameters
        model = self.set_parameters(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        n_valset = int(len(self.trainset) * self.validation_split)

        #print("The selected dataset is {}".format(selectedDataset))
        if(selectedDataset == "fedemnist"):
            id_list = config['id_list']
            id_list = id_list.split(" ")
            id_list = id_list[1:]
            print(id_list)
            print("Using {} as my ID".format(id_list[clientID]))
            self.trainset = torch.load(f'./dataset/Fed_EMNIST/user_trainsets/user_trainsets/user_{id_list[clientID]}_trainset.pt')
            if(int(id_list[clientID]) < 338):
                print("POISONING MY DATA")
                utils.poison_dataset(self.trainset, selectedDataset, None, id_list[clientID])

        #valset = torch.utils.data.Subset(self.trainset, range(0, n_valset))
        valset = self.testset
        #trainset = torch.utils.data.Subset(
        #    self.trainset, range(n_valset, len(self.trainset))
        #)
        trainset = self.trainset

        idxs = (self.testset.targets == 5).nonzero().flatten().tolist()
        trainLoader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        valLoader = DataLoader(valset, batch_size=batch_size, shuffle=False)
        
        #create a copy to be poisoned and another copy as a control 
        poisoned_val_set = utils.DatasetSplit(copy.deepcopy(self.testset), idxs)
        #clean_val_set = utils.DatasetSplit(copy.deepcopy(self.testset), idxs)

        utils.poison_dataset(poisoned_val_set.dataset, selectedDataset, idxs, clientID, poison_all=False)
        #print(poisoned_val_set.dataset.data.shape)

        poisoned_val_loader = DataLoader(poisoned_val_set, batch_size=256, shuffle=False, pin_memory=False)
        #poisoned_val_loader = DataLoader(valset, batch_size=256, shuffle=False, pin_memory=False)
        #test images for visualization to confirm poisoning was successful 
        #test_poison = poisoned_val_set.dataset.data[49988]
        #test_clean = clean_val_set.dataset.data[3000]
        #test_clean = poisoned_val_set.dataset.data[49988]

        #visualize the poisoned image
        #fig = plt.figure()
        #ax1 = fig.add_subplot(2,2,1)
        #ax1.imshow(test_clean)
        #ax2 = fig.add_subplot(2,2,2)
        #ax2.imshow(test_poison)
        #plt.show()

        #training
        parameters_old = utils.get_model_params(model)
        #test_params = parameters_old - parameters_old
        
        #parameters_old = parameters_to_ndarrays(utils.get_model_params(model))
        
        #This is the format we want
        parameters_test = parameters_to_vector(model.parameters()).detach()
        #print("Old paramters")
        #print(parameters_test)
        #print(len(parameters_test))

        results = utils.train(model, trainLoader, valLoader, poisoned_val_loader, epochs, self.device)
        parameters_prime = utils.get_model_params(model)
        
        #print("Prime type:")
        #print(type(parameters_prime))
        #print(parameters_prime)
        
        parameters_new = parameters_to_vector(model.parameters()).detach()
        
        #print("new parameters")
        #print(parameters_prime)

        #This is the format from UTD, but flwr won't let me return it
        vectorTest = np.subtract(parameters_new, parameters_test)
        #print(vectorTest)
        #print(len(vectorTest))
        #print(type(vectorTest.numpy()))
        
        #test_params = parameters_prime - parameters_old
        #test_params = parameters_prime - parameters_old
        test_params = []
        #TODO: try using numpy subtract
        for param1, param2 in zip(parameters_prime, parameters_old):
            test_params.append(param1 - param2)
        #test_params = np.subtract(parameters_prime, parameters_old)

        #print("Update test")
        #print(torch.count_nonzero(test_params))
        #print(test_params)
        #print("type 1: ")
        #print(type(test_params))
        num_examples_train = len(trainset)
        #vector_to_parameters(test_params, test_params)
        #print("type 2: ")
        #print(type(test_params))
        #test_params = parameters_to_ndarrays(test_params)

        #add the ID of the client to be sent back to the server
        results["clientID"] = clientID

        return test_params, num_examples_train, results
        #return [vectorTest.numpy()], num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""
        # Update local model parameters
        model = self.set_parameters(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        testloader = DataLoader(self.testset, batch_size=256)

        loss, accuracy, per_class_accuracy = utils.test(model, testloader, steps, self.device)
        return float(loss), len(self.testset), {"accuracy": float(accuracy)}


def client_dry_run(device: str = "cpu"):
    """Weak tests to check whether all client methods are working as
    expected."""

    #model = utils.load_efficientnet(classes=10)
    model = utils.Net()
    trainset, testset = utils.load_partition(0)
    print("Targets")
    print(trainset.targets)
    idxs = (trainset.targets == 5).nonzero().flatten().tolist()
    #print(idxs)
    utils.poison_dataset(trainset, idxs, poison_all=True)
    #trainset = torch.utils.data.Subset(trainset, range(10))
    #testset = torch.utils.data.Subset(testset, range(10))
    client = CifarClient(trainset, testset, device)
    client.fit(
        utils.get_model_params(model),
        {"batch_size": 16, "local_epochs": 1, "current_round": 1},
    )

    client.evaluate(utils.get_model_params(model), {"val_steps": 32})

    print("Dry Run Successful")


def main() -> None:
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--dry",
        type=bool,
        default=False,
        required=False,
        help="Do a dry-run to check the client",
    )
    parser.add_argument(
        "--partition",
        type=int,
        default=0,
        choices=range(0, 10),
        required=False,
        help="Specifies the artificial data partition of CIFAR10 to be used. \
        Picks partition 0 by default",
    )
    parser.add_argument(
        "--toy",
        type=bool,
        default=False,
        required=False,
        help="Set to true to quicky run the client using only 10 datasamples. \
        Useful for testing purposes. Default: False",
    )
    parser.add_argument(
        "--use_cuda",
        type=bool,
        default=True,
        required=False,
        help="Set to true to use GPU. Default: False",
    )
    parser.add_argument(
        "--poison",
        type=bool,
        default=False,
        required=False,
        help="Set to true to make the client poison their train data"
    )
    parser.add_argument(
        "--clientID",
        type=int,
        default=0,
        choices=range(0,3383),
        required=False,
        help="Used so each client knows which data slice to use"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="cifar10",
        required=False,
        help="Used to select the dataset to train on"
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu"
    )

    if args.dry:
        client_dry_run(device)
    else:
        #Seed everything test to try enabling reproducability 
        #seed_everything(42)
        #global variable used to keep track of what dataset the experiment is using
        seed_everything(42)
        global selectedDataset
        global clientID
        selectedDataset = args.data
        clientID = args.clientID

        print("Client ID {}".format(args.clientID))
        #trainset, testset = utils.load_partition(args.partition)
        trainset, testset, num_examples = utils.load_data(args.data)

        #split the dataset into slices and store the slices in user_groups
        if(args.data != "fedemnist"):
            user_groups = utils.distribute_data(trainset)
        #print(str(user_groups))

        #Use the client's ID to select which slice of the data to use
        if(args.data != "fedemnist"):
            trainset = utils.DatasetSplit(trainset, user_groups[args.clientID])
        #fedemnist is handled differently
        else:
            trainset = torch.load(f'./dataset/Fed_EMNIST/user_trainsets/user_trainsets/user_{clientID}_trainset.pt')

        #Poison the data if the poison option is selected
        if args.poison:
            print("poisoning the data")
            #idxs = (trainset.targets == 5).nonzero().flatten().tolist()
            utils.poison_dataset(trainset.dataset, selectedDataset, user_groups[args.clientID], agent_idx=args.clientID)

        if args.toy:
            trainset = torch.utils.data.Subset(trainset, range(10))
            testset = torch.utils.data.Subset(testset, range(10))

        # Start Flower client
        client = CifarClient(trainset, testset, device)

        fl.client.start_numpy_client(server_address="10.100.116.10:8080", client=client)


if __name__ == "__main__":
    main()