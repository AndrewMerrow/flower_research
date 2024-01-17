from prettytable import PrettyTable
from texttable import Texttable
import argparse


def retrieveAccuracy(table, accuracies, poison_accuracies):
    table.add_row(["Round", "Accuracy", "Poison Accuracy"])
    #round = 0
    #for line in accuracies:
    #    accuracy = line.split(": ")[1]
    #    table.add_row([round, '{:.2%}'.format(float(accuracy))])
    #    round += 1

    for i in range(len(accuracies)):
        accuracy = accuracies[i].split(": ")[1]
        poison_accuracy = poison_accuracies[i].split(": ")[1]
        table.add_row([i, '{:.2%}'.format(float(accuracy)), '{:.2%}'.format(float(poison_accuracy))])
    return(table)

def countMaliciousFlags(args, table, predicted_malicious, selected_clients = None):
    #print(predicted_malicious)
    #table = PrettyTable(['Client', 'Malicious Flags'])
    table.add_row(["Client", "Malicious Flags", "Benign Flags"])
    times_flagged = {}
    num_clients = 0
    if("cifar" in args.file):
        num_clients = 40
        for client in range(num_clients):
            mal_counter = 0
            ben_counter = 0
            for round in predicted_malicious:
                if(" {},".format(client) in round or "[{}".format(client) in round or " {}]".format(client) in round):
                    #print(round)
                    mal_counter += 1
                else:
                    ben_counter += 1
            times_flagged[client] = (mal_counter, ben_counter)
        #print(times_flagged)
        for key, value in times_flagged.items():
            table.add_row([key, value[0], value[1]])

    elif("fedemnist" in args.file):
        num_clients = 3383
        for client in range(num_clients):
            mal_counter = 0
            ben_counter = 0
            for round in predicted_malicious:
                if((" {},".format(client) in round or "[{}".format(client) in round or " {}]".format(client) in round) and (" {},".format(client) in selected_clients or "[{}".format(client) in selected_clients or " {}]".format(client) in selected_clients)):
                    mal_counter += 1
                elif(" {},".format(client) in selected_clients or "[{}".format(client) in selected_clients or " {}]".format(client) in selected_clients):
                    ben_counter += 1
            times_flagged[client] = (mal_counter, ben_counter)

        for key, value in times_flagged.items():
            table.add_row([key, value[0], value[1]])
    
    #print(table.draw())
    return(table)

def main():
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--file",
        type=str,
        default="cifarOutput.txt",
        required=False,
        help="Used to select the dataset to train on"
    )
    args = parser.parse_args()

    table = Texttable()
    accuracyTable = Texttable()
    predicted_malicious = []
    accuracies = []
    poison_accuracies = []
    selected_clients = []
    with open(args.file, "r") as f:
        lines = f.readlines()
        for line in lines:
            if("[" in line and "malicious" in line):
                #mal_list = line.split(": ")[1]
                predicted_malicious.append(line.rstrip('\n'))
            elif("poison" not in line and "accuracy:" in line):
                accuracies.append(line.rstrip('\n'))
            elif("poison" in line):
                poison_accuracies.append(line.rstrip("\n"))

            if("fedemnist" in args.file):
                if("selected clients" in line):
                    selected_clients.append(line.rstrip('\n'))

    print("Clients selected each round: " + str(selected_clients))
    print("MALICIOUS PREDICTIONS: " + str(predicted_malicious))
    table = countMaliciousFlags(args, table, predicted_malicious, selected_clients)
    print(table.draw())
    accuracyTable = retrieveAccuracy(accuracyTable, accuracies, poison_accuracies)
    #print(accuracyTable.draw())


if __name__ == "__main__":
    main()