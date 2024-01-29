from prettytable import PrettyTable
from texttable import Texttable
import argparse


def retrieveAccuracy(table, accuracies, poison_accuracies):
    '''This function pulls the accuracy and poison accuracy metric for each 
    round and then creates the table containing the accuracy information'''
    #Add the column names to the table
    table.add_row(["Round", "Accuracy", "Poison Accuracy"])

    #This loop retrieves the base and poison accuracies to the nearest 2 decimal places and adds them to the accuracy table
    for i in range(len(accuracies)):
        accuracy = accuracies[i].split(": ")[1]
        poison_accuracy = poison_accuracies[i].split(": ")[1]
        table.add_row([i, '{:.2%}'.format(float(accuracy)), '{:.2%}'.format(float(poison_accuracy))])
    return(table)

def countMaliciousFlags(args, table, predicted_malicious, selected_clients = None):
    '''This function computes how many times each client was labeled as benign or malicious'''
    #Add the column names to the table
    table.add_row(["Client", "Malicious Flags", "Benign Flags"])
    #This dictionary stores the final count for each client ID (malicious/benign count)
    times_flagged = {}
    num_clients = 0
    if("cifar" in args.file):
        num_clients = 40
        for client in range(num_clients):
            mal_counter = 0
            ben_counter = 0
            #In this case, every client is selected each round
            for round in predicted_malicious:
                if(" {},".format(client) in round or "[{}".format(client) in round or " {}]".format(client) in round):
                    mal_counter += 1
                #If not predicted as malicous, the client is labeled benign
                else:
                    ben_counter += 1
            times_flagged[client] = (mal_counter, ben_counter)
        #print(times_flagged)
        for key, value in times_flagged.items():
            table.add_row([key, value[0], value[1]])

    #Since not every client is selected each round, we must check to see if the client was selected before labeling it as benign
    elif("fedemnist" in args.file):
        num_clients = 3383
        for client in range(num_clients):
            mal_counter = 0
            ben_counter = 0
            for i in range(len(predicted_malicious)):
                #If it was labeled as malicious, we know it was selected
                if(" {},".format(client) in predicted_malicious[i] or "[{}".format(client) in predicted_malicious[i] or " {}]".format(client) in predicted_malicious[i]):
                    mal_counter += 1
                #If it was not labeled malicious, we need to check to see if it was selected this round before labeling it benign
                elif(" {},".format(client) in selected_clients[i] or "[{}".format(client) in selected_clients[i] or " {}]".format(client) in selected_clients[i]):
                    ben_counter += 1
            times_flagged[client] = (mal_counter, ben_counter)

        #Add each client to the table using this form: (clientID, times labeled malicious, times labeled benign)
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

    #Two tables are created. One to store the amount of times each client is labeled malicious/benign, and one to store the accuracy info for each round
    table = Texttable()
    accuracyTable = Texttable()
    perRoundTable = Texttable()

    #values used to store the number of of each metric happening per round
    current_FPs = 0
    current_FNs = 0
    current_round = 0
    FPs_per_round = {}
    FNs_per_round = {}
    
    predicted_malicious = []
    accuracies = []
    poison_accuracies = []
    selected_clients = []
    false_positives = []
    false_positives_count = 0
    false_negatives = []
    false_negatives_count = 0
    server_round_count = 0
    cluster_method = ""
    with open(args.file, "r") as f:
        lines = f.readlines()
        #This loop puts the relevant lines from the test output file into the coordinating lists for analysis
        for line in lines:
            #retrieve the predicted malicious clients
            if("[" in line and "malicious" in line):
                predicted_malicious.append(line.rstrip('\n'))
            #retrieve the base accuracy
            elif("poison" not in line and "accuracy:" in line):
                accuracies.append(line.rstrip('\n'))
            #retrieve the poison accuracy
            elif("poison" in line):
                poison_accuracies.append(line.rstrip("\n"))
            elif("clustering" in line):
                cluster_method = line.strip("\n").split(" ")[1]
            #retrieve the false negatives
            elif("false negatives" in line):
                #convert the string into a list
                list_test = list(line.rstrip('\n')[:-1].split(": [")[1].split(", "))
                int_list = []
                #convert the list into a list of ints
                for value in list_test:
                    try:
                        int_list.append(int(value))
                    #if the value is not an int, skip it
                    except:
                        pass
                false_negatives.append(int_list)
                #we create a list of ints to avoid counting a value that is not a client ID
                current_FNs = len(int_list)
                false_negatives_count += len(int_list)
            #retrieve the false positives
            elif("false positives" in line):
                #convert the string into a list
                list_test = list(line.rstrip('\n')[:-1].split(": [")[1].split(", "))
                int_list = []
                #convert the list into a list of ints
                for value in list_test:
                    try:
                        int_list.append(int(value))
                    #if the value is not an int, skip it
                    except:
                        pass
                false_positives.append(int_list)
                #we create a list of ints to avoid counting a value that is not a client ID
                current_FPs = len(int_list)
                false_positives_count += len(int_list)
            #keep track of the current number of rounds
            elif("Server Round" in line):
                server_round_count += 1
                current_round = line.rstrip('\n').split(" ")[2]
            #retrieve the clients that were selected each round (useful for the benign counter for fedemnist)
            if("fedemnist" in args.file):
                if("selected clients" in line):
                    selected_clients.append(line.rstrip('\n'))
            FPs_per_round[current_round] = current_FPs
            FNs_per_round[current_round] = current_FNs

    print(FPs_per_round)
    print(FNs_per_round)
    #create the malicious/benign counter table
    table = countMaliciousFlags(args, table, predicted_malicious, selected_clients)
    #print(table.draw())
    print("\n--------------------------------------------------------------\n")

    #create the accuracy table
    accuracyTable = retrieveAccuracy(accuracyTable, accuracies, poison_accuracies)
    #print(accuracyTable.draw())
    print("\n--------------------------------------------------------------\n")

    print("Type of clustering used: " + str(cluster_method))
    perRoundTable.add_row(["Total Rounds", "Total FNs", "FN/Round", "Total FPs", "FP/Round"])
    perRoundTable.add_row([server_round_count, false_negatives_count, false_negatives_count/server_round_count, false_positives_count, false_positives_count/server_round_count])
    print(perRoundTable.draw())


if __name__ == "__main__":
    main()