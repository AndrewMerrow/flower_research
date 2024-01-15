from prettytable import PrettyTable
from texttable import Texttable

def retrieveAccuracy(table, accuracies):
    for line in accuracies:
        accuracy = line.split(": ")[1]
        print(f"{float(accuracy):.3%}")
    return(table)

def countMaliciousFlags(table, predicted_malicious):
    #print(predicted_malicious)
    #table = PrettyTable(['Client', 'Malicious Flags'])
    table.add_row(["Client", "Malicious Flags", "Benign Flags"])
    times_flagged = {}
    for client in range(30):
        mal_counter = 0
        ben_counter = 0
        for round in predicted_malicious:
            if(" {},".format(client) in round):
                #print(round)
                mal_counter += 1
            else:
                ben_counter += 1
        times_flagged[client] = (mal_counter, ben_counter)
    #print(times_flagged)
    for key, value in times_flagged.items():
        table.add_row([key, value[0], value[1]])
    #print(table.draw())
    return(table)

def main():
    table = Texttable()
    predicted_malicious = []
    accuracies = []
    poison_accuracies = []
    with open("cifarOutput.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            if("[" in line):
                predicted_malicious.append(line.rstrip('\n'))
            elif("poison" not in line and "accuracy:" in line):
                accuracies.append(line.rstrip('\n'))
            elif("poison" in line):
                poison_accuracies.append(line.rstrip("\n"))

    table = countMaliciousFlags(table, predicted_malicious)
    print(table.draw())
    table = retrieveAccuracy(table, accuracies)


if __name__ == "__main__":
    main()