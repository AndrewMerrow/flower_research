from prettytable import PrettyTable
from texttable import Texttable

def countMaliciousFlags(predicted_malicious):
    print(predicted_malicious)
    #table = PrettyTable(['Client', 'Malicious Flags'])
    table = Texttable()
    table.add_rows([["Client", "Malicious Flags", "Benign Flags"]])
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
    print(table.draw())

def main():
    print('running main')
    predicted_malicious = []
    with open("cifarOutput.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            if("[" in line):
                predicted_malicious.append(line.rstrip('\n'))
    countMaliciousFlags(predicted_malicious)