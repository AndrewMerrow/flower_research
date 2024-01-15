from prettytable import PrettyTable

predicted_malicious = []
with open("cifarOutput.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        if("[" in line):
            predicted_malicious.append(line.rstrip('\n'))

#print(predicted_malicious)
table = PrettyTable(['Client', 'Malicious Flags'])
times_flagged = {}
for client in range(30):
    counter = 0
    for round in predicted_malicious:
        if(" {},".format(client) in round):
            #print(round)
            counter += 1
    times_flagged[client] = counter
print(times_flagged)
for key, value in times_flagged.items():
    table.add_row([key, value])