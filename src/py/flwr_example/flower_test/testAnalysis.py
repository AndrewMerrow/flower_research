predicted_malicious = []
with open("cifarOutput.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        if("[" in line):
            predicted_malicious.append(line.rstrip('\n'))

#print(predicted_malicious)
for client in [0,1,2,3]:
    counter = 0
    for round in predicted_malicious:
        if(" {},".format(client) in round):
            #print(round)
            counter += 1
    print(counter)