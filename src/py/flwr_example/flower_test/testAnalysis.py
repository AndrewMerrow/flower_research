predicted_malicious = []
with open("cifarOutput.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        if("[" in line):
            predicted_malicious.append(line.rstrip('\n'))

#print(predicted_malicious)
for round in predicted_malicious:
    if(" 3," in round):
        print(round)