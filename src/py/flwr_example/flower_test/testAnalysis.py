with open("cifarOutput.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        if("[" in line):
            print(line)