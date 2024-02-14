from texttable import Texttable
import argparse

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

    current_FPs = 0
    current_FNs = 0
    current_TPs = 0
    current_round = 0
    FPs_per_round = {}
    FNs_per_round = {}

    with open(args.file, "r") as f:
        lines = f.readlines()
        for line in lines:
            if("Server Round" in line):
                current_round = line.rstrip("\n").split(": ")[1]
            elif("False Positives" in line):
                current_FPs += line.rstrip("\n").split(": ")[1]
            elif("False Negatives" in line):
                current_FNs += line.rstrip("\n").split(": ")[1]
            elif("True Positives" in line):
                current_TPs += line.rstrip("\n").split(": ")[1]
    
    table.add_row(["Total Rounds", "Total FPs", "Total FNs", "Total TPs"])
    table.add_row([current_round, current_FPs, current_FNs, current_TPs])
    print(table.draw())