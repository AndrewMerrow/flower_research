from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from texttable import Texttable

def gatherInfo(filename):

    #values used to store the number of of each metric happening per round
    current_FPs = 0
    current_FNs = 0
    current_round = 0
    FPs_per_round = {}
    FNs_per_round = {}
    
    predicted_malicious = []
    accuracies = []
    poison_accuracies = []
    aggregated_training_accuracies = []
    aggregated_poison_accuracies = []
    selected_clients = []
    false_positives = []
    false_positives_count = 0
    false_negatives = []
    false_negatives_count = 0
    server_round_count = 0
    cluster_method = ""
    with open(filename, "r") as f:
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
            elif("poison" in line and "aggregated:" not in line):
                poison_accuracies.append(line.rstrip("\n"))
            #retrieve the aggregated client training accuracy
            elif("aggregated" in line and "training" in line):
                aggregated_training_accuracies.append(line.rstrip("\n"))
            #retrieve the aggregated client poison accuracy
            elif("aggregated" in line and "poison" in line):
                aggregated_poison_accuracies.append(line.rstrip("\n"))
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
            if("fedemnist" in filename.name):
                if("selected clients" in line):
                    #selected_clients.append(line.rstrip('\n').split(": ")[1])
                    list_test = list(line.rstrip('\n')[:-1].split(": [")[1].split(", "))
                    int_list = []
                    #convert the list into a list of ints
                    for value in list_test:
                        try:
                            int_list.append(int(value))
                        #if the value is not an int, skip it
                        except:
                            pass
                    selected_clients.append(int_list)
            FPs_per_round[int(current_round)] = current_FPs
            FNs_per_round[int(current_round)] = current_FNs

    return FPs_per_round, FNs_per_round, accuracies, poison_accuracies, false_positives_count, false_negatives_count, server_round_count, aggregated_training_accuracies, aggregated_poison_accuracies

def retrieveAccuracy(table, accuracies, poison_accuracies, aggregated_training_accuracies, aggregated_poison_accuracies, FNs_Per_Round, args):
    '''This function pulls the accuracy and poison accuracy metric for each 
    round and then creates the table containing the accuracy information'''
    #Add the column names to the table
    if("UTD" in args.file):
        table.add_row(["Round", "Accuracy", "Poison Accuracy"])
    else:
        if(len(aggregated_training_accuracies) > 0):
            table.add_row(["Round", "Val Accuracy", "Poison Accuracy", "Train Accuracy", "Client Poison Acc"])
        else:
            table.add_row(["Round", "Accuracy", "Poison Accuracy"])
    all_df = pd.DataFrame()
    acc_df = pd.DataFrame()

    #This loop retrieves the base and poison accuracies to the nearest 2 decimal places and adds them to the accuracy table
    for i in range(len(accuracies)):
        if(len(aggregated_training_accuracies) > 0):
            accuracy = accuracies[i].split(": ")[1]
            poison_accuracy = poison_accuracies[i].split(": ")[1]
            training_accuracy = aggregated_training_accuracies[i].split(": ")[1]
            aggregated_poison_accuracy = aggregated_poison_accuracies[i].split(": ")[1]
        else:
            accuracy = accuracies[i].split(": ")[1]
            poison_accuracy = poison_accuracies[i].split(": ")[1]

        if("UTD" in args.file):
            table.add_row([i, '{:.2%}'.format(float(accuracy)), '{:.2%}'.format(float(poison_accuracy))])
        else:
            #added training accuracy recording, but not all my input files will have this new info in them
            if(len(aggregated_training_accuracies) > 0):
                training_accuracy = aggregated_training_accuracies[i].split(": ")[1]
                aggregated_poison_accuracy = aggregated_poison_accuracies[i].split(": ")[1]
                table.add_row([i, '{:.2%}'.format(float(accuracy)), '{:.2%}'.format(float(poison_accuracy)), '{:.2%}'.format(float(training_accuracy)), '{:.2%}'.format(float(aggregated_poison_accuracy))])
            #if the new info is not present, use the old format
            else:
                table.add_row([i, '{:.2%}'.format(float(accuracy)), '{:.2%}'.format(float(poison_accuracy))])

        #includes FNs in each round
        if("UTD" not in args.file):
            all_df2 = pd.DataFrame([[i, '{:.2%}'.format(float(accuracy)), '{:.2%}'.format(float(poison_accuracy)), FNs_Per_Round[i]]], columns=['Round', 'Accuracy', 'Poison Accuracy', 'FNs'])
            all_df = pd.concat([all_df, all_df2])

        #just includes accuracies and round number
        if(len(aggregated_training_accuracies) > 0):
            acc_df2 = pd.DataFrame([[i, '{:.2}'.format(float(accuracy)), '{:.2}'.format(float(poison_accuracy)), '{:.2}'.format(float(training_accuracy)), '{:.2}'.format(float(aggregated_poison_accuracy))]], columns=['Round', 'Accuracy', 'Poison_Accuracy', 'Training_Accuracy', 'Aggregated_Poison_Accuracy'])
            acc_df = pd.concat([acc_df, acc_df2])
        else:
            acc_df2 = pd.DataFrame([[i, '{:.2}'.format(float(accuracy)), '{:.2}'.format(float(poison_accuracy))]], columns=['Round', 'Accuracy', 'Poison_Accuracy'])
            acc_df = pd.concat([acc_df, acc_df2])


    return(table, all_df, acc_df)


def main():
    accuracyTable = Texttable()
    perRoundTable = Texttable()
    roundGroupTable = Texttable()

    #the path of the directory containing the files we want to analyize 
    p = Path('./directoryAnalysis/hybrid/all')
    for child in p.iterdir():
        if child.is_file():
            #save the path of the current file
            q = p / child.name

            #retrive the info from the current file
            FPs_per_round, FNs_per_round, accuracies, poison_accuracies, false_positives_count, false_negatives_count, server_round_count, aggregated_training_accuracies, aggregated_poison_accuracies = gatherInfo(q)
            accuracyTable, all_accuracy_df, just_accuracy_df, aggregated_training_accuracies, aggregated_poison_accuracies = retrieveAccuracy(accuracyTable, accuracies, poison_accuracies, FNs_per_round, child.name)

            #create the graph for the current file
            fig, ax = plt.subplots()
            #numbers are retrieved from the dataframe and converted to floats for graphing
            val_accuracy = np.asarray(just_accuracy_df.Accuracy.values, float)
            poison_accuracy = np.asarray(just_accuracy_df.Poison_Accuracy.values, float)
            
            round_number = just_accuracy_df.Round.values
            
            #plot the retrieved values
            ax.plot(round_number, val_accuracy, label="Accuracy")
            ax.plot(round_number, poison_accuracy, label="Poison Accuracy")
            ax.set_xlabel("Round")
            ax.set_ylabel("Accuracy")
            title_pieces = child.name.split('_')
            
            #create the title for the graph based on the filename
            title = title_pieces[0]
            for i in range(len(title_pieces)):
                if(title_pieces[i] == 'poison'):
                    if('UTD' in title):
                        title_pieces[i+1] = str(float(title_pieces[i+1]) * 100)
                    title += ' ' + title_pieces[i] + ' ' + title_pieces[i+1] + '%'
                elif('clients' in title_pieces[i]):
                    title += ' ' + title_pieces[i-1] + ' clients'

            ax.set_title(title)
            #the legend is set manually because the labels aren't working for some reason
            L = ax.legend()
            L.get_texts()[0].set_text('Accuracy')
            L.get_texts()[1].set_text('Poison Accuracy')


if __name__ == "__main__":
    main()