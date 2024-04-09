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
            elif("poison" not in line and "aggregated" not in line and "accuracy:" in line):
                accuracies.append(line.rstrip('\n'))
            #retrieve the poison accuracy
            elif("poison" in line and "aggregated:" not in line):
                poison_accuracies.append(line.rstrip("\n"))
            #retrieve the aggregated client training accuracy
            elif("aggregated" in line and "training" in line):
                aggregated_training_accuracies.append(line.rstrip("\n"))
            #retrieve the aggregated client poison accuracy
            elif("aggregated poison" in line):
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

def retrieveAccuracy(table, accuracies, poison_accuracies, aggregated_training_accuracies, aggregated_poison_accuracies, FNs_Per_Round, filename):
    '''This function pulls the accuracy and poison accuracy metric for each 
    round and then creates the table containing the accuracy information'''
    #Add the column names to the table
    if("UTD" in filename and "flower" not in filename):
        table.add_row(["Round", "Accuracy", "Poison Accuracy", "Train Accuracy"])
    else:
        if(len(aggregated_training_accuracies) > 0):
            table.add_row(["Round", "Val Accuracy", "Poison Accuracy", "Train Accuracy"])
        else:
            table.add_row(["Round", "Accuracy", "Poison Accuracy"])
    all_df = pd.DataFrame()
    acc_df = pd.DataFrame()

    #if it is a UTD test, add a row for round 0
    if("UTD" in filename and "flower" not in filename):
            acc_df2 = pd.DataFrame([[0, 0, 0]], columns=['Round', 'Accuracy', 'Poison_Accuracy'])
            acc_df = pd.concat([acc_df, acc_df2])
    
    #This loop retrieves the base and poison accuracies to the nearest 2 decimal places and adds them to the accuracy table
    for i in range(len(accuracies)):
        accuracy = accuracies[i].split(": ")[1]
        poison_accuracy = poison_accuracies[i].split(": ")[1]
        training_accuracy = ''

        if("UTD" in filename and "flower" not in filename):
            table.add_row([i, '{:.2%}'.format(float(accuracy)), '{:.2%}'.format(float(poison_accuracy)), 'N/A'])
        else:
            #added training accuracy recording, but not all my input files will have this new info in them
            if(len(aggregated_training_accuracies) > 0):
                #Round 0 is used by the server to get the initial model metrics, it is not applicaple to the training accuracy reported by the clients
                if(i == 0):
                    training_accuracy = 'N/A'
                    aggregated_poison_accuracy = 'N/A'
                    table.add_row([i, '{:.2%}'.format(float(accuracy)), '{:.2%}'.format(float(poison_accuracy)), 'N/A'])

                else:
                    training_accuracy = aggregated_training_accuracies[i-1].split(": ")[1]
                    #aggregated_poison_accuracy = aggregated_poison_accuracies[i-1].split(": ")[1]
                    table.add_row([i, '{:.2%}'.format(float(accuracy)), '{:.2%}'.format(float(poison_accuracy)), '{:.2%}'.format(float(training_accuracy))])
            #if the new info is not present, use the old format
            else:
                table.add_row([i, '{:.2%}'.format(float(accuracy)), '{:.2%}'.format(float(poison_accuracy))])

        #includes FNs in each round
        if("UTD" not in filename and "flower" not in filename):
            all_df2 = pd.DataFrame([[i, '{:.2%}'.format(float(accuracy)), '{:.2%}'.format(float(poison_accuracy)), FNs_Per_Round[i]]], columns=['Round', 'Accuracy', 'Poison Accuracy', 'FNs'])
            all_df = pd.concat([all_df, all_df2])

        #just includes accuracies and round number
        if(len(aggregated_training_accuracies) > 0):
            if(i == 0):
                acc_df2 = pd.DataFrame([[i, '{:.2}'.format(float(accuracy)), '{:.2}'.format(float(poison_accuracy)), 0]], columns=['Round', 'Accuracy', 'Poison_Accuracy', 'Train_Accuracy'])
                acc_df = pd.concat([acc_df, acc_df2])
            else:
                acc_df2 = pd.DataFrame([[i, '{:.2}'.format(float(accuracy)), '{:.2}'.format(float(poison_accuracy)), '{:.2}'.format(float(training_accuracy))]], columns=['Round', 'Accuracy', 'Poison_Accuracy', 'Train_Accuracy'])
                acc_df = pd.concat([acc_df, acc_df2])
        else:
            acc_df2 = pd.DataFrame([[i, '{:.2}'.format(float(accuracy)), '{:.2}'.format(float(poison_accuracy))]], columns=['Round', 'Accuracy', 'Poison_Accuracy'])
            acc_df = pd.concat([acc_df, acc_df2])

    return(table, all_df, acc_df)

def concatTestResults(results, multi_df, title):
    '''This function adds a new column to the provided dataframe'''
    if(title in multi_df.columns):
        title = "new title"
    multi_df[title] = results
    return(multi_df)

def plotMultiGraph(multi_df, ax):
    '''This function takes a dataframe containing the results from multiple tests and places them all in the same graph.
        We assume there is a column containing the round numbers called 'Round'. 
    '''
    #fig, ax = plt.subplots()
    rounds = multi_df.Round.values
    for col in multi_df.columns:
        if(col != "Round"):
            if(col != "Average"):
                ax.set_title(col.split(" ")[0])
            accuracy = multi_df[col].tolist()
            ax.plot(rounds, accuracy, label=col)
            ax.set_xlabel("Round")
            ax.set_ylabel("Accuracy")
            #Set the accuracy axis to go to 100%
            ax.set_ylim([0, 1]) 
            L = ax.legend()
            ax.set_title("UTD vs V3 Hybrid")
            #L.get_texts()[0].set_text('Val Accuracy')

def plotAverages(multi_df, ax=None):
    '''This function works like the plotMultiGraph() funciton, but only plots the average values'''
    if(ax == None):
        fig, ax = plt.subplots()
    ax.set_title("Average Accuracies")
    rounds = multi_df.Round.values
    for col in multi_df.columns:
        if("Average" in col):
            accuracy = multi_df[col].tolist()
            ax.plot(rounds, accuracy, label=col)
            ax.set_xlabel("Round")
            ax.set_ylabel("Accuracy")
            #Set the accuracy axis to go to 100%
            ax.set_ylim([0, 1]) 
            L = ax.legend()

def evaluateMetrics(df):
    '''This function takes the results dataframe and tests to see when the first round 90% accuracy was reached'''
    for col in df.columns:
        if(col != "Round"):
            #Print the final poison and val accuracies
            if('Poison' not in col):
                print(col.split(" ")[0] + " final accuracy: {:.2%}".format(float(df[col].iloc[-1])))
            else:
                print(col.split(" ")[0] + " final poison accuracy: {:.2%}\n".format(float(df[col].iloc[-1])))
            #find the first round where training reaches 90%
            for value in df[col]:
                if(value >= .9):
                    idx = df.index[df[col] == value]
                    acc_round = df.loc[idx]['Round'].values[0]
                    print(col.split(" ")[0] + " first round of 90% accuracy: {}".format(str(acc_round)))
                    break


def main():
    accuracyTable = Texttable()

    perRoundTable = Texttable()
    roundGroupTable = Texttable()

    #using this to implement graphing multiple tests in the same graph
    multi_test_accuracies = pd.DataFrame()
    multi_test_accuracies["Round"] = list(range(0, 101))
    multi_test_poisons = pd.DataFrame()
    multi_test_poisons["Round"] = list(range(0, 101))

    avg_values = pd.DataFrame()
    avg_values['Round'] = list(range(0, 101))

    #used to toggle averaging functionality 
    AVG = True
    AVG_counter = 1
    available_paths = {'UTD': './directoryAnalysis/bestMethod/UTD', 'lofHybrid': './directoryAnalysis/bestMethod/lofHybrid', 'hybrid': './directoryAnalysis/bestMethod/hybrid', 'UTD_flower': './directoryAnalysis/bestMethod/UTD_flower'}
    #the path of the directory containing the files we want to analyize
    #paths = [available_paths['UTD'], available_paths['lofHybrid'], available_paths["hybrid"], available_paths["UTD_flower"]] 
    paths = [available_paths["UTD"], available_paths["hybrid"]]
    #p = Path('./directoryAnalysis/bestMethod/lofHybrid')
    for path in paths:
        p = Path(path)
        AVG_counter = 1
        for child in p.iterdir():
            if child.is_file():
                #save the path of the current file
                q = p / child.name

                #retrive the info from the current file
                FPs_per_round, FNs_per_round, accuracies, poison_accuracies, false_positives_count, false_negatives_count, server_round_count, aggregated_training_accuracies, aggregated_poison_accuracies = gatherInfo(q)
                accuracyTable, all_accuracy_df, just_accuracy_df = retrieveAccuracy(accuracyTable, accuracies, poison_accuracies, aggregated_training_accuracies, aggregated_poison_accuracies, FNs_per_round, child.name)

                #create the graph for the current file
                if(not AVG):
                    fig, ax = plt.subplots()
                #numbers are retrieved from the dataframe and converted to floats for graphing
                val_accuracy = np.asarray(just_accuracy_df.Accuracy.values, float)
                poison_accuracy = np.asarray(just_accuracy_df.Poison_Accuracy.values, float)
                if(len(aggregated_training_accuracies) > 0):
                    train_accuracy = np.asanyarray(just_accuracy_df.Train_Accuracy.values, float)
                
                round_number = just_accuracy_df.Round.values
                
                #plot the retrieved values
                if(not AVG):
                    ax.plot(round_number, val_accuracy, label="Val Accuracy")
                    ax.plot(round_number, poison_accuracy, label="Poison Accuracy")
                    if(len(aggregated_training_accuracies) > 0):
                        ax.plot(round_number, train_accuracy, label="Train Accuracy")
                    ax.set_xlabel("Round")
                    ax.set_ylabel("Accuracy")
                title_pieces = child.name.split('_')
                
                #create the title for the graph based on the filename
                if AVG:
                    if(title_pieces[0] == "UTD"):
                        if(title_pieces[1] == "flower"):
                            title = "{}_{} Test {}".format(title_pieces[0], title_pieces[1], str(AVG_counter))
                        else:
                            title = "{} Test {}".format(title_pieces[0], str(AVG_counter))
                    else:
                        title = "{} Test {}".format(title_pieces[0], str(AVG_counter))
                    AVG_counter += 1
                else:
                    title = title_pieces[0]
                    for i in range(len(title_pieces)):
                        if(title_pieces[i] == 'poison'):
                            if('UTD' in title):
                                title_pieces[i+1] = str(float(title_pieces[i+1]) * 100)
                            title += ' ' + title_pieces[i] + ' ' + title_pieces[i+1] + '%'
                        elif('clients' in title_pieces[i]):
                            title += ' ' + title_pieces[i-1] + ' clients'

                if(not AVG):
                    ax.set_title(title)
                    #the legend is set manually because the labels aren't working for some reason
                    L = ax.legend()
                    L.get_texts()[0].set_text('Val Accuracy')
                    L.get_texts()[1].set_text('Poison Accuracy')
                    if(len(aggregated_training_accuracies) > 0):
                        L.get_texts()[2].set_text('Train Accuracy')

                #call a function to add the accuracy from the current test to the overall dataframe
                multi_test_accuracies = concatTestResults(val_accuracy, multi_test_accuracies, title + " accuracy")
                multi_test_poisons = concatTestResults(poison_accuracy, multi_test_poisons, title + " poison")
            
        #print(multi_test_accuracies)
        #print("BEFORE")
        #print(multi_test_accuracies.columns)
        avg_values[title.split(" ")[0] + ' Average Accuracy'] = multi_test_accuracies.loc[:, multi_test_accuracies.columns != "Round"].mean(axis=1)
        avg_values[title.split(" ")[0] + ' Average Poison'] = multi_test_poisons.loc[:, multi_test_poisons.columns != "Round"].mean(axis=1)
        multi_test_accuracies = pd.DataFrame(None)
        multi_test_poisons.iloc[0:0]
        #print("AFTER")
        #print(multi_test_accuracies.columns)

    #compute the averages of all the tests
    #exclude the column containing the round numbers in the average calculation
    multi_test_accuracies['Average Accuracy'] = multi_test_accuracies.loc[:, multi_test_accuracies.columns != "Round"].mean(axis=1)
    multi_test_poisons['Average Poison'] = multi_test_poisons.loc[:, multi_test_poisons.columns != "Round"].mean(axis=1)
    #print(avg_values)
    #print(avg_values[avg_values['hybrid Average Accuracy'] >= .9])

    evaluateMetrics(avg_values)    

    fig, ax = plt.subplots()
    ax.set_title("Average Results")
    #plotMultiGraph(multi_test_accuracies, ax)
    #plotMultiGraph(multi_test_poisons, ax)
    #plotAverages(multi_test_accuracies, ax)
    #plotAverages(multi_test_poisons, ax)
    #print(multi_test_poisons)
    #print(multi_test_accuracies)

    plotMultiGraph(avg_values, ax)



if __name__ == "__main__":
    main()