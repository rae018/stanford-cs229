import csv
import numpy as np

samples = 100 #max number of entries to use in calculating a rolling average

fileName = "ydeskx_overhand_weight_4-60Hz"

fileName_new = fileName + "_derivatives.csv"

fileName = fileName + ".csv"

file = open(fileName_new, "a") #"a" appends to an existing file
print("archivo modificado creado/localizado")

with open(fileName,mode='r') as csvfile: #reading from original file
    #window = np.empty(4) #for the rolling average of the four muscle signals
    plots = csv.reader(csvfile, delimiter=',')
    #print(plots)
    for idx, row in enumerate(plots):
        if idx != 0:
            print("Transcribing to line ",str(idx),"...")
            # Add data to file
            file = open(fileName_new, "a") #append data to file
            #print(row)

            fila = ''
            for datum in row:
                fila += datum + "," #transcription of original information

            muscles = row[-4:] #input data for manipulation
            for indice in range(len(muscles)): #approximation of muscle signal derivatives
                if idx <= 1:
                    fila += '0,'
                    prev_row.append(muscles[indice]) #store first values for next iteration
                else:
                    #print(prev_row)
                    derivada = float(muscles[indice]) - float(prev_row[indice]) #approximate derivative for indiceth signal
                    prev_row[indice] = muscles[indice] #store this value for next iteration
                    fila += str(derivada) + ","


            if idx <= 1: #rolling average values of each muscle signal
                for indice in range(len(muscles)):
                    fila += muscles[indice] + "," #initial values are entry/1
                #prev_row.append(muscles[indice]) #store first values for next iteration
                window = np.array([float(ele) for ele in muscles])
            else:
                if idx < samples: #we have less than the number of samples to use in a rolling average
                    #print(prev_row)
                    window = np.vstack((window,np.array([float(ele) for ele in muscles]))) #progressively build up window for moving average values
                else: #we have at least as many samples as typically used in a rolling average
                    window[:-1,:] = window[1:,:] #shift rows up by one (first row overwritten)
                    window[-1,:] = [float(ele) for ele in muscles] #replace last entries for next average calculation

                for indice in range(len(muscles)): #rolling average values of each muscle signal
                    integrada = sum(window[:,indice])/len(window[:,indice]) #rolling average for relevant entries
                    fila += str(integrada) + ","

            file.write(fila + "\n") #append new columns
        else: #if idx == 0
            print("Transcribing column headers...")
            file = open(fileName_new,"a") #append data to modified file
            #print(row)
            fila = ''
            for header in row:
                fila += header + ","
            for header in row[-4:]:
                fila += header + " derivatives,"
            for header in row[-4:]:
                fila += header + " rolling averages,"
            file.write(fila + "\n") #write column headers with a new line
            prev_row = [] #initialize preceding row for derivative approximation

# Close the file
file.close()
