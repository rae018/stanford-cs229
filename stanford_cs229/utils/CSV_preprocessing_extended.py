import csv
import numpy as np

samples = 100 #max number of entries to use in calculating a rolling average

fileName = "desk_overhand_weight_4-60Hz"

fileName_new = fileName + "_extended.csv"

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

            if idx <= 1: #rolling average values of each muscle signal
                for indice in range(len(muscles)):
                    fila += '0,0,' #initial derivatives are estimated as 0
                    prev_row.append(muscles[indice]) #store first values for next iteration
            else:
                for indice in range(len(muscles)): #rolling average values of each muscle signal
                    derivada = float(muscles[indice]) - float(prev_row[indice]) #rolling average for relevant entries
                    fila += str(derivada) + ","
                for indice in range(len(muscles)):
                    derivadabs = abs(float(muscles[indice]) - float(prev_row[indice]))
                    prev_row[indice] = muscles[indice] #overwrite for calculation in next row
                    fila += str(derivadabs) + ","

            if idx <= 1: #rolling average values of each muscle signal
                for indice in range(len(muscles)):
                    fila += muscles[indice] #initial values are entry/1
                    if indice < len(muscles)-1: #if this isn't the last entry in the row
                        fila += "," #separate from next entry in the row
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
                    fila += str(integrada)
                    if indice < len(muscles)-1:
                        fila += ","

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
                fila += header + " derivative absolute values,"

            ctr = 0
            for header in row[-4:]:
                ctr += 1
                fila += header + " rolling averages"
                if ctr < len(row[-4:]):
                    fila += ","
            file.write(fila + "\n") #write column headers with a new line
            prev_row = [] #initialize preceding row for derivative approximation

# Close the file
file.close()
