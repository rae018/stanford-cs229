import serial

samples = 2000 #number of samples to collect on this run

arduino_port = 'COM5' #Arduino serial port (double check)
baud = 9600 #same baud rate specified in Arduino code
fileName = "circles_overhand_weight-60.csv"

ser = serial.Serial(arduino_port,baud)
print("Conectado al puerto: ", arduino_port)
file = open(fileName, "a") #"a" appends to an existing file
print("Archivo creado!")

line = 0 #tracks the number of sample we are reading (line 0 is column headers)
while line <= samples:

    # Display data in terminal
    if line ==0:
        print("Printing column headers...")
    else:
        print("Transcribing to line ",str(line),"...")
    getData = str(ser.readline())
    data = getData[2:][:-5] #double check this
    print(data)

    # Add data to file
    file = open(fileName, "a") #append data to file
    file.write(data + "\n") #write data with a new line
    line += 1 #line advances by one

# Close the file
file.close()
