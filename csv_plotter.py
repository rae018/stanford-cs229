import matplotlib.pyplot as plt
import csv
import numpy as np

x_acc = []
y_acc = []
z_acc = []

# This needs to be hard-coded in
freq = 10
delta_t = (1./10)

sens1 = []
#sens2 = []
#sens3 = []

with open('static-zup.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    print(plots)
    for idx, row in enumerate(plots):
        if idx != 0:
            x_acc.append((-338 + int(row[0]))*0.157667239)
            y_acc.append((-338 + int(row[1]))*0.1566367342)
            z_acc.append((-338 + int(row[2]))*0.1597694689)
            sens1.append(int(row[3]))
            #sens2.append(int(row[4]))
            #sens3.append(int(row[5]))

N = len(x_acc) # Total data-points
t = np.arange(N)
fig, axs = plt.subplots(2)

axs[0].plot(t, x_acc, label='x acc.')
axs[0].plot(t, y_acc, label='y acc.')
axs[0].plot(t, z_acc, label='z acc.')
axs[0].set_xlabel(r't ($s$)')
axs[0].set_ylabel(r'acc. ($m/s^2$)')
axs[0].set_title('Acceleration')

axs[1].plot(t, sens1, label='sens. 1')
#axs[1].plot(t, sens2, label='sens. 2')
#axs[1].plot(t, sens3, label='sens. 3')
axs[1].set_xlabel(r't ($s$)')
axs[1].set_ylabel(r'sens ($V$)')
axs[1].set_title('Muscle Sensor Data')
plt.tight_layout()
plt.show()
