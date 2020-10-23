import matplotlib.pyplot as plt
import csv
import numpy as np

x_acc = []
y_acc = []
z_acc = []

# This needs to be hard-coded in
freq = 10
delta_t = (1./60)

sens1 = []
sens2 = []
sens3 = []

tagX = []
tagY = []
tagZ = []

y_down = -4 #threshold for categorizing motion in meters per second squared
y_up = 4

with open('circles_overhand_weight-60.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    print(plots)
    for idx, row in enumerate(plots):
        if idx != 0:
            x_acc.append((-346.3 + int(row[0]))*0.1536003738) #converted to meters per second squared
            y_acc.append((-333.5 + int(row[1]))*0.1417651282)
            z_acc.append((-349.6 + int(row[2]))*0.1442854621)

            # x_acc.append(int(row[0])) #original ADC outputs
            # y_acc.append(int(row[1]))
            # z_acc.append(int(row[2]))

            sens1.append(int(row[3]))
            sens2.append(int(row[4]))
            sens3.append(int(row[5]))

for i in range(len(x_acc)):
    if y_acc[i] < y_down:
        tagY.append('front')
    elif y_acc[i] > y_up:
        tagY.append('back')
    else:
        tagY.append('neutral')

print(tagY)

# u = 0
# u_begin = np.array([])
# u_end = np.array([])
# l = 0
# l_begin = np.array([])
# l_end =  np.array([])
# for i in range(len(z_acc)-1):
#     if z_acc[i] <= 450 and z_acc[i+1] > 450:
#         u += 1
#         u_begin = np.hstack([u_begin,i])
#
#     elif z_acc[i] > 450 and z_acc[i+1] <= 450:
#         if bool(len(u_begin)):
#             u_end = np.hstack([u_end,i+1])
#
#     elif z_acc[i] <= 400 and z_acc[i+1] > 400:
#         if bool(len(u_end)):
#             l_end = np.hstack([l_end,i+1])
#
#     elif z_acc[i] > 400 and z_acc[i+1] <= 400:
#         l += 1
#         l_begin = np.hstack([l_begin,i])
#
# print('u = ',u)
# print('l = ',l)
# #print(l_begin)
# #print(l_end)
#
# u_begin = u_begin[:len(u_end)]
# u_end = u_end[:len(u_begin)]
# l_begin = l_begin[:len(l_end)]
# l_end = l_end[:len(l_begin)]
#
# print('u average: ',sum(u_end - u_begin)/u)
# print('l average: ',sum(l_end - l_begin)/l)

N = len(x_acc) # Total data-points
t = np.arange(N)*delta_t
fig, axs = plt.subplots(2)

# print('x average: ',sum(x_acc)/N)
# print('y average: ',sum(y_acc)/N)
# print('z average: ',sum(z_acc)/N)

axs[0].plot(t, x_acc, label='x acc.')
axs[0].plot(t, y_acc, label='y acc.')
axs[0].plot(t, z_acc, label='z acc.')
axs[0].set_xlabel(r't ($s$)')
axs[0].set_ylabel(r'acc. ($m/s^2$)')
axs[0].set_title('Acceleration')

axs[1].plot(t, sens1, label='sens. 1')
axs[1].plot(t, sens2, label='sens. 2')
axs[1].plot(t, sens3, label='sens. 3')
axs[1].set_xlabel(r't ($s$)')
axs[1].set_ylabel(r'sens ($V$)')
axs[1].set_title('Muscle Sensor Data')
plt.tight_layout()
plt.show()
