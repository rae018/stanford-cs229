import numpy as np
import gym

import time
import matplotlib.pyplot as plt
from Newton_method import *
"""
def main(ylabels):
    env = gym.make('RobotArm-v0')

    env.reset()
    env.render()

    copy_amount = 1000

    ylabels = np.tile(ylabels,copy_amount)
    for i in range(len(ylabels)): # was 10000 before
        theta = env.data.qpos
        print(theta)
        cur_pos = pos(theta[1:])
        print('from mujoco')
        print(env.data.body_xpos)
        print('calculated')
        test_pos = np.copy(cur_pos)

        test_pos += np.array([-29.2*np.cos(sum(theta[1:]))+16.2*np.sin(sum(theta[1:])),-29.2*np.sin(sum(theta[1:]))-16.2*np.cos(sum(theta[1:]))])
        test_pos += np.array([0, 101.15 - 76])
        # Test this out. Not sure if correct rotation matrix
        R=np.array([[np.cos(theta[0]),-np.sin(theta[0]),0],[np.sin(theta[0]),np.cos(theta[0]),0],[0.,0.,1.]])
        #test_pos = R@np.hstack([3.,test_pos]) # 3 added because the center is slightly off
        test_pos = R @ np.hstack([3, test_pos])  # 3 added because the center is slightly off

        test_pos = test_pos[1:]
        #test_pos += np.array([0, 76])
        test_pos += np.array([0, 76])
          # Add height of hinge
        print(test_pos)

        env.render()
        # Each specific angle should be h
        # with the time-frame

        if i%copy_amount==0: # Because we tiled 20 times
            dx = 0.1
            dy = 0.1
        else:
            dx = 0
            dy = 0

        # 0 for up
        # 1 for down
        # 2 for forward
        # 3 for back
        new_pos = np.copy(cur_pos)
        if ylabels[i]==0:
            new_pos += np.array([0,dy])
        elif ylabels[i]==1:
            new_pos += np.array([0, -dy])
        elif ylabels[i]==2:
            new_pos += np.array([dx, 0])
        elif ylabels[i]==3:
            new_pos += np.array([-dx, 0])
        print('the old position')
        print(cur_pos)
        print('the new position')
        print(new_pos)

        if i%copy_amount==0: # We try to match the desired thetas repeatedly
            calc_pos, new_theta = pos_to_angles(new_pos, theta[1:])
        print('found thetas:')
        print(new_theta)
        print('pos_after')
        print(calc_pos)
        print('why u do dis?')
        print(pos(new_theta))
        print()
        #if i>2500:
            #env.step(np.hstack([0,new_theta])) # We keep the base angle as zero
        #else:
            #env.step(np.zeros(4))
        env.step(np.hstack([0, new_theta]))
"""
def main(ylabels):
    theta = np.array([0,0,0]) # Initialize angles

    # How much we want to move after each label
    dx = 3
    dy = 3

    for i in range(len(ylabels)):
        _,_,cur_pos = pos(theta)
        new_pos = np.copy(cur_pos)
        if ylabels[i] == 0:
            new_pos += np.array([0, dy])
        elif ylabels[i] == 1:
            new_pos += np.array([0, -dy])
        elif ylabels[i] == 2:
            new_pos += np.array([dx, 0])
        elif ylabels[i] == 3:
            new_pos += np.array([-dx, 0])
        calc_pos, new_theta = pos_to_angles(new_pos, theta)

        x_vals, y_vals, end_pos = pos(new_theta)
        x1,x2,x3 = x_vals
        y1,y2,y3 = y_vals

        joint_size = 80

        plt.plot(x1, y1, 'r') # plot first link
        plt.plot(x2, y2, 'g') # plot second link
        plt.plot(x3, y3, 'b') # plot third link
        plt.axis('equal')
        plt.scatter([x1[0],x2[0],x3[0]],[y1[0],y2[0],y3[0]],s=joint_size) # Plot the joints
        #bottom, top = ylim()  # return the current ylim
        plt.ylim(-240, 240)
        plt.xlim(-240, 240)
        #plt.gca().set_ylim(-240, 240)
        plt.pause(0.02)
        plt.clf()

        theta = np.copy(new_theta)

if __name__ == "__main__":
    labels = np.ones(50)
    labels = np.hstack([labels,4*np.ones(20)])
    labels = np.hstack([labels,2*np.ones(60)])
    labels = np.hstack([labels,3*np.ones(80)])
    # Import csv and ylabels here!!
    main(labels)
    #print("Starting main loop\n")
    #while True:
    #    time.sleep(500)
