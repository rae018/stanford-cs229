import numpy as np

def pos(theta,lengths):
    l1,l2,l3 = lengths
    th1,th2,th3 = theta
    x = -l1*np.sin(th1)+l2*np.cos(th2+th1)+l3*np.cos(th3+th2+th1)
    y = l1*np.cos(th1)+l2*np.sin(th2+th1)+l3*np.sin(th3+th2+th1)
    return np.array([x,y])

def f(theta,lengths,goal):
    pos_cur = pos(theta,lengths)
    return pos_cur-goal

def main():
    # Hard-coded in values
    l1 = 103.4
    l2 = 97.75
    l3 = 39.35
    lengths = np.array([l1,l2,l3])

    th = np.zeros(3) # Start with zero initialization
    pos_cur = np.array([1000,1000]) # Initialize to absurdity

    print('Just a test')
    print('Position at zero',pos(th,lengths))
    pos_g = pos(th,lengths)+np.array([60,0]) # start with easy one
    print('goal_pos',pos_g)
    eps = 10**(-3)
    while np.linalg.norm(pos_cur-pos_g,2)>eps:
        th1,th2,th3 = th
        J_11 = -l1*np.cos(th1)-l2*np.sin(th2+th1)-l3*np.sin(th3+th2+th1)
        J_12 = -l2*np.sin(th2+th1)-l3*np.sin(th3+th2+th1)
        J_13 = -l3*np.sin(th3+th2+th1)
        J_21 = -l1*np.sin(th1)+l2*np.cos(th2+th1)+l3*np.cos(th3+th2+th1)
        J_22 = l2*np.cos(th2+th1)+l3*np.cos(th3+th2+th1)
        J_23 = l3*np.cos(th3+th2+th1)
        J = np.array([[J_11,J_12,J_13],[J_21,J_22,J_23]])
        #th = th-np.linalg.pinv(J)@f(th,lengths,pos_g)
        th = th-np.linalg.pinv(J)@f(th,lengths,pos_g)
        pos_cur = pos(th,lengths)
        print(pos_cur)
    print('Hey gurl :3')
    print('norm: ',np.linalg.norm(pos_cur-pos_g,2))
    print('end position: ',pos_cur)
    print('goal position: ',pos_g)
    print('end theta',th)

if __name__ == "__main__":
    main()
