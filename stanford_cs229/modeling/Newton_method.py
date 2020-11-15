import numpy as np

def pos(theta):
    l1 = 103.4
    l2 = 96.75
    l3 = 39.35
    lengths = np.array([l1, l2, l3])
    th1,th2,th3 = theta
    x = -l1*np.sin(th1)+l2*np.cos(th2+th1)+l3*np.cos(th3+th2+th1)
    y = l1*np.cos(th1)+l2*np.sin(th2+th1)+l3*np.sin(th3+th2+th1)
    x1 = np.array([0,-l1*np.sin(th1)])
    x2 = np.array([-l1*np.sin(th1),-l1*np.sin(th1)+l2*np.cos(th2+th1)])
    x3 = np.array([-l1*np.sin(th1)+l2*np.cos(th2+th1),-l1*np.sin(th1)+l2*np.cos(th2+th1)+l3*np.cos(th3+th2+th1)])
    x_vals = [x1,x2,x3]
    x1 = np.array([0, -l1 * np.sin(th1)])
    x2 = np.array([-l1 * np.sin(th1), -l1 * np.sin(th1) + l2 * np.cos(th2 + th1)])
    x3 = np.array([-l1 * np.sin(th1) + l2 * np.cos(th2 + th1),
                   -l1 * np.sin(th1) + l2 * np.cos(th2 + th1) + l3 * np.cos(th3 + th2 + th1)])
    y1 = np.array([0, l1*np.cos(th1)])
    y2 = np.array([l1*np.cos(th1), l1*np.cos(th1)+l2*np.sin(th2+th1)])
    y3 = np.array([l1*np.cos(th1)+l2*np.sin(th2+th1),
                   l1*np.cos(th1)+l2*np.sin(th2+th1)+l3*np.sin(th3+th2+th1)])
    y_vals = [y1,y2,y3]
    return x_vals, y_vals, np.array([x,y])

def f(theta,goal):
    _,_,pos_cur = pos(theta)
    return pos_cur-goal

def pos_to_angles(pos_g,th_init):
    # Hard-coded in values
    l1 = 103.4
    l2 = 96.75
    l3 = 39.35
    lengths = np.array([l1,l2,l3])
    buffer = 10

    # Below is just to make sure we never go out of bounds of workspace
    if np.linalg.norm(pos_g)>sum(lengths)-buffer:
        pos_g = (pos_g/np.linalg.norm(pos_g))*(sum(lengths)-buffer)

    th = np.copy(th_init) # Start with zero initialization
    #pos_cur = np.array([1000,1000]) # Initialize to absurdity
    _,_,pos_cur = pos(th) # Initialize using the previous pos

    #print('Just a test')
    #print('Position at zero',pos(th,lengths))
    #print('goal_pos',pos_g)
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
        th = th-np.linalg.pinv(J)@f(th,pos_g)
        _,_,pos_cur = pos(th)
    return pos_cur,th

#if __name__ == "__main__":
#    end_pos,th = pos_to_angles(pos_g)
