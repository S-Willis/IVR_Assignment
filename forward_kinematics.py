import math
import numpy as np


def get_a(link):
    d = link[0]
    theta = link[1]
    a = link[2]
    alpha = link[3]

    s_theta = math.sin(theta)
    c_theta = math.cos(theta)
    s_alpha = math.sin(alpha)
    c_alpha = math.cos(alpha)



    A = np.array([[c_theta,-s_theta*c_alpha,s_theta*s_alpha,a*c_theta],
                  [s_theta,c_theta*c_alpha,-c_theta*s_alpha,a*s_theta],
                  [0,s_alpha,c_alpha,d],
                  [0,0,0,1]])

    return A

def forward_kinematics(angle1, angle2, angle3, angle4):
    theta1 = angle1
    theta2 = angle2
    theta3 = angle3
    theta4 = angle4
    link1 = [2.5, theta1,0,0]
    link2 = [0,theta2,0,-math.pi/2]
    link3 = [3.5,theta3,0,math.pi/2]
    link4 = [3.0,theta4,0,-math.pi/2]

    link_list = [link1,link2,link3,link4];

    mat_list = [None for x in range(4)]

    for i in range(4):
        mat_list[i] = get_a(link_list[i])

    A02 = np.matmul(mat_list[0],mat_list[1])
    A03 = np.matmul(A02,mat_list[2])
    A04 = np.matmul(A03,mat_list[3])

    return A04

def main():
    print(forward_kinematics(0,1,-1,0))

if __name__ == '__main__':
    main()
