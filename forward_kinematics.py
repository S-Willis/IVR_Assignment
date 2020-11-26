import math
import numpy as np

def get_xyz(a_matrix):
    x = a_matrix[0,3]
    y = a_matrix[1,3]
    z = a_matrix[2,3]

    return [x,y,z]

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

    link1 = [2.5, angle1,0,0]
    link2 = [3.5,angle2,0,-math.pi/2]
    link3 = [3.5,angle3,0,math.pi/2]
    link4 = [0,angle4,3.0,math.pi/2]

    link_list = [link1,link2,link3,link4];

    mat_list = [None for x in range(4)]

    for i in range(4):
        mat_list[i] = get_a(link_list[i])
        # print(mat_list[i])
    print(mat_list[0])
    print()
    print(mat_list[2])
    print()

    A02 = np.matmul(mat_list[0],mat_list[1])
    print(A02)
    print()
    A03 = np.matmul(A02,mat_list[2])
    print(A03)
    print()
    A04 = np.matmul(A03,mat_list[3])

    return get_xyz(A04)

def main():
    print(forward_kinematics(0,0,0,0))

if __name__ == '__main__':
    main()
