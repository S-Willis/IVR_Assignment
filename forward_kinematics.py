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

    # link = [d , theta, r , alpha]

    link1 = [2.5, -math.pi/2 + angle1, 0, -math.pi/2]
    link2 = [0, -math.pi/2 + angle2, 0, math.pi / 2]
    link3 = [0, angle3, 3.5, -math.pi / 2]
    link4 = [0, angle4, 3, 0]

    link_list = [link1, link2, link3, link4];

    mat_list = [None for x in range(4)]

    for i in range(4):
        mat_list[i] = get_a(link_list[i])
        # print(mat_list[i])
    # print(mat_list[0])
    # print()
    # print(mat_list[2])
    # print()


    #L1_rotation = mat_list[0][np.ix_([0, 1, 2],[0, 1, 2])]
    #print(np.matmul(np.array([1, 0, 0]), L1_rotation))
    #print("-------------------------------")
    # L2_rotation = mat_list[1][np.ix_([0, 1, 2],[0, 1, 2])]
    # print(np.matmul(np.array([0, 0, -1]), L2_rotation))
    # print("-------------------------------")
    # L3_rotation = mat_list[2][np.ix_([0, 1, 2],[0, 1, 2])]
    # print(np.matmul(np.array([0, 0, 1]), L3_rotation))
    # print("-------------------------------")
    # L4_rotation = mat_list[3][np.ix_([0, 1, 2],[0, 1, 2])]
    # print(np.matmul(np.array([0, 0, 1]), L4_rotation))
    # print("-------------------------------")


    A01 = mat_list[0]
    # A01_rotation = A01[np.ix_([0, 1, 2],[0, 1, 2])]
    # print(np.matmul(np.array([1, 0, 0]), A01_rotation))
    # print("-------------------------------")

    A02 = np.matmul(mat_list[0],mat_list[1])
    # A02_rotation = A02[np.ix_([0, 1, 2],[0, 1, 2])]
    # print(np.matmul(np.array([0, 1, 0]), A02_rotation))
    # print("-------------------------------")

    A03 = np.matmul(A02,mat_list[2])
    # A03_rotation = A03[np.ix_([0, 1, 2],[0, 1, 2])]
    # print(np.matmul(np.array([0, 1, 0]), A03_rotation))
    # print("-------------------------------")

    A04 = np.matmul(A03,mat_list[3])
    # A04_rotation = A04[np.ix_([0, 1, 2],[0, 1, 2])]
    # print(np.matmul(np.array([0, 1, 0]), A04_rotation))
    # print("-------------------------------")
    return(get_xyz(A04))
    #return get_xyz(mat_list[3].dot(mat_list[2]).dot(mat_list[1]).dot(mat_list[0]))

def main():

    ultimate_list = [[0,1.0,0.5,-1.7],
                     [0,-1,1,1],
                     [0,-1.7,-1.7,-1.7],
                     [0,1,1,1],
                     [0,0.5,0.5,0.5],
                     [0,0.8,-0.8,1.7],
                     [0,1,1,0.65],
                     [0,0.6,-1.2,0.7],
                     [0,-0.5,1,-0.4],
                     [0,0.9,0.8,0.7]]

    print(forward_kinematics(0,0, 0, 0))

    for list in ultimate_list:
        print(str(list) + " :\n " + str(forward_kinematics(list[0],list[1],list[2],list[3])))

if __name__ == '__main__':
    main()
