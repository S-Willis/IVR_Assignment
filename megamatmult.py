def multiply(matrix1,matrix2):
    return_matrix = [['' for x in range(len(matrix1))] for y in range(len(matrix1))]

    return_matrix[0][0] = matrix1[0][0] + '*' + matrix2[0][0] + " + " + matrix1[0][1] + '*' + matrix2[1][0]

    for row in range(len(return_matrix)):

        for col in range(len(return_matrix)):

            # for i in range(len(return_matrix)):
            #     if matrix1[row][i] == '0' or matrix2[i][col]=='0':
            #         return_matrix[row][col] = return_matrix[row][col] + '0 + '
            #     else:
            #         return_matrix[row][col] = return_matrix[row][col] + matrix1[row][i] + "*" + matrix2[i][col] + " + "

            return_matrix[row][col] = matrix1[row][0]+ "*" + matrix2[0][col] + ' + ' + matrix1[row][1] + "*" + matrix2[1][col] + ' + ' + matrix1[row][2] + "*" + matrix2[2][col] + ' + ' + matrix1[row][3] + "*" + matrix2[3][col]


    # print(len(return_matrix))
    # print(len(return_matrix[0]))
    return return_matrix


def main():
    # mat1 = [['a','b','c','d'],
    #         ['e','f','g','h'],
    #         ['0','j','k','l'],
    #         ['0','0','0','1']]
    #
    # mat2 = [['i','m','n','o'],
    #         ['p','q','r','s'],
    #         ['0','u','v','w'],
    #         ['0','0','0','1']]


    mat1 = [['cos(theta1)','(-sin(theta1)*cos(alpha1))','sin(theta1)*sin(alpha1)','A1*cos(theta1)'],
            ['sin(theta1)','cos(theta1)*cos(alpha1)','(-cos(theta1)*sin(alpha1))','A1*sin(theta1)'],
            ['0','sin(alpha1)','cos(alpha1)','(d1)'],
            ['0','0','0','1']]

    mat2 = [['cos(theta2)','(-sin(theta2)*cos(alpha2))','sin(theta2)*sin(alpha2)','A2*cos(theta2)'],
            ['sin(theta2)','cos(theta2)*cos(alpha2)','(-cos(theta2)*sin(alpha2))','A2*sin(theta2)'],
            ['0','sin(alpha2)','cos(alpha2)','(d2)'],
            ['0','0','0','1']]

    mat3 = [['cos(theta3)','(-sin(theta3)*cos(alpha3))','sin(theta3)*sin(alpha3)','A3*cos(theta3)'],
            ['sin(theta3)','cos(theta3)*cos(alpha3)','(-cos(theta3)*sin(alpha3))','A3*sin(theta3)'],
            ['0','sin(alpha3)','cos(alpha3)','(d3)'],
            ['0','0','0','1']]
    mat4 = [['cos(theta4)','(-sin(theta4)*cos(alpha4))','sin(theta4)*sin(alpha4)','A4*cos(theta4)'],
            ['sin(theta4)','cos(theta4)*cos(alpha4)','(-cos(theta4)*sin(alpha4))','A4*sin(theta4)'],
            ['0','sin(alpha4)','cos(alpha4)','(d4)'],
            ['0','0','0','1']]


    mat12 = multiply(mat1,mat2)
    mat13 = multiply(mat12,mat3)
    mat14 = multiply(mat13,mat4)

    # print(mat14)
    print(mat14[1][3])



if __name__ == '__main__':
    main()
