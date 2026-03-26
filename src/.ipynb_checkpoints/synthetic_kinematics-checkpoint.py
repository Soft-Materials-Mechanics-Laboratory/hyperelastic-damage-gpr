import numpy as np

def right_cauchy_green(F):
    """
    Function to compute the modified right Cauchy-Green deformation tensor, given the deformation gradiant tensor is
    provided as a list. Each list element of the deformation gradient and its rate tensors are written in the
    [11, 12, 13, 21, 22, 23, 31, 32, 33] vector format.
    """
    C, C_bar = [], []
    
    for i in range(len(F)):
        F_temp = np.array([[F[i][0], F[i][1], F[i][2]],
                           [F[i][3], F[i][4], F[i][5]],
                           [F[i][6], F[i][7], F[i][8]]])
        C_temp = np.transpose(F_temp) @ F_temp
        J_temp = np.linalg.det(F_temp)
        F_bar_temp = (J_temp ** (-1/3)) * F_temp
        
        C_bar_temp = np.transpose(F_bar_temp) @ F_bar_temp
        
        C.append(C_temp)
        C_bar.append(C_bar_temp)
        
    return C, C_bar


def invariants(F):
    """
    Function to compute the modified strain invariants from the deformation gradient tensors provided as a list.
    Each list element of the deformation gradient tensor is written in the [11, 12, 13, 21, 22, 23, 31, 32, 33]
    vector format.
    """
    I1_bar, I2_bar, J = [], [], []
        
    for i in range(len(F)):
        F_temp = np.array([[F[i][0], F[i][1], F[i][2]],
                           [F[i][3], F[i][4], F[i][5]],
                           [F[i][6], F[i][7], F[i][8]]])
        J_temp = np.linalg.det(F_temp)
        F_bar_temp = (J_temp ** (-1/3)) * F_temp
        
        C_bar_temp = np.transpose(F_bar_temp) @ F_bar_temp
        C_bar_temp_2 = C_bar_temp @ C_bar_temp
        C_temp = np.transpose(F_temp) @ F_temp

        
        I1_bar.append(np.trace(C_bar_temp))
        I2_bar.append(0.5 * (np.trace(C_bar_temp)**2 - np.trace(C_bar_temp_2)))
        J.append(J_temp)
        
    return I1_bar, I2_bar, J


def ref_deviatoric(Z, C):
    """
    Computes the referential deviatoric component of a given matrix (rank-2 tensor), Z.
    """
    
    Dev_Z = Z - ((1/3) * np.tensordot(Z,C) * np.linalg.inv(C))
    
    return Dev_Z
