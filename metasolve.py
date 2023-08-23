from sympy import symbols, Eq, nsolve, solve
from scipy.optimize import fsolve
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# functions to get lists of previously generated neighors in A2222/A22.22.22.23 schemes
def get_meta_metalist(ind):
    if ind == 2:
        return get_metalist2()
    if ind == 3:
        return get_metalist3()
    return None
def get_metalist():  # using my custom naming convention and second nearest neighbor with geomoetry
    j1 = [[3, 3, 3, 3, (3, 2), (3, 2), (3, 2), (3, 2)],
          [3, 3, 3, 3, (3, 2), (3, 2), (3, 2), (3, 4)]]
    j2 = [[5, 5, (5, 2), (5, 2), (5, 2), (5, 2)],
          [5, 5, (5, 2), (5, 2), (5, 2), (5, 4)],
          [5, 5, (5, 2), (5, 2), (5, 2), (5, 6)],
          [3, 5, (3, 4), (5, 2), (5, 2)],
          [3, 5, (3, 4), (5, 2), (5, 4)],
          [3, 5, (3, 4), (5, 2), (5, 6)],
          [3, 5, (3, 4), (5, 4), (5, 6)],
          [3, 5, (3, 4), (5, 4), (5, 4)],
          [3, 5, (3, 1), (5, 2), (5, 2)],
          [3, 5, (3, 1), (5, 2), (5, 4)],
          [3, 5, (3, 1), (5, 2), (5, 6)],
          [3, 3, (3, 1), (3, 1)],
          [3, 3, (3, 1), (3, 4)]]
    j3 = [[1, 4, (1, 3), (1, 3), (1, 3), (4, 3), (4, 3)],
          [1, 2, (1, 3), (1, 3), (1, 3), (2, 5)],
          [1, 2, (1, 3), (1, 3), (1, 3), (2, 3)],
          [2, 4, (2, 5), (4, 3), (4, 5)],
          [2, 4, (2, 5), (4, 3), (4, 3)],
          [2, 4, (2, 3), (4, 3), (4, 3)],
          [2, 4, (2, 3), (4, 3), (4, 5)],
          [4, 4, (4, 3), (4, 3), (4, 3), (4, 3)],
          [4, 4, (4, 3), (4, 3), (4, 3), (4, 5)]]
    j4 = [[3, 3, 3, (3, 1), (3, 2), (3, 2)],
          [3, 3, 3, (3, 2), (3, 2), (3, 4)],
          [3, 3, 3, (3, 2), (3, 4), (3, 4)],
          [3, 3, 5, (3, 2), (3, 2), (5, 2), (5, 2)],
          [3, 3, 5, (3, 2), (3, 4), (5, 2), (5, 6)],
          [3, 3, 5, (3, 2), (3, 2), (5, 2), (5, 4)],
          [3, 3, 5, (3, 2), (3, 4), (5, 2), (5, 4)]]

    j5 = [[2, 2, 2, (2, 5), (2, 5), (2, 5)],
          [2, 2, 2, (2, 3), (2, 3), (2, 5)],
          [2, 2, 2, (2, 3), (2, 5), (2, 5)],
          [2, 2, 4, (2, 3), (2, 5), (4, 3), (4, 3)],
          [2, 2, 6, (2, 3), (2, 5), (6, 5)],
          [2, 4, 4, (2, 3), (4, 3), (4, 3), (4, 3), (4, 3)],
          [2, 4, 6, (2, 3), (4, 3), (4, 3), (6, 5)]]
    j6 = [[5, 5, (5, 2), (5, 2), (5, 2), (5, 2)],
          [5, 5, (5, 2), (5, 2), (5, 2), (5, 4)]]

    return [j1, j2, j3, j4, j5, j6]
def get_metalist2():  # recursive connectivity based on neighbors but only nearest neighbor hopping
    j1 = [['B23', 'A223', 'A23'],
          ['B24', 'A2222', 'A23'],
          ['B24', 'A22', 'A2222'],
          ['B23', 'A222', 'A23'],
          ['B34', 'A222', 'A2222'],
          ['B33', 'A222', 'A222'],
          ['B23', 'A22', 'A222'],
          ['B33', 'A222', 'A223'],
          ['B23', 'A22', 'A223']]
    j2 = [['B222', 'A23', 'A33', 'A33'],
          ['B222', 'A33', 'A33', 'A33'],
          ['B223', 'A223', 'A23', 'A33'],
          ['B222', 'A23', 'A23', 'A33'],
          ['B233', 'A223', 'A223', 'A23']]
    j3 = [['A33', 'B222', 'B222'],
          ['A33', 'B222', 'B223'],
          ['A23', 'B223', 'B23'],
          ['A23', 'B222', 'B23'],
          ['A23', 'B222', 'B24'],
          ['A22', 'B24', 'B24'],
          ['A22', 'B23', 'B24'],
          ['A23', 'B223', 'B24'],
          ['A23', 'B23', 'B233']]
    j4 = [['A223', 'B223', 'B23', 'B23'],
          ['A222', 'B23', 'B23', 'B34'],
          ['A222', 'B23', 'B23', 'B33'],
          ['A222', 'B23', 'B33', 'B33'],
          ['A223', 'B223', 'B23', 'B33'],
          ['A223', 'B23', 'B23', 'B233'],
          ['A223', 'B23', 'B233', 'B33']]
    j5 = [['A2222', 'B24', 'B24', 'B24', 'B24'],
          ['A2222', 'B24', 'B24', 'B24', 'B34']]
    return [j1, j2, j3, j4, j5]
def get_metalist3():
    return [[['B_22_322', 'A_22_23_23', 'A_23_24'],
             ['B_22_323', 'A_22_23_322', 'A_23_24'],
             ['B_22_323', 'A_22_23_323', 'A_23_24'],
             ['B_22_4222', 'A_22_22_22_22', 'A_23_24'],
             ['B_22_4222', 'A_22_22_22_22', 'A_24_24'],
             ['B_22_4222', 'A_22_22_22_23', 'A_23_24'],
             ['B_22_4222', 'A_22_22_22_23', 'A_24_24'],
             ['B_23_322', 'A_22_22_23', 'A_23_322'],
             ['B_23_322', 'A_22_22_23', 'A_23_323'],
             ['B_23_322', 'A_22_22_24', 'A_23_322'],
             ['B_23_322', 'A_22_22_24', 'A_23_323'],
             ['B_23_322', 'A_22_22_24', 'A_23_333'],
             ['B_23_323', 'A_22_22_322', 'A_23_322'],
             ['B_23_323', 'A_22_22_322', 'A_23_323'],
             ['B_23_323', 'A_22_22_323', 'A_23_322'],
             ['B_23_323', 'A_22_22_323', 'A_23_323'],
             ['B_23_4222', 'A_22_22_22_22', 'A_24_322'],
             ['B_23_4222', 'A_22_22_22_22', 'A_24_323'],
             ['B_23_4222', 'A_22_22_22_23', 'A_24_322'],
             ['B_322_322', 'A_22_22_23', 'A_22_23_23'],
             ['B_322_323', 'A_22_23_23', 'A_22_23_322'],
             ['B_322_323', 'A_22_23_23', 'A_22_23_323'],
             ['B_322_4222', 'A_22_22_22_23', 'A_22_22_24']],
            [['B_22_22_23', 'A_23_322', 'A_24_322', 'A_322_322'],
             ['B_22_22_23', 'A_23_322', 'A_24_322', 'A_322_323'],
             ['B_22_23_23', 'A_23_322', 'A_322_322', 'A_322_322'],
             ['B_22_23_23', 'A_23_322', 'A_322_322', 'A_322_323'],
             ['B_22_23_23', 'A_24_322', 'A_322_322', 'A_322_322'],
             ['B_22_23_23', 'A_24_322', 'A_322_322', 'A_322_323'],
             ['B_22_23_322', 'A_22_22_322', 'A_23_323', 'A_322_323'],
             ['B_22_23_322', 'A_22_22_322', 'A_24_323', 'A_322_323'],
             ['B_22_23_322', 'A_22_23_322', 'A_23_323', 'A_322_323'],
             ['B_22_322_322', 'A_22_22_323', 'A_22_23_323', 'A_23_333'],
             ['B_23_23_23', 'A_322_322', 'A_322_322', 'A_322_322']],
            [['A_23_24', 'B_22_322', 'B_22_4222'],
             ['A_23_24', 'B_22_323', 'B_22_4222'],
             ['A_23_322', 'B_22_22_23', 'B_23_322'],
             ['A_23_322', 'B_22_22_23', 'B_23_323'],
             ['A_23_322', 'B_22_23_23', 'B_23_322'],
             ['A_23_323', 'B_22_23_322', 'B_23_322'],
             ['A_23_323', 'B_22_23_322', 'B_23_323'],
             ['A_23_333', 'B_22_322_322', 'B_23_322'],
             ['A_24_24', 'B_22_4222', 'B_22_4222'],
             ['A_24_322', 'B_22_22_23', 'B_23_4222'],
             ['A_24_322', 'B_22_23_23', 'B_23_4222'],
             ['A_24_323', 'B_22_23_322', 'B_23_4222'],
             ['A_322_322', 'B_22_22_23', 'B_23_23_23'],
             ['A_322_322', 'B_22_23_23', 'B_22_23_23'],
             ['A_322_322', 'B_22_23_23', 'B_23_23_23'],
             ['A_322_323', 'B_22_22_23', 'B_22_23_322'],
             ['A_322_323', 'B_22_23_23', 'B_22_23_322']],
            [['A_22_22_23', 'B_23_322', 'B_23_322', 'B_322_322'],
             ['A_22_22_24', 'B_23_322', 'B_23_322', 'B_322_4222'],
             ['A_22_22_322', 'B_22_23_322', 'B_23_323', 'B_23_323'],
             ['A_22_22_323', 'B_22_322_322', 'B_23_323', 'B_23_323'],
             ['A_22_23_23', 'B_22_322', 'B_322_322', 'B_322_323'],
             ['A_22_23_322', 'B_22_23_322', 'B_22_323', 'B_322_323'],
             ['A_22_23_323', 'B_22_322_322', 'B_22_323', 'B_322_323']],
            [['A_22_22_22_22', 'B_22_4222', 'B_22_4222', 'B_23_4222', 'B_23_4222'],
             ['A_22_22_22_23', 'B_22_4222', 'B_22_4222', 'B_23_4222', 'B_322_4222']]]
# critical functionalities used in other programs for evaluating arrangements of points
def angle_between_points(p1, p2):
    return np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
def calculate_angle(point1, point2, point3):
    angle1 = angle_between_points(point2, point1)
    angle2 = angle_between_points(point2, point3)
    angle = angle2 - angle1
    if angle < 0:
        angle += 2 * np.pi
    return np.degrees(angle)
def get_vtype(i, adjmatrix, unique):
    adjRow = adjmatrix[i]
    pos = unique[i]
    supertype = np.sum(adjmatrix[int(i)])
    ones_indices = np.where((adjRow == 1))[0]
    p1 = unique[ones_indices[0]]
    p2 = unique[ones_indices[1]]
    ang = calculate_angle(p1,pos,p2)
    if supertype == 4:
        return 1
    elif supertype == 3:
        if abs((ang / 120) - 1) < .01 or abs((ang/240)-1) < .01:
            return 5
        else:
            return 4
    elif supertype == 2:
        if abs((ang / 180) - 1) < .01:
            return 6
        elif abs((ang / 120) - 1) < .01 or abs((ang/240)-1) < .01:
            return 3
        else:
            return 2
    else:
        print("VTYPE ERROR:", pos, p1, p2, ang)
        return -1
def get_coordination(vnum): # returns 4 for 1, 3 for 4 and 5, 2 for 2,3,6
        if vnum == 1:
            return 4
        if vnum == 5 or vnum == 4:
            return 3
        if vnum == 2 or vnum == 3 or vnum == 6:
            return 2
def coord(ind):
    if ind == 2 or ind == 3 or ind == 6:
        return 2
    elif ind == 5 or ind == 4:
        return 3
    else:
        return 4
def alat(ind):
    if ind == 3 or ind == 5:
        return False
    return True
# various techniques for trying to get an analytical solution for the spectres at vertex level, none really worked
def failed_listconvert():
    metalist = get_metalist()
    coordlist = [[], [], [], [], [], []]
    for i, l in enumerate(metalist):
        char = "A" if alat(i + 1) else "B"  # +1 because old tile names go from 1-6 not 0-5 like i
        oppo = "B" if alat(i + 1) else "A"
        for eq in l:
            newlist = []
            setter = char
            for item in eq:
                if isinstance(item, int):
                    setter += str(coord(item))
                else:
                    inorder = sorted([coord(item[0]), coord(item[1])])
                    newlist.append(oppo + str(inorder[0]) + str(inorder[1]))
            newlist.insert(0, setter)
            coordlist[i].append(newlist)
    for i in range(len(coordlist)):
        print(i + 1)
        for subl in coordlist[i]:
            print(subl)
def failed_secondNN():
    namelist = 'E a1 a2 a3 a4 a5 a6 d13 d23 d25 d34 d45 d56'
    E, a1, a2, a3, a4, a5, a6, d13, d23, d25, d34, d45, d56 = symbols('E a1 a2 a3 a4 a5 a6 d13 d23 d25 d34 d45 d56')
    # failed geometry second nearest neighbor
    metalist = get_metalist()
    symbolic = dict()
    symbolic[1] = a1
    symbolic[2] = a2
    symbolic[3] = a3
    symbolic[4] = a4
    symbolic[5] = a5
    symbolic[6] = a6
    symbolic[(1, 3)] = d13
    symbolic[(3, 1)] = d13
    symbolic[(2, 3)] = d23
    symbolic[(3, 2)] = d23
    symbolic[(2, 5)] = d25
    symbolic[(5, 2)] = d25
    symbolic[(3, 4)] = d34
    symbolic[(4, 3)] = d34
    symbolic[(4, 5)] = d45
    symbolic[(5, 4)] = d45
    symbolic[(5, 6)] = d56
    symbolic[(6, 5)] = d56
    eqs = []
    lhs = [E * a1, E * a2, E * a3, E * a4, E * a5, E * a6]
    for j, meta in enumerate(metalist):
        for eq in meta:
            symbo = 0
            for sy in eq:
                if isinstance(sy, tuple):
                    symbo += symbolic[sy] * symbolic[sy[1]]
                else:
                    symbo += symbolic[sy]
            print(lhs[j], "=", symbo)
            eqs.append(Eq(symbo, lhs[j]))
    solutions = nsolve(eqs, [E, a1, a2, a3, a4, a5, a6, d13, d23, d25, d34, d45, d56],
                       [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1])
    print(solutions)
    solutions2 = solve(eqs)
    print(solutions2)
    if solutions:
        print("Solutions:")
    else:
        print("No solutions found for the system of equations.")
def nn_onsite():
    # eqs = []
    # # Failed nearest neighbor only - too unconstrained
    # E, e1, e2, e3, e4, e5, e6, a1, a2, a3, a4, a5, a6 = symbols('E e1 e2 e3 e4 e5 e6 a1 a2 a3 a4 a5 a6')
    # eqs.append(Eq((E - e1) * a1, 4 * a3))
    # #eqs.append(Eq((E - e2) * a2, 2 * a5))
    # ######eqs.append(Eq((E - e2) * a2, 2 * a3))
    # #eqs.append(Eq((E - e2) * a2, a3 + a5))
    # eqs.append(Eq((E - e3) * a3, 2 * a4))
    # ######eqs.append(Eq((E - e3) * a3, a1 + a2))
    # #eqs.append(Eq((E - e3) * a3, a1 + a4))
    # ######eqs.append(Eq((E - e3) * a3, a2 + a4))
    # eqs.append(Eq((E - e4) * a4, 3 * a3))
    # ######eqs.append(Eq((E - e4) * a4, 2 * a3 + a5))
    # eqs.append(Eq((E - e5) * a5, 3 * a2))
    # #eqs.append(Eq((E - e5) * a5, 2 * a2 + a4))
    # ######eqs.append(Eq((E - e5) * a5, 2 * a2 + a6))
    # ######eqs.append(Eq((E - e5) * a5, a2 + 2 * a4))
    # #eqs.append(Eq((E - e5) * a5, a2 + a4 + a6))
    # eqs.append(Eq((E - e6) * a6, 2 * a5))
    # sols = solve(eqs)
    # for sol in sols:
    #     print("")
    #     for ans in sol:
    #         print(ans, "=", sol[ans])
    '''a1 = 0
a2 = 0
a3 = 0
a4 = 0
a5 = 0
a6 = 0
e1 = E

E = e6
a1 = 0
a2 = 0
a3 = 0
a4 = 0
a5 = 0
a6 = 0
e1 = e6

E = e4/2 + e5/2 - sqrt(e4**2 - 2*e4*e5 + e5**2 + 36)/2
a1 = a2
a3 = 6*a2/(e4 - e5 - sqrt(e4**2 - 2*e4*e5 + e5**2 + 36))
a4 = a2
a5 = 6*a2/(e4 - e5 - sqrt(e4**2 - 2*e4*e5 + e5**2 + 36))
a6 = a2
e1 = 7*e4/6 - e5/6 + sqrt(e4**2 - 2*e4*e5 + e5**2 + 36)/6
e2 = (e4**2 - e4*e5 - e4*sqrt(e4**2 - 2*e4*e5 + e5**2 + 36) + 6)/(e4 - e5 - sqrt(e4**2 - 2*e4*e5 + e5**2 + 36))
e3 = (e4*e5 - e5**2 + e5*sqrt(e4**2 - 2*e4*e5 + e5**2 + 36) - 6)/(e4 - e5 + sqrt(e4**2 - 2*e4*e5 + e5**2 + 36))
e6 = (e4**2 - e4*e5 - e4*sqrt(e4**2 - 2*e4*e5 + e5**2 + 36) + 6)/(e4 - e5 - sqrt(e4**2 - 2*e4*e5 + e5**2 + 36))

E = e4/2 + e5/2 + sqrt(e4**2 - 2*e4*e5 + e5**2 + 36)/2
a1 = a2
a3 = 6*a2/(e4 - e5 + sqrt(e4**2 - 2*e4*e5 + e5**2 + 36))
a4 = a2
a5 = 6*a2/(e4 - e5 + sqrt(e4**2 - 2*e4*e5 + e5**2 + 36))
a6 = a2
e1 = 7*e4/6 - e5/6 - sqrt(e4**2 - 2*e4*e5 + e5**2 + 36)/6
e2 = (e4**2 - e4*e5 + e4*sqrt(e4**2 - 2*e4*e5 + e5**2 + 36) + 6)/(e4 - e5 + sqrt(e4**2 - 2*e4*e5 + e5**2 + 36))
e3 = (e4*e5 - e5**2 - e5*sqrt(e4**2 - 2*e4*e5 + e5**2 + 36) - 6)/(e4 - e5 - sqrt(e4**2 - 2*e4*e5 + e5**2 + 36))
e6 = (e4**2 - e4*e5 + e4*sqrt(e4**2 - 2*e4*e5 + e5**2 + 36) + 6)/(e4 - e5 + sqrt(e4**2 - 2*e4*e5 + e5**2 + 36))'''
    print("ROUND 2")
    # REDUCED NEAREST NEIGHBOR - works
    eqs = []
    # Original in SYMPY
   # E, e1, e2, e3, e4, e5, e6, a1, a3, alpha = symbols('E e1 e2 e3 e4 e5 e6 a1 a3 alpha')
    # eqs.append(Eq((E - e1) * a1, 4 * a3))
    # eqs.append(Eq((E - e2) * a1, 2 * a3))
    # eqs.append(Eq((E - e3) * a3, 2 * a1))
    # eqs.append(Eq((E - e4) * a1, 3 * a3))
    # eqs.append(Eq((E - e5) * a3, 3 * a1))
    # eqs.append(Eq((E - e6) * a1, 2 * a3))
    '''a1*(E - e1) = 4*a3
    a1*(E - e2) = 2*a3
    a3*(E - e3) = 2*a1
    a1*(E - e4) = 3*a3
    a3*(E - e5) = 3*a1
    a1*(E - e6) = 2*a3

    E = e5/2 + e6/2 - sqrt(e5**2 - 2*e5*e6 + e6**2 + 24)/2
    a1 = a3*(-e5 + e6 - sqrt(e5**2 - 2*e5*e6 + e6**2 + 24))/6
    e1 = -e5/2 + 3*e6/2 + sqrt(e5**2 - 2*e5*e6 + e6**2 + 24)/2
    e2 = e6
    e3 = 5*e5/6 + e6/6 - sqrt(e5**2 - 2*e5*e6 + e6**2 + 24)/6
    e4 = -e5/4 + 5*e6/4 + sqrt(e5**2 - 2*e5*e6 + e6**2 + 24)/4'''
    # ansatz that A sites = a1, B sites = a3, e2 = e3 (2 coordinated) so only 4site= e1, 3site = e5, 2site = e2
    useSympy = True
    # SYMBOLIC
    if useSympy:
        E, e1, e2, e5, a1, a3 = symbols('E e1 e2 e5 a1 a3')
        sylist = [E,e1,e2,e5,a1,a3]
        E = 0
        eqs.append(Eq((E - e1) * a1, 4 * a3))
        eqs.append(Eq((E - e2) * a1, 2 * a3))
        eqs.append(Eq((E - e2) * a3, 2 * a1))
        eqs.append(Eq((E - e5) * a1, 3 * a3))
        eqs.append(Eq((E - e5) * a3, 3 * a1))
        for eq in eqs:
            print(eq.lhs, "=", eq.rhs)
        print("")
        sols = solve(eqs,[e1,e2,e5,a1,a3])
        for sol in sols:
            if a3 in sol:
                print(sol)
                for key in sol:
                    print(key,type(key))
    # NUMERICAL
    else:
        N=300
        E_vals = np.linspace(-1,1,N)
        data = []
        for E_val in E_vals:
            def meta_equations(vars):
                e1_val, e2_val, e5_val, a1_val, a3_val = vars
                return [
                    (E_val - e1_val) * a1_val - 4 * a3_val,
                    (E_val - e2_val) * a1_val - 2 * a3_val,
                    (E_val - e2_val) * a3_val - 2 * a1_val,
                    (E_val - e5_val) * a1_val - 3 * a3_val,
                    (E_val - e5_val) * a3_val - 3 * a1_val
                ]
            initial_guess = [1,1,1,1,1]  # [e1, e2, e5, a1, a3]
            solution = fsolve(meta_equations, initial_guess)
            data.append(solution)
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        val_names = ["epsilon1","epsilon2","epsilon5","a1","a3"]
        fig, ax = plt.subplots(figsize=(8, 6))
        for i in range(5):
            values = [row[i] for row in data]
            ax.plot(E_vals, values, label=val_names[i])
        ax.set_xlabel('Energy (free parameter)')
        ax.set_ylabel('Values')
        ax.set_title('Values vs. E')
        ax.legend()
        plt.tight_layout()
        plt.show()
