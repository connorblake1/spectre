import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.fftpack import fft2, ifft2
from mpl_toolkits.mplot3d import Axes3D
from translated import *
from metasolve import *
import pickle
import numpy as np
from PIL import Image
import os

# HYPERPARAMETERS
plotBorders = True
supergrid = False # has to do with putting balls into each spectre, false no balls
draw1 = True  # uses original polygon method in sys translated draw metatiles submethod, other used by meta.py
auxcurve = .25
linethickness = 3
linedensity = 10
recursions = 1
balls = 30
tile_sel = "Gamma"
# Build spectres into vertices variable, implements old ball packing and picture generation that was fed into 2DTISE
if True:
    plotCO = not plotBorders
    if recursions == 3:  # Rec 3 Circle that displays a solid block of connections - used in analytic generation and is shown in showCircle
        center = [14, -28]
        rad = 14
    elif recursions == 1:
        center = [1.2, -1.2]
        rad = 3.2
    if supergrid:
        name = "Tile" + tile_sel + "_Rec" + str(recursions)
    else:
        if plotBorders:
            name = "full_" +tile_sel + "_" + str(recursions) + "_" + str(linethickness)
        else:
            name = str(balls)
        name += "_" + str(auxcurve)
    def draw_moves(dummy):
        plt.cla()
        plt.axis('off')
        sys = buildSpectreBase(False)
        for i in range(recursions):  # determines iterations
            sys = buildSupertiles(sys) # loads datat in to ax.patches
        patchlist = list()  # things to draw inside the spectres
        if plotCO and not supergrid:
            for point in pointlist:
                ball = plt.Circle((point[0], point[1]), radius=ballsize, color='black')
                patchlist.append(ball)
        sys[tile_sel].draw(ident, ax, plotBorders, patchlist,draw1)
        global vertices
        vertices = []
        for patch in ax.patches:  # pull the vertices from the generated polygons (hidden) to make bezier splines
            if isinstance(patch, plt.Polygon):
                patch.set_alpha(0)
                points = patch.get_verts()
                for vert in points:
                    vertices.append(vert)
                generate_bezier_points(points, curve, linedensity, True,
                                       linethickness)  # points, curve variable, density of curve, whether to plot, whether to polygon, axis
            elif isinstance(patch, plt.Circle):
                points = patch.get_verts()
                x = points[:, 0]
                y = points[:, 1]
                plt.gca().plot(x, y, color='black')
                plt.gca().fill(x, y, color='black', alpha=1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
    def remove_duplicates_with_tolerance(coordinates, tolerance=0.01):
        unique_coordinates = set()
        unique_list = []

        for coord in coordinates:
            rounded_coord = np.round(coord / tolerance) * tolerance
            if tuple(rounded_coord) not in unique_coordinates:
                unique_coordinates.add(tuple(rounded_coord))
                unique_list.append(coord)

        return unique_list

    # Load data from file to draw in circles
    if supergrid:
        curve = auxcurve
    else:
        if plotCO:
            with open('metadata_' + name + '.pkl', 'rb') as f:
                meta = pickle.load(f)
            curve = meta[0]
            ballsize = meta[1]
            with open('pointlist_' + name + '.pkl', 'rb') as f:
                pointlist = pickle.load(f)
        else:
            curve = auxcurve
    # Plot
    fig, ax = plt.subplots(facecolor='black')
    plt.subplots_adjust(bottom=0.25)
    draw_moves(0)
    plt.rcParams.update({
        "lines.color": "white",
        "patch.edgecolor": "white",
        "text.color": "black",
        "axes.facecolor": "white",
        "axes.edgecolor": "lightgray",
        "axes.labelcolor": "white",
        "xtick.color": "black",
        "ytick.color": "black",
        "grid.color": "lightgray",
        "figure.facecolor": "white",
        "figure.edgecolor": "black",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "black"})
    plt.axis('off')
    folderName = "RandomFiles\\"
    fig.savefig("RandomFiles\\"+'out_' + name + '.png', bbox_inches='tight', transparent=False, pad_inches=0)
    image = Image.open("RandomFiles\\"'out_' + name + '.png')
    new_image = image.resize((64, 64))
    new_image.save("RandomFiles\\"+'out_64_' + name + '.png')
    filename = "Rec_" + str(recursions) + "_Tile" + tile_sel

# Create adjacency matrix for use in all further
unique = remove_duplicates_with_tolerance(vertices, .01)  # remove duplicates
vertnum = len(unique)
adjmatrix = np.zeros((vertnum, vertnum))
for i in range(vertnum):
    for j in range(vertnum):
        if i != j:
            if unitdist(unique[i], unique[j],1,.02):
                adjmatrix[i][j] = 1  # find unit neighbors

# Modify to 3 and 4 only in the adjacency matrix, size of matrix and vertices stay the same
modConnect = False
if modConnect:
    filename += "_modded"
    for i in range(vertnum):  # rows
        if np.sum(adjmatrix[int(i)]) <= 2:
            continue
        # only 3 and 4 now
        for j in range(vertnum):  # cols
            if adjmatrix[i][j] != 0:  # j = vertex off i which has 3 or 4
                trow = j
                oldtrow = i
                while np.sum(adjmatrix[int(trow)]) == 2:
                    indices = np.where(adjmatrix[int(trow)] == 1)
                    indices = indices[0]
                    adjmatrix[oldtrow][trow] = 0  # kills back connection
                    adjmatrix[trow][oldtrow] = 0  # kills back connection
                    if indices[0] == oldtrow:
                        oldtrow = trow
                        trow = indices[1]
                    else:
                        oldtrow = trow
                        trow = indices[0]
                adjmatrix[i][trow] = 1
                adjmatrix[trow][i] = 1
                if oldtrow != i:
                    adjmatrix[oldtrow][trow] = 0  # kills back connection
                    adjmatrix[trow][oldtrow] = 0  # kills back connection
    print("Matrix:")
    print(np.allclose(adjmatrix, adjmatrix.T))

# NEAREST NEIGHBOR PROGRAM (ie A2222 and A22.22.22.23) SHOWING AND COMPUTING
A22Models = False
if A22Models:
    # Analytic Generation - find all vertex and situation types
    nn = 2 # 2 neighbors deep or 3 (really more like 1 and 2 but it's convention now)
    label_list = [] # will enable labelling each vertex with requisite scheme
    analyticGenerate = True # Generating the list of equations in some kind of array to be fed into metasolver, prints
    # metalist pasteable makes label_list which enables tagging each named vertex
    if analyticGenerate:
        def eq_sort_key(item):
            if isinstance(item, tuple):
                return (1, item[0], item[1])
            else:
                return (0, item)
        def sublist_in_list(sublist, mainlist):
            for item in mainlist:
                if isinstance(item, tuple) and isinstance(sublist, tuple):
                    if item == sublist:
                        return True
                elif not isinstance(item, tuple) and not isinstance(sublist, tuple):
                    if item == sublist:
                        return True
            return False
        def remove_duplicate_sublists(input_list):
            result = []
            for sublist in input_list:
                if not sublist_in_list(sublist, result):
                    result.append(sublist)
            return result
        def in_circle(pos):
            return np.sqrt(np.square(pos[0] - center[0]) + np.square(pos[1] - center[1])) < rad
        def vert_name(AB,i,which):
            if which == 2:
                return vert_name2(AB,i)
            elif which == 3:
                return vert_name3(AB,i)
            else:
                return None
        def vert_name2(AB, i):
            adjRow = adjmatrix[i]
            ones_indices = np.where((adjRow == 1))[0]
            adjlist = sorted([int(np.sum(adjmatrix[i])) for i in ones_indices])
            return AB + ''.join(map(str, adjlist)), ones_indices
        def vert_name3(AB,i):
            adjRow = adjmatrix[i]
            first_indices = np.where((adjRow==1))[0]
            nn1 = []
            nn2 = []
            outstr = []
            for f1 in first_indices:
                adjRow2 = adjmatrix[f1]
                second_indices = np.where((adjRow2==1))[0]
                strf = "_" + str(int(np.sum(adjRow2)))
                stradd = []
                for ind in second_indices:
                    if ind != i:
                        stradd.append(str(int(np.sum(adjmatrix[ind]))))
                outstr.append(strf+''.join(map(str,sorted(stradd))))
            outstr = AB + "".join(map(str,sorted(outstr)))
            return outstr,first_indices
        eqs = [[], [], [], [], [], []]
        old_verts = False  # whether uses old classification or new classification (old = 1-6 based on geometry, new on bipartite (A2222, A22.22.22.23)
        if old_verts:
            for i in range(vertnum):
                pos = unique[i]
                if in_circle(pos):
                    type = get_vtype(i, adjmatrix, unique)
                    adjRow = adjmatrix[i]
                    ones_indices = np.where((adjRow == 1))[0]
                    neweq = []
                    for adj in ones_indices:
                        adjType = get_vtype(adj, adjmatrix, unique)
                        neweq.append(adjType)
                        secondAdjRow = adjmatrix[adj]
                        second_ones_indices = np.where((secondAdjRow == 1))[0]
                        for adj2 in second_ones_indices:
                            if adj2 != i:
                                neweq.append((adjType, get_vtype(adj2, adjmatrix, unique)))
                    neweq = sorted(neweq, key=eq_sort_key)
                    eqs[type - 1].append(neweq)
            # main bottleneck as it churns through duplicate vertices
            for i in range(len(eqs)):
                eqs[i] = remove_duplicate_sublists(eqs[i])
            for i in range(6):
                print(i + 1)
                for eq in eqs[i]:
                    print(eq)
        else:
            # generate custom vertex A2222 and A22.22.22.23 lists to be stored into metalist (bipartite)
            coordlist = [[], [], [], [], []]
            for i in range(vertnum):
                pos = unique[i]
                if in_circle(pos):
                    vnum = get_vtype(i,adjmatrix,unique)
                    on_a = alat(vnum)
                    AB = "A" if on_a else "B"
                    opp = "B" if on_a else "A"
                    neweq = []
                    firstname, ones_indices = vert_name(AB, i,nn)
                    coordindex = 2 * on_a + get_coordination(vnum)-2
                    for ind in ones_indices:
                        name, _ = vert_name(opp,ind,nn)
                        neweq.append(name)
                    neweq = sorted(neweq)
                    neweq.insert(0, firstname)
                    fullname = "".join(neweq)
                    print(fullname,pos)
                    label_list.append((pos,fullname,firstname))
                    coordlist[coordindex].append(neweq)
            # remove duplicates - takes a massive amount of time
            for il, coordind in enumerate(coordlist):
                non_duplicate_lists = []
                for inner_list in coordind:
                    first_element = inner_list[0]  # Keep the first element unchanged
                    rest_of_elements = sorted(inner_list[1:])  # Sort remaining elements
                    sorted_inner_list = [first_element] + rest_of_elements
                    if sorted_inner_list not in non_duplicate_lists:
                        non_duplicate_lists.append(sorted_inner_list)
                coordlist[il] = non_duplicate_lists
            # print in a readable and pasteable way to create the metalists in metasolve
            for i, coordind in enumerate(coordlist):
                coordind = sorted(coordind)
                for j,eq in enumerate(coordind): # print out in array pasteable format that is visually easy to read
                    outstr = ""
                    if j == 0:
                        if i == 0:
                            outstr+="return ["
                        outstr+="["
                    outstr += str(eq)
                    if j == len(coordind)-1:
                        outstr+="]"
                        if i == len(coordlist)-1:
                            outstr+="]"
                        else:
                            outstr+=","
                    else:
                        outstr+=","
                    print(outstr)
            print("")
    # Analtyic Show - ie show the name of the coordination and its equation on point in graph show
    analyticShow = (recursions == 3) # otherwise causes key dict errors because smaller tilings don't have all options

    # Analytic Solve: try to solve the massive lists of equations
    solveOld = True  # solves a single neighbor vertex model with onsite energies - see metasolve for other bad methods
    if solveOld:
        nn_onsite()

    # FAILED ATTEMPTS TO SOLVE A2222 and A22.22.22.23 models
    analyticSolve = False
    if analyticSolve:
        solveNNModels = False # True use sympy (which did not work because too overconstrained all 0), false use brute force eigenvalues (ie not possible because multiple equations per vertex type)
        metalist = get_meta_metalist(nn) # get a list of all the possible equations for the 1st or 2nd nearest neighbor (A2222 vs A22.22.22.23)
        names = {term for co in metalist for eq in co for term in eq}  # get all unique names
        names = sorted(names)
        if solveNNModels:
            # Creates stuff to paste into sympy to solve a massive chunk of equations and then solves it (you have to run
            # this multiple times - 1 to generate the equations and one to actually solve them - ie paste output into nn3,nn2
            names_string = "E " +' '.join(names)
            print("Analytic Solving from Metalist " + str(nn))
            print(names_string)
            print(len(names))
            print("<CODE GENERATION>")
            print("E,"+", ".join(names) + " = symbols(names_string)")
            for name in names:
                print("sy['"+name+"'] = " +name)
            print("</CODE GENERATION>")
            sy = dict()
            if nn == 2:
                E,A22,A222,A2222,A223,A23,A33,B222,B223,B23,B233,B24,B33,B34 = symbols(names_string)
                sy['A22'] = A22
                sy['A222'] = A222
                sy['A2222'] = A2222
                sy['A223'] = A223
                sy['A23'] = A23
                sy['A33'] = A33
                sy['B222'] = B222
                sy['B223'] = B223
                sy['B23'] = B23
                sy['B233'] = B233
                sy['B24'] = B24
                sy['B33'] = B33
                sy['B34'] = B34
            elif nn == 3:
                E, A_22_22_22_22, A_22_22_22_23, A_22_22_23, A_22_22_24, A_22_22_322, A_22_22_323, A_22_23_23, A_22_23_322, A_22_23_323, A_23_24, A_23_322, A_23_323, A_23_333, A_24_24, A_24_322, A_24_323, A_322_322, A_322_323, B_22_22_23, B_22_23_23, B_22_23_322, B_22_322, B_22_322_322, B_22_323, B_22_4222, B_23_23_23, B_23_322, B_23_323, B_23_4222, B_322_322, B_322_323, B_322_4222 = symbols(
                    names_string)
                sy['A_22_22_22_22'] = A_22_22_22_22
                sy['A_22_22_22_23'] = A_22_22_22_23
                sy['A_22_22_23'] = A_22_22_23
                sy['A_22_22_24'] = A_22_22_24
                sy['A_22_22_322'] = A_22_22_322
                sy['A_22_22_323'] = A_22_22_323
                sy['A_22_23_23'] = A_22_23_23
                sy['A_22_23_322'] = A_22_23_322
                sy['A_22_23_323'] = A_22_23_323
                sy['A_23_24'] = A_23_24
                sy['A_23_322'] = A_23_322
                sy['A_23_323'] = A_23_323
                sy['A_23_333'] = A_23_333
                sy['A_24_24'] = A_24_24
                sy['A_24_322'] = A_24_322
                sy['A_24_323'] = A_24_323
                sy['A_322_322'] = A_322_322
                sy['A_322_323'] = A_322_323
                sy['B_22_22_23'] = B_22_22_23
                sy['B_22_23_23'] = B_22_23_23
                sy['B_22_23_322'] = B_22_23_322
                sy['B_22_322'] = B_22_322
                sy['B_22_322_322'] = B_22_322_322
                sy['B_22_323'] = B_22_323
                sy['B_22_4222'] = B_22_4222
                sy['B_23_23_23'] = B_23_23_23
                sy['B_23_322'] = B_23_322
                sy['B_23_323'] = B_23_323
                sy['B_23_4222'] = B_23_4222
                sy['B_322_322'] = B_322_322
                sy['B_322_323'] = B_322_323
                sy['B_322_4222'] = B_322_4222
                # E ,A2222 ,A2223 ,A223, A2233, A224, A233 ,A234, A333, A34, A44, B222, B2222, B22222, B2223 ,B223, B233, B333 = symbols(names_string)
                # sy['A2222'] = A2222
                # sy['A2223'] = A2223
                # sy['A223'] = A223
                # sy['A2233'] = A2233
                # sy['A224'] = A224
                # sy['A233'] = A233
                # sy['A234'] = A234
                # sy['A333'] = A333
                # sy['A34'] = A34
                # sy['A44'] = A44
                # sy['B222'] = B222
                # sy['B2222'] = B2222
                # sy['B22222'] = B22222
                # sy['B2223'] = B2223
                # sy['B223'] = B223
                # sy['B233'] = B233
                # sy['B333'] = B333
            eqs = []
            # Actually solving for the naming scheme
            for j,meta in enumerate(metalist):
                for eq in meta:
                    symbo = 0
                    for sym in eq[1:]:
                        symbo += sy[sym]
                    lhs= E*sy[eq[0]]
                    Equation = Eq(symbo,lhs)
                    print(len(eqs),Equation.rhs,'=',Equation.lhs)
                    eqs.append(Equation)
            print("Equations: " + str(len(eqs)))
            print("Vertex Variables: " + str(len(names)))
            #solutions = nsolve(eqs,[E,a1,a2,a3,a4,a5,a6,d13,d23,d25,d34,d45,d56],[1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1])
            #print(solutions)
            solutions2 = solve(eqs)
            for sol in solutions2:
                print("")
                for ans in sol:
                    print(ans,"=",sol[ans])
        else:
            # FAILED - was going to build adjacency matrix with nn2 and nn3 schemes but then I realized you have multiple
            # equations per type so it wouldn't work
            metalist = get_meta_metalist(nn)
            count_index = 0
            name_dict = dict() # key = name, value = index of matrix rowcol
            adj2 = np.zeros((len(names),len(names)))
            for i,co in enumerate(metalist):
                for j,eq in enumerate(co):
                    if eq[0] not in name_dict:
                        name_dict[eq[0]] = (count_index,i,j)
                        count_index +=1
            for name in name_dict:
                mi, i, j = name_dict[name]
                for term in metalist[i][j][1:]:
                    adj2[mi,name_dict[term][0]] += 1
                print(mi,name)
            for row in adj2:
                print(row)

# Verify A and B have the same sites 8/16
siteVerify = False
if siteVerify:
    Alat = 0
    atype = 0
    acoord = 0
    btype = 0
    bcoord = 0
    for i in range(vertnum):
        type = get_vtype(i, adjmatrix, unique)
        print(type)
        if alat(type):
            atype += 1
            acoord += get_coordination(type)
        else:
            btype -= 1
            bcoord += get_coordination(type)
    print(atype, btype, acoord, bcoord)

# Plot Graph show the tiles in a consideration circle with or without labels
showGraph = True
showCircle = False # Big Block of solid connections used for vertex analysis in nn2 and nn3 models (A2222 etc)
if showGraph:
    custom_color_dict = {
        1: (1.0, 0.0, 0.0),  # Red
        2: (0.0, 1.0, 0.0),  # Green
        3: (0.0, 0.0, 1.0),  # Blue
        4: (1.0, 1.0, 0.0),  # Yellow
        5: (1.0, 0.0, 1.0),  # Magenta
        6: (0.0, 1.0, 1.0)  # Cyan
    }
    bipartite_color_dict = {
        -2: (1.0, 0, 0),
        -3: (1.0, 0.7, 0),
        2: (0, 1.0, 0),
        3: (0, 0, 1.0),
        4: (0, 1, 1)
    }
    true_bipartite_colors = {
        -2: (1.0, 0, 0),
        -3: (1.0, 0, 0),
        2: (0, 0, 1),
        3: (0, 0, 1.0),
        4: (0, 0, 1)
    }

    plt.cla()
    for j in range(vertnum):
        if np.all(adjmatrix[j] == 0):
            continue
        for i in range(vertnum):
            if adjmatrix[j][i] != 0:
                line = plt.Line2D((unique[i][0], unique[j][0]), (unique[i][1], unique[j][1]), linewidth=1,
                                  color='black')
                plt.gca().add_line(line)
        vnum = get_vtype(j, adjmatrix, unique)
        # circ = plt.Circle((unique[j][0], unique[j][1]), .5, facecolor=custom_color_dict[vnum]) #custom color mapping
        circ = plt.Circle((unique[j][0], unique[j][1]), .2, facecolor=true_bipartite_colors[(-1 + 2 * alat(vnum)) * coord(vnum)])  # bipartite color mapping
        plt.gca().add_patch(circ)
    # Labels all the vertices and their connections and equations
    if A22Models:
        if analyticShow: # so doesn't throw error because otherwise not declared outside of within A22Models
            metalist = get_meta_metalist(nn)
            eq_dict = dict()
            counter = 0
            for co in metalist:
                for eq in co:
                    eq_dict["".join(eq)] = counter
                    counter +=1
            for point,fname,oname in label_list:
                plt.gca().annotate(oname.replace("_",""), (point[0], point[1]), textcoords="offset points", xytext=(0, 5), ha='center',color='black',fontsize=8)
                # plt.gca().annotate("Eq"+str(eq_dict[fname]), (point[0], point[1]), textcoords="offset points", xytext=(0, -3), ha='center',color='white')
    # if Metatiles:
    #     for i in range(vertnum):
    #         plt.gca().annotate(str(i), (unique[i][0],unique[i][1]), textcoords="offset points", xytext=(0, 5),
    #                            ha='center', color='black', fontsize=8)
    if showCircle:
        plt.gca().add_patch(plt.Circle(center, rad))
    plt.axis('tight')
    plt.gca().set_aspect('equal')
    plt.show()

# Compute Energies for adjacency matrix subject to some parameters (ie bands, individual plots etc)
doStates = False
if doStates:
    e_values, e_vec = np.linalg.eig(adjmatrix)
    idx = e_values.argsort()[::-1]
    e_values = e_values[idx]
    e_vec = e_vec[:, idx]
    disp = np.zeros_like(unique)
    plt.cla()

    # States
    makeIndivGraphs = True
    makeSpectrum = True
    makeBand = True
    makeFolder = True
    showgraphs = False
    lowE = 0
    highE = 40
    bands = [1, 4, 8, 16, 24, 32]
    if makeFolder:
        os.mkdir(filename)
    if makeSpectrum:
        def lorentzian(x, x0, gamma):
            return 1.0 / (np.pi * np.pi * gamma * (1.0 + ((x - x0) / gamma) ** 2))
        positions = [x for x in e_values if abs(x) > .01]
        gamma = 0.1
        x = np.linspace(min(e_values) - .5, max(e_values) + .5, 1000)
        big_function = np.zeros_like(x)
        for pos in positions:
            big_function += lorentzian(x, pos, gamma)
        plt.figure(figsize=(8, 6))
        plt.plot(x, big_function, label='Composite Lorentzian Function')
        plt.xlabel('Energy Spectrum')
        plt.ylabel('Density of States')
        plt.title('Composite Lorentzian Function')
        plt.savefig(str(filename) + "\\" + filename + "_EigVals.pdf")
        plt.cla()
    if makeIndivGraphs:
        for i in range(lowE, highE):
            figi, axi = plt.subplots(1, 1)
            for j in range(vertnum):
                mapval = np.square(np.abs(e_vec[j, i]))
                mapval = e_vec[j, i]
                circ = plt.Circle((unique[j][0], unique[j][1]), .3, facecolor=map_value_to_color(mapval))
                plt.gca().add_patch(circ)
            plt.setp(axi, xticks=[], yticks=[])
            plt.axis('tight')
            axi.set_aspect('equal')
            divider = make_axes_locatable(axi)
            cax = divider.append_axes("right", size="3%", pad=0.1)
            if i == 0:
                axi.set_title('The ground state', fontsize=12)
            elif i == 1:
                axi.set_title('The 1$^{st}$ Eigenvalue', fontsize=12)
            elif i == 2:
                axi.set_title('The 2$^{nd}$ Eigenvalue', fontsize=12)
            elif i == 3:
                axi.set_title('The 3$^{rd}$ Eigenvalue', fontsize=12)
            else:
                axi.set_title("The " + str(i) + '$^{th}$ Eigenvalue', fontsize=12)
            formatted_string = "{:.3f}".format(e_values[i])
            plt.savefig(str(filename) + "\\" + str(i) + '_e_' + formatted_string + '.pdf')
        if showgraphs:
            plt.show()
    if makeBand:
        indices = np.where(abs(np.sum(adjmatrix, axis=1) - 4) < .1)[0]  # finds rows with 4 coordination
        for bandopt in bands:
            fourcoords = []
            plt.clf()
            plt.cla()
            figi, axi = plt.subplots(1, 1)
            for j in range(vertnum):
                mapval = 0
                for i in range(lowE, bandopt):
                    mapval += np.square(np.abs(e_vec[j, i]))
                if j in indices:
                    fourcoords.append(mapval)
                circ = plt.Circle((unique[j][0], unique[j][1]), .3, facecolor=map_value_to_color(mapval / 1))
                plt.gca().add_patch(circ)
            with open(filename + "\\" + filename + "_4Coords_Band" + str(bandopt) + '.pkl', 'wb') as f:
                pickle.dump(fourcoords, f)
            plt.setp(axi, xticks=[], yticks=[])
            plt.axis('tight')
            axi.set_aspect('equal')
            divider = make_axes_locatable(axi)
            axi.set_title("Bands from " + str(lowE) + " to " + str(bandopt))
            plt.savefig(str(filename) + "\\" + filename + "_Bands_" + str(lowE) + "." + str(bandopt) + '.pdf')
            if showgraphs:
                plt.show()
