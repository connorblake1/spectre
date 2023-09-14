import os
import numpy as np
import scipy.fft
from translated import *
from metasolve import *

# returns the edge index. when flat side down, 0 is top left going CW around to 5 at 9 oclock. Mirrored along 0-3 axis when even recursions
def mirrorIndex(i, recs):
    if recs % 2 == 1:
        return i
    else:
        return (7 - i) % 6

def matchIndex(i):
    return (i + 3) % 6

def Rangle(c1, c2):
    tol = .2
    ang = np.degrees(angle_between_points(c1, c2))
    if abs(ang / 90 - 1) < tol:
        return 0
    elif abs(ang / 90 + 1) < tol:
        return 3
    elif abs(ang / 30 - 1) < tol:
        return 1
    elif abs(ang / 30 + 1) < tol:
        return 2
    elif abs(ang / 150 - 1) < tol:
        return 5
    elif abs(ang / 150 + 1) < tol:
        return 4
    else:
        print("ERROR")
        return -1

def conjLabel(ul):
    return (ul[1] + 3) % 6, (ul[0] + 3) % 6, ul[3], ul[2]

def mirrorLabel(ul):
    return ul[1], ul[0], ul[3], ul[2]

def grayscale(val):
    max = .097  # from gdict
    return (val / max, val / max, val / max)

def matchPos(c1, c2):
    ang = np.degrees(angle_between_points(c1, c2))
    tol = .2
    if abs(ang / 90 - 1) < tol:
        return 4, 1
    elif abs(ang / 90 + 1) < tol:
        return 1, 4
    elif abs(ang / 30 - 1) < tol:
        return 5, 2
    elif abs(ang / 30 + 1) < tol:
        return 0, 3
    elif abs(ang / 150 - 1) < tol:
        return 3, 0
    elif abs(ang / 150 + 1) < tol:
        return 2, 5
    else:
        print("ERROR")
        return 0, 0

def buildHexBase():
    hex6 = [pt(0, 0), pt(rad, 0), pt(rad * 1.5, rad * s32), pt(rad, rad * 2 * s32), pt(0, rad * 2 * s32),
            pt(-.5 * rad, rad * s32)]
    hex_keys = [hex6[4], hex6[5], hex6[0], hex6[2]]
    ret = {}
    for lab in ['Delta', 'Theta', 'Lambda', 'Xi', 'Pi', 'Sigma', 'Phi', 'Psi', 'Gamma']:
        ret[lab] = Shape(hex6, hex_keys, lab)
    return ret

# Requires Centers to work
def showHex(ind, show_edges_labels, rotate_name, axin, axHold, type, adjM=None, edgeDict=None):
    v = np.array(axHold.patches[ind].get_verts())
    centerxyH, labelH, orie = centers[ind]
    if type == 0:
        col = colmap[rev_alpha_dict[axHold.patches[ind].get_alpha()]]
        poly = patches.Polygon(v, closed=True, edgecolor='black', facecolor=col)
        axin.add_patch(poly)
        if show_edges_labels:
            edge_list = edge_dict[labelH]
            for i in range(len(edge_list)):
                edge_pos = .7 * np.array(edge_pos_dict[i]) + centerxyH
                edge_name = edge_dict[labelH][mirrorIndex((-orie + i + 6) % 6, recursions)]
                axin.text(edge_pos[0], edge_pos[1], Greek[edge_name], color='white', size=10, ha='center', va='center')
        if not rotate_name:
            orie = 0
        axin.text(centerxyH[0], centerxyH[1] - .15, Greek[labelH], color='white', size=12, ha='center', va='center',
                  rotation=-orie * 60 + 30)
        if showIndex:
            axin.text(centerxyH[0], centerxyH[1] + .15, ind, color='white', size=8, ha='center', va='center')
    elif type == 1:
        if adjM is None or edgeDict is None:
            print("FAILED TO SEND ADJACENCY/DICT")
            return
        adjRow = adjM[ind]
        gA = (labelH == "Gamma")
        ones_indices = np.where((adjRow == 1))[0]
        for one in ones_indices:
            centerxyJ, labelJ, oriJ = centers[one]
            gB = (labelJ == "Gamma")
            R = Rangle(centerxyH, centerxyJ)
            ul = ((orie - R + 6) % 6, (oriJ - R + 6) % 6, gA, gB)
            j1, i1 = matchPos(centerxyH, centerxyJ)
            cLabel = conjLabel(ul)  # equivalent to rotating 180 and swapping order
            if ul in edgeDict:
                wedgeCol = edgeDict[ul]
            else:
                wedgeCol = edgeDict[cLabel]

            a1 = -i1 * 60 + 180
            a2 = a1 - 60
            a1_rad = np.radians(a1)
            a2_rad = np.radians(a2)
            x, y = (centerxyH[0], centerxyH[1])
            vertex1 = (x, y)
            vertex2 = (x + np.cos(a1_rad), y + np.sin(a1_rad))
            vertex3 = (x + np.cos(a2_rad), y + np.sin(a2_rad))
            triangle = patches.Polygon([vertex1, vertex2, vertex3], closed=True, edgecolor=None, facecolor=wedgeCol)
            axin.add_patch(triangle)

    elif type == 2:
        col = "white" if (labelH == "Gamma") else "grey"
        poly = patches.Polygon(v, closed=True, edgecolor='black', facecolor=col)
        axin.add_patch(poly)

        arrow_length = .5
        angle_radians = np.radians(-orie * 60 + 30)
        dx = arrow_length * np.cos(angle_radians) / 2
        dy = arrow_length * np.sin(angle_radians) / 2
        arrow = FancyArrowPatch((centerxyH[0] - dx, centerxyH[1] - dy), (centerxyH[0] + dx, centerxyH[1] + dy),
                                arrowstyle='->', color='blue', mutation_scale=8)
        axin.add_patch(arrow)
        # if show_edges_labels:
        #     edge_list = edge_dict[labelH]
        #     for i in range(len(edge_list)):
        #         edge_pos = .7 * np.array(edge_pos_dict[i]) + centerxyH
        #         edge_name = edge_dict[labelH][mirrorIndex((-orie + i + 6) % 6, recursions)]
        #         axin.text(edge_pos[0], edge_pos[1], edge_name, color='white', size=8, ha='center', va='center')
        # if not rotate_name:
        #     orie = 0
        # axin.text(centerxyH[0], centerxyH[1] - .15, Greek[labelH], color='white', size=10, ha='center', va='center',
        #         rotation=-orie * 60 + 30)
        # axin.text(centerxyH[0], centerxyH[1] + .15, ind, color='white', size=8, ha='center', va='center')


# Based on flawed "edge balance" hypothesis
def computeEdgeBalanceIndiv(ind):
    return edge_bal[centers[ind][1]]

# Based on flawed "edge balance" hypothesis
def computeEdgeBalancePatch(inds):
    bal = [0] * 8
    for ind in inds:
        ibal = computeEdgeBalanceIndiv(ind)
        bal[0] += ibal.a
        bal[1] += ibal.b
        bal[2] += ibal.g
        bal[3] += ibal.d
        bal[4] += ibal.e
        bal[5] += ibal.z
        bal[6] += ibal.eta
        bal[7] += ibal.t
    return bal

# Based on flawed "edge balance" hypothesis
def checkBal(ibal):
    return (ibal[0] == 0 and ibal[1] == 0 and ibal[2] == 0 and ibal[3] == 0 and ibal[4] == 0 and ibal[5] == 0 and
            ibal[6] % 2 == 0 and ibal[7] == 0)

# Used in connecting all the edges to each other without adding twice as many etas as necessary
def noDupEta(i, orin, inEdge, inlist):
    if inEdge != "eta":
        return False
    for item in inlist:
        m1, m2 = item
        if (m1[0] == i and m1[1] == orin) or (m2[0] == i and m2[1] == orin):
            return False
    return True

# Requires matches to work
def findMatch(i, j):
    for match in matches:
        m1, m2 = match
        if m1[0] == i and m2[0] == j:
            return (m1[1], m2[1])
        elif m1[0] == j and m2[0] == i:
            return (m2[1], m1[1])
    return (None, None)

def remove_close_duplicates(vertices, threshold=0.1):
    unique_indices = []
    for i, vertex in enumerate(vertices):
        is_unique = True
        for j, unique_vertex in enumerate(vertices[:i]):
            distance = np.linalg.norm(vertex - unique_vertex)
            if distance <= threshold:
                is_unique = False
                break
        if is_unique:
            unique_indices.append(i)
    unique_vertices = vertices[unique_indices]

    return unique_vertices

def lorentzian(x, x0, gamma):
        return 1.0 / np.pi / gamma / (1.0 + ((x - x0) / gamma) ** 2)

def drawIndivState(filename,plotnamein,fout,adj,uni,state,wfc=True,doLog=False):
    vertnumt = len(uni)

    plt.close('all')
    figi = plt.figure()
    gs = gridspec.GridSpec(1, 2, width_ratios=[0.85, 0.05])  # Main plot and colorbar
    ax_main = plt.subplot(gs[0])

    if wfc:
        if doLog:
            ncut = 1E-4
            state = state*state
            state = np.maximum(state,ncut)
            absMax = max(state)
            cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', ['white', 'red'])
            cnorm = mcolors.LogNorm(vmin=ncut, vmax=absMax)  # Adjust vmin and vmax as needed
            ax_colorbar = plt.subplot(gs[1])
            cbar = plt.colorbar(mappable=plt.cm.ScalarMappable(norm=cnorm, cmap=cmap),
                                cax=ax_colorbar, orientation='vertical')
            cbar.set_label('Wave Function Norm')
        else:
            ncut = "lin"
            rHigh = max(state)
            rLow = min(state)
            absMax = max(abs(rHigh),abs(rLow)) # allows dirty AF phase converter
            cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', ['blue', 'white', 'red'])
            cnorm = mcolors.Normalize(vmin=-absMax, vmax=absMax)  # Adjust vmin and vmax as needed
            ax_colorbar = plt.subplot(gs[1])
            cbar = plt.colorbar(mappable=plt.cm.ScalarMappable(norm=cnorm, cmap=cmap),
                            cax=ax_colorbar, orientation='vertical')
            cbar.set_label('Wave Function Amplitude')
    else: # band
        if doLog:
            ncut = 5E-5
            state = np.maximum(state,ncut)
            absMax = max(state)
            cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', ['white', 'red'])
            cnorm = mcolors.LogNorm(vmin=ncut, vmax=absMax)  # Adjust vmin and vmax as needed
            ax_colorbar = plt.subplot(gs[1])
            cbar = plt.colorbar(mappable=plt.cm.ScalarMappable(norm=cnorm, cmap=cmap),
                                cax=ax_colorbar, orientation='vertical')
        else:
            ncut = "lin"
            rHigh = max(state)
            absMax = rHigh
            cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', ['white', 'red'])
            cnorm = mcolors.Normalize(vmin=0, vmax=rHigh)  # Adjust vmin and vmax as needed
            ax_colorbar = plt.subplot(gs[1])
            cbar = plt.colorbar(mappable=plt.cm.ScalarMappable(norm=cnorm, cmap=cmap),
                            cax=ax_colorbar, orientation='vertical')
        cbar.set_label("Normalized Electron Density")
    # draw vertices
    for i in range(vertnumt):
        connections = np.where((adj[i] != 0))[0]
        posvert = uni[i]
        for conn in connections:
            posconn = uni[conn]
            dist = np.linalg.norm(posvert - posconn)
            if abs(dist - 1) < .05:
                line = plt.Line2D((posvert[0], posconn[0]), (posvert[1], posconn[1]), linewidth=1, color='gray',zorder=2)
                ax_main.add_line(line)
    # draw circles
    for j in range(vertnumt):
        if np.sum(adj[j]) == 0:
            continue  # dead vertices states that override white
        # mapval = np.square(np.abs(e_vec[j, i]))
        mapval = state[j]
        circ = plt.Circle((uni[j][0], uni[j][1]), .35+.25*abs(mapval)/absMax, facecolor=cmap(cnorm(mapval)),zorder=10,edgecolor='black', linewidth=.5)
        ax_main.add_patch(circ)

    plt.tight_layout()
    plt.setp(ax_main, xticks=[], yticks=[])
    ax_main.set_aspect('equal')
    ax_main.autoscale_view()
    plt.subplots_adjust(top=0.95, bottom=0.05)
    ax_main.set_title(plotnamein, fontsize=12)
    plt.savefig(str(filename) + "\\" + str(ncut) + fout + "_" + '.png', dpi=400)

def drawSpectra(gam,posi,fname,makeHistogram):
    plt.close('all')
    x = np.linspace(min(posi) - 1, max(posi) + 1, 1000)
    big_function = np.zeros_like(x)
    for pos in posi:
        big_function += lorentzian(x, pos, gam)
    plt.figure(figsize=(8, 6))
    plt.plot(x, big_function)
    plt.xlabel('Energy Spectrum (units of $\mathit{t}$)')
    plt.ylabel('Density of States')
    plt.xlim(min(x)-.5, max(x)+.5)
    plt.ylim(0,max(big_function)*1.1)
    plt.title('Smoothed Density of States ($\gamma$ = ' + str(np.round(gam,3)) + ")")
    plt.savefig(fname + "\\" + fname + "_EigVals_" + str(np.round(gam,3)) + ".png",dpi=400)
    plt.cla()
    if makeHistogram:
        width = 0.01
        y = np.zeros_like(x)
        for pos in posi:
            rv = uniform(loc=pos - width / 2, scale=width)  # Create a uniform distribution (top-hat)
            y += width * rv.pdf(x)  # Use boxcar function to create top-hat
        plt.figure(figsize=(8, 6))
        plt.plot(x, y)
        plt.xlabel('Energy Spectrum (units of $\mathit{t}$)')
        plt.ylabel('Density of States')
        plt.title('Exact Spectrum')
        plt.savefig(fname + "\\" + fname + "_EigValsExact.png",dpi=400)
        plt.cla()

def drawStates(adj, uni, filename, drawSpectrum=False, drawIndivs=False, drawBands=False, bandsIn=[1, 4, 8, 16, 24, 32],
               lowE=0, highE=40, S_mat=None,gammas=list(np.linspace(0.001,1,20)),defects=False):
    # strip dead states
    row_sums = np.sum(adj, axis=1)
    col_sums = np.sum(adj, axis=0)
    if defects:
        print("Defecting",defectList)
        valid_rows = [int(i) for i, x in enumerate(row_sums) if
                      i not in defectList and (abs(int(x)) == 2 or abs(int(x)) == 3 or abs(int(x)) == 4)]
        valid_cols = [int(i) for i, x in enumerate(row_sums) if
                      i not in defectList and (abs(int(x)) == 2 or abs(int(x)) == 3 or abs(int(x)) == 4)]
    else:
        valid_rows = np.where(row_sums != 0)[0]
        valid_cols = np.where(col_sums != 0)[0] # TODO
    print("Plotted Vertices",len(valid_rows))
    print("Row Validity:",np.allclose(valid_rows,valid_cols))
    adjNew = adj[valid_rows][:, valid_cols]
    if S_mat is not None:
        S_matNew = S_mat[valid_rows][:,valid_cols]
    else:
        S_matNew = None
    uniNew = uni[valid_rows]
    # diagonalize
    start_time = time.time()
    if S_mat is None:

        e_values, e_vec = np.linalg.eigh(adjNew)
        # sparse_adjNew = csc_matrix(adjNew)
        # e_values, e_vec = scipy_eigsh(sparse_adjNew)
    else:
        e_values, e_vec = scipy_eigh(adjNew, S_matNew)
    # expects adj to have negative hops, diagonal energy
    end_time = time.time()
    tictoc = end_time - start_time
    print("Diagonalizing Time:",tictoc)
    plt.cla()
    # States
    makeIndivGraphs = drawIndivs
    makeSpectrum = drawSpectrum
    makeBand = drawBands
    showgraphs = False
    bands = bandsIn
    vertnum = len(uniNew)
    if not os.path.exists(filename):
        os.makedirs(filename)
    # modify spectrum for zeromode correctoin with rank deficiency
    print("Zeromodes",filename,len([x for x in e_values if abs(x) < .0001]))
    positions = e_values
    if makeSpectrum:
        for gamma in gammas:
            drawSpectra(gamma,positions,filename,True)
    if makeIndivGraphs:
        for i in range(lowE, min(highE, len(e_values))):
            plotname = "Eigenstate: " + str(i+1)+", Energy: " + "{:.3f}".format(e_values[i]) + "$\mathit{t}$"
            fout = 'Lin_' + str(i+1) + '_e_' + "{:.3f}".format(e_values[i])
            drawIndivState(filename,plotname,fout,adjNew,uniNew,e_vec[:,i],wfc=True,doLog=False)

            fout = "Log_" + str(i+1) + '_e_' + "{:.3f}".format(e_values[i])
            drawIndivState(filename,plotname,fout,adjNew,uniNew,e_vec[:,i],wfc=True,doLog=True)
    if makeBand:
        for bandopt in bands:
            mapval = []
            for j in range(vertnum):
                hold = 0
                for i in range(lowE, bandopt):
                    hold += np.square(np.abs(e_vec[j, i]))
                mapval.append(hold/bandopt)
            plotname = "Normalized Density (" + str(bandopt) + " elec. / " + str(vertnum) + " orbitals)"
            fout = "Log_Bands_" + '_' +str(lowE) + "." + str(bandopt)
            drawIndivState(filename,plotname,fout,adjNew,uniNew,mapval,wfc=False,doLog=True)
            fout = "Lin_Bands_" + '_' + str(lowE) + "." + str(bandopt)
            drawIndivState(filename, plotname, fout, adjNew, uniNew, mapval, wfc=False, doLog=False)
"""This block of code finds the conditions (ie number of each metatile type) under which all the edges will be able to wrap.
I then realized this is not useful because it creates hexagons which have 3 neighbors on a single vertex which is unphysical.
This code solves the Sympy equations and then exists. It is entirely standalone and exits the program upon completion."""
solveBalancedPatch = False
if solveBalancedPatch:
    matrix = Matrix(abgd)
    eta = symbols('eta', positive=True, integer=True, even=True)
    vector_b = Matrix([0, 0, 0, 0, 0, 0, eta, 0])
    namelist = 'Gamma Delta Theta Lambda Xi Pi Sigma Phi Psi'
    names = namelist.split(" ")
    Gamma, Delta, Theta, Lambda, Xi, Pi, Sigma, Phi, Psi = symbols('Gamma Delta Theta Lambda Xi Pi Sigma Phi Psi',
                                                                   positive=True, integer=True)
    solution = solve(matrix * Matrix([Gamma, Delta, Theta, Lambda, Xi, Pi, Sigma, Phi, Psi]) - vector_b,
                     (Gamma, Delta, Theta, Lambda, Xi, Pi, Sigma, Phi, Psi, eta))
    print(solution)
    pi = 2
    phi = 1
    psi = 3
    eta1 = 8
    sol = [eta1 / 4, eta1 / 4, pi + psi - eta1 / 2, -phi - pi - psi + eta1, -phi - pi - 2 * psi + 3 * eta1 / 2, pi,
           eta1 / 4, phi, psi]
    print("Gamma, Delta, Theta, Lambda, Xi, Pi, Sigma, Phi, Psi")
    print("Tilenums:")
    for i, name in enumerate(names):
        print(name, sol[i])
    print("Eta: " + str(eta1))
    print("Verify: " + str(np.dot(abgd, np.array(sol))))
    exit()

"""Hyperparameters Section
This is the hyperparameter section. It controls everything at the metatile level. Don't edit anything above this.
Key Program Variables:
    centers             list(tuples(list(pos),label,orientation))
    adj                 np(adjacency matrix of all the centers)
    ind_list            list(stores all the indices of the metatiles "patch", used to reference centers)
        size_list       list(how big each metatile is, same indexes as ind_list, basically 71 for Gamma, 78 else)
        blockoffsets    list(cumulative sum of size_list, same indexes as ind_list, used to access vertex matrices)
    matches             list(tuple(tuple(center index 1, display orientation of edge),tuple(center index 2, display orientation of edge)))
    totalAdj (H_tot)    np(matrix of how all the vertices connect, has many 0 rows/columns as duplicates removed in processing)
    totalVertices       list(positions of all the coordinates, does include duplicates)    
    sAdj                np(matrix of how all the unique vertices connect, no duplicates)
    sVerts              np(positions of all vertices in sAdj)      
    projectors (P_A)    dict(key:center index,value: np(projection matrix of this tile with trace = size_list size of tile))
Key Concepts:
    "vertex scheme"     using totalAdj, computing entire eigenstates in doVertexStates --> "Patch#_ConnectTrue folders
    "meta scheme"       using N_vals orbitals on each metatile site with matrix defined by t = <Psi_A|P_AH_totP_B|Psi_B>, drawn in drawProjectors which creates individual projections, MetaProject_Patch# (showing coefficients of each)
    "super scheme"      using 1 orbital on each site, does TB on just the metatiles with the saved 15 types of angle-Gamma hops into HexHopStates#
Overview:
    You change the settings in this section and run the program.
    Depending on showVertices/showHexagons, you will see the selected patch to confirm
"""
# TOP LEVEL PARAMETERS
superTileType = "Psi"  # DEFAULT: "Psi" This is what the inflation rules are based on. They mostly work the same except "Gamma", should be in tile_names.values()
recursions =3  # DEFAULT: 3 This is how big it goes. 1,2 don't work for an unknown bug reason. 3 is large (5 seconds), 4 is massive (~1 minute), 5 takes unknown amount of time. Note that everything is mirrored between levels.
doWrap = False  # DEFAULT: FALSE If the patch is balanced (mistaken belief, see description of solveBalancedPatch above)
tileConnect = True  # DEFAULT: TRUE Whether to connect the tiles together along vertices. This is assumed true for much of the program, so be EXTREMELY careful if false
# superfilename # This will be defined after iList is defined, but it's where most files get saved, so just know it exists
iList = 2  # DEFAULT 2 (19-tile patch with all possible 15 types) which set of patches - see ILIST SECTION below for choices / modifications - (all work for 3,4 recursions, 10+ should be used only with 4)
# DISPLAYING HEXAGONS
showHexagons = True  # Whether to show the metatiles, the next few don't matter if not showHexagons
showIndex = True # DEFAULT: TRUE Whether to show the metatile indices (purely aesthetic or for verifying you selected the right patch)
showEdgeLabels = True  # DEFAULT: TRUE whether to show edge labels (purely aesthetic)
yesRotate = True  # DEFAULT: TRUE whether to rotate tile labels (purely aesthetic)
showPatchOnly = True  # DEFAULT: TRUE only displays the hexagons specified in ind_list, else all in centers (lots of hexagons, have to zoom in to make sense)
saveHex = True  # DEFAUT: TRUE saves to png in superfilename folder
saveHexType = 0  # DEFAULT: 0 0 = normal, 1 = diamonds "what the electron sees", 2 = orientation arrows / gamma not gamma
colorScheme = 1  # DEFAULT: 1, 0 = random, 1 = grayscale (only matters with recursions = 3, saveHexType = 1)
# DISPLAYING VERTICES (the program will EITHER display hexagons or vertices, so even if showVertices is true showHexagons must be false to see)
showVertices = True  # DEFAULT: TRUE (but overriden by showHexagons,so doesn't actually show) displays stitched/rotated vertices on the metatiles from ind_list, the next few don't matter if not showVertices
saveVerts = True  # DEFAULT: TRUE saves to png in superfilename folder
vertColor = 0  # DEFAULT: 0 is colorful (based on metatile colors), 1 is all black, 2 is highlighted edges
hexOnly = False  # DEFAULT: FALSE use only for iList >= 10, skips all vertex operations and only deals with metatiles in order to compute the largest patches in the GammaGammaScheme
# COMPUTATIONAL SECTION
doProjections = True  # DEFAULT: TRUE whether to compute the projection matrices and all subsequent operations (metascheme (assining N_vals orbitals to each metatile trying to see how many needed), goodness of fit calculations
drawProjectors = False  # DEFAULT: FALSE both draws individual patch eigenstates of P_AH_totP_A|Psi_A>=E|Psi_A>
verifyProjectors = False  # DEFAULT: FALSE prints things like whether add to identity etc, creates MetaProject_Patch(showing all N_vals coeffs on all tiles)
computeOverlap = True # DEFAULT: TRUE does a ton of verification operations (printing, some graphing), see this section for more info
doSanityChecks = False # DEFAULT: FALSE prints out some data on metascheme
doHypothesis = False # DEFAULT: FALSE verifies key data to meta/superschemes
doTileNorm = False # DEFAULT: FALSE computes a ton of tilenorms for making sure things work
generateTree = False  # DEFAULT: FALSE does hierarchical tree analysis
doDoubleWeighting = True  # DEFAULT: TRUE save plots showing the weight on each type of vertex
localChern = False  # DEFAULT: FALSE
N_vals = 2  # DEFAULT: 2 number of "orbitals" per tile in the metascheme
doDefects = False # DEFAULT: FALSE removes the vertices at the indices specified in defectList in file translated.py for computations (but not display)
# MISC TERMINAL SECTION
doA2222Scheme = False  # DEFAULT: FALSE EXITS PROGRAM ON COMPLETION tries to self-consistently solve for t_a, A, e_A, E by looking at all possible configurations of 7 and solving 37 equations
doGammaGammaScheme = True  # DEFAULT: TRUE key to super scheme, finds 15 edge types and then assigns hopping values and runs TB on the hexagons with these values
hexTB = False  # DEFAULT: FALSE if doGammaGammaScheme: computes the supervertex model with the 15 types (good data), saves to HexHopStates (good)
               #                else: does the supervertex model but the hoppings are from suspicious solutions from A2222 previous results or handpicked, saves to HexStates (bad)
doCrystallography = False  # DEFAULT: FALSE, EXITS UPON COMPLETION, stores file to superfilename with FT of patch
doGraphene = False  # DEFAULT: FALSE EXITS UPON COMPLETION, solves graphene TB model on just ind_list sites (times 2)
doSingleSpectre = False  # DEFAULT: FALSE EXITS UPON COMPLETION, draws states for single spectre, just a cycle graph
# EIGENSTATE GENERATION
doVertexStates = True  # DEFAULT TRUE calculates states of all the vertices in H_tot (totalAdj)
doSpectrum = True  # DEFAULT: TRUE draws out the spectra the specified smoothing parameters (gammas), exact DOS too
gammasIn = list(np.geomspace(.001, .5, 20))  # DEFAULT some values between 0.001 and .5, smoothing params
doIndivs = True  # DEFAULT: TRUE draws individual eigenstates with high quality graphing in superfilename, takes up to 1minute per state for very large patches
stateNumLow = 0  # DEFAULT: 0 the lowest state drawn if doIndivs
stateNumHigh = 20  # DEFAULT: 20 the highest state drawn if doIndivs
doBands = False  # DEFAULT: FALSE prints out the density plots for all the fillings in bandList, WARNING if any itme is larger than Active Vertices, will break
bandList = [4, 8, 16, 24, 32, 64, 128, 256]

"""Most Useful Configurations
All default: uses standard 19 tile patch, makes images of vertices/metatiles/graph of edge weights, prints out some key details, saves the first 20 states in Patch2Connect_True

End Hyperparamter Section"""

if doSingleSpectre:
    print("Computing Single, then exiting.")
    n = 14
    adj_matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        adj_matrix[i, (i + 1) % n] = 1
        adj_matrix[i, (i - 1) % n] = 1
    verts = []
    verts.append([0, 0]),
    verts.append([1.0, 0.0])
    verts.append([1.5, -0.8660254037844386])
    verts.append([2.366025403784439, -0.36602540378443865])
    verts.append([2.366025403784439, 0.6339745962155614])
    verts.append([3.366025403784439, 0.6339745962155614])
    verts.append([3.866025403784439, 1.5])
    verts.append([3.0, 2.0])
    verts.append([2.133974596215561, 1.5])
    verts.append([1.6339745962155614, 2.3660254037844393])
    verts.append([0.6339745962155614, 2.3660254037844393])
    verts.append([-0.3660254037844386, 2.3660254037844393])
    verts.append([-0.866025403784439, 1.5])
    verts.append([0.0, 1.0])
    verts = np.array(verts)
    drawStates(-adj_matrix,verts,"SingleSpectre",True,True,False,gammas=gammasIn)
    exit()
# build centers = (position, name, orientation), adj
if True:
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    fig, ax = plt.subplots()
    plt.cla()
    sys = buildHexBase()
    for i in range(recursions):
        sys = buildSupertiles(sys)
    sys[superTileType].draw(ident, ax, True, [], False)  # loads data into ax.patches
    centers = []
    for i in range(len(ax.patches)):
        v = np.array(ax.patches[i].get_verts())
        label = rev_alpha_dict[ax.patches[i].get_alpha()]
        ori = np.degrees(angle_between_points(v[0], v[1])) - 90
        outori = (math.floor(1 - ori / 60) + 6) % 6
        centerxy = (v[0] + v[3]) / 2
        centers.append((centerxy, label, outori))
    vertnum = len(centers)
    adj = np.zeros((vertnum, vertnum))
    for i, center1 in enumerate(centers):
        centerxy1, label1, orient1 = center1
        for j, center2 in enumerate(centers):
            centerxy2, label2, orient2 = center2
            if unitdist(centerxy1, centerxy2, s3, .1) and i != j:
                adj[i][j] = 1
                # Verifies that edges match up
                # , i1 = matchPos(centerxy1, centerxy2)
                # print(edge_dict[label1][mirrorIndex((-orient1+i1+6)%6,recursions)],edge_dict[label2][mirrorIndex((-orient2+j1+6)%6,recursions)])
# BEGIN ILIST SECTION
if True:
    if iList == -1:# non gamma tile
        ind_list = [450]
    elif iList == -2:# gamma tile
        ind_list = [7]
    elif iList == -3:# 2 tiles
        ind_list = [472,491]
    elif iList == -4:  # everything up to 110, big, possibly broken
        ind_list = list(range(110))
    elif iList == 0:  # set of wrapping vertices, don't use
        ind_list = [450, 456, 471, 449, 451, 470, 469, 463, 113]
    elif iList == 1:  # more wrapping vertices, don't use
        ind_list = [102, 118, 112, 120, 121, 101, 471, 71, 95, 449, 470, 119, 115, 469, 117, 457, 113, 111]
    elif iList == 2:  # 19 tile patch, default, uses all the possible 15 edge connection types
        ind_list = [309, 310, 291, 284, 308, 314, 311, 267, 290, 288, 285, 306, 307, 300, 462, 451, 313, 312, 266]
    elif iList == 3:  # 3x6 grouping
        ind_list = [472,466,467,469,468,470,113,114,457,112,458,111,118,459,119,302,125,120]
    elif iList == 4:  # 7-tile patch at the center of 5, 9
        # ind_list = [313, 314, 308, 300, 460, 458, 457, 470, 471, 456, 453, 452] only outsides of 7 patch
        ind_list = [451, 462, 461, 464, 449, 450, 463]
    elif iList == 5:  # 19-tile hexagon patch
        ind_list = [313, 314, 308, 300, 460, 458, 457, 470, 471, 456, 453, 452, 451, 462, 461, 464, 449, 450, 463]
    elif iList == 6:  # zigzag
        ind_list = [300,301,460,302,459,125,119,120,95,102]
    elif iList == 7:  # idek
        ind_list = [463,461,460,302,124,123,76,239,240,84]
    elif iList == 8: # two psi connected to each other, wacky zeromode
        ind_list = [457, 111]
    elif iList == 9: # 37-tile hexagon patch
        ind_list = [313, 314, 308, 300, 460, 458, 457, 470, 471, 456, 453, 452, 451, 462, 461, 464, 449, 450, 463,
                    312, 311, 309, 284, 307, 301, 302, 459, 111, 112, 113, 469, 472, 491, 455, 454, 257, 258]
    elif iList == 10: # big circle
        h = 1.9
        k = 17.4
        r = 8
    elif iList == 11: # massive circle, needs >16GB RAM to load vertices, do hexOnly
        h = 40
        k = 27.5
        r = 22
    elif iList == 12: # largest computable vertex patch
        h = 1.9
        k = 17.4
        r = 14
    if iList >= 10: # actually collect all the tiles for 10,11,12
        ind_list = []
        for i in range(len(centers)):
            pos = centers[i][0]
            if np.sqrt(np.square(pos[0] - h) + np.square(pos[1] - k)) < r:
                ind_list.append(i)
        print(ind_list)
    indSize = len(ind_list)
    superfilename = "Patch" + str(iList) + "_Connect" + str(tileConnect)
# END ILIST SECTION

# use sympy and fsolve to try to self-consistently find values for t_a, A, e_A, E, exits program (needs centers)
if doA2222Scheme:
    useSympy = False
    E, Gamma, Delta, Theta, Lambda, Xi, Pi, Sigma, Phi, Psi, t_a, t_b, t_g, t_d, t_ep, t_z, t_t, t_eta, e_Gamma, e_Delta, e_Theta, e_Lambda, e_Xi, e_Pi, e_Sigma, e_Phi, e_Psi, Null1, Null2, Null3, Null4, Null5, Null6, Null7, Null8, Null9, Null10 = symbols(
        "E Gamma Delta Theta Lambda Xi Pi Sigma Phi Psi t_a t_b t_g t_d t_ep t_z t_t t_eta e_Gamma e_Delta e_Theta e_Lambda e_Xi e_Pi e_Sigma e_Phi e_Psi Null1 Null2 Null3 Null4 Null5 Null6 Null7 Null8 Null9 Null10")
    sy = dict()
    sy["Gamma"] = Gamma
    sy["Delta"] = Delta
    sy["Theta"] = Theta
    sy["Lambda"] = Lambda
    sy["Xi"] = Xi
    sy["Pi"] = Pi
    sy["Sigma"] = Sigma
    sy["Phi"] = Phi
    sy["Psi"] = Psi
    sy["t_a"] = t_a
    sy["t_b"] = t_b
    sy["t_g"] = t_g
    sy["t_d"] = t_d
    sy["t_ep"] = t_ep
    sy["t_z"] = t_z
    sy["t_t"] = t_t
    sy["t_eta"] = t_eta
    sy["e_Gamma"] = e_Gamma  #
    sy["e_Delta"] = e_Gamma  # e_Delta
    sy["e_Theta"] = e_Gamma  # e_Theta
    sy["e_Lambda"] = e_Gamma  # e_Lambda
    sy["e_Xi"] = e_Gamma  # e_Xi
    sy["e_Pi"] = e_Gamma  # e_Pi
    sy["e_Sigma"] = e_Gamma  # e_Sigma
    sy["e_Phi"] = e_Gamma  # e_Phi
    sy["e_Psi"] = e_Gamma  # e_Psi
    conndict = dict()
    for i in tile_names:
        conndict[tile_names[i]] = []
    for i, center in enumerate(centers):
        centerxyI, labelI, oriI = center
        adjRow = adj[i]
        ones_indices = np.where((adjRow == 1))[0]
        if len(ones_indices) != 6:
            continue
        neighbors = [None] * 6
        for one in ones_indices:
            centerxyJ, labelJ, oriJ = centers[one]
            j1, i1 = matchPos(centerxyI, centerxyJ)
            oriI1 = mirrorIndex((i1 - oriI + 6) % 6, recursions)
            neighbors[oriI1] = labelJ
        neighbors = tuple(neighbors)
        if neighbors not in conndict[labelI]:
            conndict[labelI].append(neighbors)
    for label in conndict:
        print(label, len(conndict[label]))
    eqs = []
    for label in conndict:
        for eq in conndict[label]:
            symbo = 0
            for i, sym in enumerate(eq):
                coupling = edge_dict[label][i]
                if coupling.endswith('+') or coupling.endswith('-'):
                    coupling = coupling[:-1]
                coupling = "t_" + coupling
                symbo += sy[sym] * sy[coupling]
            onsite = "e_" + label
            lhs = (E - sy[onsite]) * sy[label]
            Equation = Eq(symbo, lhs)
            print(len(eqs), Equation.rhs, '=', Equation.lhs)
            eqs.append(Equation)
    if useSympy:
        solutions2 = solve(eqs, simplify=False)
        for sol in solutions2:
            print("")
            for ans in sol:
                print(ans, "=", sol[ans])
    else:
        variables = [E, Gamma, Delta, Theta, Lambda, Xi, Pi, Sigma, Phi, Psi, t_a, t_b, t_g, t_d, t_ep, t_z, t_t, t_eta,
                     e_Gamma, e_Delta, e_Theta, e_Lambda, e_Xi, e_Pi, e_Sigma, e_Phi, e_Psi, Null1, Null2, Null3, Null4,
                     Null5, Null6, Null7, Null8, Null9, Null10]
        derivatives = [[-eq.lhs.diff(var) + eq.rhs.diff(var) for var in variables] for eq in eqs]


        def calculate_equations(variables):
            E, Gamma, Delta, Theta, Lambda, Xi, Pi, Sigma, Phi, Psi, t_a, t_b, t_g, t_d, t_ep, t_z, t_t, t_eta, e_Gamma, e_Delta, e_Theta, e_Lambda, e_Xi, e_Pi, e_Sigma, e_Phi, e_Psi, NullOne, NullOne, NullOne, NullOne, NullOne, NullOne, NullOne, NullOne, NullOne, NullOne = variables
            e_Delta = e_Gamma
            e_Theta = e_Gamma
            e_Lambda = e_Gamma
            e_Xi = e_Gamma
            e_Pi = e_Gamma
            e_Sigma = e_Gamma
            e_Phi = e_Gamma
            e_Psi = e_Gamma
            eqs = [
                Gamma * (E - e_Gamma) - Delta * t_g - Phi * t_b - Pi * t_a - Psi * t_a - Sigma * t_d - Xi * t_b,
                Gamma * (E - e_Gamma) - Delta * t_g - Lambda * t_a - Phi * t_b - Sigma * t_d - Theta * t_b - Xi * t_a,
                Gamma * (E - e_Gamma) - Delta * t_a - Delta * t_g - Lambda * t_b - Sigma * t_d - Theta * t_b - Xi * t_a,
                Gamma * (E - e_Gamma) - Delta * t_g - Phi * t_b - 2 * Pi * t_a - Sigma * t_d - Xi * t_b,
                Delta * (E - e_Delta) - Gamma * t_g - Phi * t_ep - Phi * t_g - Sigma * t_z - Xi * t_a - Xi * t_b,
                Delta * (E - e_Delta) - Gamma * t_g - Phi * t_g - Psi * t_a - Psi * t_b - Psi * t_ep - Sigma * t_z,
                Delta * (E - e_Delta) - Gamma * t_a - Gamma * t_g - Pi * t_b - Sigma * t_z - Theta * t_g - Xi * t_ep,
                Delta * (E - e_Delta) - Gamma * t_g - Phi * t_g - Pi * t_a - Psi * t_b - Psi * t_ep - Sigma * t_z,
                Delta * (E - e_Delta) - Gamma * t_g - Phi * t_ep - Phi * t_g - Psi * t_b - Sigma * t_z - Xi * t_a,
                Theta * (E - e_Theta) - Delta * t_g - 2 * Gamma * t_b - Lambda * t_t - Phi * t_eta - Sigma * t_b,
                Lambda * (E - e_Lambda) - Gamma * t_a - Gamma * t_b - Pi * t_b - Sigma * t_g - Theta * t_t - Xi * t_ep,
                Xi * (E - e_Xi) - Delta * t_b - Gamma * t_a - Gamma * t_b - Lambda * t_ep - Phi * t_eta - Pi * t_t,
                Xi * (E - e_Xi) - Delta * t_a - Gamma * t_b - Phi * t_b - Phi * t_ep - Phi * t_eta - Pi * t_t,
                Xi * (E - e_Xi) - Delta * t_b - Delta * t_ep - Gamma * t_a - Gamma * t_b - Phi * t_eta - Pi * t_t,
                Xi * (E - e_Xi) - Gamma * t_b - Phi * t_b - Phi * t_ep - Phi * t_eta - Pi * t_t - Sigma * t_a,
                Pi * (E - e_Pi) - Delta * t_b - 2 * Gamma * t_a - Pi * t_ep - Psi * t_ep - Xi * t_t,
                Pi * (E - e_Pi) - Delta * t_a - Gamma * t_a - Phi * t_b - Pi * t_ep - Psi * t_ep - Xi * t_t,
                Pi * (E - e_Pi) - Gamma * t_a - Phi * t_b - 2 * Psi * t_ep - Sigma * t_a - Xi * t_t,
                Pi * (E - e_Pi) - Gamma * t_a - Phi * t_b - Pi * t_ep - Psi * t_ep - Sigma * t_a - Xi * t_t,
                Pi * (E - e_Pi) - Gamma * t_a - Lambda * t_b - 2 * Psi * t_ep - Sigma * t_a - Xi * t_t,
                Sigma * (E - e_Sigma) - Delta * t_z - Gamma * t_d - Phi * t_ep - Phi * t_g - Theta * t_b - Xi * t_a,
                Sigma * (E - e_Sigma) - Delta * t_z - Gamma * t_d - Phi * t_b - Phi * t_g - Psi * t_a - Psi * t_ep,
                Sigma * (E - e_Sigma) - Delta * t_z - Gamma * t_d - Phi * t_b - Phi * t_g - Pi * t_a - Psi * t_ep,
                Sigma * (E - e_Sigma) - Delta * t_z - Gamma * t_d - Lambda * t_g - Phi * t_b - Pi * t_a - Psi * t_ep,
                Phi * (E - e_Phi) - Gamma * t_b - 2 * Phi * t_ep - Psi * t_b - Sigma * t_g - Xi * t_eta,
                Phi * (E - e_Phi) - Gamma * t_b - Pi * t_b - Sigma * t_ep - Sigma * t_g - Theta * t_eta - Xi * t_ep,
                Phi * (E - e_Phi) - Delta * t_ep - Gamma * t_b - Pi * t_b - Sigma * t_g - Xi * t_ep - Xi * t_eta,
                Phi * (E - e_Phi) - Gamma * t_b - 2 * Phi * t_ep - Sigma * t_g - Xi * t_b - Xi * t_eta,
                Phi * (E - e_Phi) - Delta * t_ep - Delta * t_g - Pi * t_b - Psi * t_eta - Sigma * t_b - Xi * t_ep,
                Phi * (E - e_Phi) - Delta * t_g - 2 * Phi * t_ep - Psi * t_eta - Sigma * t_b - Xi * t_b,
                Phi * (E - e_Phi) - Delta * t_g - 2 * Phi * t_ep - Psi * t_b - Psi * t_eta - Sigma * t_b,
                Psi * (E - e_Psi) - Delta * t_b - Delta * t_ep - Gamma * t_a - 2 * Pi * t_ep - Psi * t_eta,
                Psi * (E - e_Psi) - Delta * t_a - Phi * t_b - Phi * t_eta - Pi * t_ep - Psi * t_ep - Sigma * t_ep,
                Psi * (E - e_Psi) - Delta * t_b - Delta * t_ep - Gamma * t_a - Pi * t_ep - Psi * t_ep - Psi * t_eta,
                Psi * (E - e_Psi) - Phi * t_b - Phi * t_eta - Pi * t_ep - Psi * t_ep - Sigma * t_a - Sigma * t_ep,
                Psi * (E - e_Psi) - Delta * t_b - Gamma * t_a - Phi * t_eta - Pi * t_ep - Psi * t_ep - Sigma * t_ep,
                Psi * (E - e_Psi) - Delta * t_b - Gamma * t_a - Phi * t_eta - 2 * Pi * t_ep - Sigma * t_ep
            ]
            return eqs


        def jacobian_function(values):
            numerical_derivatives = [[lambdify(variables, derivative, "numpy") for derivative in row] for row in
                                     derivatives]
            current_derivatives = np.array(
                [[derivative(*values) for derivative in row] for row in numerical_derivatives])
            return current_derivatives


        initial_guess = [2.0] * 27
        initial_guess[0] = -1
        # initial_guess.extend([random.uniform(5, 10) for _ in range(17)])
        # initial_guess[0]=1
        initial_guess.extend([0] * 10)
        print(initial_guess)
        solutions2 = fsolve(calculate_equations, initial_guess, xtol=1e-8, fprime=jacobian_function)
        print(calculate_equations(solutions2))
        for i, value in enumerate(solutions2):
            variable_name = \
                ["E", "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Phi", "Psi", "t_a", "t_b", "t_g",
                 "t_d", "t_ep",
                 "t_z", "t_t", "t_eta", "e_Gamma", "e_Delta", "e_Theta", "e_Lambda", "e_Xi", "e_Pi", "e_Sigma", "e_Phi",
                 "e_Psi", "Null1", "Null1", "Null1", "Null1", "Null1", "Null1", "Null1", "Null1", "Null1", "Null1"][i]
            print(f"{variable_name} =", np.round(value, 4))
        '''
        initial_guess = [90.0] * 27
        initial_guess[0]=1
        initial_guess.extend([0]*10)
        E = 3.623780757342434
        Gamma = 0.6999789318398041
        Delta = 0.6999789318398285
        Theta = 0.6999789318397968
        Lambda = 0.6999789318398316
        Xi = 0.6999789318398038
        Pi = 0.6999789318398213
        Sigma = 0.6999789318398091
        Phi = 0.6999789318398104
        Psi = 0.6999789318398077
        t_a = 0.5651377774200056
        t_b = 0.35305696404441556
        t_g = 0.6854876182211087
        t_d = 1.2447491980913328
        t_ep = 0.5844027186689876
        t_z = -1.8639828050004719
        t_t = 2.602634129761698
        t_eta = 0.700685690934162
        e_Gamma = -0.1428455418988478
        e_Delta = 2.6141908657670783
        e_Theta = -1.4241975737081862
        e_Lambda = -1.5199954148177037
        e_Xi = -1.5351934875314648
        e_Pi = -1.630991328641426
        e_Sigma = 2.0549292858973365
        e_Phi = 0.36268808276034625
        e_Psi = 0.25169216893677504'''
    exit()
# (re)Generates the list of 15 types of unique overlapes (ie 2 orientations, whether each tile is Gamma or not) by searching all centers (large)
if doGammaGammaScheme:
    gdict = dict()
    gdictCols = dict()
    for i, center in enumerate(centers):
        centerxyI, labelI, oriI = center
        adjRow = adj[i]
        gA = (labelI == "Gamma")
        ones_indices = np.where((adjRow == 1))[0]
        for one in ones_indices:
            centerxyJ, labelJ, oriJ = centers[one]
            gB = (labelJ == "Gamma")
            R = Rangle(centerxyI, centerxyJ)
            ul = ((oriI - R + 6) % 6, (oriJ - R + 6) % 6, gA, gB)
            cLabel = conjLabel(ul)  # equivalent to rotating 180 and swapping order
            if ul not in gdict and cLabel not in gdict:
                gdict[ul] = []
    counting = 0

    # colors of the edges in hex scheme 1
    if colorScheme == 0 or saveHexType != 1:
        colormap = matplotlib.colormaps.get_cmap('tab20')  # random
        colors = [colormap(i)[:3] for i in range(15)]
        for key in gdict:
            gdictCols[key] = colors[counting]
            counting += 1
    elif colorScheme == 1:
        # because i'm lazy, value mapping only works for recursion 3
        for key in gdict:
            gdictCols[key] = grayscale(bind_dict[key])
    else:
        print("NAHHHHHH")
        exit()
    # for key in gdict:
    #     print(key,gdict[key])
# build matches, totalAdj, totalVertices
if True:
    # EDGE MATCHING
    match_edges = dict()  # key: edgetype, val: list of (index of tile, edge orientation (absolute))
    matches = []  # tuple of 2 tuples with format (tile index, edge orientation (absolute)) each
    # finding normal edges
    for i in ind_list:
        adjRow = adj[i]
        ones_indices = np.where((adjRow == 1))[0]
        centerxyI, labelI, oriI = centers[i]
        for one in ones_indices:
            centerxyJ, labelJ, oriJ = centers[one]
            j1, i1 = matchPos(centerxyI, centerxyJ)
            oriI1 = mirrorIndex((i1 - oriI + 6) % 6, recursions)
            edgeI = edge_dict[labelI][oriI1]  # edge type through which it is matched
            if one not in ind_list:
                match_edges[edgeI] = match_edges.get(edgeI, [])
                match_edges[edgeI].append((i, oriI1))
            else:
                if "+" in edgeI or noDupEta(i, oriI1, edgeI, matches):
                    matches.append(((i, oriI1), (one, mirrorIndex((j1 - oriJ + 6) % 6, recursions))))
    normal_edges = len(matches)
    # finding wraparounds is applicable
    if doWrap and checkBal(computeEdgeBalancePatch(ind_list)):
        for key in match_edges:
            for i, edge in enumerate(match_edges[key]):
                total = len(match_edges[key])
                if key == "eta" and i < len(match_edges[key]) / 2:
                    matches.append((edge, match_edges[match_dict[key]][total - 1 - i]))
                if "+" in key:
                    matches.append((edge, match_edges[match_dict[key]][total - i - 1]))
    if not hexOnly:
        # FULL ADJACENCY CONSTRUCTION
        all_unique = {}
        all_adj = {}
        tile_sizes = {}  # size of each TYPE of tile (Gamma, Delta,...)
        for key in tile_names:
            with open('UniqueAdj_' + tile_names[key] + '.pkl', 'rb') as f:
                hu, ha = pickle.load(f)
                all_unique[tile_names[key]] = hu
                all_adj[tile_names[key]] = ha
                tile_sizes[tile_names[key]] = len(hu)
        # the block diagonal sizes of each tile IN THE PATCH
        size_list = []
        blockoffsets = []
        holdOffset = 0
        for ind in ind_list:
            s1 = tile_sizes[centers[ind][1]]
            size_list.append(s1)
            blockoffsets.append(holdOffset)
            holdOffset += s1
        print("Tiles:",indSize)
        print("Block diagonal sizes:", size_list)
        print("Offsets:",blockoffsets)
        # build vertex list in the correct locations (includes duplicates), labelled in same index order as totalAdj
        startList = rotate_points(np.array(all_unique[centers[ind_list[0]][1]]),
                                  centers[ind_list[0]][2] * np.pi / 3 + np.pi / 6)
        totalVertices = [startList]  # starting
        for loop in range(indSize - 1):
            totalVertices.append([])
        locked = [False] * indSize
        locked[0] = True
        print("Matching...")
        matchCycles = 0
        while not all(locked):
            matchCycles += 1
            if matchCycles > indSize:
                print("Disconnected List of Tiles. Exiting.")
                exit()
            for match in matches:
                # i1 = index in ind_list, o1 = absolute edge orientation, e1 = edge name, p1 = pointlist of edgename, t1 is tilename
                m1, m2 = match
                i1, o1 = m1
                l1 = ind_list.index(i1)
                t1 = centers[i1][1]
                e1 = edge_dict[t1][o1]
                p1 = edge_index_dict[t1][o1]
                block1 = blockoffsets[l1]
                i2, o2 = m2
                l2 = ind_list.index(i2)
                t2 = centers[i2][1]
                e2 = edge_dict[t2][o2]
                p2 = edge_index_dict[t2][o2]
                block2 = blockoffsets[ind_list.index(i2)]
                if (locked[l1] and locked[l2]) or (not locked[l1] and not locked[l2]):
                    continue
                # f1 is anchor, f2 is moved
                if locked[l1]:
                    f1 = i1
                    f2 = i2
                else:
                    f1 = i2
                    f2 = i1
                    l1, l2 = l2, l1
                    p1, p2 = p2, p1
                    t1, t2 = t2, t1
                locked[l2] = True
                shiftVertices = all_unique[t2]
                if e2 == "eta":
                    p2 = list(reversed(p2))
                newpoints = align_shapes(totalVertices[l1], shiftVertices, p1[0], p1[1], p2[0], p2[1])
                totalVertices[l2] = newpoints
                if all(locked):
                    break
        totalVertices = np.vstack(totalVertices)
        print("Vertex Count (duplicates)", len(totalVertices))
        # Build block diagonal
        totalSize = sum(size_list)
        totalAdj = np.zeros((totalSize, totalSize))
        rollingIndex = 0
        for i, ind in enumerate(ind_list):
            tileName = centers[ind][1]
            blockSize = size_list[i]
            totalAdj[rollingIndex:rollingIndex + blockSize, rollingIndex:rollingIndex + blockSize] = all_adj[tileName]
            rollingIndex += blockSize
        # Cross connect across block diagonals (all in matchlist)
        newmap = dict()
        if tileConnect:
            for match in matches:
                # i1 = index in ind_list, o1 = absolute edge orientation, e1 = edge name, p1 = pointlist of edgename, t1 is tilename
                m1, m2 = match
                i1, o1 = m1
                l1 = ind_list.index(i1)
                t1 = centers[i1][1]
                mi1 = mirrorIndex(o1, recursions)
                e1 = edge_dict[t1][o1]
                p1 = edge_index_dict[t1][o1]
                block1 = blockoffsets[l1]
                i2, o2 = m2
                l2 = ind_list.index(i2)
                t2 = centers[i2][1]
                mi2 = mirrorIndex(o2, recursions)
                e2 = edge_dict[t2][o2]
                p2 = edge_index_dict[t2][o2]
                block2 = blockoffsets[l2]
                print("Edge Connection: ", i1, t1, e1, i2, t2, e2)
                if e2 == "eta":
                    p2 = list(reversed(p2))
                # fix adjacency matrix
                for j in range(len(p1)):  # k is overwritten
                    h, k = (block1 + p1[j], block2 + p2[j])
                    while h in newmap:
                        conns = np.where((totalAdj[h] == 1))[0]
                        for con in conns:
                            ncon = con
                            while ncon in newmap:
                                ncon = newmap[ncon]
                            totalAdj[con][h] = 0
                            totalAdj[h][con] = 0
                            totalAdj[ncon][newmap[h]] = 1
                            totalAdj[newmap[h]][ncon] = 1
                        h = newmap[h]
                    while k in newmap:
                        conns = np.where((totalAdj[k] == 1))[0]
                        for con in conns:
                            ncon = con
                            while ncon in newmap:
                                ncon = newmap[ncon]
                            totalAdj[con][k] = 0
                            totalAdj[k][con] = 0
                            totalAdj[ncon][newmap[k]] = 1
                            totalAdj[newmap[k]][ncon] = 1
                        k = newmap[k]
                    k_indices = np.where((totalAdj[k] == 1))[0]
                    newmap[k] = h
                    for one in k_indices:
                        totalAdj[k][one] = 0
                        totalAdj[one][k] = 0
                        totalAdj[one][h] = 1
                        totalAdj[h][one] = 1
        print("Symmetric:", np.allclose(totalAdj, totalAdj.T))

        if doCrystallography:
            print("Starting FT")
            xmin = np.min(totalVertices[:,0])
            xmax = np.max(totalVertices[:,0])
            ymin = np.min(totalVertices[:,1])
            ymax = np.max(totalVertices[:,1])
            xs = xmax - xmin
            ys = ymax-ymin
            res = 1024
            maxWidth = 1.1*max(xs,ys)
            centerX = (xmax+xmin)/2
            centerY = (ymax+ymin)/2

            aperture = np.zeros((res, res), dtype=complex)
            for vertex in totalVertices:
                x, y = vertex
                cx = int(res/2)+int((x-centerX)*res/maxWidth)
                cy = int(res/2)+int((y-centerY)*res/maxWidth)
                aperture[cx,cy] = 1.0
            fft_result = np.fft.fft2(aperture)
            fft_result = np.fft.fftshift(fft_result)
            fft_result = np.abs(fft_result)
            # fft_result = gaussian_filter(fft_result,2)
            fft_result /= np.max(fft_result)
            thresh = .2
            for x in range(res):
                for y in range(res):
                    val = fft_result[x,y]
                    if val > thresh:
                        kill = 8
                        maxval = val
                        im = 0
                        jm = 0
                        for i in range(-kill, kill+1):
                            for j in range(-kill, kill+1):
                                if i + x < res and j + y < res:
                                    if fft_result[x+i,y+j] > maxval:
                                        maxval = fft_result[x+i,y+j]
                                        im = i
                                        jm = j
                                    fft_result[x+i,y+j] = 0
                        plt.gca().add_patch(plt.Circle((x+im,y+jm), radius=10*maxval, color='black', fill=True))


            plt.xticks([])
            plt.yticks([])
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().autoscale_view()
            plt.gca().set_aspect('equal')
            # plt.imshow(fft_result)
            plt.tight_layout()
            plt.savefig(str(superfilename) + "\\FFT_"+str(res)+"_"+str(thresh)+".png",dpi=600)
            plt.close('all')
            exit()
# draws out things and save them as necessary - stops program to verify evaluating what you want
if True:
    if not os.path.exists(superfilename):
        os.makedirs(superfilename)
    if saveHex and showHexagons:
        axNew = fig.add_subplot()
        if saveHexType == 1:
            colDict = gdictCols
            adjM = adj
        else:
            colDict = None
            adjM = None
        if not showPatchOnly:
            for i in range(len(centers)):
                showHex(i, showEdgeLabels, yesRotate, axNew, ax, saveHexType, edgeDict=colDict, adjM=adjM)
        else:
            for ind in ind_list:
                showHex(ind, True, True, axNew, ax, saveHexType, edgeDict=colDict, adjM=adjM)
        hexfilename = "Metatiles" + str(iList) + "_" + str(saveHexType) + "_Patch" + str(showPatchOnly) + "_Col" + str(
            colorScheme)
        ax.remove()
        plt.sca(axNew)
        plt.axis('tight')
        plt.xticks([])
        plt.yticks([])
        axNew.set_aspect('equal')
        # plt.title("Patch Metatiles")
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.savefig(str(superfilename) + "\\" + hexfilename + ".png",dpi=400)
    plt.show()
    plt.cla()
# draw vertices to patch
if not hexOnly and showVertices:

    def get_col(ind):
        for i, block in enumerate(blockoffsets):
            if ind < block:
                return tuple(colmap[centers[ind_list[i - 1]][1]])
        return tuple(colmap[centers[ind_list[-1]][1]])
    is_edge_dict = dict()
    for i in range(len(totalVertices)):
        for k, v in newmap.items():
            if v == i:
                is_edge_dict[v] = True
    for i in range(totalSize):
        if i in newmap:
            if newmap[i] != i:
                continue
        connections = np.where((totalAdj[i] == 1))[0]
        posvert = totalVertices[i]
        for conn in connections:
            posconn = totalVertices[conn]
            dist = np.linalg.norm(posvert - posconn)
            if abs(dist - 1) < .05:
                line = plt.Line2D((posvert[0], posconn[0]), (posvert[1], posconn[1]), linewidth=1, color='black')
                plt.gca().add_line(line)
    x = []  # shortened list WITH DIFFERENT INDICES
    y = []
    cols = []
    actives = totalSize
    for row in totalAdj:
        if np.sum(row) == 0:
            actives -= 1
    blockInd = 0
    for i in range(len(totalVertices)):
        if vertColor == 2:
            if i > blockoffsets[blockInd]+size_list[blockInd]:
                blockInd += 1
            modI = i - blockoffsets[blockInd]
            NG = True
            edgeList = elseEdge
            if size_list[blockInd] == 71:
                NG = False
                edgeList=gammaEdge
            if modI in edgeList:
                col = (1,0,0)
            else:
                col = (0,0,0)
        elif vertColor == 0:
            col = get_col(i)
            if i in is_edge_dict:
                col = (0, 0, 1)
            if i in newmap:
                if i != newmap[i]:
                    continue
                else:
                    col = (0, 0, 1)
        elif vertColor == 1:
            col = (0,0,0)

        x.append(totalVertices[i, 0])
        y.append(totalVertices[i, 1])
        cols.append(col)
    plt.scatter(x, y, c=cols,s=12)
    print("Active Vertices:", actives)
    # for i in range(len(x)):
    #     plt.text(x[i], y[i], str(i), fontsize=12, ha='center', va='center')
if saveVerts and showVertices:
    vertfilename = "Verts" + str(iList)+"_"+str(vertColor)
    hexfilename = "Metatiles" + str(iList) + "_Verts"
    plt.axis('tight')
    # plt.title("Patch Vertices")
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_aspect('equal')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.savefig(str(superfilename) + "\\" + vertfilename + ".png",dpi=400)
if doGraphene:
    Hgraph = np.zeros((2*indSize,2*indSize))
    Vgraph = np.zeros((2*indSize,2))
    onsite = 0
    hop = 1
    for i,ind in enumerate(ind_list):
        centI, labI, oriI = centers[ind]
        Vgraph[2*i,0] = centI[0]
        Vgraph[2*i,1] = centI[1]
        Vgraph[2*i+1, 0] = centI[0]+1
        Vgraph[2*i+1, 1] = centI[1]
    for i,vertI in enumerate(Vgraph):
        for j,vertJ in enumerate(Vgraph):
            if unitdist(vertJ,vertI,1,.1):
                Hgraph[i,j] = 1
    print("Sym?",np.allclose(Hgraph,Hgraph.T))
    drawStates(-Hgraph,Vgraph,"Graphene"+str(iList),True,True,False)
    exit()
# on a large hex grid, assign arbitrary hoppings between tiles dependent on edge only (not orientation, gamma or not)
if hexTB and not doGammaGammaScheme:
    print("UNVERIFIED CODE BAD RESULTS GOING TO HEXSTATES FOLDERS")
    Hhex = np.zeros((indSize, indSize))
    scheme = 2
    onsite = 0  # -8.2176
    # a_ep = 1.8778
    # gdte = 2.1206
    # bz = 1.635
    # a_ep = .85
    # bz = .65
    # gdte = 1.1
    a_ep = 1
    bz = 1
    gdte = 1
    edge_binds = {
        'a+': a_ep,
        'a-': a_ep,
        'b+': bz,
        'b-': bz,
        'g+': gdte,
        'g-': gdte,
        'd+': gdte,
        'd-': gdte,
        'ep+': a_ep,
        'ep-': a_ep,
        'z+': bz,
        'z-': bz,
        't+': gdte,
        't-': gdte,
        'eta': gdte}
    for match in matches:
        m1, m2 = match
        i1, o1 = m1
        l1 = ind_list.index(i1)
        t1 = centers[i1][1]
        e1 = edge_dict[t1][o1]
        i2, o2 = m2
        l2 = ind_list.index(i2)
        t2 = centers[i2][1]
        e2 = edge_dict[t2][o2]
        Hhex[l1, l2] = edge_binds[e1]
        Hhex[l2, l1] = edge_binds[e2]
    verts = []
    for i, ind in enumerate(ind_list):
        Hhex[i, i] = onsite
        verts.append(centers[ind][0])
    print("Hhex Symmetric:", np.allclose(Hhex, Hhex.T))
    print("Number of metatiles:", indSize)
    fname = "HexStates" + str(iList) + "_" + str(onsite) + "_" + str(bz) + "_" + str(a_ep) + "_" + str(gdte)
    drawStates(Hhex, verts, fname, True, True, True, highE=(indSize - 1), bandsIn=[1, 30, 50, 100, 150])
    exit()
# super scheme - uses 15 computed values from meta hopping matrix of ground states and does it for large patches
if hexTB and doGammaGammaScheme:
    randomDict = True
    # if randomDict: implement random hoppings

    # invert dicts as necessary
    if recursions % 2 == 0:
        auxdict = dict()
        sauxdict = dict()
        for key in bind_dict:
            auxdict[mirrorLabel(key)] = bind_dict[key]
            sauxdict[mirrorLabel(key)] = sbind_dict[key]
        for key in auxdict:
            bind_dict[key] = auxdict[key]
            sbind_dict[key] = sauxdict[key]
        # for key in bind_dict:
        #     print(key,bind_dict[key])
    # makes H and S matrices for computing states
    HhexG = np.zeros((indSize, indSize))
    SmatG = np.zeros((indSize, indSize))
    for i, ind in enumerate(ind_list):
        if centers[ind][1] == "Gamma":
            HhexG[i, i] = -2.39
        else:
            HhexG[i, i] = -2.394
        SmatG[i, i] = 1
        adjRow = adj[ind]
        ones_indices = np.where((adjRow == 1))[0]
        centerxyI, labelI, oriI = centers[ind]
        for one in ones_indices:
            if one not in ind_list:
                continue
            j = ind_list.index(one)
            centerxyJ, labelJ, oriJ = centers[one]
            R = Rangle(centerxyI, centerxyJ)
            ul = (mirrorIndex((oriI - R + 6) % 6, recursions), mirrorIndex((oriJ - R + 6) % 6, recursions),
                  (labelI == "Gamma"), (labelJ == "Gamma"))
            if ul in bind_dict:
                val = bind_dict[ul]
                sval = sbind_dict[ul]
            else:
                val = bind_dict[conjLabel(ul)]
                sval = sbind_dict[conjLabel(ul)]
            HhexG[i, j] = -val
            HhexG[j, i] = -val
            SmatG[i, j] = sval
            SmatG[j, i] = sval
    vertsG = []
    for i, ind in enumerate(ind_list):
        vertsG.append(centers[ind][0])
    vertsG = np.array(vertsG)
    drawStates(HhexG, vertsG, "HexHopStates" + str(iList), lowE=0, highE=indSize, bandsIn=[1, 50, 100], drawIndivs=True,
               drawSpectrum=True, S_mat=SmatG)
# SHOW PROJECTED STATES
# build projection matrices and find states (solve P_AHP_A|Psi_A> = E|Psi_A>)
if doProjections and not hexOnly:
    print("")
    print("Generating Projectors...")
    projectors = dict()
    projectSpectra = dict()
    projectStates = dict()
    # kill states with 0 rows
    # strip dead states
    row_sums = np.sum(totalAdj, axis=1)
    col_sums = np.sum(totalAdj, axis=0)
    if doDefects:
        print("Defecting:",defectList)
        valid_rows = [int(i) for i, x in enumerate(row_sums) if
                      (i not in defectList) and (abs(int(x)) == 2 or abs(int(x)) == 3 or abs(int(x)) == 4)]
        valid_cols = [int(i) for i, x in enumerate(col_sums) if
                      (i not in defectList) and (abs(int(x)) == 2 or abs(int(x)) == 3 or abs(int(x)) == 4)]
        valid_rows = np.array(valid_rows)
        valid_cols = np.array(valid_cols)
    else:
        valid_rows = np.where(row_sums != 0)[0]
        valid_cols = np.where(col_sums != 0)[0]
    sAdj = totalAdj[valid_rows][:, valid_cols]
    sVert = totalVertices[valid_rows]
    sSize = len(sAdj)
    print("Presize:",totalSize,"Postsize:",sSize)
    # generate projectors
    totalEdges = set()
    for i, ind in enumerate(ind_list):
        Pi = np.zeros((sSize,sSize))
        blockLow = blockoffsets[i]
        blockHigh = blockLow + size_list[i]
        edgeIndices = []
        foundEdges = []
        for j in range(totalSize):
            if j in range(blockLow, blockHigh):
                j1 = j
                while j1 in newmap:
                    if j1 == newmap[j1]:
                        break
                    j1 = newmap[j1]
                js = np.where(valid_rows == j1)[0]
                for ii, subInd in enumerate(ind_list):
                    NG = (size_list[ii] != 71)
                    if NG:
                        edgeIndices = elseEdge
                    else:
                        edgeIndices = gammaEdge
                    if j1 - blockoffsets[ii] in edgeIndices:
                        foundEdges.append(js[0])
                        break
                Pi[js, js] = 1
        totalEdges = totalEdges | set(foundEdges)
        projectors[ind] = Pi
        # full projector in full basis
        Hmod = np.dot(np.dot(Pi, sAdj), Pi)  # PiHPi

        dummy_values, dummy_vec = np.linalg.eigh(Hmod)
        idx = dummy_values.argsort()[::-1]
        dummy_values = dummy_values[idx]
        dummy_vec = dummy_vec[:,idx]
        # down projection
        hrow_sums = np.sum(Hmod, axis=1)
        hcol_sums = np.sum(Hmod, axis=0)
        hvalid_rows = np.where(hrow_sums != 0)[0]
        hvalid_cols = np.where(hcol_sums != 0)[0]
        sHmod = Hmod[hvalid_rows][:, hvalid_cols]  # just the projector subspace
        # diagonalize in full rank projector basis (ie not a ton of vacuum states)
        e_values, e_vec = np.linalg.eigh(-sHmod)
        projectSpectra[ind] = e_values
        # back project
        output = np.zeros((sSize,sSize))
        prerank = int(np.trace(Pi))
        for j, row_idx in enumerate(hvalid_rows):
            for k in range(prerank):
                output[row_idx, k] = e_vec[j, k]  # first prerank columns are the actual eigenstates
        vacHold = prerank
        for j in range(sSize): # locks gauge in nullranked rows to make them vacuum states
            if j not in hvalid_rows:
                output[j,vacHold] = 1
                vacHold += 1
        # print("Rank",np.linalg.matrix_rank(output),vacHold,prerank)
        projectStates[ind] = output
        projectStates[ind][:, 0] = np.abs(projectStates[ind][:, 0])  # define ground to be positive phase for ansatz TODO be careful with complex
        if drawProjectors:
            fname = "Projector_Patch" + str(iList) + "_Tile" + str(ind) + "_" + centers[ind][1]
            if not os.path.exists(fname):
                os.makedirs(fname)
            for j in range(38):
                pname = "Projector " + centers[ind][1] + " (" + str(ind) + ") State " + str(j + 1)
                oname = str(j + 1) + "_e_" + str(np.round(projectSpectra[ind][j], 3))
                drawIndivState(fname,pname,oname,sHmod,sVert[hvalid_rows],projectStates[ind][:,j][hvalid_rows],wfc=True,doLog=False)
            # drawStates(-Hmod, sVert, "Projector_Patch" + str(iList) + "_Tile" + str(ind) + "_" + centers[ind][1],
            #            doSpectrum, doIndivs, doBands, [1, 4, 8], lowE=0, highE=70)
    totalEdges = sorted(list(totalEdges))
    print("All Edges", len(totalEdges))
    if verifyProjectors:
        print("Verifying Projectors...")
        sum = np.zeros((sSize,sSize))
        for i, indi in enumerate(ind_list):
            Pi = projectors[indi]
            sum += projectors[indi]
            for j, indj in enumerate(ind_list):
                if j < i:
                    sum -= np.dot(projectors[indi], projectors[indj])
                    for k, indk in enumerate(ind_list):
                        if k < i and k < j:
                            sum += np.dot(np.dot(projectors[indi],projectors[indj]),projectors[indk])
        print("Projectors (with nonortho correction) add to identity: ", np.allclose(sum, np.eye(sSize)))
    # find <psi_i|H|psi_j> = t_matrix, S matrix for all ground-->N states and then diagonalize all of it (diagonal blocks are energies on diag and 0 otherwise because orthonormal)
    print("Generating H_meta...")
    t_matrix = np.zeros((N_vals * indSize, N_vals * indSize))
    overlapMatrix = np.zeros((N_vals * indSize, N_vals * indSize))
    for i, indi in enumerate(ind_list):
        for j, indj in enumerate(ind_list):
            for ki in range(N_vals):
                for kj in range(N_vals):
                    psi_i = projectStates[indi][:, ki]
                    psi_j = projectStates[indj][:, kj]
                    overlapMatrix[i * N_vals + ki, j * N_vals + kj] = np.dot(np.conjugate(psi_i), psi_j)
                    t_matrix[i * N_vals + ki, j * N_vals + kj] = np.dot(np.conjugate(psi_i), np.dot(sAdj, psi_j))
    t_matrix[t_matrix < 1e-4] = 0  # kills small values
    t_matrix *= -1 # True Hamiltonian
    print("T_Matrix Hermitian:",np.allclose(t_matrix,t_matrix.conj().T))
    print("Overlap Hermitian:", np.allclose(overlapMatrix,overlapMatrix.conj().T))
    # graphically display each
    filename = "MetaProject_Patch" + str(iList) + "_N" + str(N_vals)
    if verifyProjectors:
        # print out adjacency matrix in legible form
        showMatrix = True
        if showMatrix:
            print("<psi_i|H|psi_j> Matrix for all i,j, each states up to", N_vals)
            truncated_matrix = np.round(t_matrix, 3)
            for i, line in enumerate(truncated_matrix):
                outs = str(ind_list[i // N_vals]) + "&\t"
                for val in line:
                    if abs(val) < .001:
                        val = 0
                    outs += str(val) + "&"
                outs += "\\\\"
                print(outs)
            print("<psi_i|psi_j> Matrix for all i,j, each states up to", N_vals)
            truncated_matrix = np.round(overlapMatrix, 3)
            for i, line in enumerate(truncated_matrix):
                outs = str(ind_list[i // N_vals]) + "\t"
                for val in line:
                    outs += str(val) + "\t"
                print(outs)
        # create positions for each of eig val in order to display it
        uni_t = np.zeros((N_vals * indSize, 2))
        shift = .3
        for i in range(len(uni_t)):
            c_index = i // N_vals
            pos = centers[ind_list[c_index]][0]
            uni_t[i] = np.array([pos[0] * 5, pos[1] * 5 + shift * (i % N_vals)])
        drawStates(t_matrix, uni_t, filename, doSpectrum, doIndivs, doBands, [1, min(4, indSize), min(8, indSize)],
                   lowE=0, highE=min(20, len(t_matrix)), S_mat=overlapMatrix,gammas=[.01])
    if computeOverlap:
        # tile eigenstates are in projectStates, projectSpectrum
        # compute the meta spectrum/states
        print("Starting Computations...")
        print("Diagonalizing H_meta...")
        e_values, e_vecs = scipy_eig(t_matrix, overlapMatrix)
        idx = e_values.argsort()
        metaSpectrum = e_values[idx]
        metaStates = e_vecs[:, idx]

        # compute vertex states
        print("Diagonalizing H_tot...")
        e_values, e_vecs = np.linalg.eigh(-sAdj)
        idx = e_values.argsort()
        totalSpectrum = e_values[idx]
        totalStates = e_vecs[:, idx] # no need to deal with dead rows

        # generate tree and look at it
        doTreePlot = True  # saves graph
        do3dplot = True  # opens 3d graph within tree of surface
        doGrouping = True
        if generateTree:
            maxWidth = 1.5 # roughly how much smoothing needed to give 1 peak
            cWidth = maxWidth
            splittings = []  # lists of indices of x storing local max vals
            end = 0.01 # how low to go
            x = np.linspace(min(totalSpectrum), max(totalSpectrum), 1000)
            y = np.geomspace(maxWidth, end, num=200, endpoint=True, dtype=float)

            def line_function(xi, yi):
                big_function = np.zeros_like(xi)
                for pos in totalSpectrum:
                    big_function += lorentzian(xi, pos, yi)
                return big_function

            for cWidth in y:
                local_max_indices = find_peaks(line_function(x, cWidth))
                splittings.append(local_max_indices[0])

            # plot the branching tree
            if doTreePlot:
                plt.figure(figsize=(8, 6))
                for yi, xs in enumerate(splittings):
                    plt.scatter(x[xs], [y[yi]] * len(xs), color='black', s=10)
                # aux plotting
                plt.gca().set_aspect('equal', adjustable='box')
                tickHeight = .1
                for pos in totalSpectrum:
                    plt.plot([pos, pos], [-tickHeight, 0], color='black', linewidth=.1)
                plt.xlim(min(totalSpectrum) - .5, max(totalSpectrum) + .5)
                plt.ylim(-tickHeight, maxWidth)
                plt.xlabel('Energy Spectrum (units of $\mathit{t}$)')
                plt.ylabel('Lorentzian Smoothing Parameter ($\mathit{t}$)')
                plt.title('Recursive Spectrum Analysis')
                plt.subplots_adjust(left=0.05, right=.95, bottom=.05, top=.95)
                plt.savefig(superfilename + "\\" + superfilename + "_TreeSpectrum.png", dpi=400)
                plt.cla()
            # 3d Surface
            if do3dplot:
                plt.close('all')
                X, Y = np.meshgrid(x, y)
                Z = line_function(X, Y)
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                surface = ax.plot_surface(X, Y, Z, cmap='viridis')
                for i, split in enumerate(splittings):
                    ax.scatter(x[split], [y[i]] * len(split), Z[i][split] + 1, color='black', s=10)
                ax.set_xlabel('Energy (units of $\mathit{t}$)')
                ax.set_ylabel('Smoothing Parameter (units of $\mathit{t}$)')
                ax.set_zlabel('Density of States')
                ax.set_title('Smoothed Density of States Sweep')
                fig.colorbar(surface, ax=ax)
                plt.show()
                plt.cla()
            # grouping by smearing things and looking at smeared functions
            if doGrouping:
                levelIndex = 150  # fixed smearing parameter at levelVal energy
                levelVal = y[levelIndex]  # smoothing
                levelList = x[splittings[levelIndex]]  # energies at split - form branches under which to assign
                print("Smoothing:",levelVal,", Spectrum Splits:",levelList)
                assignmentDict = dict()
                for i, pos in enumerate(totalSpectrum):
                    mindex = -1
                    mindist = 50 * abs(max(totalSpectrum)) # random big number
                    for j, lev in enumerate(levelList):
                        if abs(lev - pos) < mindist:
                            mindex = j
                            mindist = abs(lev - pos)
                    assignmentDict[mindex] = assignmentDict.get(mindex, [])
                    assignmentDict[mindex].append(i)

                # equation 8 in https://www.tandfonline.com/doi/pdf/10.1080/01442359509353303?needAccess=true
                tindex = 0
                tstate = 0
                Psi_start = projectStates[ind_list[tindex]][:, tstate]  # ground state of some tile in patch
                Psi_group = []
                for key in assignmentDict:
                    Psi_hold = np.zeros_like(Psi_start)
                    for substate in assignmentDict[key]:
                        Psi_hold += np.dot(np.conj(Psi_start), totalStates[:, substate]) * totalStates[:, substate]
                    Psi_group.append(Psi_hold)
                # visualize the mixed states to "undo" the chaos at some smoothing level
                filename = "SmoothedState" + str(iList) + "_TileStart" + str(tindex) + "_TileState" + str(
                    tstate) + "_Smooth" + str(np.round(levelVal, 3))
                if not os.path.exists(filename):
                    os.makedirs(filename)
                for i, Psi_G in enumerate(Psi_group):
                    rangeV = "Smoothed States from " + str(np.round(totalSpectrum[assignmentDict[i][0]], 2)) + " to " + str(
                        np.round(totalSpectrum[assignmentDict[i][-1]], 2)) + "$\mathit{t}$" # energy range for smoothing
                    fout = str(i)+"_"+str(np.round(totalSpectrum[assignmentDict[i][0]], 2)) + "." + str(np.round(totalSpectrum[assignmentDict[i][-1]], 2))
                    drawIndivState(filename,rangeV,fout,sAdj,sVert,Psi_G,wfc=True,doLog=False)

        # Sanity Check 1: looking at <Psi_tot_0|P_AH_totP_A|Psi_tot_0> vs <Psi_tot_0|H_tot|Psi_tot_0>
        # Sanity Check 2: looking at <Psi_tot_n|Psi_A_m>
        # Sanity Check 3: lookint at <Psi_A|H_tot|Psi_B> to see if the hopping values are constant (spoiler: not)



        if doSanityChecks:
            print("")
            print("Sanity Check 1")
            for state in range(1):
                for ind in ind_list:
                    Pi = projectors[ind]  # get the projector of the ind tile
                    Psi_tot = totalStates[:, state]  # in the state state
                    H =- np.dot(np.dot(Pi, sAdj), Pi)  # Pi H Pi (totalAdj is H_tot)
                    intro = "<Psi_tot_" + str(state) + "|Pi_" + centers[ind][1] + "HPi_" + centers[ind][
                        1] + "|Psi_tot_" + str(state) + ">: "
                    print(intro + str(np.round(np.dot(np.conjugate(Psi_tot), np.dot(H, Psi_tot)), 3)) + "\t\t E_" + str(
                        state) + ": " + str(np.round(totalSpectrum[state], 3)))
            print("Sanity Check 2")
            for ind in ind_list:
                for n in range(1):
                    for m in range(N_vals):
                        print("<Psi_tot_" + str(n) + "|Psi_" + centers[ind][1] + "_" + str(m) + ">: " + str(
                            np.round(np.dot(np.conj(projectStates[ind][:, m]), totalStates[:, n]), 3)))
            print("Sanity Check 3: Determine if hoppings are the same")
            for match in matches:
                # i1 = index in ind_list, o1 = absolute edge orientation, e1 = edge name, p1 = pointlist of edgename, t1 is tilename
                m1, m2 = match
                i1, o1 = m1
                l1 = ind_list.index(i1)
                t1 = centers[i1][1]
                e1 = edge_dict[t1][o1]
                i2, o2 = m2
                l2 = ind_list.index(i2)
                t2 = centers[i2][1]
                e2 = edge_dict[t2][o2]
                centerxyI, labelI, oriI = centers[i1]
                centerxyJ, labelJ, oriJ = centers[i2]
                for n in range(N_vals):
                    print(str(n) + ": <" + labelI + str(i1) + "|H_tot|" + labelJ + str(i2) + ">", e1,
                          np.round(t_matrix[l1 * N_vals + n, l2 * N_vals + n], 4))
                # print(np.trace(np.dot(projectors[i1],projectors[i2])))

        # Hypothesis: each state is well-represented by the groundstate ie that <Psi_tot_0|P_AH_totP_A|Psi_tot_0>/<Psi_tot_0|H_tot|Psi_tot_0>
        # is roughly equal to <Psi_A_m|Psi_tot_n> (it sort of is)
        if doHypothesis:
            print("")
            print("Hypothesis Testing:")
            for ind in ind_list:
                for state in range(len(ind_list)):
                    P_tile = projectors[ind]
                    Psi_tot = totalStates[:, state]
                    H = np.dot(np.dot(P_tile, -sAdj), P_tile)  # Pi H Pi (totalAdj is H_tot)
                    val2 = np.round(np.dot(np.conjugate(Psi_tot), np.dot(H, Psi_tot)) / totalSpectrum[state], 3)
                    for m in range(3):
                        val1 = np.round(np.square(np.dot(np.conj(projectStates[ind][:, m]), totalStates[:, state])), 3)
                        print("Tile Index:" + str(ind) + "\tState:" + str(state) + "\tNval:" + str(m) + "\t\t" + str(
                            val1) + "\t" + str(val2))

        # looking at the "tilenorm" = sum_i->dim(A) |<Psi_tot_0|Psi_A_i>|^2 versus "how completely the ground state captures"
        # |<Psi_tot_0|Psi_A_0>|^2/tilenorm
        if doTileNorm:
            print("")
            print("Doing Tile Norm...")
            totalStatesCounted = indSize
            storage = np.zeros((totalStatesCounted,78,indSize))
            tnorm = np.zeros((totalStatesCounted,indSize))
            for totState in range(totalStatesCounted):
                for i, ind in enumerate(ind_list):
                    sum = 0
                    for j in range(int(np.trace((projectors[ind])))):
                        sum += np.square(np.abs(np.dot(np.conj(totalStates[:, totState]), projectStates[ind][:, j])))
                    tnorm[totState,i] = sum
                    for tileState in range(int(np.trace(projectors[ind]))):
                        state1 = np.square(np.abs(np.dot(np.conj(totalStates[:, totState]), projectStates[ind][:, tileState])))
                        # print(str(ind) + " tilenorm = sum_i->" + str(size_list[i]) + " |<Psi_tot_"+str(totState)+"|Psi_" + str(
                        #     ind) + "_i>|^2 = " + str(
                        #     np.round(sum, 3)) + "\tcompleteness of "+str(tileState)+"^th tile state = |<Psi_tot_"+str(totState)+"|Psi_" + str(
                        #     ind) + "_"+str(tileState)+">|^2/tilenorm = " + str(np.round(state1 / sum, 3)))
                        storage[totState,tileState,i] = state1
            print("Completeness of total ground by each tile's ground:",storage[0,0,:]/tnorm[0,:]) # ie completeness of ground totalState by ground tileStates - decent
            stateCDF = np.zeros((totalStatesCounted,78,indSize))
            for totState in range(totalStatesCounted):
                for tile in range(indSize):
                    sum = 0
                    for tileState in range(78):
                        sum += storage[totState,tileState,tile]/tnorm[totState,tile]
                        stateCDF[totState,tileState,tile] += sum
                    print(totState,tile,sum) # verifies add to 1
            # generate plots
            print("Generating Cumulative Plot..")
            plt.close('all')
            toPlot = 36
            fig = plt.figure(figsize=(10,6))
            gs = gridspec.GridSpec(1, 2, width_ratios=[0.95, 0.05])  # Graph space, Legend space
            ax = plt.subplot(gs[0])
            for i in range(indSize):
                # ax.plot(storage[0,:36,i]/tnorm[0,i], label=centers[ind_list[i]][1] + str(ind_list[i]))
                ax.plot(np.arange(1,toPlot+1),stateCDF[0,:toPlot,i],label=centers[ind_list[i]][1] + str(ind_list[i]))
            ax.set_xlabel('Tile State ($m$)')
            plt.xticks(np.arange(1,toPlot+1,5))

            # ax.set_ylabel('State Completeness $C_{A,0,m}$')
            ax.set_ylabel('Cumulative Completeness $\sum_i^mC_{A,1,i}$')
            plt.ylim(0,1)
            ax.set_title('How well the $m^{th}$ state of each tile represents $\Psi^1_{tot}$')#_{\\text{tot}}
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.savefig(superfilename+"\\C0mCDF.png",dpi=400)
            # plt.show()

            if iList == 2:
                tileIndex = ind_list.index(309)
            else:
                tileIndex = 0
            plt.close('all')
            print("Generating C_Middle_n_1 Plot...")
            fig = plt.figure(figsize=(10, 6))
            plt.cla()
            gs = gridspec.GridSpec(1, 2, width_ratios=[0.90,.1])  # Graph space, Legend space
            ax = plt.subplot()
            tilenamestr = centers[ind_list[tileIndex]][1] + str(ind_list[tileIndex])
            ax.plot(np.arange(1,totalStatesCounted+1),storage[:, 0, tileIndex],label=r"$|<\Psi_{tot}^n|\Psi_{"+tilenamestr+"}^1>|^2$")#"$C_{"+tilenamestr+",n,0}$")
            ax.plot(np.arange(1,totalStatesCounted+1),tnorm[:,tileIndex],label="$T_{"+tilenamestr+",n}$")
            ax.set_xlabel('Total State ($n$)')
            ax.set_ylabel("Square of Overlap")
            plt.xticks(np.arange(1,totalStatesCounted))
            plt.ylim(0,1)
            ax.set_title("How Well $\Psi^n_{tot}$ is Captured by Tile "+tilenamestr+"'s Ground State") #_{\\text{tot}}
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.subplots_adjust(right=.78)
            plt.savefig(superfilename+"\\Cn0.png",dpi=400)
            # plt.show()

        # look at how much weight on the edges near the midgap
        doEdgeWeighting = False
        if doEdgeWeighting:
            print("")
            print("Examining Edge Weights...")
            eNums = 60
            Midgaps = list(range(sSize))
            olaps = []
            ens = []
            for index in Midgaps:
                overlap = 0
                for edge in totalEdges:
                    overlap+=np.square(np.abs(totalStates[edge,index]))
                olaps.append(overlap)
                ens.append(totalSpectrum[index])
            plt.close('all')
            plt.plot(ens,olaps, marker='o', linestyle='-', color='b', label='Edge Weight')
            plt.xlabel('Energy (units of $\mathit{t}$)')
            plt.ylabel('Percent of State on Tile Edges')
            plt.ylim(0,1.2*max(olaps))
            plt.title('Edge Weight for Patch '+str(iList))
            plt.legend()
            plt.savefig(superfilename+'\\edgeweight'+str(eNums)+'.png')

        if doDoubleWeighting:
            print("")
            print("Examining Vertex Distributions...")
            Midgaps = list(range(sSize))
            eNums = len(Midgaps)
            doubles = []
            cDoubles = []
            eDoubles = []
            sDoubles = []
            triples = []
            quads = []
            alledges = []
            for s in range(sSize):
                rsum  = np.sum(sAdj[s])
                if rsum == 2:
                    if s in totalEdges:
                        if s in newmap or s in newmap.values():
                            sDoubles.append(s)
                        else:
                            eDoubles.append(s)
                    else:
                        cDoubles.append(s)
                    doubles.append(s)
                elif rsum == 3:
                    triples.append(s)
                elif rsum == 4:
                    quads.append(s)
                else:
                    print("FUCK")
            print("Doubles, Centers Doubles, Edge Doubles, Shared Doubles:",len(doubles),len(cDoubles),len(eDoubles),len(sDoubles))
            olaps = []
            centerTwo = []
            edgeTwo = []
            sharedTwo = []
            Three = []
            Four = []
            ens = []
            doWeightPlotting = True
            if doWeightPlotting:
                for index in Midgaps:
                    overlap, oEdge, oCenter, oShared = 0,0,0,0
                    oTrip, oQuad = 0,0
                    state = totalStates[:,index]
                    for doub in cDoubles:
                        oCenter += np.square(np.abs(state[doub]))
                    for doub in eDoubles:
                        oEdge += np.square(np.abs(state[doub]))
                    for doub in sDoubles:
                        oShared += np.square(np.abs(state[doub]))
                    for trip in triples:
                        oTrip += np.square(np.abs(state[trip]))
                    for quad in quads:
                        oQuad += np.square(np.abs(state[quad]))
                    centerTwo.append(oCenter)
                    edgeTwo.append(oEdge)
                    sharedTwo.append(oShared)
                    Three.append(oTrip)
                    Four.append(oQuad)
                    overlap = oCenter + oEdge + oShared
                    olaps.append(overlap)
                    ens.append(totalSpectrum[index])
                plt.close('all')
                fig = plt.figure(figsize=(10, 6))
                plt.cla()
                gs = gridspec.GridSpec(1, 2, width_ratios=[0.90, .1])  # Graph space, Legend space
                ax = plt.subplot()
                ax.plot(ens,olaps, color='b', label='Total Double Weight')
                ax.plot(ens,centerTwo,  color='r', label='Center Double Weight')
                ax.plot(ens,sharedTwo,  color='g', label='Shared Double Weight')
                ax.plot(ens,edgeTwo, color='y', label='Edge Double Weight')
                ax.plot(ens,Three,color='orange',label="Triple Weight")
                ax.plot(ens,Four,color='magenta',label="Quad. Weight")
                plt.axhline(y=len(doubles)/sSize, color='blue', linestyle='--', label='% 2-coordinated')
                plt.axhline(y=len(triples)/sSize,color='orange',linestyle='--',label="% 3-coordinated")
                plt.axhline(y=len(quads)/sSize,color='magenta',linestyle='--',label="% 4-coordinated")
                ax.set_xlabel('Energy (units of $\mathit{t}$)')
                ax.set_ylabel('Percent of State on Vertex Type')
                plt.ylim(0,1.2*max(olaps))
                ax.set_title('Vertex Weights for Patch of '+str(indSize)+ " Tiles")
                plt.legend()
                plt.savefig(superfilename+'\\doubleweight'+str(eNums)+'.png')
            doEdgeStates = False
            if doEdgeStates:
                dlist = np.array(totalEdges)
                Hedge = sAdj[dlist][:, dlist]
                Sedge = sVert[dlist]
                print("Drawing Edges Only")
                drawStates(-Hedge,Sedge,"EdgesOnly"+str(iList),True,True,False,lowE=260,highE=280)

        verifyEigvals = False
        if verifyEigvals:
            tests = [random.randint(0, sSize) for _ in range(50)]
            for test in tests:
                psi = totalStates[:,test]
                print("Testing Eigval",test," Gap:",np.abs(np.dot(np.conj(psi),np.dot(-sAdj,psi))-totalSpectrum[test]))

        # TODO NOT REAL LINEAR ALGEBRA, SECTIONS BELOW NOT VERIFIED
        # <psi_tot|psi_{alpha}_{J}> = sum_i(sum_J(sum_alpha(<psi_tot|i><i|phi_j_alpha><phi_j_alpha|psi_tile>)))
        # looking at the overlap is kind of meaningless because not squared I think
        doOverlap = False
        if doOverlap:
            state = 0
            Tform = np.zeros((len(totalVertices), indSize * N_vals))
            for i in range(len(totalVertices)):
                for j in range(indSize * N_vals):
                    Tform[i, j] = projectStates[ind_list[j // N_vals]][i, j % N_vals]
            for state in range(N_vals):
                sum = np.dot(np.dot(totalStates[:, state], Tform), metaStates[:, state])
                print("OVERLAP in state " + str(state) + ": " + str(sum))

        # needs flux threading and magnetization to work
        if localChern:
            X = np.zeros((totalSize, totalSize))
            Y = np.zeros((totalSize, totalSize))
            for i in range(totalSize):
                X[i, i] = totalVertices[i][0]
                Y[i, i] = totalVertices[i][1]
            e_cut = .5
            P = np.zeros((totalSize, totalSize))
            for state in range(totalSize):
                if totalSpectrum[state] > e_cut:
                    P += np.outer(totalStates[:, state], np.conj(totalStates[:, state]))
            Q = np.identity(totalSize) - P
            C = 2 * np.pi * 1j * (
                        np.dot(np.dot(np.dot(np.dot(P, X), Q), Y), P) - np.dot(np.dot(np.dot(np.dot(P, Y), Q), X), P))
            print(C)

        # actually computes the values that pasted into a dict earlier with t_i = <Psi_A_oriA|H_tot|Psi_B_oriB> (ie directly from t_matrix)
        recomputeGammaGamma = False
        if doGammaGammaScheme and recomputeGammaGamma:
            for i, ind in enumerate(ind_list):
                adjRow = adj[ind]
                ones_indices = np.where((adjRow == 1))[0]
                centerxyI, labelI, oriI = centers[ind]
                for one in ones_indices:
                    if one not in ind_list:
                        continue
                    centerxyJ, labelJ, oriJ = centers[one]
                    R = Rangle(centerxyI, centerxyJ)
                    ul = ((oriI - R + 6) % 6, (oriJ - R + 6) % 6, (labelI == "Gamma"), (labelJ == "Gamma"))
                    psi_i = projectStates[ind][:, 0]
                    psi_j = projectStates[one][:, 0]
                    j1, i1 = matchPos(centerxyI, centerxyJ)
                    oriI1 = mirrorIndex((i1 - oriI + 6) % 6, recursions)
                    edgeI = edge_dict[labelI][oriI1]
                    val = np.round(np.dot(np.conjugate(psi_i), np.dot(sAdj, psi_j)), 3)
                    sval = np.round(np.dot(np.conj(psi_i), psi_j), 3)
                    val = abs(val)
                    if ul in gdict:
                        gdict[ul].append((edgeI, val, sval,ind,one))
                    else:
                        gdict[conjLabel(ul)].append((edgeI, val, sval,ind,one))
            for key in gdict:
                print(key, gdict[key])

        # TODO USES WRONG LINALG
        # <Psi_tot_n|Psi_super_n> and plot it by multiplying each |Psi_A> by <Psi_A|Psi_Super>, comparing to |Psi_tot>
        doSuperOverlap = False  # also draws into SuperStates# folder
        if doSuperOverlap and doGammaGammaScheme:
            lowE = 0
            filename = "SuperStates" + str(iList)
            if not os.path.exists(filename):
                os.makedirs(filename)
            e_valuesG, e_vecG = scipy_eigh(HhexG, SmatG)
            idx = e_valuesG.argsort()  # [::-1]
            e_valuesG = e_valuesG[idx]
            e_vecG = e_vecG[:, idx]
            extrap = np.zeros((totalSize, indSize))  # only indSize states in HhexG
            weighting = np.zeros((1, totalSize))
            for i, indi in enumerate(ind_list):
                for j, indj in enumerate(ind_list):
                    if j < i:
                        weighting += np.diag(projectors[indi]) * np.diag(
                            projectors[indj])  # 3 vertex shows as 3, 2 as 1
            for i, ind in enumerate(ind_list):
                activeIndices = np.where((np.diag(projectors[ind]) == 1))
                for state in range(indSize):
                    for indexI in activeIndices[0]:
                        weight = weighting[0][indexI]  # TODO NOT REAL LINEAR ALGEBRA
                        if weight == 0:
                            apply = 1
                        elif weight == 1:
                            apply = .5
                        elif weight == 3:
                            apply = 1 / 3
                        else:
                            print("FAILURE")
                        extrap[indexI, state] += e_vecG[i, state] * apply * projectStates[ind][indexI, 0]
            for state in range(indSize):
                extrap[:, state] /= np.linalg.norm(extrap[:, state])
                print("State " + str(state) + " overlap: |<Psi_tot_" + str(state) + "|Psi_super_" + str(
                    state) + ">|^2 = " + str(
                    np.round(np.square(np.abs(np.dot(np.conj(totalStates[:, state]), extrap[:, state]))), 3)))
            for i in range(0, indSize):
                plt.close()
                figi, axi = plt.subplots(1, 1)
                for j in range(totalSize):
                    if np.sum(totalAdj[j]) == 0:
                        continue  # dead
                    mapval = extrap[j, i]
                    circ = plt.Circle((totalVertices[j][0], totalVertices[j][1]), .3,
                                      facecolor=map_value_to_color(mapval))
                    plt.gca().add_patch(circ)
                plt.setp(axi, xticks=[], yticks=[])
                plt.axis('tight')
                axi.set_aspect('equal')
                divider = make_axes_locatable(axi)
                cax = divider.append_axes("right", size="3%", pad=0.1)
                axi.set_title('Extrapolated: Eigenvalue ' + str(i), fontsize=12)
                formatted_string = "{:.3f}".format(e_valuesG[i])
                plt.savefig(str(filename) + "\\" + str(i) + '_e_' + formatted_string + '.png')
# solve the states for all vertices using H_tot (totalAdJ)
if doVertexStates and not hexOnly:
    print("")
    print("Generating Vertex States...")
    drawStates(-totalAdj, totalVertices, superfilename, doSpectrum, doIndivs, doBands, bandList, stateNumLow,
               stateNumHigh,gammas=gammasIn,defects=doDefects)
