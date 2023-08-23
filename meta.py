import os
from translated import *
from metasolve import *

# returns the edge index. when flat side down, 0 is top left going CW around to 5 at 9 oclock. Mirrored along 0-3 axis when even recursions
def mirrorIndex(i, recs):
    if recs % 2 == 1:
        return i
    else:
        return (7-i) % 6

def matchIndex(i):
    return (i + 3) % 6
def Rangle(c1,c2):
    tol = .2
    ang = np.degrees(angle_between_points(c1,c2))
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
    return ((ul[1]+3)%6,(ul[0]+3)%6,ul[3],ul[2])
def mirrorLabel(ul):
    return (ul[1],ul[0],ul[3],ul[2])
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
def showHex(ind, show_edges_labels, rotate_name,axin,axHold):
    v = np.array(axHold.patches[ind].get_verts())
    col = colmap[rev_alpha_dict[axHold.patches[ind].get_alpha()]]
    poly = patches.Polygon(v, closed=True, edgecolor='black', facecolor=col)
    axin.add_patch(poly)
    centerxyH, labelH, orie = centers[ind]
    if show_edges_labels:
        edge_list = edge_dict[labelH]
        for i in range(len(edge_list)):
            edge_pos = .7 * np.array(edge_pos_dict[i]) + centerxyH
            edge_name = edge_dict[labelH][mirrorIndex((-orie + i + 6) % 6, recursions)]
            axin.text(edge_pos[0], edge_pos[1], edge_name, color='white', size=8, ha='center', va='center')
    if not rotate_name:
        orie = 0
    axin.text(centerxyH[0], centerxyH[1] - .15, greek[labelH], color='white', size=10, ha='center', va='center',
            rotation=-orie * 60 + 30)
    axin.text(centerxyH[0], centerxyH[1] + .15, ind, color='white', size=8, ha='center', va='center')
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
def drawStates(adj,uni, filename, drawSpectrum=False, drawIndivs=False, drawBands=False, bandsIn=[1, 4, 8, 16, 24, 32],
               lowE=0, highE=40,S_mat = None):
    if S_mat is None:
        e_values, e_vec = np.linalg.eigh(adj)
    else:
        # sparseTAdj = csc_matrix(adj)
        # sparseS = csc_matrix(S_mat)
        # e_values, e_vec = scipy_eigsh(sparseTAdj,M=sparseS)
        e_values, e_vec = scipy_eigh(adj,S_mat)
    idx = e_values.argsort()[::-1]
    e_values = e_values[idx]
    e_vec = e_vec[:, idx]
    plt.cla()
    # States
    makeIndivGraphs = drawIndivs
    makeSpectrum = drawSpectrum
    makeBand = drawBands
    showgraphs = False
    bands = bandsIn
    vertnum = len(uni)
    if not os.path.exists(filename):
        os.makedirs(filename)
    if makeSpectrum:
        def lorentzian(x, x0, gamma):
            return 1.0 / (np.pi * np.pi * gamma * (1.0 + ((x - x0) / gamma) ** 2))
        positions = [x for x in e_values if abs(x) > .0001]
        positions.extend([0]*max(0,int(np.sum(np.abs(e_values) < .0001)-np.sum(np.sum(adj, axis=1) == 0)))) # properly accounts for zero modes that aren't just deleted rows
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
        plt.savefig(filename + "\\" + filename + "_EigVals.pdf")
        plt.cla()
    if makeIndivGraphs:
        for i in range(lowE, highE):
            plt.close()
            figi, axi = plt.subplots(1, 1)
            for j in range(vertnum):
                if np.sum(adj[j]) == 0:
                    continue
                mapval = np.square(np.abs(e_vec[j, i]))
                mapval = e_vec[j, i]
                circ = plt.Circle((uni[j][0], uni[j][1]), .3, facecolor=map_value_to_color(mapval))
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
        for bandopt in bands:
            plt.clf()
            plt.cla()
            figi, axi = plt.subplots(1, 1)
            for j in range(vertnum):
                if np.sum(adj[j]) == 0:
                    continue
                mapval = 0
                for i in range(lowE, bandopt):
                    mapval += np.square(np.abs(e_vec[j, i]))
                circ = plt.Circle((uni[j][0], uni[j][1]), .3,
                                  facecolor=map_value_to_color(mapval / 1))
                plt.gca().add_patch(circ)
            plt.setp(axi, xticks=[], yticks=[])
            plt.axis('tight')
            axi.set_aspect('equal')
            divider = make_axes_locatable(axi)
            axi.set_title("Bands from " + str(lowE) + " to " + str(bandopt))
            plt.savefig(str(filename) + "\\" + filename + "_Bands_" + str(lowE) + "." + str(bandopt) + '.pdf')
            if showgraphs:
                plt.show()
"""
Written 8/11->8/15
This block of code finds the conditions (ie number of each metatile type) under which all the edges will be able to wrap.
I then realized this is not useful because it creates hexagons which have 3 neighbors on a single vertex which is unphysical.
This code solves the Sympy equations and then exists. It is entirely standalone and exits the program upon completion.
 """
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
"""
This is the hyperparameter section. It controls everything at the metatile level.
Key Program Variables:
    centers             list(tuples(list(pos),label,orientation))
    adj                 np(adjacency matrix of all the centers)
    ind_list            list(stores all the indices of the metatiles "patch", used to reference centers)
        size_list       list(how big each metatile is, same indexes as ind_list, basically 71 for Gamma, 78 else)
        blockoffsets    list(cumulative sum of size_list, same indexes as ind_list, used to access vertex matrices)
    matches             list(tuple(tuple(center index 1, display orientation of edge),tuple(center index 2, display orientation of edge)))
    totalAdj (H_tot)    np(matrix of how all the vertices connect, has many 0 rows/columns as duplicates removed)
    totalVertices       list(positions of all the coordinates, does include duplicates)          
    projectors (P_A)    dict(key:center index,value: np(projection matrix of this tile with trace = size_list size of tile))
Key Concepts:
    "vertex scheme"     using totalAdj, computing entire eigenstates in doVertexStates --> "Patch#_ConnectTrue folders
    "meta scheme"       using N_vals orbitals on each metatile site with matrix defined by t = <Psi_A|P_AH_totP_B|Psi_B>, drawn in drawProjectors which creates individual projections, MetaProject_Patch# (showing coefficients of each)
    "super scheme"      using 1 orbital on each site, does TB on just the metatiles with the saved 15 types of angle-Gamma hops into HexHopStates#
"""
superTileType = "Psi"  # UNLESS YOU PICK GAMMA, SHOULD WORK. Psi is what all the ind_lists are based on, so keep
recursions = 3 # This is how big it goes. 1,2 don't work for an unknown bug reason. 3 is large (5 seconds), 4 is massive (~1 minute), 5 takes unknown amount of time. Note that everything is mirrored between levels.
doWrap = False  # If the patch is balanced (mistaken belief, see description of solveBalancedPatch above)
wrapConnect = True  # Whether to connect the tiles together along vertices. This is assumed true for much of the program, so be extremely careful if false

showHexagons = True  # Whether to show the metatiles
showEdgeLabels = False  # iif showHexagons, whether to show edge labels (slows down Matplotlib)
yesRotate = True  # iif showHexagons, whether to rotate tile labels
showPatchOnly = True  # iif showHexagons, only displays the hexagons specified in ind_list, else all in centers

showVertices = False  # displays stitched/rotated vertices on the metatiles from ind_list
hexOnly = False  # use only for iList >= 10, skips all vertex operations and only deals with metatiles in order to compute the largest patches in the GammaGammaScheme

doProjections = True  # whether to compute the projection matrices and all subsequent operations (metascheme (assining N_vals orbitals to each metatile trying to see how many needed), goodness of fit calculations
drawProjectors = False  # iif doProjections, both draws creates MetaProject_Patch# (showing all N_vals coeffs on all tiles), and creates folders showing individual projectors / eigenstates of P_AH_totP_A|Psi_A>=E|Psi_A>
computeOverlap = True  # iif doProjections, does a ton of verification operations (printing), see this section for more info
localChern = False  #n eeds magnetization TODO
N_vals = 15  # number of "orbitals" per tile in the metascheme

doA2222Scheme = False # tries to self-consistently solve for t_a, A, e_A, E by looking at all possible configurations of 7 and solving 37 equations, exits program
doGammaGammaScheme = True # key to super scheme, finds 15 edge types and then assigns hopping values and runs TB on the hexagons with these values
hexTB = True # key to superscheme, runs hex TB with 15 types

doVertexStates = True  # calculates states of all the vertices in H_tot (totalAdj)
doSpectrum = True
doIndivs = True
doBands = True
stateNum = 15  # number of individual states to draw in vertex scheme
bandList = [1, 4, 8, 16, 24, 32]
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
    showAll = not showPatchOnly and showHexagons
    for i in range(len(ax.patches)):
        v = np.array(ax.patches[i].get_verts())
        label = rev_alpha_dict[ax.patches[i].get_alpha()]
        ori = np.degrees(angle_between_points(v[0], v[1])) - 90
        outori = (math.floor(1 - ori / 60) + 6) % 6
        centerxy = (v[0] + v[3]) / 2
        centers.append((centerxy, label, outori))
        if showAll:
            showHex(i, showEdgeLabels, yesRotate,ax,ax)
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

# Build ind_list
iList = 2 # which set of patches - see inside to choose - (all work for 3,4 recursions, 10+ should be used only with 4)
if True:
    if iList == 0:
        ind_list = [450, 456, 471, 449, 451, 470, 469, 463, 113]  # WORKING LIST FOR DELTA TILE
    elif iList == 1:
        ind_list = [102, 118, 112, 120, 121, 101, 471, 71, 95, 449, 470, 119, 115, 469, 117, 457, 113, 111]  # WORKING LIST FOR DELTA TILE
    elif iList == 2:  # all the possible 15 edge connection types
        ind_list = [309,310,291,284,308,314,311,267,290,288,285,306,307,300,462,451,313,312,266]
    elif iList == 3:  # line zig zag
        ind_list = [466, 469, 113, 457, 111, 459, 125, 124, 69, 70, 64, 237, 244, 221]
    elif iList == 4:  # 7 patch center of 5, 9
        # ind_list = [313, 314, 308, 300, 460, 458, 457, 470, 471, 456, 453, 452] only outsides of 7 patch
        ind_list = [451, 462, 461, 464, 449, 450, 463]
    elif iList == 5:  # hexagon patch (4 is just the inside)
        ind_list = [313, 314, 308, 300, 460, 458, 457, 470, 471, 456, 453, 452, 451, 462, 461, 464, 449, 450, 463]
    elif iList == 6:  # dead slot
        ind_list = []
    elif iList == 7:  # etas + surroundings (7patch)
        ind_list = [457, 111, 112, 113, 114, 115, 118]
    elif iList == 8:
        ind_list = [457, 111]
    elif iList == 9:
        ind_list = [313, 314, 308, 300, 460, 458, 457, 470, 471, 456, 453, 452, 451, 462, 461, 464, 449, 450, 463,
                    312,311,309,284,307,301,302,459,111,112,113,469,472,491,455,454,257,258]
    elif iList == 10:
        h = 1.9
        k = 17.4
        r = 8
    elif iList == 11:
        h = 40
        k = 27.5
        r = 22
    if iList >= 10:
        ind_list = []
        for i in range(len(centers)):
            pos = centers[i][0]
            if np.sqrt(np.square(pos[0] - h) + np.square(pos[1] - k)) < r:
                ind_list.append(i)
        print(ind_list)


    if iList <= 1:
        print("Verifying edge balances...")
        print(computeEdgeBalancePatch(ind_list))
        if not checkBal(computeEdgeBalancePatch(ind_list)):
            print("FATAL ERROR: UNBALANCED")
        else:
            print("Verified edge balances.")
    if showPatchOnly and showHexagons:
        for ind in ind_list:
            showHex(ind, True, True,ax,ax)

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
                    matches.append(((i, oriI1), (one, mirrorIndex((j1 - oriJ + 6) % 6,recursions))))
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
        print("Block diagonal sizes:", size_list)
        # build vertex list in the correct locations (includes duplicates), labelled in same index order as totalAdj
        startList = rotate_points(np.array(all_unique[centers[ind_list[0]][1]]),centers[ind_list[0]][2]*np.pi/3+np.pi/6)
        totalVertices = [startList]  # starting
        for loop in range(len(ind_list) - 1):
            totalVertices.append([])
        locked = [False] * len(ind_list)
        locked[0] = True
        while not all(locked):
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
        if wrapConnect:
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

# draw vertices
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
    for i in range(len(totalVertices)):
        col = get_col(i)
        if i in is_edge_dict:
            col = (0, 0, 1)
        if i in newmap:
            if i != newmap[i]:
                continue
            else:
                col = (0, 0, 1)
        x.append(totalVertices[i, 0])
        y.append(totalVertices[i, 1])
        cols.append(col)
    plt.scatter(x, y, c=cols)
    # for i in range(len(x)):
    #     plt.text(x[i], y[i], str(i), fontsize=12, ha='center', va='center')

plt.axis('tight')
ax.set_aspect('equal')
plt.show()


# use sympy and fsolve to try to self-consistently find values for t_a, A, e_A, E, exits program (needs centers)
if doA2222Scheme:
    useSympy = False
    E, Gamma, Delta, Theta, Lambda, Xi, Pi, Sigma, Phi, Psi, t_a, t_b, t_g, t_d, t_ep, t_z, t_t, t_eta, e_Gamma, e_Delta, e_Theta, e_Lambda, e_Xi, e_Pi, e_Sigma, e_Phi, e_Psi,Null1,Null2,Null3,Null4,Null5,Null6,Null7,Null8,Null9,Null10 = symbols(
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
    sy["e_Gamma"] = e_Gamma#
    sy["e_Delta"] = e_Gamma#e_Delta
    sy["e_Theta"] = e_Gamma#e_Theta
    sy["e_Lambda"] = e_Gamma#e_Lambda
    sy["e_Xi"] = e_Gamma#e_Xi
    sy["e_Pi"] = e_Gamma#e_Pi
    sy["e_Sigma"] = e_Gamma#e_Sigma
    sy["e_Phi"] = e_Gamma#e_Phi
    sy["e_Psi"] = e_Gamma#e_Psi
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
            for i,sym in enumerate(eq):
                coupling = edge_dict[label][i]
                if coupling.endswith('+') or coupling.endswith('-'):
                    coupling = coupling[:-1]
                coupling = "t_" + coupling
                symbo += sy[sym]*sy[coupling]
            onsite = "e_"+label
            lhs = (E-sy[onsite]) * sy[label]
            Equation = Eq(symbo, lhs)
            print(len(eqs), Equation.rhs, '=', Equation.lhs)
            eqs.append(Equation)
    if useSympy:
        solutions2 = solve(eqs,simplify=False)
        for sol in solutions2:
            print("")
            for ans in sol:
                print(ans, "=", sol[ans])
    else:
        variables = [E, Gamma, Delta, Theta, Lambda, Xi, Pi, Sigma, Phi, Psi, t_a, t_b, t_g, t_d, t_ep, t_z, t_t, t_eta, e_Gamma, e_Delta, e_Theta, e_Lambda, e_Xi, e_Pi, e_Sigma, e_Phi, e_Psi,Null1,Null2,Null3,Null4,Null5,Null6,Null7,Null8,Null9,Null10]
        derivatives = [[-eq.lhs.diff(var) + eq.rhs.diff(var) for var in variables] for eq in eqs]
        def calculate_equations(variables):
            E, Gamma, Delta, Theta, Lambda, Xi, Pi, Sigma, Phi, Psi, t_a, t_b, t_g, t_d, t_ep, t_z, t_t, t_eta, e_Gamma, e_Delta, e_Theta, e_Lambda, e_Xi, e_Pi, e_Sigma, e_Phi, e_Psi,NullOne,NullOne,NullOne,NullOne,NullOne,NullOne,NullOne,NullOne,NullOne,NullOne = variables
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
        initial_guess.extend([0]*10)
        print(initial_guess)
        solutions2 = fsolve(calculate_equations, initial_guess,xtol=1e-8,fprime=jacobian_function)
        print(calculate_equations(solutions2))
        for i, value in enumerate(solutions2):
            variable_name = \
            ["E", "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Phi", "Psi", "t_a", "t_b", "t_g", "t_d", "t_ep",
             "t_z", "t_t", "t_eta", "e_Gamma", "e_Delta", "e_Theta", "e_Lambda", "e_Xi", "e_Pi", "e_Sigma", "e_Phi",
             "e_Psi", "Null1", "Null1", "Null1", "Null1", "Null1", "Null1", "Null1", "Null1", "Null1", "Null1"][i]
            print(f"{variable_name} =", np.round(value,4))
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
# Generates the list of 15 types of unique overlapes (ie 2 orientations, whether each tile is Gamma or not) by searching all centers (large)
if doGammaGammaScheme:
    gdict = dict()
    for i, center in enumerate(centers):
        centerxyI, labelI, oriI = center
        adjRow = adj[i]
        gA = (labelI == "Gamma")
        ones_indices = np.where((adjRow == 1))[0]
        for one in ones_indices:
            centerxyJ, labelJ, oriJ = centers[one]
            gB = (labelJ == "Gamma")
            R = Rangle(centerxyI, centerxyJ)
            ul = ((oriI-R+6)%6,(oriJ-R+6)%6,gA,gB)
            cLabel = conjLabel(ul)  # equivalent to rotating 180 and swapping order
            if ul not in gdict and cLabel not in gdict:
                gdict[ul] = []
    # for key in gdict:
    #     print(key,gdict[key])

# on a large hex grid, assign arbitrary hoppings between tiles dependent on edge only (not orientation, gamma or not)
if hexTB and not doGammaGammaScheme:
    Hhex = np.zeros((len(ind_list),len(ind_list)))
    scheme = 2
    onsite = 0#-8.2176
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
        Hhex[l1,l2] = edge_binds[e1]
        Hhex[l2, l1] = edge_binds[e2]
    verts = []
    for i,ind in enumerate(ind_list):
        Hhex[i,i] = onsite
        verts.append(centers[ind][0])
    print("Hhex Symmetric:",np.allclose(Hhex,Hhex.T))
    print("Number of metatiles:",len(ind_list))
    fname = "HexStates" + str(iList)+"_"+str(onsite)+"_" + str(bz)+"_" + str(a_ep)+"_" + str(gdte)
    drawStates(Hhex,verts,fname,True,True,True,highE=(len(ind_list)-1),bandsIn=[1,30,50,100,150])
# super scheme - uses 15 computed values from meta hopping matrix of ground states and does it for large patches
if hexTB and doGammaGammaScheme:
    # loads in 15 types - done manually
    bind_dict = {
        (1, 0, False, False):.009,
        (2, 0, False, True):.012,
        (3, 1, False, False):.017,
        (5, 5, False, False):.024,
        (0, 5, False, False):.05,
        (1, 5, False, False):.02,
        (2, 1, False, True):.051,
        (1, 2, False, True):.097,
        (0, 3, False, False):.063,
        (5, 1, False, False):.033,
        (1, 3, False, True):.02,
        (2, 2, True, False):.025,
        (1, 2, True, False):.016,
        (4, 3, False, False):.072,
        (1, 0, True, False):.009}
    sbind_dict = {
        (1, 0, False, False):.003,
        (2, 0, False, True):.004,
        (3, 1, False, False):.006,
        (5, 5, False, False):.008,
        (0, 5, False, False):.018,
        (1, 5, False, False):.007,
        (2, 1, False, True):.019,
        (1, 2, False, True):.036,
        (0, 3, False, False):.023,
        (5, 1, False, False):.012,
        (1, 3, False, True):.007,
        (2, 2, True, False):.009,
        (1, 2, True, False):.005,
        (4, 3, False, False):.027,
        (1, 0, True, False):.003}
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
        for key in bind_dict:
            print(key,bind_dict[key])
    # makes H and S matrices for computing states
    Hhex = np.zeros((len(ind_list), len(ind_list)))
    Smat = np.zeros((len(ind_list), len(ind_list)))
    for i, ind in enumerate(ind_list):
        if centers[ind][1] == "Gamma":
            Hhex[i,i] = -2.39
        else:
            Hhex[i,i] = -2.394
        Smat[i,i] = 1
        adjRow = adj[ind]
        ones_indices = np.where((adjRow == 1))[0]
        centerxyI, labelI, oriI = centers[ind]
        for one in ones_indices:
            if one not in ind_list:
                continue
            j = ind_list.index(one)
            centerxyJ, labelJ, oriJ = centers[one]
            R = Rangle(centerxyI, centerxyJ)
            ul = (mirrorIndex((oriI - R + 6) % 6,recursions), mirrorIndex((oriJ - R + 6) % 6,recursions), (labelI == "Gamma"), (labelJ == "Gamma"))
            if ul in bind_dict:
                val = bind_dict[ul]
                sval = sbind_dict[ul]
            else:
                val = bind_dict[conjLabel(ul)]
                sval = sbind_dict[conjLabel(ul)]
            Hhex[i,j] = -val
            Hhex[j,i] = -val
            Smat[i,j] = sval
            Smat[j,i] = sval
    verts = []
    for i,ind in enumerate(ind_list):
        verts.append(centers[ind][0])
    drawStates(Hhex,verts,"HexHopStates" +str(iList),lowE=0,highE=len(ind_list),bandsIn=[1,50,100],drawIndivs=True,drawSpectrum=True,S_mat=Smat)

# SHOW PROJECTED STATES
# build projection matrices and find states (solve P_AHP_A|Psi_A> = E|Psi_A>)
if doProjections and not hexOnly:
    verifyProjectors = False

    projectors = dict()
    projectSpectra = dict()
    projectStates = dict()
    # find states that are not referenced at all, used to verify the identity matrix (otherwise would be 0 and fail), basically the overwritten vertices
    if verifyProjectors:
        deadStates = [0]*totalSize
        for i in range(totalSize):
            if i in newmap:
                deadStates[i] = 1
        deadStates = np.diag(deadStates)
    # build projection matrix for each block and store in projectors[ind], projectSpectra[ind], projectStates[ind]
    # also draw each projector and its eigenstates as desired
    for i, ind in enumerate(ind_list):
        matp = np.zeros((totalSize, totalSize))
        blockLow = blockoffsets[i]
        blockHigh = blockLow + size_list[i]
        for j in range(totalSize):
            if j in range(blockLow, blockHigh):
                while j in newmap:
                    if j == newmap[j]:
                        break
                    j = newmap[j]
                matp[j, j] = 1
        Pi = matp
        projectors[ind] = Pi
        Hmod = np.dot(np.dot(Pi, totalAdj), Pi) # PiHPi
        e_values, e_vec = np.linalg.eigh(Hmod)
        idx = e_values.argsort()[::-1]
        projectSpectra[ind] = e_values[idx]
        projectStates[ind] = e_vec[:, idx]
        if drawProjectors:
            drawStates(Hmod, totalVertices, "Projector_Patch" + str(iList) + "_Tile" + str(ind) + "_" + centers[ind][1],
                       doSpectrum, doIndivs, doBands, [1, 4, 8], lowE=0, highE=N_vals)
    # verify rules for summation to identity
    if verifyProjectors:
        sum = np.zeros((totalSize,totalSize))
        for i, indi in enumerate(ind_list):
            Pi = projectors[indi]
            sum += projectors[indi]
            for j, indj in enumerate(ind_list):
                if j < i:
                    sum -= np.dot(projectors[indi], projectors[indj])
        sum += deadStates  # correct for null basis
        print("Projectors (with nonortho correction) add to identity: ",np.allclose(sum,np.eye(totalSize)))
    # find <psi_i|H|psi_j> =t_matrix, S matrix for all ground-->N states and then diagonalize all of it (diagonal blocks are energies on diag and 0 otherwise because orthonormal)
    t_matrix = np.zeros((N_vals*len(ind_list), N_vals*len(ind_list)))
    overlapMatrix = np.zeros((N_vals*len(ind_list), N_vals*len(ind_list)))
    for i, indi in enumerate(ind_list):
        for j, indj in enumerate(ind_list):
            for ki in range(N_vals):
                for kj in range(N_vals):
                    psi_i = projectStates[indi][:, ki]
                    psi_j = projectStates[indj][:, kj]
                    overlapMatrix[i*N_vals+ki,j*N_vals+kj] = np.dot(np.conjugate(psi_i),psi_j)
                    t_matrix[i*N_vals+ki,j*N_vals+kj] = np.dot(np.conjugate(psi_i), np.dot(totalAdj, psi_j))
    t_matrix[t_matrix < 1e-4] = 0  # kills small values
    # graphically display each
    filename = "MetaProject_Patch" + str(iList) + "_N" + str(N_vals)
    if drawProjectors:
        # print out adjacency matrix in legible form
        showMatrix = True
        if showMatrix:
            print("<psi_i|H|psi_j> Matrix for all i,j, each states up to",N_vals)
            truncated_matrix = np.round(t_matrix, 3)
            for i,line in enumerate(truncated_matrix):
                outs = str(ind_list[i//N_vals])+"\t"
                for val in line:
                    outs += str(val) + "\t"
                print(outs)
            print("<psi_i|psi_j> Matrix for all i,j, each states up to", N_vals)
            truncated_matrix = np.round(overlapMatrix, 3)
            for i, line in enumerate(truncated_matrix):
                outs = str(ind_list[i // N_vals]) + "\t"
                for val in line:
                    outs += str(val) + "\t"
                print(outs)
        # create positions for each of eig val in order to display it
        uni_t = np.zeros((N_vals * len(ind_list), 2))
        shift = .3
        for i in range(len(uni_t)):
            c_index = i // N_vals
            pos = centers[ind_list[c_index]][0]
            uni_t[i] = np.array([pos[0] * 5, pos[1] * 5 + shift * (i % N_vals)])
        drawStates(t_matrix,uni_t,filename,doSpectrum, doIndivs, doBands, [1, min(4,len(ind_list)), min(8,len(ind_list))], lowE=0, highE=min(20,len(t_matrix)), S_mat=overlapMatrix)
    if computeOverlap:
        # tile eigenstates are in projectStates, projectSpectrum
        # compute the meta spectrum/states
        e_values, e_vecs = scipy_eigh(t_matrix, overlapMatrix)
        idx = e_values.argsort()[::-1]
        metaSpectrum = e_values[idx]
        metaStates = e_vecs[:, idx]

        # compute vertex states
        e_values, e_vecs = np.linalg.eigh(totalAdj)
        idx = e_values.argsort()[::-1]
        totalSpectrum = e_values[idx]
        totalStates = e_vecs[:,idx]

        # benchmarks sparse matrix operations on totalAdj
        doBenchmarking = False
        if doBenchmarking:
            start_time = time.time()
            eigenvalues_dense, eigvecs = scipy_eigh(totalAdj)
            end_time = time.time()
            print("Dense diagonalization time:\t"+str(end_time - start_time))
            start_time = time.time()
            sparseTAdj = csc_matrix(totalAdj)
            eigenvalues_sparse, eigvecs = scipy_eigsh(sparseTAdj)
            end_time = time.time()
            print("Sparse diagonalization time:\t"+str(end_time - start_time))

        # Sanity Check 1: looking at <Psi_tot_0|P_AH_totP_A|Psi_tot_0> vs <Psi_tot_0|H_tot|Psi_tot_0>
        # Sanity Check 2: looking at <Psi_tot_n|Psi_A_m>
        # Sanity Check 3: lookint at <Psi_A|H_tot|Psi_B> to see if the hopping values are constant (spoiler: not)
        doSanityChecks = False
        if doSanityChecks:
            print("Sanity Check 1")
            for state in range(1):
                for ind in ind_list:
                    Pi = projectors[ind] # get the projector of the ind tile
                    Psi_tot = totalStates[:,state] # in the state state
                    H = np.dot(np.dot(Pi, totalAdj), Pi) # Pi H Pi (totalAdj is H_tot)
                    intro = "<Psi_tot_"+str(state)+"|Pi_"+centers[ind][1]+"HPi_"+centers[ind][1] +"|Psi_tot_"+str(state) +">: "
                    print(intro +str(np.round(np.dot(np.conjugate(Psi_tot), np.dot(H, Psi_tot)),3))+"\t\t E_" + str(state) + ": " +str(np.round(totalSpectrum[state],3)))
            print("Sanity Check 2")
            for ind in ind_list:
                for n in range(1):
                    for m in range(N_vals):
                        print("<Psi_tot_"+str(n)+"|Psi_"+centers[ind][1]+"_" + str(m) + ">: "+ str(np.round(np.dot(np.conj(projectStates[ind][:,m]),totalStates[:,n]),3)))
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
                centerxyI,labelI,oriI = centers[i1]
                centerxyJ,labelJ,oriJ = centers[i2]
                for n in range(N_vals):
                    print(str(n)+": <" + labelI + str(i1)+"|H_tot|" + labelJ + str(i2)+">",e1,np.round(t_matrix[l1 * N_vals+n, l2 * N_vals+n], 4))
                #print(np.trace(np.dot(projectors[i1],projectors[i2])))

        # Hypothesis: each state is well-represented by the groundstate ie that <Psi_tot_0|P_AH_totP_A|Psi_tot_0>/<Psi_tot_0|H_tot|Psi_tot_0>
        # is roughly equal to <Psi_A_m|Psi_tot_n> (it sort of is)
        doHypothesis = False
        if doHypothesis:
            print("Hypothesis Testing:"+str(iList))
            for ind in ind_list:
                for state in range(3):
                    P_tile = projectors[ind]
                    Psi_tot = totalStates[:,state]
                    H = np.dot(np.dot(P_tile, totalAdj), P_tile)  # Pi H Pi (totalAdj is H_tot)
                    val2 = np.round(np.dot(np.conjugate(Psi_tot), np.dot(H, Psi_tot)) / totalSpectrum[state], 3)
                    for m in range(3):
                        val1 = np.round(np.square(np.dot(np.conj(projectStates[ind][:,m]),totalStates[:,state])),3)
                        print("Tile Index:"+str(ind)+"\tState:"+str(state)+"\tNval:"+str(m)+"\t\t"+str(val1)+"\t"+str(val2))

        # looking at the "tilenorm" = sum_i->dim(A) |<Psi_tot_0|Psi_A_i>|^2 versus "how completely the ground state captures"
        # |<Psi_tot_0|Psi_A_0>|^2/tilenorm
        doTileNorm = True
        if doTileNorm:
            tnorm = []
            for i,ind in enumerate(ind_list):
                sum = 0
                for j in range(len(projectors[ind])):
                    if abs(projectSpectra[ind][j]) < .00001:
                        continue
                    sum += np.square(np.abs(np.dot(np.conj(totalStates[:,0]),projectStates[ind][:,j])))
                state1 = np.square(np.abs(np.dot(np.conj(totalStates[:,0]),projectStates[ind][:,0])))
                print(str(ind) +" tilenorm = sum_i->"+str(size_list[i]) + " |<Psi_tot_0|Psi_" + str(ind) + "_i>|^2 = " + str(np.round(sum,3))+"\tcompleteness of first state = |<Psi_tot_0|Psi_" + str(ind) + "_i>|^2/tilenorm = " + str(np.round(state1/sum,3)))
                tnorm.append(sum)
            print("Total tilenorm:",str(np.round(np.sum(tnorm),3)))

        # TODO POSSIBLY BROKEN
        # <psi_tot|psi_{alpha}_{J}> = sum_i(sum_J(sum_alpha(<psi_tot|i><i|phi_j_alpha><phi_j_alpha|psi_tile>)))
        # looking at the overlap is kind of meaningless because not squared I think
        doOverlap = False
        if doOverlap:
            state = 0
            Tform = np.zeros((len(totalVertices), len(ind_list)*N_vals))
            for i in range(len(totalVertices)):
                for j in range(len(ind_list)*N_vals):
                    Tform[i,j] = projectStates[ind_list[j//N_vals]][i,j%N_vals]
            for state in range(N_vals):
                sum = np.dot(np.dot(totalStates[:, state], Tform), metaStates[:, state])
                print("OVERLAP in state " + str(state) + ": " + str(sum))

        # needs flux threading and magnetization to work TODO
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
                    P += np.outer(totalStates[:,state],np.conj(totalStates[:,state]))
            Q = np.identity(totalSize) - P
            C = 2*np.pi*1j*(np.dot(np.dot(np.dot(np.dot(P, X), Q), Y), P) - np.dot(np.dot(np.dot(np.dot(P, Y), Q), X), P))
            print(C)

        # actually computes the values that pasted into a dict earlier with t_i = <Psi_A_oriA|H_tot|Psi_B_oriB> (ie directly from t_matrix)
        if doGammaGammaScheme:
            for i,ind in enumerate(ind_list):
                adjRow = adj[ind]
                ones_indices = np.where((adjRow == 1))[0]
                centerxyI, labelI, oriI = centers[ind]
                for one in ones_indices:
                    if one not in ind_list:
                        continue
                    centerxyJ, labelJ, oriJ = centers[one]
                    R = Rangle(centerxyI,centerxyJ)
                    ul = ((oriI - R + 6) % 6, (oriJ - R + 6) % 6, (labelI == "Gamma"), (labelJ=="Gamma"))
                    psi_i = projectStates[ind][:, 0]
                    psi_j = projectStates[one][:, 0]
                    j1, i1 = matchPos(centerxyI, centerxyJ)
                    oriI1 = mirrorIndex((i1 - oriI + 6) % 6, recursions)
                    edgeI = edge_dict[labelI][oriI1]
                    val = np.round(np.dot(np.conjugate(psi_i), np.dot(totalAdj, psi_j)),3)
                    sval = np.round(np.dot(np.conj(psi_i),psi_j),3)
                    val = abs(val)
                    if ul in gdict:
                        gdict[ul].append((edgeI, val, sval))
                    else:
                        gdict[conjLabel(ul)].append((edgeI, val, sval))
            for key in gdict:
                print(key,gdict[key])

# solve the states for all vertices using H_tot (totalAdJ)
filename = "Patch" + str(iList) + "_Connect" + str(wrapConnect)
if doVertexStates and not hexOnly:
    drawStates(totalAdj, totalVertices, filename, doSpectrum, doIndivs, doBands, bandList, 0,
               stateNum)
