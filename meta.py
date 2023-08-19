import numpy as np
import os
from translated import *
from metasolve import *


def mirrorIndex(i, recs):
    if recs % 2 == 1:
        return i
    else:
        return 5 - i
def matchIndex(i):
    return (i + 3) % 6
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
def computeEdgeBalanceIndiv(ind):
    return edge_bal[centers[ind][1]]
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
def checkBal(ibal):
    return (ibal[0] == 0 and ibal[1] == 0 and ibal[2] == 0 and ibal[3] == 0 and ibal[4] == 0 and ibal[5] == 0 and
            ibal[6] % 2 == 0 and ibal[7] == 0)
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
        positions.extend([0]*max(0,int(np.sum(np.abs(e_values) < .0001)-np.sum(np.sum(adj, axis=1) == 0)))) # properly accounts for zero modes that aren't just vacuum states
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
# SYMPY SOLUTION that finds what kinds of tile combinations balance out to map to self
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
# REQUIRED: Generate Hexagons into "centers" (tuple:(center,label,orientation)) list, "adj" adjacency matrix (indexed by centers)
superTileType = "Psi"
recursions = 3  # MUST BE ODD
doWrap = False  # wraparound edges
wrapConnect = True  # actually connect tiles

showEdgeLabels = True
yesRotate = True
showPatchOnly = True
showHexagons = False
showVertices = True

doProjections = True
drawProjectors = False  # Writes
drawOverlap = False # DON'T TURN ON
computeOverlap = True
loadState = False  # if false, saves state, if true, loads state
stateName = "States_Patch5_N10.pkl"  # if true what name it loads in to compare
N_vals = 15

doA2222Scheme = False



doVertexStates = False  # Writes
doSpectrum = True
doIndivs = True
doBands = True
stateNum = 15  # number of individual states to draw
bandList = [1, 4, 8, 16, 24, 32]
iList = 5

if iList == 0:
    ind_list = [450, 456, 471, 449, 451, 470, 469, 463, 113]  # WORKING LIST FOR DELTA TILE
elif iList == 1:
    ind_list = [102, 118, 112, 120, 121, 101, 471, 71, 95, 449, 470, 119, 115, 469, 117, 457, 113, 111]  # WORKING LIST FOR DELTA TILE
elif iList == 2:  # 2
    ind_list = [450, 449]
elif iList == 3:  # line zig zag
    ind_list = [466, 469, 113, 457, 111, 459, 125, 124, 69, 70, 64, 237, 244, 221]
elif iList == 4:  # 7 patch center of 5, 9
    # ind_list = [313, 314, 308, 300, 460, 458, 457, 470, 471, 456, 453, 452] only outsides of 7 patch
    ind_list = [451, 462, 461, 464, 449, 450, 463]
elif iList == 5:  # hexagon patch (4 is just the inside)
    ind_list = [313, 314, 308, 300, 460, 458, 457, 470, 471, 456, 453, 452, 451, 462, 461, 464, 449, 450, 463]
elif iList == 6:  # straight line
    ind_list = [438, 439, 443, 255, 256, 258, 313, 314, 308, 307, 304, 68]
elif iList == 7:  # etas + surroundings (7patch)
    ind_list = [457, 111, 112, 113, 114, 115, 118]
elif iList == 8:
    ind_list = [457, 111]
elif iList == 9:
    ind_list = [313, 314, 308, 300, 460, 458, 457, 470, 471, 456, 453, 452, 451, 462, 461, 464, 449, 450, 463,
                312,311,309,284,307,301,302,459,111,112,113,469,472,491,455,454,257,258]
# hexagon generation and display of hexagons (adjacency matrix in adj, others in centers = (position, name, orientation)
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

# Main program generating tiles, connecting vertices physically, reconnecting the graph
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
                    matches.append(((i, oriI1), (one, (j1 - oriJ + 6) % 6)))
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
    print("Block diagonal sizes:",size_list)
    # build vertex list in the correct locations (includes duplicates), labelled in same index order as totalAdj
    totalVertices = [np.array(all_unique[centers[ind_list[0]][1]])]  # starting
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
            t1 = centers[i1][1]
            e1 = edge_dict[t1][o1]
            p1 = edge_index_dict[t1][o1]
            block1 = blockoffsets[ind_list.index(i1)]
            i2, o2 = m2
            t2 = centers[i2][1]
            e2 = edge_dict[t2][o2]
            p2 = edge_index_dict[t2][o2]
            block2 = blockoffsets[ind_list.index(i2)]
            print("Edge Connection: ", t1, e1, t2, e2)
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
# DRAWING the points
if showVertices:
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


# run nearest neighbor stuff from penrose paper on the metatile lattice
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
    sy["e_Gamma"] = e_Gamma
    sy["e_Delta"] = e_Delta
    sy["e_Theta"] = e_Theta
    sy["e_Lambda"] = e_Lambda
    sy["e_Xi"] = e_Xi
    sy["e_Pi"] = e_Pi
    sy["e_Sigma"] = e_Sigma
    sy["e_Phi"] = e_Phi
    sy["e_Psi"] = e_Psi
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
        initial_guess = [90.0] * 27
        initial_guess[0]=1
        initial_guess.extend([0]*10)
        solutions2 = fsolve(calculate_equations, initial_guess,xtol=1e-9,fprime=jacobian_function)
        for i, value in enumerate(solutions2):
            variable_name = \
            ["E","Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Phi", "Psi", "t_a", "t_b", "t_g", "t_d", "t_ep",
             "t_z", "t_t", "t_eta", "e_Gamma", "e_Delta", "e_Theta", "e_Lambda", "e_Xi", "e_Pi", "e_Sigma", "e_Phi",
             "e_Psi","Null1","Null1","Null1","Null1","Null1","Null1","Null1","Null1","Null1","Null1"][i]
            print(f"{variable_name} =", value)
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

# SHOW PROJECTED STATES
# build projection matrices and find states (solve PiHPi |psi> = E|psi>)
if doProjections:
    projectors = dict()
    projectSpectra = dict()
    projectStates = dict()
    # find states that are not referenced at all, used to verify the identity matrix (otherwise would be 0 and fail), basically the overwritten vertices
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
    t_matrix[t_matrix < 1e-4] =  0  # kills small values
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
        # <psi_tot|psi_{alpha}_{J}> = sum_i(sum_J(sum_alpha(<psi_tot|i><i|phi_j_alpha><phi_j_alpha|psi_tile>)))
        # compute the meta spectrum/states
        e_values, e_vecs = scipy_eigh(t_matrix, overlapMatrix)
        idx = e_values.argsort()[::-1]
        metaSpectrum = e_values[idx]
        metaStates = e_vecs[:, idx]

        doBenchmarking = False # benchmarks on totalAdj
        if doBenchmarking:
            sparseTAdj = csc_matrix(totalAdj)
            start_time = time.time()
            eigenvalues_dense, eigvecs = scipy_eigh(totalAdj)
            end_time = time.time()
            print("Dense diagonalization time:\t"+str(end_time - start_time))
            start_time = time.time()
            eigenvalues_sparse, eigvecs = scipy_eigsh(sparseTAdj)
            end_time = time.time()
            print("Sparse diagonalization time:\t"+str(end_time - start_time))
        # compute vertex states
        e_values, e_vecs = np.linalg.eigh(totalAdj)
        idx = e_values.argsort()[::-1]
        totalSpectrum = e_values[idx]
        totalStates = e_vecs[:,idx]

        doSanityChecks = False
        if doSanityChecks:
            print("Sanity Check 1")
            for state in range(1):
                for ind in ind_list:
                    Pi = projectors[ind] # get the projector of the ind tile
                    Psi_tot = totalStates[:,state] # in the state state
                    H = np.dot(np.dot(Pi, totalAdj), Pi) # Pi H Pi (totalAdj is H_tot)
                    intro = "<Phi_tot_"+str(state)+"|Pi_"+centers[ind][1]+"HPi_"+centers[ind][1] +"|Phi_tot_"+str(state) +">: "
                    print(intro +str(np.round(np.dot(np.conjugate(Psi_tot), np.dot(H, Psi_tot)),3))+"\t\t E_" + str(state) + ": " +str(np.round(totalSpectrum[state],3)))
            print("Sanity Check 2")
            for ind in ind_list:
                for n in range(1):
                    for m in range(N_vals):
                        print("<Phi_tot_"+str(n)+"|Psi_"+centers[ind][1]+"_" + str(m) + ">: "+ str(np.round(np.dot(np.conj(projectStates[ind][:,m]),totalStates[:,n]),3)))
        print("Hypothesis Testing:"+str(iList))
        for ind in ind_list:
            for state in range(1):
                m = 0
                val1 = np.round(np.square(np.dot(np.conj(projectStates[ind][:,m]),totalStates[:,state])),3)
                Psi_tot = totalStates[:,state]
                P_tile = projectors[ind]
                H = np.dot(np.dot(P_tile, totalAdj), P_tile) # Pi H Pi (totalAdj is H_tot)
                val2 = np.round(np.dot(np.conjugate(Psi_tot), np.dot(H, Psi_tot))/totalSpectrum[state],3)
                print(ind,val1,val2)
        for i,indi in enumerate(ind_list):
            for j,indj in enumerate(ind_list):
                if i != j:
                    pass

        state = 0
        Tform = np.zeros((len(totalVertices),len(ind_list)*N_vals))
        for i in range(len(totalVertices)):
            for j in range(len(ind_list)*N_vals):
                Tform[i,j] = projectStates[ind_list[j//N_vals]][i,j%N_vals]
        for state in range(N_vals):
            sum = np.dot(np.dot(totalStates[:,state], Tform), metaStates[:,state])
            print("OVERLAP in state " + str(state) + ": " + str(sum))
    if drawOverlap:
        # DO NOT RUN HIGHLY FLAWED AND USELESS FUNCTION
        if not os.path.exists(filename):
            os.makedirs(filename)
        figi, axi = plt.subplots(1, 1)
        plt.cla()
        e_values, e_vecs = scipy_eigh(t_matrix,overlapMatrix)
        idx = e_values.argsort()[::-1]
        metaSpectrum = e_values[idx]
        metaStates = e_vecs[:, idx]
        #normalization
        for i in range(len(metaStates)):
            norm = np.dot(np.conjugate(metaStates[:,i]),metaStates[:,i])
            metaStates[:,i] /= np.sqrt(norm)
        # either load state from file or save it to memory
        if loadState:
            with open(stateName, 'rb') as f:
                meta = pickle.load(f)
                compInds = meta["inds"]
                compNV = meta["N_vals"]
                if compNV != N_vals:
                    print("FATAL MISMATCH IN N:",N_vals)
                    exit()
                compSpec = meta["spectrum"]
                compStates = meta["states"]
            sameInds = list(set(ind_list) & set(compInds))
            state = 0
            if state==0:
                plt.close()
                figi, axi = plt.subplots(1, 1)
                for same in sameInds:
                    showHex(same, True, True, axi, ax)

                for same in sameInds:
                    # compute dot product between the values
                    i1 = ind_list.index(same)*N_vals
                    psi_1 = metaStates[i1:i1+N_vals,state]
                    i2 = compInds.index(same)*N_vals
                    psi_2 = compStates[i2:i2+N_vals,state]
                    olap = np.dot(np.conjugate(psi_1),psi_2)/(np.linalg.norm(psi_1)*np.linalg.norm(psi_2)) # TODO mickey mouse
                    # plot circle
                    c1 = centers[same][0]
                    circ = plt.Circle((c1[0], c1[1]), .3, facecolor=map_value_to_color(olap / 5), edgecolor='black')
                    plt.gca().add_patch(circ)
                plt.axis('tight')
                axi.set_aspect('equal')
                plt.setp(axi, xticks=[], yticks=[])
                plt.savefig(str(filename) + "\\Overlap" + str(len(sameInds))+"_State" + str(state)+"_N"+str(N_vals)+".pdf")
        else:
            with open("States_Patch"+str(iList)+"_N" +str(N_vals)+'.pkl', 'wb') as f:
                stateFile = dict()
                stateFile["inds"] = ind_list
                stateFile["N_vals"] = N_vals
                stateFile["spectrum"] = metaSpectrum
                stateFile["states"] = metaStates
                pickle.dump(stateFile, f)

# solve the states for ALL vertices and full adjacency
filename = "Patch" + str(iList) + "_Connect" + str(wrapConnect)
if doVertexStates:
    drawStates(totalAdj, totalVertices, filename, doSpectrum, doIndivs, doBands, bandList, 0,
               stateNum)
