import numpy as np
from sympy import symbols, Eq, nsolve, solve, Matrix, lambdify
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigsh as scipy_eigsh
import time
from scipy.linalg import null_space
from scipy.linalg import eigh as scipy_eigh
from scipy.linalg import eigh as scipy_eig
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist
from scipy import signal
from scipy.signal import argrelextrema, find_peaks, argrelmin
from scipy.stats import uniform


import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import FancyArrowPatch
import math
import matplotlib
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from collections import namedtuple
from matplotlib.artist import Artist
import matplotlib.transforms as mtransforms
import matplotlib.patches as patches
import random
ident = [1,0,0,0,1,0]
s32 = np.sqrt(3) / 2
rad = 1
tile_names = {
    0:'Gamma',
    1:'Delta',
    2:'Theta',
    3:'Lambda',
    4:'Xi',
    5:'Pi',
    6:'Sigma',
    7:'Phi',
    8:'Psi'
}
tile_indices = {v: k for k, v in tile_names.items()}
match_dict = {
    'a+':'a-',
    'a-':'a+',
    'b+':'b-',
    'b-':'b+',
    'g+':'g-',
    'g-':'g+',
    'd+':'d-',
    'd-':'d+',
    'ep+':'ep-',
    'ep-':'ep+',
    'z+':'z-',
    'z-':'z+',
    't+':'t-',
    't-':'t+',
    'eta':'eta'
}
gamma_dict = {
    0: [5,4,3,65,64,63],    # Blank list
    1: [5,6,7,8,9,10],    # Blank list
    2: [27,26,25,24,20,19,18,17,16,11,10],    # Blank list
    3: [47,46,45,38,37,32,31,30,29,28,27],    # Blank list
    4: [47,48,49,50,51,52,55,56],    # Blank list
    5: [63,70,69,60,59,58,57,56]     # Blank list
}
delta_dict = {
    0: [6,5,4,3,72,71,70],    # Blank list
    1: [26,25,24,17,16,11,10,9,8,7,6],    # Blank list
    2: [26,27,28,29,30,31,34,35],    # Blank list
    3: [53,52,45,44,39,38,37,36,35],    # Blank list
    4: [53,54,55,56,57,58],    # Blank list
    5: [58,59,62,63,64,65,66,67,76,77,70]     # Blank list
}
theta_dict = {
    0: [5,4,3,72,71,70],    # Blank list
    1: [5,6,7,8,9,10,11,16,17,24,25],    # Blank list
    2: [25,26,27,28,29,30],    # Blank list
    3: [30,31,34,35,36,37,38,39,44,45,52,53],    # Blank list
    4: [53,54,55,56,57,58],    # Blank list
    5: [58,59,62,63,64,65,66,67,76,77,70]     # Blank list
}
lambda_dict = {
    0: [5,4,3,72,71,70],    # Blank list
    1: [26,25,24,17,16,11,10,9,8,7,6,5],    # Blank list
    2: [26,27,28,29,30,31,34,35],    # Blank list
    3: [53,52,45,44,39,38,37,36,35],    # Blank list
    4: [53,54,55,56,57,58],    # Blank list
    5: [58,59,62,63,64,65,66,67,76,77,70]     # Blank list
}
xi_dict = {
    0: [5,4,3,72,71,70],    # Blank list
    1: [5,6,7,8,9,10,11,16,17,24,25],    # Blank list
    2: [25,26,27,28,29,30],    # Blank list
    3: [30,31,34,35,36,37,38,39,44,45,52,53],    # Blank list
    4: [53,54,55,56,57,58,59,62,63],    # Blank list
    5: [70,77,76,67,66,65,64,63]
}
pi_dict = {
    0: [5,4,3,72,71,70],    # Blank list
    1: [26,25,24,17,16,11,10,9,8,7,6,5],    # Blank list
    2: [26,27,28,29,30,31,34,35],    # Blank list
    3: [53,52,45,44,39,38,37,36,35],    # Blank list
    4: [53,54,55,56,57,58,59,62,63],    # Blank list
    5: [70,77,76,67,66,65,64,63]     # Blank list
}
sigma_dict = {
    0: [66,67,76,77,70,71,72,3,4,5,6],    # Blank list
    1: [26,25,24,17,16,11,10,9,8,7,6],    # Blank list
    2: [26,27,28,29,30,31,34,35],    # Blank list
    3: [53,52,45,44,39,38,37,36,35],    # Blank list
    4: [53,54,55,56,57,58],    # Blank list
    5: [58,59,62,63,64,65,66]     # Blank list
}
phi_dict = {
    0: [5,4,3,72,71,70],    # Blank list
    1: [5,6,7,8,9,10,11,16,17,24,25],    # Blank list
    2: [25,26,27,28,29,30,31,34,35],    # Blank list
    3: [53,52,45,44,39,38,37,36,35],    # Blank list
    4: [53,54,55,56,57,58],    # Blank list
    5: [58,59,62,63,64,65,66,67,76,77,70]     # Blank list
}
psi_dict = {
    0: [5,4,3,72,71,70],    # Blank list
    1: [5,6,7,8,9,10,11,16,17,24,25],    # Blank list
    2: [25,26,27,28,29,30,31,34,35],    # Blank list
    3: [53,52,45,44,39,38,37,36,35],    # Blank list
    4: [53,54,55,56,57,58,59,62,63],    # Blank list
    5: [70,77,76,67,66,65,64,63]     # Blank list
}
edge_index_dict = {
    'Gamma':gamma_dict,
    'Delta':delta_dict,
    'Theta':theta_dict,
    'Lambda':lambda_dict,
    'Xi':xi_dict,
    'Pi':pi_dict,
    'Sigma':sigma_dict,
    'Phi':phi_dict,
    'Psi':psi_dict
}
edge_dict = {
    'Gamma': ['b-', 'b+', 'd-', 'g-', 'a+', 'a-'],
    'Delta': ['z-', 'g-', 'a+', 'ep-', 'b+', 'g+'],
    'Theta': ['b-', 'eta', 'b+', 't+', 'b+', 'g+'],
    'Lambda': ['b-', 't-', 'a+', 'ep-', 'b+', 'g+'],
    'Xi': ['b-', 'eta', 'b+', 't+', 'ep+', 'a-'],
    'Pi': ['b-', 't-', 'a+', 'ep-', 'ep+', 'a-'],
    'Sigma': ['d+', 'g-', 'a+', 'ep-', 'b+', 'z+'],
    'Phi': ['b-', 'eta', 'ep+', 'ep-', 'b+', 'g+'],
    'Psi': ['b-', 'eta', 'ep+', 'ep-', 'ep+', 'a-']
}
Balance = namedtuple('Balance', ['a','b','g','d','e','z','eta','t'])
s3 = np.sqrt(3)
edge_pos_dict = {
    0:[-3*rad/4,s3*rad/4],
    1:[0,s3*rad/2],
    2:[3*rad/4,s3*rad/4],
    3:[3*rad/4,-s3*rad/4],
    4:[0,-s3*rad/2],
    5:[-3*rad/4,-s3*rad/4]
}
edge_bal = { #      a b g d e z eta t
    'Gamma': Balance(0,0,-1,-1,0,0,0,0),
    'Delta': Balance(1,1,0,0,-1,-1,0,0),
    'Theta': Balance(0,1,1,0,0,0,1,1),
    'Lambda': Balance(1,0,1,0,-1,0,0,-1),
    'Xi': Balance(-1,0,0,0,1,0,1,1),
    'Pi': Balance(0,-1,0,0,0,0,0,-1),
    'Sigma': Balance(1,1,-1,1,-1,1,0,0),
    'Phi': Balance(0,0,1,0,0,0,1,0),
    'Psi': Balance(-1,-1,0,0,1,0,1,0)
}
abgd = np.array([[0,1,0,1,-1,0,1,0,-1],#a # just the dict above but transposed into a matrix form
[0,1,1,0,0,-1,1,0,-1],# b
[-1,0,1,1,0,0,-1,1,0], # g
[-1,0,0,0,0,0,1,0,0],# d
[0,-1,0,-1,1,0,-1,0,1], # e
[0,-1,0,0,0,0,1,0,0],# z
[0,0,1,0,1,0,0,1,1], # eta
[0,0,1,-1,1,-1,0,0,0]])
Greek = {
    'Gamma':"$\Gamma$",
    'Delta': "$\Delta$",
    'Theta': "$\Theta$",
    'Lambda': "$\Lambda$",
    'Xi': "$\Xi$",
    'Pi': "$\Pi$",
    'Sigma': "$\Sigma$",
    'Phi': "$\Phi$",
    'Psi': "$\Psi$",
    'a+': "$\\alpha$+",
    'a-': "$\\alpha$-",
    'b+': '$\\beta$+',
    'b-': '$\\beta$-',
    'g+': '$\gamma$+',
    'g-': '$\gamma$-',
    'd+': '$\delta$+',
    'd-': '$\delta$-',
    'ep+': '$\epsilon$+',
    'ep-': '$\epsilon$-',
    'z+': '$\zeta$+',
    'z-': '$\zeta$-',
    't+': '$\\theta$+',
    't-': '$\\theta$-',
    'eta': '$\eta$'
}
alpha_dict = {
    'Gamma': .1,
    'Delta': .2,
    'Theta': .3,
    'Lambda': .4,
    'Xi': .5,
    'Pi': .6,
    'Sigma': .7,
    'Phi': .8,
    'Psi': .9
} # THIS IS STUPID: the alpha of the polygon holds its type information because of stupid printing reasons
rev_alpha_dict = {v: k for k, v in alpha_dict.items()}
colmap = {
    'Gamma': [.5, .5, .5],
    'Gamma1': [0.38, 0.22, 0.08],
    'Gamma2': [0.0, 0.0, 0.0],
    'Delta': [0.01, 0.51, 0.13],
    'Theta': [0.0, 0.30, 1.0],
    'Lambda': [0.46, 0.0, 0.53],
    'Xi': [0.90, 0.0, 0.0],
    'Pi': [1.0, 0.69, 0.78],
    'Sigma': [0.45, 0.84, 0.93],
    'Phi': [1.0, 0.55, 0.0],
    'Psi': [.1, .1, 0.0]
}
# loads in 15 types - typed in manually from computeOverlap down below
bind_dict = {
    (1, 0, False, False): .009,
    (2, 0, False, True): .012,
    (3, 1, False, False): .017,
    (5, 5, False, False): .024,
    (0, 5, False, False): .05,
    (1, 5, False, False): .02,
    (2, 1, False, True): .051,
    (1, 2, False, True): .097,
    (0, 3, False, False): .063,
    (5, 1, False, False): .033,
    (1, 3, False, True): .02,
    (2, 2, True, False): .025,
    (1, 2, True, False): .016,
    (4, 3, False, False): .072,
    (1, 0, True, False): .009}
sbind_dict = {
    (1, 0, False, False): .003,
    (2, 0, False, True): .004,
    (3, 1, False, False): .006,
    (5, 5, False, False): .008,
    (0, 5, False, False): .018,
    (1, 5, False, False): .007,
    (2, 1, False, True): .019,
    (1, 2, False, True): .036,
    (0, 3, False, False): .023,
    (5, 1, False, False): .012,
    (1, 3, False, True): .007,
    (2, 2, True, False): .009,
    (1, 2, True, False): .005,
    (4, 3, False, False): .027,
    (1, 0, True, False): .003}
def pt(x, y):
    return {'x': x, 'y': y}
def inv(T):
    det = T[0]*T[4] - T[1]*T[3]
    return [T[4]/det, -T[1]/det, (T[1]*T[5]-T[2]*T[4])/det,
            -T[3]/det, T[0]/det, (T[2]*T[3]-T[0]*T[5])/det]
def mul(A, B):
    return [A[0]*B[0] + A[1]*B[3],
            A[0]*B[1] + A[1]*B[4],
            A[0]*B[2] + A[1]*B[5] + A[2],
            A[3]*B[0] + A[4]*B[3],
            A[3]*B[1] + A[4]*B[4],
            A[3]*B[2] + A[4]*B[5] + A[5]]
def padd(p, q):
    return {'x': p['x'] + q['x'], 'y': p['y'] + q['y']}
def psub(p, q):
    return {'x': p['x'] - q['x'], 'y': p['y'] - q['y']}
def trot(ang):
    c = np.cos(ang)
    s = np.sin(ang)
    return [c, -s, 0, s, c, 0]
def ttrans(tx, ty):
    return [1, 0, tx, 0, 1, ty]
def transTo(p, q):
    return ttrans(q['x'] - p['x'], q['y'] - p['y'])
def rotAbout(p, ang):
    return mul(ttrans(p.x, p.y), mul(trot(ang), ttrans(-p.x, -p.y)))
def transPt(M, P):
    return pt(M[0]*P['x'] + M[1]*P['y'] + M[2], M[3]*P['x'] + M[4]*P['y'] + M[5])
def draw_transformed_patch(patch, transformation,ax):
    c = patch.get_center()
    r = patch.get_radius()
    t2 = mul(transformation,mul(ttrans(1.5,-np.sqrt(3)/2),trot(-np.pi/2)))
    nc = transPt(t2, {'x': c[0], 'y': c[1]})
    npatch = plt.Circle((nc['x'],nc['y']),r,facecolor='black',edgecolor='black')
    Artist.set_transform(npatch,mtransforms.Affine2D())
    plt.gca().add_patch(npatch)
def drawPolygon(shape, T,ax):
    points = []
    for p in shape:
        tp = transPt(T,p)
        points.append([tp['x'],tp['y']])
    polygon = patches.Polygon(points, edgecolor='black', facecolor='lightgray')
    Artist.set_transform(polygon,mtransforms.Affine2D())
    plt.gca().add_patch(polygon)
class Shape:
    def __init__(self, pts, quad, label):
        self.pts = pts
        self.quad = quad
        self.label = label
    def draw(self, S,ax,poly,patchlist,doOriginal):
        if not doOriginal:
            self.drawPoly2(S,ax)
        elif poly:
            drawPolygon(self.pts, S,ax)
        for patch in patchlist:
            draw_transformed_patch(patch,S,ax)
    def drawPoly2(self,T, ax):
        points = []
        R = [-1, 0, 0, 0, 1, 0]
        for p in self.pts:
            tp = transPt(mul(R,T), p)
            points.append([tp['x'], tp['y']])
        polygon = patches.Polygon(points, edgecolor='black', facecolor='lightgray',alpha=alpha_dict[self.label])
        Artist.set_transform(polygon, mtransforms.Affine2D())
        plt.gca().add_patch(polygon)
class Meta:
    def __init__(self):
        self.geoms = []
        self.quad = []
    def addChild(self, g, T):
        self.geoms.append({'geom': g, 'xform': T})
    def draw(self, S,ax,poly,patchlist,doOriginal):
        for g in self.geoms:
            g['geom'].draw(mul(S, g['xform']),ax,poly,patchlist,doOriginal)
def buildSpectreBase(curved):
    spectre = [
        pt(0, 0),
        pt(1.0, 0.0),
        pt(1.5, -0.8660254037844386),
        pt(2.366025403784439, -0.36602540378443865),
        pt(2.366025403784439, 0.6339745962155614),
        pt(3.366025403784439, 0.6339745962155614),
        pt(3.866025403784439, 1.5),
        pt(3.0, 2.0),
        pt(2.133974596215561, 1.5),
        pt(1.6339745962155614, 2.3660254037844393),
        pt(0.6339745962155614, 2.3660254037844393),
        pt(-0.3660254037844386, 2.3660254037844393),
        pt(-0.866025403784439, 1.5),
        pt(0.0, 1.0)
    ]
    spectre_keys = [
        spectre[3], spectre[5], spectre[7], spectre[11]
    ]
    ret = {}
    for lab in ['Delta', 'Theta', 'Lambda', 'Xi', 'Pi', 'Sigma', 'Phi', 'Psi']:
        ret[lab] = Shape(spectre, spectre_keys, lab)
    mystic = Meta()
    mystic.addChild(Shape(spectre, spectre_keys, 'Gamma1'), ident)
    mystic.addChild(Shape(spectre, spectre_keys, 'Gamma2'),
                        mul(ttrans(spectre[8]['x'], spectre[8]['y']), trot(math.pi / 6)))
    mystic.quad = spectre_keys
    ret['Gamma'] = mystic
    return ret
def buildSupertiles(sys):
    # First, use any of the nine-unit tiles in sys to obtain
    # a list of transformation matrices for placing tiles within
    # supertiles.
    quad = sys['Psi'].quad
    R = [-1, 0, 0, 0, 1, 0]
    t_rules = [
        [60, 3, 1], [0, 2, 0], [60, 3, 1], [60, 3, 1],
        [0, 2, 0], [60, 3, 1], [-120, 3, 3]
    ]
    Ts = [ident]
    total_ang = 0
    rot = ident
    tquad = list(quad)
    for ang, from_, to in t_rules:
        total_ang += ang
        if ang != 0:
            rot = trot(math.radians(total_ang))
            for i in range(4):
                tquad[i] = transPt(rot, quad[i])
        ttt = transTo(tquad[to], transPt(Ts[-1], quad[from_]))
        Ts.append(mul(ttt, rot))
    for idx in range(len(Ts)):
        Ts[idx] = mul(R, Ts[idx])
    # Now build the actual supertiles, labeling appropriately.
    super_rules = {
        'Gamma': ['Pi', 'Delta', 'null', 'Theta', 'Sigma', 'Xi', 'Phi', 'Gamma'],
        'Delta': ['Xi', 'Delta', 'Xi', 'Phi', 'Sigma', 'Pi', 'Phi', 'Gamma'],
        'Theta': ['Psi', 'Delta', 'Pi', 'Phi', 'Sigma', 'Pi', 'Phi', 'Gamma'],
        'Lambda': ['Psi', 'Delta', 'Xi', 'Phi', 'Sigma', 'Pi', 'Phi', 'Gamma'],
        'Xi': ['Psi', 'Delta', 'Pi', 'Phi', 'Sigma', 'Psi', 'Phi', 'Gamma'],
        'Pi': ['Psi', 'Delta', 'Xi', 'Phi', 'Sigma', 'Psi', 'Phi', 'Gamma'],
        'Sigma': ['Xi', 'Delta', 'Xi', 'Phi', 'Sigma', 'Pi', 'Lambda', 'Gamma'],
        'Phi': ['Psi', 'Delta', 'Psi', 'Phi', 'Sigma', 'Pi', 'Phi', 'Gamma'],
        'Psi': ['Psi', 'Delta', 'Psi', 'Phi', 'Sigma', 'Psi', 'Phi', 'Gamma']
    }
    super_quad = [
        transPt(Ts[6], quad[2]),
        transPt(Ts[5], quad[1]),
        transPt(Ts[3], quad[2]),
        transPt(Ts[0], quad[1])
    ]
    ret = {}
    for lab, subs in super_rules.items():
        sup = Meta()
        for idx in range(8):
            if subs[idx] == 'null':
                continue
            sup.addChild(sys[subs[idx]], Ts[idx])
        sup.quad = super_quad
        ret[lab] = sup
    return ret
def calculate_bezier_points(p0, p1, p2, p3, t):
    # Calculate the Bezier spline points
    x = (1 - t) ** 3 * p0[0] + 3 * (1 - t) ** 2 * t * p1[0] + 3 * (1 - t) * t ** 2 * p2[0] + t ** 3 * p3[0]
    y = (1 - t) ** 3 * p0[1] + 3 * (1 - t) ** 2 * t * p1[1] + 3 * (1 - t) * t ** 2 * p2[1] + t ** 3 * p3[1]
    return x, y
def control_points(curve,parity,ps,pe):
    x = pe[0]-ps[0]
    y = pe[1]-ps[1]
    nx = y
    ny = x
    p0 = [-curve*nx+x/2+ps[0], curve*ny+y/2+ps[1]] # double curve
    p1 = [ curve*nx+x/2+ps[0],  -curve*ny+y/2+ps[1]] # double curve
    parity = 1-parity
    p0 = [(-2*parity+1)*curve*nx + x/2 + ps[0], (2*parity-1)*curve*ny + y/2 + ps[1]]  # single curve
    p1 = [(-2*parity+1)*curve*nx + x/2 + ps[0], (2*parity-1)*curve*ny + y/2 + ps[1]]  # single curve
    return p0,p1
def generate_bezier_points(points,curve,density,yesPlot,thickness):
    xpoints = []
    ypoints = []
    for i in range(len(points)):
        ps = points[i]
        pe = points[(i + 1) % len(points)]
        num_points = density
        t_values = np.linspace(0, 1, num_points)
        p0, p1 = control_points(curve,i % 2, ps, pe)
        x_points, y_points = calculate_bezier_points(ps, p0, p1, pe, t_values)
        if yesPlot:
            plt.gca().plot(x_points, y_points,c='black',linewidth=thickness)
        xpoints = np.append(xpoints,x_points)
        ypoints = np.append(ypoints,y_points)
    return xpoints,ypoints
def generate_points(a,b,dx,dy,r):
    c = np.cos(np.pi / 3)
    s = np.sin(np.pi / 3)
    moves = [(c * b, s * b), (b, 0), (0, a), (s * a, c * a), (c * b, -s * b), (-c * b, -s * b), (s * a, -c * a),
             (0, -a), (0, -a), (-s * a, -c * a), (-c * b, s * b), (-b, 0), (0, a), (-s * a, c * a)]
    accumulated_moves = [(sum([move[0] for move in moves[:i + 1]]), sum([move[1] for move in moves[:i + 1]])) for i in
                         range(len(moves))]
    accumulated_moves=rotate_points(accumulated_moves,r)
    accumulated_moves+=[dx,dy]
    return accumulated_moves
def rotate_points(points, theta):
    # Define the 2D rotation matrix
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
    # Multiply each pair in the points array by the rotation matrix
    rotated_points = np.dot(points, rotation_matrix)
    return rotated_points
def reflect_points(points):
    mat = np.array([[-1,0],[0,1]])
    return np.dot(points,mat)
def unitdist(point1, point2,d,tol):
    return abs(np.sqrt(np.square(point1[0] - point2[0]) + np.square(point1[1] - point2[1])) - d) < tol
def align_shapes(N, M, i1, i2, j1, j2):
    translation_vector = M[j1] - N[i1]
    # plt.scatter(M[j1][0],M[j1][1],color='red')
    # plt.scatter(N[i1][0],M[j1][1],color='red')
    M -= translation_vector
    angle = (np.arctan2(M[j2][1] - M[j1][1], M[j2][0] - M[j1][0]) - np.arctan2(N[i2][1] - N[i1][1], N[i2][0] - N[i1][0]))
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    M_aligned = np.dot(M - N[i1], rotation_matrix) + N[i1]
    return M_aligned
def map_value_to_color(value):
    max = .2
    normalized_value = (value / max + 1) / 2.0
    color = cm.bwr(normalized_value)
    return color