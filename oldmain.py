from matplotlib.widgets import Slider
from translated import *
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
#HYPERPARAMETERS
plotBorders = True
linethickness = 1
linedensity = 10
recursions = 1
tile_sel = 'Psi'
plotCO = True


def calculate_snap_value(val):
    snap_values = [1, np.sqrt(3)]
    thresh = .3
    for snap in snap_values:
        if abs(val-snap) < thresh:
            return snap
    return val
def get_start(angles,a):
    x = 0
    y = 0
    for angle in angles:
        angle = float(angle)
        angle *= np.pi / 180
        x+=a*np.cos(angle)
        y+=a*np.sin(angle)
    return x,y
def get_startab(langles):
    x = 0
    y = 0
    for langle in langles:
        angle = float(langle[0])*np.pi/180
        dist = float(langle[1])
        x+=dist*np.cos(angle)
        y+=dist*np.sin(angle)
    return x,y
def old_draw_moves(a,b,curve):
    shift1 = get_startab([(-30,a),(-90,a),(0,b),(-60,b),(30,a),(90,2*a),(150,a)])#get_startab([(60,b),(0,b),(90,a),(30,b)])
    tiling = [[0,0,0],
              [shift1[0],shift1[1],0]
              ]
    shift1 = get_start([60,0,90,30],a)
    shift2 = get_start([-30,-90,0,-60,30,90,0,60,-30,30],a)
    shift3 = get_start([60,0],a)
    shift4 = get_start([60,120,210,-90,180,240],a)
    shift5 = get_start([-30,-90,-90,-30,240,-60],a)
    shift6 = get_start([60,0,90,30], a)
    tiling = [  [0,0,0],
                [shift1[0],shift1[1],np.pi/6],
                [shift2[0],shift2[1],4*np.pi/6],
                [shift3[0],shift3[1],-4*np.pi/6],
                [shift4[0],shift4[1],0],
                [shift5[0],shift5[1],-2*np.pi/6],
                [shift6[0],shift6[1],-2*np.pi/6]
                ]
    for tile in tiling:
        plot_point_set(generate_points(a,b,tile[0],tile[1],tile[2]),curve)
def set_nearest_multiple_of_round(ax, points,round):
    max_x,max_y,min_x,min_y = points
    xlim_lower = -round+np.floor(min_x / round) *round
    ylim_lower = -round+np.floor(min_y / round) * round
    xlim_upper = round+np.ceil(max_x / round) * round
    ylim_upper = round+np.ceil(max_y / round) * round

    ax.set_xlim(xlim_lower, xlim_upper)
    ax.set_ylim(ylim_lower, ylim_upper)
    ax.set_aspect('equal',adjustable='box')
def draw_moves(dummy):
    # a = calculate_snap_value(slider_a.val)
    # b = calculate_snap_value(slider_b.val)
    #curve = slider_c.val # imported from metadaya
    # theta=slider_theta.val
    # old_draw_moves(a,b,curve)

    # Translated
    ax.clear()
    sys = buildSpectreBase(False)
    for i in range(recursions): # determines iterations
        sys = buildSupertiles(sys)
    plt.cla()
    ax.set_aspect('equal',adjustable='box')
    patchlist = list()  # things to draw inside the spectres
    if plotCO:
        for point in pointlist:
            ball = plt.Circle((point[0], point[1]),radius=ballsize,color='black')
            patchlist.append(ball)
    sys[tile_sel].draw(ident,ax,plotBorders,patchlist)
    # maxx = float('-inf')
    # minx = float('inf')
    # maxy = float('-inf')
    # miny = float('inf')
    allpoints = []
    for patch in ax.patches: # pull the vertices from the generated polygons (hidden) to make bezier splines
        patch.set_alpha(0)
        points = patch.get_verts()
        # max_x, max_y = np.max(points, axis=0)
        # min_x, min_y = np.min(points, axis=0)
        # maxx = max(max_x, maxx)
        # maxy = max(max_y, maxy)
        # minx = min(min_x, minx)
        # miny = min(min_y, miny)
        generate_bezier_points(points,curve,linedensity,True,ax,linethickness) #points, curve variable, density of curve, whether to plot, whether to polygon, axis
    ax.set_aspect('equal',adjustable='box')


# Load data from file to draw in circles
with open('metadata.pkl','rb') as f:
    meta = pickle.load(f)
curve = meta[0]
ballsize = meta[1]
with open('pointlist.pkl', 'rb') as f:
    pointlist = pickle.load(f)
# Plot
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
# initial_a = 1.0
# initial_b = 1
# ax_slider_a = plt.axes([0.1, 0.09, 0.65, 0.03])
# ax_slider_b = plt.axes([0.1, 0.06, 0.65, 0.03])
# ax_slider_c = plt.axes([0.1,0.01,.65,.03])
# ax_slider_theta = plt.axes([0.1, 0.15, 0.65, 0.03])
# slider_a = Slider(ax_slider_a, 'a', 0.1, 5.0, valinit=initial_a)
# slider_b = Slider(ax_slider_b, 'b', 0.1, 5.0, valinit=initial_b)
# slider_c = Slider(ax_slider_c, 'curve', -1,1, valinit=initial_b)
# slider_theta = Slider(ax_slider_theta,'angle',0,2*np.pi,valinit=0)
# slider_a.on_changed(draw_moves)
# slider_b.on_changed(draw_moves)
# slider_c.on_changed(draw_moves)
# slider_theta.on_changed(draw_moves)
ax.plot(color='black')
draw_moves(0)
plt.rcParams.update({
    "lines.color": "white",
    "patch.edgecolor": "white",
    "text.color": "black",
    "axes.facecolor": "white",
    "axes.edgecolor": "lightgray",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "grid.color": "lightgray",
    "figure.facecolor": "black",
    "figure.edgecolor": "black",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "black"})
plt.axis('off')
fig.savefig('out_30_nocurve.png', bbox_inches='tight',transparent=False, pad_inches=0)
image = Image.open('out_30_nocurve.png')
new_image = image.resize((64,64))
new_image.save('out_64_30_nocurve.png')

