from translated import *
# from matplotlib.widgets import Slider
# import matplotlib.pyplot as plt

# def replot(dum):
#     curve = slider_c.val
#     ax.cla()
#     xpo,ypo=generate_bezier_points(points, curve, density, False, None)
#     ax.scatter(xpo, ypo,color='black')
#
# fig, ax = plt.subplots()
# ax_slider_c = plt.axes([0.1, 0.01, .65, .03])
# slider_c = Slider(ax_slider_c, 'curve', -1, 1, valinit=0)
# slider_c.on_changed(replot)
#
# points= generate_points(1,1,0,0,0)
# curve = .50
# density = 40
#
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from random import *
from matplotlib import animation
from matplotlib.path import Path
from random import randrange
import pickle

rmult = 2.75 # distance from walls
bbmult = 3 # distance from each other
curve = 0.1
sbox=1
balls = 35 # 30
ballsize = .16*sbox # .2

storageFolder = "RandomFiles"
def forcefield(dist, range): # ball to ball
    def f(r, range):
        # return 80*sbox/25 * np.exp(-r*1 / range)
        # return 80*sbox/25*np.exp(-r*r/.2)
        return 180*sbox/25*(range-r)
    return f(dist, range) - f(range, range)
def forcefield2(dist,range): # ball to wall
    return -.1 * abs(dist -rmult*range) #INVERT IF PATH DIRECTION CHANGES
def forcefield3(dist,range): # grid to ball
    k = 3
    ab = gridvec/2
    cut = range
    if dist > cut:
        return 0
    def field(di,a):
        return -2*(di/a/a)/np.square(di*di/a/a+k/a)
    return field(dist,ab)-field(dist,cut)
class Ball:
    def __init__(self, x, y, radius):
        self.r = radius
        self.acceleration = np.array([0, 0])
        self.velocity = np.array([uniform(0, 1),
                                  uniform(0, 1)])
        self.position = np.array([x, y])
    @property
    def x(self):
        return self.position[0]
    @property
    def y(self):
        return self.position[1]
    def applyForce(self, force):
        self.acceleration = np.add(self.acceleration, force)
    def _normalize(self, v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm
    def update(self):
        self.velocity = np.add(self.velocity, self.acceleration)
        self.position = np.add(self.position, self.velocity)
        self.acceleration *= 0
        self.velocity *= .95
class Pack:
    def __init__(self, list_balls,borders):
        self.iter = 0
        self.list_balls = list_balls
        self.list_separate_forces = [np.array([0, 0])] * len(self.list_balls)
        self.list_near_balls = [0] * len(self.list_balls)
        self.wait = True
        self.borders = borders
        self.poly = plt.Polygon(borders,alpha=.1,edgecolor='black')
        self.polygon_path = Path(borders)
        self.nudge=False
    def _normalize(self, v):

        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm
    def run(self):
        self.iter += 1
        for ball in self.list_balls:
            self.checkBorders(ball)
            self.applySeparationForcesToBall(ball)
            self.snapGrid(ball)
        return self.freeze()
    def snapGrid(self,ball):
        # central repulsion force doesn't work
        # cx = -2
        # cy = -.2
        # n_v = np.array([cx-ball.x,cy-ball.y])
        # d = np.linalg.norm(n_v)
        # n_v = n_v / d
        # ball.applyForce(n_v*forcefield3(d,1))
        # forcefield on particles takes too long, different spectres with different rotations won't work
        # totalForce = np.array([0.0, 0.0])
        # pos = np.array([ball.x,ball.y])
        # for x in range(gx):
        #     for y in range(gy):
        #         lat = x * a1 + y * a2 + gridshift
        #         if self.polygon_path.contains_point(lat.tolist()):
        #             n_v = pos-lat
        #             dist = np.linalg.norm(n_v)
        #             n_v /= dist
        #             totalForce += n_v*forcefield3(dist,gridvec)
        # ball.applyForce(totalForce)
        pass
    def minWallDist(self,ball):
        closest_line = None
        min_distance = float('inf')
        vertices = self.borders
        point = [ball.x,ball.y]
        for i in range(len(vertices)):
            start_point = vertices[i]
            end_point = vertices[(i + 1) % len(vertices)]
            # Calculate the distance from the point to the line segment
            dx = end_point[0] - start_point[0]
            dy = end_point[1] - start_point[1]
            length_sq = dx * dx + dy * dy
            t = max(0, min(1, ((point[0] - start_point[0]) * dx + (point[1] - start_point[1]) * dy) / length_sq))
            closest_point = (start_point[0] + t * dx, start_point[1] + t * dy)
            distance = np.sqrt((point[0] - closest_point[0]) ** 2 + (point[1] - closest_point[1]) ** 2)
            # Update minimum distance and closest line segment
            if distance < min_distance:
                min_distance = distance
                closest_line = (start_point, end_point)
        return closest_line,min_distance
    def checkBorders(self, ball):

        closest_line, min_distance = self.minWallDist(ball)

        boundtrigger = 0
        if not self.polygon_path.contains_point([ball.x,ball.y]):
            boundtrigger = 1
        elif min_distance < rmult*ball.r:
            boundtrigger = 1
        if boundtrigger != 0:
            dx = closest_line[1][0] - closest_line[0][0]
            dy = closest_line[1][1] - closest_line[0][1]
            n_v = np.array([dy, -dx])
            n_v = n_v/np.linalg.norm(n_v)
            u = np.dot(ball.velocity, n_v) * n_v
            w = np.subtract(ball.velocity, u)
            ball.applyForce(n_v*boundtrigger*forcefield2(min_distance,ball.r))
    def freeze(self):
        for ball in self.list_balls:
            if np.linalg.norm(ball.velocity) > .25*sbox/25:
                return False,None
        output = []
        if not self.nudge:
            self.nudge = True
            for ball in self.list_balls:
                if not self.polygon_path.contains_point([ball.x, ball.y]):
                    scale = 5*sbox/25
                    ball.applyForce([scale*(random()*2-1),scale*(random()*2-1)])
            return False,None
        for ball in self.list_balls:
            _,ld = self.minWallDist(ball)
            if self.polygon_path.contains_point([ball.x, ball.y]) and abs(ld) < 1.5*rmult*ball.r:
                ball.velocity *= 0
                output.append([ball.x,ball.y])
        return True,output
    def getSeparationForce(self, c1, c2):
        steer = np.array([0, 0])
        d = self._distanceBalls(c1, c2)
        if d > 0 and d < bbmult*c1.r:
            diff = np.subtract(c1.position, c2.position)
            diff = self._normalize(diff)
            diff *= forcefield(d,bbmult*c1.r)
            steer = diff
        return steer
    def _distanceBalls(self, c1, c2):
        x1, y1 = c1.x, c1.y
        x2, y2 = c2.x, c2.y
        dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return dist
    def applySeparationForcesToBall(self, ball):
        i = self.list_balls.index(ball)
        for neighbour in self.list_balls[i + 1:]:
            j = self.list_balls.index(neighbour)
            forceij = self.getSeparationForce(ball, neighbour)
            if np.linalg.norm(forceij) > 0:
                self.list_separate_forces[i] = np.add(self.list_separate_forces[i], forceij)
                self.list_separate_forces[j] = np.subtract(self.list_separate_forces[j], forceij)
                self.list_near_balls[i] += 1
                self.list_near_balls[j] += 1
        if np.linalg.norm(self.list_separate_forces[i]) > 0:
            self.list_separate_forces[i] = np.subtract(self.list_separate_forces[i], ball.velocity)
        if self.list_near_balls[i] > 0:
            self.list_separate_forces[i] = np.divide(self.list_separate_forces[i], self.list_near_balls[i])
        separation = self.list_separate_forces[i]
        ball.applyForce(separation)
        ball.update()

list_balls = list()
for i in range(balls):
    b = Ball(randrange(-4*sbox,0), randrange(-2*sbox,2*sbox), ballsize)
    list_balls.append(b)
points= generate_points(sbox,sbox,0,0,0)
points = reflect_points(points)
density = 10
xpo,ypo=generate_bezier_points(points, curve, density, False,1)
points_array = [[x, y] for x, y in zip(xpo,ypo)]
p = Pack(list_balls,points_array)
fig, ax = plt.subplots()
cycleplot = False
if cycleplot:
    # META PLOTTING
    cycles = 10
    metao = []
    for i in range(cycles):
        list_balls = list()
        for i in range(balls):
            b = Ball(randrange(-2*sbox, 0), randrange(-sbox, sbox), ballsize)
            list_balls.append(b)
        p = Pack(list_balls, points_array)
        notdone=True
        while notdone:
            done,output = p.run()
            if done:
                print(output)
                metao.append(output)
                notdone=False
    flat_points = [point for row in metao for point in row]
    x_points, y_points = zip(*flat_points)
    plt.scatter(x_points, y_points, c='blue', marker='o')
    plt.gca().add_patch(p.poly)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()
else:
    #ANIMATE SINGLE
    def draw(i):
        patches = []
        done,output = p.run()
        if done:
            name = str(balls)+"_"+str(curve)
            with open(storageFolder+'\\pointlist_'+name+'.pkl', 'wb') as f:
                pickle.dump(output, f)
            with open(storageFolder+'\\metadata_'+name+'.pkl', 'wb') as f:
                pickle.dump([curve,ballsize], f)
            print("Dumped. Exiting.",name)
            exit()
        ax.cla()
        ax.add_patch(p.poly)
        plt.axis('scaled')
        for c in list_balls:
            ball = plt.Circle((c.x, c.y), radius=c.r, picker=True, fc='none', ec='k')
            patches.append(plt.gca().add_patch(ball))
        return patches
    anim = animation.FuncAnimation(fig, draw,frames=500, interval=2, blit=True)
    plt.show()