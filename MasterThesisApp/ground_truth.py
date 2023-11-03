import numpy as np
import numpy.matlib as nmat
from mpl_toolkits.mplot3d import Axes3D, art3d
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
import tikzplotlib
from matplotlib import animation,rcParams
rcParams['animation.writer'] = 'ffmpeg'
import time

def add_point(ax, x, y, z, fc=None, ec='m', radius=0.0008):
    xy_len, z_len = ax.get_figure().get_size_inches()
    axis_length = [x[1] - x[0] for x in [ax.get_xbound(), ax.get_ybound(), ax.get_zbound()]]
    axis_rotation = {'z': ((x, y, z), axis_length[1] / axis_length[0]),
                     'y': ((x, z, y), axis_length[2] / axis_length[0] * xy_len / z_len),
                     'x': ((y, z, x), axis_length[2] / axis_length[1] * xy_len / z_len)}
    for a, ((x0, y0, z0), ratio) in axis_rotation.items():
        p = Ellipse((x0, y0), width=radius, height=radius * ratio, fc=fc, ec=ec,label='(10mm,0)')
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=z0, zdir=a)


m1 = -5/28.3564
m2 = 5/18.66025

x1 = np.linspace(-28.3564, 0, num=20, endpoint=True)
x2 = np.linspace(0, 18.66025, num=20, endpoint=True)
x = np.concatenate((x1,x2), axis=0)

z1 = (m1*(x1 + 28.3564) + 25)
z2 = (m2*(x2 - 18.66025) + 25)
z = np.concatenate((z1,z2), axis=0)

y = np.linspace(-10, 10, 10)

X,Y = np.meshgrid(x,y)
Z = nmat.repmat(z,len(y),1)
x_range = np.arange(-20,20,2)
fig = plt.figure(figsize=(7,7))
ax = fig.gca(projection='3d')
#ax = fig.add_subplot(111, projection='3d')
ax.set_zlim3d(15,25)
surf = ax.plot_surface(X, Y, Z,label='Specimen')
surf._facecolors2d=surf._facecolor3d
surf._edgecolors2d=surf._edgecolor3d
ax.set_xlabel('x-axis [mm]')
ax.set_ylabel('y-axis [mm]')
ax.set_zlabel('depth [mm]')
quiver1 = ax.quiver([[6]],[[0]],[[15]],[[0]],[[0]],[[4]],linewidths=(7,), edgecolor="red", label='Transmit Pulse')
quiver2 = ax.quiver([[6]], [[0]], [[19]], [[0]], [[0]], [[-4]], linewidths=(7,), edgecolor="green", label='Reflected Pulse')
point = ax.plot([6], [0], [15], color='black', marker='o', markersize=10, alpha=0.8, label='Transducer')
ax.legend()
quiver1.remove()

#quiver1 = ax.quiver([[2]],[[0]],[[20]],[[0]],[[0]],[[-4]],linewidths=(7,), edgecolor="green", label='Reflected Pulse')
#ax.text(12e-3, 2e-3, 20e-3,'(10e-3,0,22e-3)',color='black')
#ax.text(-5e-3, 2e-3, 20e-3,'(0,0,20e-3)',color='black')

#p = Circle((10e-3, 0), 20e-3, ec='k', fc="none")
#ax.add_patch(p)
#art3d.pathpatch_2d_to_3d(p, z=1, zdir="z")

#add_point(ax, 6, 0, 15, radius=20)
#add_point(ax, 0, 0, 20e-3, radius=20e-4)
#add_point(ax, 0, 0, 0, radius=30e-3)
#point = ax.scatter(4,0,15,c='k', depthshade=False, alpha = 1, s=100,label='Transducer')
#point.remove()
#ax.plot([4], [0], [15], color='black', marker='o', markersize=10, alpha=0.8,label='Transducer')
ax.yaxis.set_ticklabels([])
ax.invert_zaxis()
ax.view_init(elev=2, azim=90)
ax.set_title('Measurement Approach')
#plt.show()
flag = 0


def init():
    global quiver1
    global quiver2
    quiver1 = ax.quiver([[6]], [[0]], [[15]], [[0]], [[0]], [[4]], linewidths=(7,), edgecolor="red", label='Transmit Pulse')
    quiver2 = ax.quiver([[6]], [[0]], [[19]], [[0]], [[0]], [[-4]], linewidths=(7,), edgecolor="green",label='Reflected Pulse')

    return quiver1,quiver2


def animate(i):
    global quiver1
    global quiver2
    global flag
    global point
    #global point
    #quiver1.remove()
    #quiver2.remove()

    if flag == 0:
        quiver2.remove()
        #point.clear()
        p = point.pop(0)
        p.remove()
        ax.view_init(elev=2, azim=90)
        point = ax.plot([x_range[i]], [0], [15], color='black', marker='o', markersize=10, alpha=0.8,label='Transducer')
        quiver1 = ax.quiver([[x_range[i]]], [[0]], [[15]], [[0]], [[0]], [[4]], linewidths=(7,), edgecolor="red",label='Transmit Pulse')
        flag = 1
    else:
        quiver1.remove()
        #point = ax.plot([x_range[i]], [0], [15], color='white', marker='o', markersize=10, alpha=0.8,label='Transducer')
        p = point.pop(0)
        p.remove()
        ax.view_init(elev=2, azim=90)
        point = ax.plot([x_range[i-1]], [0], [15], color='black', marker='o', markersize=10, alpha=0.8, label='Transducer')
        quiver2 = ax.quiver([[x_range[i-1]]], [[0]], [[19]], [[0]], [[0]], [[-4]], linewidths=(7,), edgecolor="green", label='Reflected Pulse')
        flag = 0
    #point = ax.scatter(x_range[i], 0, 15, c='k', depthshade=False, alpha=1, s=100, label='Transducer')
    #time.sleep(2)
    #quiver = ax.quiver([[x_range[i]]], [[0]], [[18]], [[0]], [[0]], [[-4]], linewidths=(7,), edgecolor="green", label='Reflected Pulse')
    #time.sleep(2)
    #point.remove()


anim = animation.FuncAnimation(fig, animate, frames=len(x_range)-1, interval=1000)
anim.save('C:/Users/user/Documents/Master_Thesis_final_animations/Measurement_setup.gif', writer='Pillow', fps=1, dpi=100, metadata={'title':'test'})
#tikzplotlib.save("C:/Users/user/Documents/test.tex")