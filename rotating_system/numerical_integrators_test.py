import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
import numpy as np
import time

start_time = time.time()

G = 1
M = 100
m = 1
R = 25
fixed_x1 = -m * R / (M + m)
fixed_x2 = M * R / (M + m)
fixed_y1 = 0
fixed_y2 = 0

fixed_period = 2 * np.pi * np.sqrt(R ** 3 / (G * (M + m)))
fixed_freq = 1 / fixed_period

print(fixed_period)

def findMassiveObjAngles(t):
    orbit_time = t % fixed_period
    M_angle = fixed_freq * orbit_time * 2 * np.pi
    m_angle = fixed_freq * orbit_time * 2 * np.pi
    return M_angle, m_angle

def findMassiveObjPositions(t):
    M_angle, m_angle = findMassiveObjAngles(t)
    return fixed_x1 * np.cos(M_angle), fixed_x1 * np.sin(M_angle), fixed_x2 * np.cos(m_angle), fixed_x2 * np.sin(m_angle)

def potential(t, x, y):
    Mx, My, mx, my = findMassiveObjPositions(t)
    return -G * M / np.sqrt((x - Mx)**2 + (y - My)**2) - G * m / np.sqrt((x - mx)**2 + (y - my)**2)

def Fx(t, x, y):
    Mx, My, mx, my = findMassiveObjPositions(t)
    return -G * M * (x - Mx) * ((x - Mx) ** 2 + (y - My) ** 2) ** -1.5 - G * m * (x - mx) * ((x - mx) ** 2 + (y - my) ** 2) ** -1.5

def Fy(t, x, y):
    Mx, My, mx, my = findMassiveObjPositions(t)
    return -G * M * (y - My) * ((x - Mx) ** 2 + (y - My) ** 2) ** -1.5 - G * m * (y - my) * ((x - mx) ** 2 + (y - my) ** 2) ** -1.5


def RungeKuttaCoupled2(t, r, r_dot, theta, theta_dot, dt, drdt, dr_dotdt, dthetadt, dtheta_dotdt):
    
    k1 = dt * drdt(t, r, r_dot, theta, theta_dot)
    l1 = dt * dr_dotdt(t, r, r_dot, theta, theta_dot)
    m1 = dt * dthetadt(t, r, r_dot, theta, theta_dot)
    n1 = dt * dtheta_dotdt(t, r, r_dot, theta, theta_dot)
    k2 = dt * drdt(t + .5 * dt, r + .5 * k1, r_dot + .5 * l1, theta + .5 * m1, theta_dot + .5 * n1)
    l2 = dt * dr_dotdt(t + .5 * dt, r + .5 * k1, r_dot + .5 * l1, theta + .5 * m1, theta_dot + .5 * n1)
    m2 = dt * dthetadt(t + .5 * dt, r + .5 * k1, r_dot + .5 * l1, theta + .5 * m1, theta_dot + .5 * n1)
    n2 = dt * dtheta_dotdt(t + .5 * dt, r + .5 * k1, r_dot + .5 * l1, theta + .5 * m1, theta_dot + .5 * n1)
    k3 = dt * drdt(t + .5 * dt, r + .5 * k2, r_dot + .5 * l2, theta + .5 * m2, theta_dot + .5 * n2)
    l3 = dt * dr_dotdt(t + .5 * dt, r + .5 * k2, r_dot + .5 * l2, theta + .5 * m2, theta_dot + .5 * n2)
    m3 = dt * dthetadt(t + .5 * dt, r + .5 * k2, r_dot + .5 * l2, theta + .5 * m2, theta_dot + .5 * n2)
    n3 = dt * dtheta_dotdt(t + .5 * dt, r + .5 * k2, r_dot + .5 * l2, theta + .5 * m2, theta_dot + .5 * n2)
    k4 = dt * drdt(t + dt, r + k3, r_dot + l3, theta + m3, theta_dot + n3)
    l4 = dt * dr_dotdt(t + dt, r + k3, r_dot + l3, theta + m3, theta_dot + n3)
    m4 = dt * dthetadt(t + dt, r + k3, r_dot + l3, theta + m3, theta_dot + n3)
    n4 = dt * dtheta_dotdt(t + dt, r + k3, r_dot + l3, theta + m3, theta_dot + n3)

    r += (k1 + 2 * k2 + 2 * k3 + k4) / 6.
    r_dot += (l1 + 2 * l2 + 2 * l3 + l4) / 6.
    theta += (m1 + 2 * m2 + 2 * m3 + m4) / 6.
    theta_dot += (n1 + 2 * n2 + 2 * n3 + n4) / 6.
    t += dt
    return t, r, r_dot, theta, theta_dot

def SympEuler(t, r, r_dot, theta, theta_dot, dt, drdt, dr_dotdt, dthetadt, dtheta_dotdt):
    
    r_dot += dt * dr_dotdt(t, r, r_dot, theta, theta_dot)
    r += dt * r_dot
    theta_dot += dt * dtheta_dotdt(t, r, r_dot, theta, theta_dot)
    theta += dt * theta_dot
    t += dt
    return t, r, r_dot, theta, theta_dot

def Verlet(t, x1, vx1, y1, vy1, dt, Fx, Fy):
    
    new_x1 = x1 + dt * vx1 + .5 * dt * dt * Fx(t, x1, y1)
    new_y1 = y1 + dt * vy1 + .5 * dt * dt * Fy(t, x1, y1)
    new_vx1 = vx1 + .5 * dt * (Fx(t, x1, y1) + Fx(t + dt, new_x1, new_y1))
    new_vy1 = vy1 + .5 * dt * (Fy(t, x1, y1) + Fy(t + dt, new_x1, new_y1))
    
    return t + dt, new_x1, new_vx1, new_y1, new_vy1

num_steps = 10000
dt = .025 # timestep
r0 = 24 #6.29960524947
theta0 = 0 * np.pi / 180
r_dot0 = 0
theta_dot0 = 0#np.sqrt(G * (M + m) / r0**3)#np.sqrt(G * (M + m) / r0**3) - np.sqrt(G * (M + m) / R**3) #* np.pi / 180

x1_0 = r0 * np.cos(theta0)
y1_0 = r0 * np.sin(theta0)
vx1_0 = r_dot0 * np.cos(theta0) - r0 * np.sin(theta0) * theta_dot0
vy1_0 = r_dot0 * np.sin(theta0) + r0 * np.cos(theta0) * theta_dot0

ts = [0] * num_steps
x1s = [0] * num_steps
y1s = [0] * num_steps
vx1s = [0] * num_steps
vy1s = [0] * num_steps

x1s[0] = x1_0
y1s[0] = y1_0
vx1s[0] = vx1_0
vy1s[0] = vy1_0
print(potential(0, r0 * np.cos(theta0), r0 * np.sin(theta0)) + .5 * (r_dot0 ** 2 + (r0 * theta_dot0) ** 2))
for i in range(num_steps - 1):
    # t, r, r_dot, theta, theta_dot = RungeKuttaCoupled2(ts[i], rs[i], r_dots[i], thetas[i], theta_dots[i], dt, drdt, dr_dotdt, dthetadt, dtheta_dotdt)
    # t, r, r_dot, theta, theta_dot = SympEuler(ts[i], rs[i], r_dots[i], thetas[i], theta_dots[i], dt, drdt, dr_dotdt, dthetadt, dtheta_dotdt)
    t, x1, vx1, y1, vy1 = Verlet(ts[i], x1s[i], vx1s[i], y1s[i], vy1s[i], dt, Fx, Fy)
    ts[i + 1] = t
    x1s[i + 1] = x1
    vx1s[i + 1] = vx1
    y1s[i + 1] = y1
    vy1s[i + 1] = vy1
    print(potential(t, x1, y1) + .5 * (vx1 ** 2 + vy1 ** 2))

print("initial energy:", potential(0, r0 * np.cos(theta0), r0 * np.sin(theta0)) + .5 * (r_dot0 ** 2 + (r0 * theta_dot0) ** 2))

pos_x = x1s
pos_y = y1s
# print(pos_x)
# print(pos_y)

spacing = .1
plot_factor = 1.25

xs = np.arange(-R * plot_factor, R * plot_factor + spacing, spacing)
ys = np.arange(-R * plot_factor, R * plot_factor + spacing, spacing)
Xs, Ys = np.meshgrid(xs, ys)
Zs = potential(0, Xs, Ys)


fig, ax = plt.subplots()
ax.set_xlim([xs[0], xs[-1]])
ax.set_ylim([ys[0], ys[-1]])
c1_x, c1_y, c2_x, c2_y = findMassiveObjPositions(0)
circle1 = plt.Circle((c1_x, c1_y),np.log10(M) * .2 + .2, fc='red',ec="black", zorder=3)
circle2 = plt.Circle((c2_x, c2_y),np.log10(m) * .2 + .2, fc='blue',ec="black", zorder=3)
ax.add_patch(circle1)
ax.add_patch(circle2)

max_line = Zs.max()
min_line = potential(0, 1 + fixed_x1, 0)
num_lines = 50
line_spacing = (max_line - min_line) / num_lines

# print(max_line)
# print(min_line)

fig.set_size_inches(8, 8)
# lines = np.arange(-25, -15, .2)
# lines = -10 * np.exp(-.6 * (lines + 25)) - 15
lines = np.arange(min_line, max_line, line_spacing)
lines = ((lines - max_line) ** 2) / (min_line - max_line) + max_line
# lines = ((max_line - min_line) / (np.log10(max_line - min_line + 1))) * np.log10(lines - min_line + 1) + min_line
# print(lines)
# contours = ax.contour(Xs, Ys, Zs, levels=lines)
# contours.set_animated(True)
# print(lines[-1])

# print("math'd")

point,  = ax.plot([], [], 'ro')
path, = ax.plot([], [], 'k')
# point.set_animated(True)
# path.set_animated(True)
# def init():
#     pendulum1.set_data([], [])
#     pendulum2.set_data([], [])
#     return pendulum1, pendulum2, 

point.set_xdata([pos_x[-1]])
point.set_ydata([pos_y[-1]])
path.set_xdata(pos_x)
path.set_ydata(pos_y)
c1_x, c1_y, c2_x, c2_y = findMassiveObjPositions(ts[-1])
circle1.center = c1_x, c1_y
circle2.center = c2_x, c2_y
plt.show();

def init():
    point.set_data([], [])
    path.set_data([], [])
    return point, path,

def animate(i):
    point.set_xdata([pos_x[i]])
    point.set_ydata([pos_y[i]])
    path.set_xdata(pos_x[:i + 1])
    path.set_ydata(pos_y[:i + 1])
    c1_x, c1_y, c2_x, c2_y = findMassiveObjPositions(ts[i])
    circle1.center = c1_x, c1_y
    circle2.center = c2_x, c2_y
    return point, path, circle1, circle2,
    

ani = matplotlib.animation.FuncAnimation(fig=fig, init_func=init, func=animate, frames=num_steps, interval=50, blit=True)
ani.save("movie.mp4", dpi=100)

print("--- %s seconds ---" % round((time.time() - start_time), 2))






