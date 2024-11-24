import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
import numpy as np
import time

start_time = time.time()

G = 1
M = 100
m = 0
R = 10
w = 0#np.sqrt(G * (M + m) / R**3)
x1 = -m * R / (M + m)
x2 = M * R / (M + m)

def potential(x, y):
    return -G * M / np.sqrt((x - x1)**2 + y**2) - G * m / np.sqrt((x - x2)**2 + y**2) - w * w * (x**2 + y**2) / 2

def specific_dU_dr(r, theta):
    return ((G * M * (r - np.cos(theta) * x1) * (r ** 2 - 2 * np.cos(theta) * r * x1 + x1 * x1) ** -1.5)
            + (G * m * (r - np.cos(theta) * x2) * (r ** 2 - 2 * np.cos(theta) * r * x2 + x2 * x2) ** -1.5))

def specific_dU_dr_xy(x, y):
    return ((G * M * (np.sqrt(x ** 2 + y ** 2) - (x / np.sqrt(x ** 2 + y ** 2)) * x1) * (x ** 2 + y ** 2 - 2 * x * x1 + x1 * x1) ** -1.5)
            + (G * m * (np.sqrt(x ** 2 + y ** 2) - (x / np.sqrt(x ** 2 + y ** 2)) * x2) * (x ** 2 + y ** 2 - 2 * x * x2 + x2 * x2) ** -1.5))

def specific_dU_dtheta(r, theta):
    return ((G * M * r * x1 * np.sin(theta) * (r ** 2 - 2 * np.cos(theta) * r * x1 + x1 * x1) ** -1.5) 
            + (G * m * r * x2 * np.sin(theta) * (r ** 2 - 2 * np.cos(theta) * r * x2 + x2 * x2) ** -1.5))

def specific_dU_dtheta(x, y):
    return ((G * M * y * x1 * (x ** 2 + y ** 2 - 2 * x * x1 + x1 * x1) ** -1.5) 
            + (G * m * y * x2 * (x ** 2 + y ** 2 - 2 * x * x2 + x2 * x2) ** -1.5))


def drdt(t, r, r_dot, theta, theta_dot):
    return r_dot

def dr_dotdt(t, r, r_dot, theta, theta_dot):
    return r * theta_dot * theta_dot + 2 * w * r * theta_dot + w * w * r- specific_dU_dr(r, theta)

def dthetadt(t, r, r_dot, theta, theta_dot):
    return theta_dot

def dtheta_dotdt(t, r, r_dot, theta, theta_dot):
    return (-specific_dU_dtheta(r, theta) - 2 * r * r_dot * (theta_dot + w)) / r ** 2

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

def Verlet(t, r, r_dot, theta, theta_dot, dt, drdt, dr_dotdt, dthetadt, dtheta_dotdt):
    # this :b:roke for polar system
    new_r = r + dt * r_dot + .5 * dt * dt * dr_dotdt(0, r, r_dot, theta, theta_dot)
    new_theta = theta + dt * theta_dot + .5 * dt * dt * dtheta_dotdt(0, r, r_dot, theta, theta_dot)
    new_r_dot = r_dot + .5 * dt * (dr_dotdt(0, r, r_dot, theta, theta_dot) + dr_dotdt(0, new_r, r_dot, new_theta, theta_dot))
    new_theta_dot = theta_dot + .5 * dt * (dtheta_dotdt(0, r, r_dot, theta, theta_dot) + dtheta_dotdt(0, new_r, r_dot, new_theta, theta_dot))
    
    r = new_r
    theta = new_theta
    r_dot = new_r_dot
    theta_dot = new_theta_dot
    t += dt
    return t, r, r_dot, theta, theta_dot

num_steps = 50
dt = .01 # timestep
r0 = 3 #6.29960524947
theta0 = 0 * np.pi / 180
r_dot0 = 0
theta_dot0 = .1 + np.sqrt(G * (M + m) / r0**3)#np.sqrt(G * (M + m) / r0**3) - np.sqrt(G * (M + m) / R**3) #* np.pi / 180

ts = [0] * num_steps
rs = [0] * num_steps
thetas = [0] * num_steps
r_dots = [0] * num_steps
theta_dots = [0] * num_steps

rs[0] = r0
thetas[0] = theta0
r_dots[0] = r_dot0
theta_dots[0] = theta_dot0
print(potential(r0 * np.cos(theta0), r0 * np.sin(theta0)) + .5 * (r_dot0 ** 2 + (r0 * theta_dot0) ** 2))
for i in range(num_steps - 1):
    # t, r, r_dot, theta, theta_dot = RungeKuttaCoupled2(ts[i], rs[i], r_dots[i], thetas[i], theta_dots[i], dt, drdt, dr_dotdt, dthetadt, dtheta_dotdt)
    # t, r, r_dot, theta, theta_dot = SympEuler(ts[i], rs[i], r_dots[i], thetas[i], theta_dots[i], dt, drdt, dr_dotdt, dthetadt, dtheta_dotdt)
    t, r, r_dot, theta, theta_dot = Verlet(ts[i], rs[i], r_dots[i], thetas[i], theta_dots[i], dt, drdt, dr_dotdt, dthetadt, dtheta_dotdt)
    rs[i + 1] = r
    r_dots[i + 1] = r_dot
    thetas[i + 1] = theta
    theta_dots[i + 1] = theta_dot
    print(potential(r * np.cos(theta), r * np.sin(theta)) + .5 * (r_dot ** 2 + (r * theta_dot) ** 2))

pos_x = rs * np.cos(thetas)
pos_y = rs * np.sin(thetas)
# print(pos_x)
# print(pos_y)

print(dr_dotdt(0, r0, r_dot0, theta0, theta_dot0))
print(dtheta_dotdt(0, r0, r_dot0, theta0, theta_dot0))

spacing = .1
plot_factor = 1.25

xs = np.arange(-R * plot_factor, R * plot_factor + spacing, spacing)
ys = np.arange(-R * plot_factor, R * plot_factor + spacing, spacing)
Xs, Ys = np.meshgrid(xs, ys)
Zs = potential(Xs, Ys)


fig, ax = plt.subplots()

circle1 = plt.Circle((x1, 0),1, fc='red',ec="black", zorder=3)
circle2 = plt.Circle((x2, 0),.2, fc='blue',ec="black", zorder=3)
ax.add_patch(circle1)
ax.add_patch(circle2)

max_line = Zs.max()
min_line = potential(1 + x1, 0)
num_lines = 100
line_spacing = (max_line - min_line) / num_lines

print(max_line)
print(min_line)

fig.set_size_inches(8, 8)
# lines = np.arange(-25, -15, .2)
# lines = -10 * np.exp(-.6 * (lines + 25)) - 15
lines = np.arange(min_line, max_line, line_spacing)
lines = ((lines - max_line) ** 2) / (min_line - max_line) + max_line
# lines = ((max_line - min_line) / (np.log10(max_line - min_line + 1))) * np.log10(lines - min_line + 1) + min_line
# print(lines)
contours = ax.contour(Xs, Ys, Zs, levels=lines)
print(lines[-1])

# print("math'd")

point,  = ax.plot(0, 0, 'ro')
path, = ax.plot(0, 0, 'k')

def animate(i):
    point.set_xdata([pos_x[i]])
    point.set_ydata([pos_y[i]])
    path.set_xdata(pos_x[:i + 1])
    path.set_ydata(pos_y[:i + 1])
    

ani = matplotlib.animation.FuncAnimation(fig, animate, frames=num_steps)
ani.save("movie.mp4", dpi=100)

print("--- %s seconds ---" % round((time.time() - start_time), 2))






