

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


import time
start_time = time.time()



a = 1 # width of well
m = 1 # mass of particle
hbar = 1 # reduced planck constant 

def state_n(n, x, t): # the wavefunction for a particular state n
    return np.sqrt(2 / a) * np.sin(n * np.pi * x / a) * np.exp(-1j * n * n * np.pi * np.pi * t / (2 * m * a * a))

# Make sure to normalize for sums of states. The sum of squares of the components of the states must be 1. 


def sum_states(comps, x, t):
    norm_factor = 0
    func = 0
    for e, c  in enumerate(comps): # n = e + 1, since the elements start at 0
        if c != 0:
            func += c * state_n(e+1, x, t)
            norm_factor += np.real(c * c.conjugate())
    return func / np.sqrt(norm_factor)

comps = [1, .5, .5, 0, 0, 0, 0, 0, 0, 0]
# test = [1 / (i + 1)**3 if i % 2 == 0 else 0 for i in range(100)]
# print(test)
# comps=test
#        1  2  3  4  5  6  7  8  9  10
num_frames = 250
ani_interval = 100 # takes milliseconds
total_time = 1  # seconds
# total_time / num_frames is sim time per frame
# ani_interval is playback time per frame
ts = np.linspace(0, total_time, num_frames)

num_xs = 100
xs = np.linspace(0, a, num_xs)

xxs, tts = np.meshgrid(xs, ts, sparse=True)

# total_state = state_n(1, xxs, tts)
total_state = sum_states(comps, xxs, tts)

ys_real = np.real(total_state)
ys_imag = np.imag(total_state)
ys_prob = np.real(np.conj(total_state) * total_state)



t = 0

part_alpha = 1

fig, ax = plt.subplots()

ax.axhline(y=0, color='k', linestyle='-') 
# ax.axvline(x=0, color='k', linestyle='-') 
# ax.axvline(x=a, color='k', linestyle='-') 

reals, = ax.plot(xs, ys_real[t], label="Real Part", alpha=part_alpha, linewidth = 1.5)
imags, = ax.plot(xs, ys_imag[t], label="Imaginary Part", alpha=part_alpha, linewidth = 1.5)
probs, = ax.plot(xs, ys_prob[t], label="Probability Density", linewidth=1.5)


ax.set(xlabel='pos', ylabel='wavefunction val', title='wavefunction', ylim=[-3, 3])
ax.grid()
ax.legend()



def update(frame):
    if frame % (num_frames // 100) == 0 and frame != 0:
        print(100 * frame / num_frames)
    # curr_y_real = ys_real[frame]
    # curr_y_imag = ys_imag[frame]
    # curr_y_prob = ys_prob[frame]
    
    reals.set_ydata(ys_real[frame])
    
    imags.set_ydata(ys_imag[frame])
    
    probs.set_ydata(ys_prob[frame])
    
    return (reals, imags, probs)
    





ani = animation.FuncAnimation(fig=fig, func=update, frames=num_frames, interval=ani_interval, blit=True)
ani.save("movie.mp4", dpi=200)
plt.show()

print("--- %s seconds ---" % round((time.time() - start_time), 2))






