# Quadruple-Tanks Process Simulator with DoS attack implementation
# Matheus Figueiredo
# Federal University of Amazonas
# 2025

# libs
import numpy as np
import matplotlib.pyplot as plt
from controler_discrete import ControladorPIDiscretoEuler


'''
Definitions to implement DoS attack:

Consider the following equation affected by an adversary:

u_till = u + gamma * S_u * gamma^T * (u - u_t),

where: 

gamma -> binary matrice of [total number of actuator] x [actuator(i) acessed by adversary] dimension

S_u -> square matrice [actuator(i) acessed by adversary] x [actuator(i) acessed by adversary] dimension

u_t -> last signal sent by the actuator

'''

# definition of the u_till

'''
u1_set and u2_set are variables that indicates if the adversary have acess to actuator u1 or u2
'''


# ==========================================================

# QTP Parameters

# system parameters set in the paper
a1, a2, a3, a4 = 0.071, 0.057, 0.071, 0.057
A1, A2, A3, A4 = 28, 32, 28, 32
g = 981
k1, k2 = 3.33, 3.35
gamma1, gamma2 = 0.7, 0.6
h_max = 30     # maximum high of the tanks
kc = 0.5

# controler parameters
kp1 = 3
kp2 = 2.7
ti1 = 30
ti2 = 40

# operation points
h1_0, h2_0, h3_0, h4_0 = 12.4, 12.7, 1.8, 1.4
v1_0, v2_0 = 3, 3
dt = 0.05     # time pass
n_steps = 7000     # simulation pass

# to h1 e h2 == 4m, we need of the following values of r1 and r2 as reference:
r1 = -4.2
r2 = -4.35

# r = np.array([[r1],
#               [r2]])

# ====================================================================================================

# inicial conditions
# controlers definition
u = np.zeros((2, 1)) # controlers u_k == [[u1], [u2]]
u_t = np.zeros((2, 1)) # controlers u_t == [[u1_t], [u2_t]]
u_til = np.zeros((2, 1))
# output signal of the system
y = np.zeros((2, 1)) # y_k == [[y1], [y2]]
y_t = np.zeros((2, 1)) # y_t == [[y1_t], [y2_t]]
y_til = np.zeros((2, 1))
# ==================================
u1_t = u_t[0,0]
u2_t = u_t[1,0]
u1_til = u_til[0,0]
u2_til = u_til[1,0]
y1_t = y_t[0,0]
y2_t = y_t[1,0]
y1_til = y_til[0,0]
y2_til = y_til[1,0]
h1, h2, h3, h4 = 10, 10, 1.8, 1.4

# =====================================================================================================

# variable to keep states of the system for plot in the graphic
states_h = [[h1, h2, h3, h4]]
states_u = [[u[0,0], u[1,0]]]
states_u_til = [[u_til[0,0], u_til[1,0]]]
states_y = [[0, 0]]
states_y_til = [[0, 0]]


# controler instances
c1 = ControladorPIDiscretoEuler(kp1, ti1, dt)
c2 = ControladorPIDiscretoEuler(kp2, ti2, dt)

# ==============================
# variables to indicates the chanels that adversay can acess
u1_set = True
u2_set = True
y1_set = False
y2_set = False

# time to attack
time_attack = np.zeros(n_steps)
time_attack[0:50] = 1
time_attack[100:150] = 1
time_attack[300:400] = 1
time_attack[500:750] = 1

'''
Definition of the matrices gamma for u and y
'''

# gamma u matrice configuration
if u1_set and u2_set:
    gamma_u = np.array([[1, 0],
                        [0, 1]])
elif u1_set:
    gamma_u = np.array([[1],
                        [0]])
elif u2_set:
    gamma_u = np.array([[0],
                        [1]])
else:
    gamma_u = np.array([[0],
                        [0]])  
# transport matrice
gamma_u_T = gamma_u.T

# ====================================================

# gamma y matrice configuration
if y1_set and y2_set:
    gamma_y = np.array([[1, 0],
                        [0, 1]])
elif y1_set:
    gamma_y = np.array([[1],
                        [0]])
elif y2_set:
    gamma_y = np.array([[0],
                        [1]])
else:
    gamma_y = np.array([[0],
                        [0]])
# transport matrice
gamma_y_T = gamma_y.T


'''
SIMULATION OF THE SYSTEM
'''

for k in range(n_steps):

    # copy last value of u when the time of attack not occur
    if time_attack[k] == 0:
        u_t = u.copy()
        y_t = y.copy()


    # S_u matrice configuration
    if u1_set and u2_set:
        S_u = np.array([[1, 0],
                        [0, 1]])
    elif u1_set or u2_set:
        S_u = np.array([[1]])
    else:
        S_u = np.array([[0]])
    


    # b_u definition
    b_u = -S_u @ gamma_u_T @ (u - u_t)

    # =====================================


    # S_u matrice configuration
    if y1_set and y2_set:
        S_y = np.array([[1, 0],
                        [0, 1]])
    else:
        S_y = np.array([[1]])
    
    # b_y definition
    b_y = -S_y @ gamma_y_T @ (y - y_t)

    # values of the system output
    y1 = kc * (h1 - h1_0)
    y2 = kc * (h2 - h2_0)
    
    y = np.array(
        [[y1],
         [y2]]
    )

    # y_til definition
    y_til = y + gamma_y @ b_y
     
    # errors measurements
    erro1 = r1 - y1
    erro2 = r2 - y2

    # controlers
    u1 = c1.calcular_saida(erro1)
    u2 = c2.calcular_saida(erro2)
    
    u = np.array([[u1],[u2]])

    # u_til definition
    u_til = u + gamma_u @ b_u

    v0 = np.array([[v1_0],
                   [v2_0]])
    
    v = u_til + v0
    v1 = v[0, 0] # element of the matrice v -> i0 and j0
    v2 = v[1, 0] # element of the matrice v -> i1 and j0


    # equations of the system
    dh1_dt = -((a1 * np.sqrt(2 * g * h1)) / A1) + ((a3 * np.sqrt(2 * g * h3)) / A1) + ((gamma1 * k1 * v1) / A1)
    dh2_dt = -((a2 * np.sqrt(2 * g * h2)) / A2) + ((a4 * np.sqrt(2 * g * h4)) / A2) + ((gamma2 * k2 * v2) / A2)
    dh3_dt = -((a3 * np.sqrt(2 * g * h3)) / A3) + (((1 - gamma2) * k2 * v2) / A3)
    dh4_dt = -((a4 * np.sqrt(2 * g * h4)) / A4) + (((1 - gamma1) * k1 * v1) / A4)

    # discrete system
    h1 = h1 + dt * dh1_dt
    h2 = h2 + dt * dh2_dt
    h3 = h3 + dt * dh3_dt
    h4 = h4 + dt * dh4_dt

    # limit of the high
    h1 = min(h1, h_max)
    h1 = max(h1, 0)
    h2 = min(h2, h_max)
    h2 = max(h2, 0)
    h3 = min(h3, h_max)
    h3 = max(h3, 0)
    h4 = min(h4, h_max)
    h4 = max(h4, 0)

    states_u.append([u[0,0], u[1,0]])
    states_y.append([y[0,0], y[1,0]])
    states_h.append([h1, h2, h3, h4])
    states_u_til.append([u_til[0,0], u_til[1,0]])
    states_y_til.append([y_til[0,0], y_til[1,0]])
 

states_y = np.array(states_y)
states_u_till = np.array(states_u_til)
states_y_till = np.array(states_y_til)
states_u = np.array(states_u)
states_h = np.array(states_h)


'''
=============================================================================================
'''

# time vector (seconds)
times = np.arange(n_steps + 1) * dt

# plot figure
plt.figure(figsize=(12, 8))

# auxiliar function for draw the attack intervals
def plot_attack_regions(time_attack, dt):
    in_attack = False
    for i in range(len(time_attack)):
        if time_attack[i] == 1 and not in_attack:
            start = i
            in_attack = True
        elif time_attack[i] == 0 and in_attack:
            end = i
            plt.axvspan(start * dt, end * dt, color='red', alpha=0.3, label='DoS Attack' if start == 500 else "")
            in_attack = False
    if in_attack:
        # close the last attack
        plt.axvspan(start * dt, len(time_attack) * dt, color='red', alpha=0.3, label='DoS Attack' if start == 500 else "")

# Subplot 1: output y
plt.subplot(3, 1, 1)
for i in range(states_y_till.shape[1]):
    plt.step(times, states_y_till[:, i], where="post", label=f'y_til{i+1}')
plot_attack_regions(time_attack, dt)
plt.xlabel('Time (s)')
plt.ylabel('Output (y)')
plt.title('System output y')
plt.legend()
plt.grid()

# Subplot 2: control (u_til)
plt.subplot(3, 1, 2)
for i in range(states_u_till.shape[1]):
    plt.step(times, states_u_till[:, i], where="post", label=f'u_til{i+1}')
plot_attack_regions(time_attack, dt)
plt.xlabel('Time (s)')
plt.ylabel('Control (u)')
plt.title('Control actions in the time')
plt.legend()
plt.grid()

# Subplot 3: tanks height
plt.subplot(3, 1, 3)
for i in range(states_h.shape[1]):
    plt.step(times, states_h[:, i], where="post", label=f'h{i+1}')
plot_attack_regions(time_attack, dt)
plt.xlabel('Time (s)')
plt.ylabel('Tanks high (h)')
plt.title('Variation in height over time')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()