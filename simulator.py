# Quadruple-Tanks Process Simulator
# Matheus Figueiredo
# Federal University of Amazonas
# 2025

# libs
import numpy as np
import matplotlib.pyplot as plt
from controler_discrete import ControladorPIDiscretoEuler

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

r = np.array([[r1],
              [r2]])

# ====================================================================================================

# inicial conditions
u = np.zeros((2, 1)) # controlers u_k == [[u1], [u2]]
h1, h2, h3, h4 = 10, 10, 1.8, 1.4

# =====================================================================================================

# variable to keep states of the system for plot in the graphic
states_h = [[h1, h2, h3, h4]]
states_u = [[u[0,0], u[1,0]]]
states_y = [[0, 0]]

# controlers in the list format
controlers = [
    ControladorPIDiscretoEuler(kp1, ti1, dt),
    ControladorPIDiscretoEuler(kp2, ti2, dt)
]

# simulation
for k in range(n_steps):

    # values of the system output
    y1 = kc * (h1 - h1_0)
    y2 = kc * (h2 - h2_0)
    y = np.array([[y1],
                  [y2]]) 

    # errors measurements
    error = r - y

    # controlers
    for i in range(2):
        u[i] = controlers[i].calcular_saida(error[i])
    
    v0 = np.array([[v1_0],
                   [v2_0]])

    v = u + v0
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

states_y = np.array(states_y)
states_u = np.array(states_u)
states_h = np.array(states_h)

'''
=============================================================================================
'''

# plot figure
plt.figure(figsize=(12, 8))
# Subplot 1: output y
plt.subplot(3, 1, 1)
for i in range(states_y.shape[1]):
    plt.step(range(n_steps + 1), states_y[:, i], where="post", label=f'y{i+1}')
plt.xlabel('Passo de tempo')
plt.ylabel('Saída (y)')
plt.title('Saída do sistema no tempo (discreto)')
plt.legend()
plt.grid()

# Subplot 2: control (u)
plt.subplot(3, 1, 2)
for i in range(states_u.shape[1]):
    plt.step(range(n_steps + 1), states_u[:, i], where="post", label=f'y{i+1}')
plt.xlabel('Passo de tempo')
plt.ylabel('Controle (u)')
plt.title('Ações de controle ao longo do tempo')
plt.legend()
plt.grid()

# Subplot 3: tanks high 
plt.subplot(3, 1, 3)
for i in range(states_h.shape[1]):
    plt.step(range(n_steps + 1), states_h[:, i], where="post", label=f'h{i+1}')
plt.xlabel('Passo de tempo')
plt.ylabel('Altura (h)')
plt.title('Variação da altura ao longo do tempo')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()