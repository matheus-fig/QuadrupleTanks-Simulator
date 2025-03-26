# Modelagem Quadruple-Tanks Process
# Matheus Figueiredo
# Universidade Federal do Amazonas
# 2025

import numpy as np
import matplotlib.pyplot as plt
from controler_discrete import ControladorPIDiscretoEuler

# parametros do sistema definidos no artigo
a1, a2, a3, a4 = 0.071, 0.057, 0.071, 0.057
A1, A2, A3, A4 = 28, 32, 28, 32
g = 981
k1, k2 = 3.33, 3.35
gamma1, gamma2 = 0.7, 0.6
h_max = 30     # altura max dos tanques
kc = 0.5

# parametros para o controlador
kp1 = 3
kp2 = 2.7
ti1 = 30
ti2 = 40

# ponto de operacao
h1_0, h2_0, h3_0, h4_0 = 12.4, 12.7, 1.8, 1.4
v1_0, v2_0 = 3, 3
dt = 0.05     # passo de tempo
n_steps = 7000     # passos de simulação

# volts desejado
r1 = -4.2
r2 = -4.35

# ====================================================================================================

# cond iniciais
u1 = 0
u2 = 0
h1, h2, h3, h4 = 10, 10, 1.8, 1.4

# =====================================================================================================

states = [[h1, h2, h3, h4]] # armazenamento de estados
errors = [[0, 0]] # armazenamento de erros

# controladores
controler_1 = ControladorPIDiscretoEuler(kp1, ti1, dt)
controler_2 = ControladorPIDiscretoEuler(kp2, ti2, dt)

# simulacao
for k in range(n_steps):

    # implementar os valores de y1 y2
    y1 = kc * (h1 - h1_0)
    y2 = kc * (h2 - h2_0)

    # medicao dos erros
    erro1 = r1 - y1
    erro2 = r2 - y2

    # controladores
    u1 = controler_1.calcular_saida(erro1)
    u2 = controler_2.calcular_saida(erro2)

    v1 = u1 + v1_0
    v2 = u2 + v2_0

    dh1_dt = -((a1 * np.sqrt(2 * g * h1)) / A1) + ((a3 * np.sqrt(2 * g * h3)) / A1) + ((gamma1 * k1 * v1) / A1)
    dh2_dt = -((a2 * np.sqrt(2 * g * h2)) / A2) + ((a4 * np.sqrt(2 * g * h4)) / A2) + ((gamma2 * k2 * v2) / A2)
    dh3_dt = -((a3 * np.sqrt(2 * g * h3)) / A3) + (((1 - gamma2) * k2 * v2) / A3)
    dh4_dt = -((a4 * np.sqrt(2 * g * h4)) / A4) + (((1 - gamma1) * k1 * v1) / A4)

    h1 = h1 + dt * dh1_dt
    h2 = h2 + dt * dh2_dt
    h3 = h3 + dt * dh3_dt
    h4 = h4 + dt * dh4_dt

    h1 = min(h1, h_max)
    h1 = max(h1, 0)
    h2 = min(h2, h_max)
    h2 = max(h2, 0)
    h3 = min(h3, h_max)
    h3 = max(h3, 0)
    h4 = min(h4, h_max)
    h4 = max(h4, 0)

    print(f"Passo {k}: u1={u1}, u2={u2}")

    states.append([h1, h2, h3, h4])
    errors.append([erro1, erro2])

states = np.array(states)

plt.figure(figsize=(10, 6))
for i in range(states.shape[1]):
    plt.step(range(n_steps + 1), states[:, i], where="post", label=f'h{i+1}')  # Passos discretos
plt.xlabel('Passo de tempo')
plt.ylabel('Altura (h)')
plt.title('Evolução dos estados no tempo (discreto)')
plt.legend()
plt.grid()
plt.show()