from random import random

from numpy import sin, cos, exp, sqrt, pi, arccos
import scipy.special as sp


def optimize_laser_phases(phases: list[float], num_iterations: int, learning_rate: float) -> list[float]:
    for _ in range(num_iterations):
        # Вычисление градиента интенсивности по фазам
        gradients = calculate_gradient(phases)

        # Обновление фаз с учетом градиента и learning_rate
        phases = update_phases(phases, gradients, learning_rate)

    return phases

def random_initialization(N: int) -> list[float]:
    # Задание случайных фаз от 0 до 2*pi
    return [random() * 2 * pi for _ in range(N)]


def calculate_gradient(phases: list[float]) -> list[float]:
    # Рассчитать градиенты интенсивности по фазам

    arg_43 = (phases[3] + phases[2] - phases[1] - phases[0]) / 2
    arg_32 = (phases[2] + phases[1] - phases[3] - phases[0]) / 2
    arg_24 = (phases[1] + phases[3] - phases[0] - phases[2]) / 2

    gradient = [0]*4
    
    gradient[0] = 2 * (sin(arg_43)*cos(arg_32) + cos(arg_43)*sin(arg_32) + sin(arg_24)*(cos(arg_32) + cos(arg_43)) + cos(arg_24)*(sin(arg_32) + sin(arg_43)))
    gradient[1] = 2 * (sin(arg_43)*cos(arg_32) - cos(arg_43)*sin(arg_32) - sin(arg_24)*(cos(arg_32) + cos(arg_43)) + cos(arg_24)*(-sin(arg_32) + sin(arg_43)))
    gradient[2] = 2 * (-sin(arg_43)*cos(arg_32) - cos(arg_43)*sin(arg_32) + sin(arg_24)*(cos(arg_32) + cos(arg_43)) - cos(arg_24)*(sin(arg_32) + sin(arg_43)))
    gradient[3] = 2 * (-sin(arg_43)*cos(arg_32) + cos(arg_43)*sin(arg_32) - sin(arg_24)*(cos(arg_32) + cos(arg_43)) + cos(arg_24)*(sin(arg_32) - sin(arg_43)))

    return gradient

def update_phases(phases: list[float], gradient: list[float], learning_rate: float) -> list[float]:
    # Обновить фазы с учетом градиентов и learning_rate
    return [phase + learning_rate * gradient for phase, gradient in zip(phases, gradient)]

def calculate_intensity(u: float, v: float, z: float, phases: list[float], settings: dict):
    # Считаем интенсивность лазерного излучения в дальней зоне
    theta_x = u / z
    theta_y = v / z

    j = complex(0, 1)

    C: complex = exp(j * settings['k'] * (u*u + v*v) / (2*z)) / (j * settings['l'] * z)

    J_arg = settings['k'] * settings['a'] * sqrt(theta_x*theta_x + theta_y*theta_y)
    J = 2 * sp.j1(J_arg) / J_arg

    exponents: complex = sum([exp(j * (phases[n] - settings['k']*(theta_x*settings['source_coords'][n][0] + theta_y*settings['source_coords'][n][1]))) for n in range(settings['N'])])

    return ((pi * settings['a']**2 * settings['E0'] * J)**2 * (C * C.conjugate()) * (exponents * exponents.conjugate())).real

def calculate_integrals_for_analytic_solution(intensities_initial: list[list[float]], boundary_left: float, boundary_right: float) -> list[float]:
    n = len(intensities_initial[0])
    delta = (boundary_right - boundary_left) / (n - 1)
    middle_index = int(n/2)

    I1, I2, I3 = 0, 0, 0
    for i in range(n-1):
        I3 += sqrt(2)*delta*intensities_initial[i, i]
        I2 += sqrt(2)*delta*intensities_initial[i, n-1-i]
        I1 += delta*intensities_initial[middle_index, i]

    return [I1, I2, I3]

def analytic_phases(phases: list[float], z: float,  analytic_integrals: list[float], settings: dict) -> list[float]:
    C = 64 / 3 * pi * (settings['a']**2 * settings['E0'] / (settings['l'] * z))**2

    delta_phi_1 = arccos(analytic_integrals[1]/C - 1)
    delta_phi_2 = arccos(analytic_integrals[2]/C - 1)
    delta_phi_3 = (delta_phi_2 - delta_phi_1)/2 + arccos((analytic_integrals[0]/(2*C) - 1/2) * 1/cos((delta_phi_2 - delta_phi_1)/2))

    new_phases = [
        phases[0],
        phases[2] + delta_phi_2 - delta_phi_1 - delta_phi_3,
        phases[0] - delta_phi_2,
        phases[1] + delta_phi_1,
    ]

    return new_phases


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    R = 0.3
    settings = {
        'a': 0.1,
        'E0': 1,
        'l': 6e-5,
        'source_coords': [
            (R/2, R/2),
            (-R/2, R/2),
            (-R/2, -R/2),
            (R/2, -R/2),
        ],
        'N': 4,
    }

    settings['k'] = 2 * pi / settings['l']

    phases_initial = random_initialization(4)
    phases = optimize_laser_phases(phases_initial, 10000, 0.01)

    A = -0.05
    B = 0.05

    x = np.linspace(A, B, 200)
    y = np.linspace(A, B, 200)
    X, Y = np.meshgrid(x, y)
    Z = 100

    intensities_initial = calculate_intensity(X, Y, Z, phases_initial, settings)
    analytic_integrals = calculate_integrals_for_analytic_solution(intensities_initial, A, B)

    phases_analytic = analytic_phases(phases_initial, Z, analytic_integrals, settings)

    fig, ax = plt.subplots(1, 3)

    ax[0].imshow(intensities_initial)
    ax[0].set_title('До фазировки')

    ax[1].imshow(calculate_intensity(X, Y, Z, phases, settings))
    ax[1].set_title('После фазировки с применением градиентного спуска')

    ax[2].imshow(calculate_intensity(X, Y, Z, phases_analytic, settings))
    ax[2].set_title('После аналитической фазировки')

    plt.show()
    