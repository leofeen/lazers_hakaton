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

def calculate_intensity(u: float, v: float, z: float, phases: list[float], settings: dict) -> float:
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
    # Численно считаем интегралы для аналитической формулы

    n = len(intensities_initial[0])
    delta = (boundary_right - boundary_left) / (n - 1)
    middle_index = int(n/2)

    I1, I2, I3 = 0, 0, 0
    for i in range(n):
        I3 += sqrt(2)*delta*intensities_initial[i, i]
        I2 += sqrt(2)*delta*intensities_initial[i, n-1-i]
        I1 += delta*intensities_initial[middle_index, i]

    I1 -= (intensities_initial[middle_index, 0] + intensities_initial[middle_index, -1])/2*delta
    I3 -= (intensities_initial[0, 0] + intensities_initial[-1, -1])/2*delta*sqrt(2)
    I2 -= (intensities_initial[-1, 0] + intensities_initial[0, -1])/2*delta*sqrt(2)

    return [I1, I2, I3]

def analytic_phases(z: float,  analytic_integrals: list[float], settings: dict) -> list[float]:
    # Аналитически считаем все возможные разности фаз

    C = 64 / 3 * pi * (settings['a']**2 * settings['E0'] / (settings['l'] * z))**2 / (settings['k'] * settings['a']) * z

    delta_phi_1 = [0]*8    
    delta_phi_2 = [0]*8    
    delta_phi_3 = [0]*8    

    delta_phi_1[0] = arccos(analytic_integrals[1]/C - 2)
    delta_phi_2[0] = arccos(analytic_integrals[2]/C - 2)
    delta_phi_3[0] = (delta_phi_2[0] - delta_phi_1[0])/2 + arccos((analytic_integrals[0]/(2*C) - 1) / cos((delta_phi_2[0] - delta_phi_1[0])/2))

    delta_phi_1[1] = -arccos(analytic_integrals[1]/C - 2)
    delta_phi_2[1] = arccos(analytic_integrals[2]/C - 2)
    delta_phi_3[1] = (delta_phi_2[1] - delta_phi_1[1])/2 + arccos((analytic_integrals[0]/(2*C) - 1) / cos((delta_phi_2[1] - delta_phi_1[1])/2))

    delta_phi_1[2] = arccos(analytic_integrals[1]/C - 2)
    delta_phi_2[2] = -arccos(analytic_integrals[2]/C - 2)
    delta_phi_3[2] = (delta_phi_2[2] - delta_phi_1[2])/2 + arccos((analytic_integrals[0]/(2*C) - 1) / cos((delta_phi_2[2] - delta_phi_1[2])/2))

    delta_phi_1[3] = -arccos(analytic_integrals[1]/C - 2)
    delta_phi_2[3] = -arccos(analytic_integrals[2]/C - 2)
    delta_phi_3[3] = (delta_phi_2[3] - delta_phi_1[3])/2 + arccos((analytic_integrals[0]/(2*C) - 1) / cos((delta_phi_2[3] - delta_phi_1[3])/2))

    delta_phi_1[4] = arccos(analytic_integrals[1]/C - 2)
    delta_phi_2[4] = arccos(analytic_integrals[2]/C - 2)
    delta_phi_3[4] = (delta_phi_2[4] - delta_phi_1[4])/2 - arccos((analytic_integrals[0]/(2*C) - 1) / cos((delta_phi_2[4] - delta_phi_1[4])/2))

    delta_phi_1[5] = -arccos(analytic_integrals[1]/C - 2)
    delta_phi_2[5] = arccos(analytic_integrals[2]/C - 2)
    delta_phi_3[5] = (delta_phi_2[5] - delta_phi_1[5])/2 - arccos((analytic_integrals[0]/(2*C) - 1) / cos((delta_phi_2[5] - delta_phi_1[5])/2))

    delta_phi_1[6] = arccos(analytic_integrals[1]/C - 2)
    delta_phi_2[6] = -arccos(analytic_integrals[2]/C - 2)
    delta_phi_3[6] = (delta_phi_2[6] - delta_phi_1[6])/2 - arccos((analytic_integrals[0]/(2*C) - 1) / cos((delta_phi_2[6] - delta_phi_1[6])/2))

    delta_phi_1[7] = -arccos(analytic_integrals[1]/C - 2)
    delta_phi_2[7] = -arccos(analytic_integrals[2]/C - 2)
    delta_phi_3[7] = (delta_phi_2[7] - delta_phi_1[7])/2 - arccos((analytic_integrals[0]/(2*C) - 1) / cos((delta_phi_2[7] - delta_phi_1[7])/2))

    return delta_phi_1, delta_phi_2, delta_phi_3


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # Задаём начальные условия
    R = 0.3  # cm
    settings = {
        'a': 0.1,  # cm
        'E0': 1,
        'l': 6e-5,  # cm
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
    phases_numeric_from_known = optimize_laser_phases(phases_initial, 10000, 0.01)


    # Без _visual используются для расчётов
    A = -4
    B = 4

    x = np.linspace(A, B, 10000)
    y = np.linspace(A, B, 10000)
    X, Y = np.meshgrid(x, y)
    Z = 100

    A_visual = -0.05
    B_visual = 0.05
    x_visual = np.linspace(A_visual, B_visual, 200)
    y_visual = np.linspace(A_visual, B_visual, 200)
    X_visual, Y_visual = np.meshgrid(x_visual, y_visual)

    intensities_initial = calculate_intensity(X, Y, Z, phases_initial, settings)
    analytic_integrals = calculate_integrals_for_analytic_solution(intensities_initial, A, B)
    phases_analytic_variants = analytic_phases(Z, analytic_integrals, settings)

    # Считаем все возможные варианты разностей фаз, получающиеся распределения
    # интенсивностей и выбираем с наименьшей ошибкой
    intensities_variants = []
    counter = -1
    error_min, i_min = 1e100, 0
    for del1, del2, del3 in zip(phases_analytic_variants[0], phases_analytic_variants[1], phases_analytic_variants[2]):
        counter += 1
        intensities_variants.append(calculate_intensity(X, Y, Z, [0, del1 - del3, -del2, -del3], settings))
        error = np.sum((intensities_initial - intensities_variants[-1])**2)
        if (error < error_min):
            error_min = error
            i_min = counter

    del_min1 = phases_analytic_variants[0][i_min]
    del_min2 = phases_analytic_variants[1][i_min]
    del_min3 = phases_analytic_variants[2][i_min]

    # Изменяем фазы на полученные разности для выравнивания
    phases_correction = [0, del_min1-del_min3, -del_min2, -del_min3]
    phases_corrected = [phases_initial[i] - phases_correction[i] for i in range(len(phases_correction))]

    print(f"Полученные разности фаз: {phases_correction}")

    intensities_minimal_error = calculate_intensity(X_visual, Y_visual, Z, phases_correction, settings)
    intensities_corrected = calculate_intensity(X_visual, Y_visual, Z, phases_corrected, settings)
    intensities_initial_visual = calculate_intensity(X_visual, Y_visual, Z, phases_initial, settings)
    intensities_perfect = calculate_intensity(X_visual, Y_visual, Z, phases_numeric_from_known, settings)

    middle_index = len(x_visual)//2
    I_corrected = intensities_corrected[middle_index, middle_index]
    I_perfect = intensities_perfect[middle_index, middle_index]
    print(f"Отношение идеальной интенсивности к полученной: {I_perfect / I_corrected}")

    fig, ax = plt.subplots(2, 2)

    ax[0, 0].imshow(intensities_initial_visual)
    ax[0, 0].set_title('До фазировки')

    ax[0, 1].imshow(intensities_perfect)
    ax[0, 1].set_title('Ожидаемый результат')

    ax[1, 0].imshow(intensities_minimal_error)
    ax[1, 0].set_title("Предсказанное распределение интенсивностей")

    ax[1, 1].imshow(intensities_corrected)
    ax[1, 1].set_title("Скорректированные фазы")

    plt.show()
    