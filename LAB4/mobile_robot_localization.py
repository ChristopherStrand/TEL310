import numpy as np
from motion_models import motion_model_velocity
from sensor_models import beam_range_finder_model


def grid_localization(p_k_t_1, u_t, z_t, m, cells, delta_t, alpha, theta_par, z_max, beam_angles):
    K = len(cells)
    p_k_t = np.zeros(K)

    for k in range(K):
        p_bar_k_t = 0
        for i in range(K):
            p_bar_k_t += p_k_t_1[i] * motion_model_velocity(cells[k], u_t, cells[i], delta_t, alpha)
        p_k_t[k] = p_bar_k_t * beam_range_finder_model(z_t, cells[k], m, theta_par, z_max, beam_angles)

    eta = 1 / np.sum(p_k_t)
    return eta * p_k_t
