import numpy as np
import math


def EKF_localization_known_correspondences(mu_t_1, sigma_t_1, u_t, z_t, c_t, m, alpha, sigma_z, delta_t):
    mu_t_1 = np.asarray(mu_t_1, dtype=float)
    theta = mu_t_1[2]
    vt, wt = u_t
    a1, a2, a3, a4 = alpha
    sigma_r, sigma_phi, sigma_s = sigma_z

    if abs(wt) > 1e-6:
        G_t = np.array([[1, 0, -vt/wt*math.cos(theta) + vt/wt*math.cos(theta + wt*delta_t)],
                        [0, 1, -vt/wt*math.sin(theta) + vt/wt*math.sin(theta + wt*delta_t)],
                        [0, 0, 1]])

        V_t = np.array([[(-math.sin(theta) + math.sin(theta + wt*delta_t))/wt, vt*(math.sin(theta) - math.sin(theta + wt*delta_t))/wt**2 + vt*math.cos(theta + wt*delta_t)*delta_t/wt],
                        [(math.cos(theta) - math.cos(theta + wt*delta_t))/wt, -vt*(math.cos(theta) - math.cos(theta + wt*delta_t))/wt**2 + vt*math.sin(theta + wt*delta_t)*delta_t/wt],
                        [0, delta_t]])

        motion = np.array([-vt/wt*math.sin(theta) + vt/wt*math.sin(theta + wt*delta_t),
                           vt/wt*math.cos(theta) - vt/wt*math.cos(theta + wt*delta_t),
                           wt*delta_t])
    else:
        G_t = np.array([[1, 0, -vt*delta_t*math.sin(theta)],
                        [0, 1, vt*delta_t*math.cos(theta)],
                        [0, 0, 1]])

        V_t = np.array([[delta_t*math.cos(theta), -vt*delta_t**2*math.sin(theta)/2],
                        [delta_t*math.sin(theta), vt*delta_t**2*math.cos(theta)/2],
                        [0, delta_t]])

        motion = np.array([vt*delta_t*math.cos(theta),
                           vt*delta_t*math.sin(theta),
                           0])

    M_t = np.array([[a1*vt**2 + a2*wt**2, 0],
                    [0, a3*vt**2 + a4*wt**2]])

    mu_hat_t = mu_t_1 + motion
    sigma_hat_t = G_t @ sigma_t_1 @ G_t.T + V_t @ M_t @ V_t.T

    Qt = np.array([[sigma_r**2, 0, 0],
                   [0, sigma_phi**2, 0],
                   [0, 0, sigma_s**2]])

    pzt = 1.0
    for i in range(len(z_t)):
        j = c_t[i]
        m_x, m_y, m_s = m[j]
        q = (m_x - mu_hat_t[0])**2 + (m_y - mu_hat_t[1])**2

        z_hat_t_i = np.array([math.sqrt(q),
                              math.atan2(m_y - mu_hat_t[1], m_x - mu_hat_t[0]) - mu_hat_t[2],
                              m_s])

        Hti = np.array([[-(m_x - mu_hat_t[0])/math.sqrt(q), -(m_y - mu_hat_t[1])/math.sqrt(q), 0],
                        [(m_y - mu_hat_t[1])/q, -(m_x - mu_hat_t[0])/q, -1],
                        [0, 0, 0]])

        Sti = Hti @ sigma_hat_t @ Hti.T + Qt

        Kti = sigma_hat_t @ Hti.T @ np.linalg.inv(Sti)

        dz = np.asarray(z_t[i], dtype=float) - z_hat_t_i
        dz[1] = (dz[1] + np.pi) % (2*np.pi) - np.pi

        mu_hat_t = mu_hat_t + Kti @ dz

        sigma_hat_t = (np.identity(3) - Kti @ Hti) @ sigma_hat_t

        pzt *= np.linalg.det(2*np.pi * Sti)**(-1/2) * np.exp(-1/2 * dz @ np.linalg.inv(Sti) @ dz)

    mu_hat_t[2] = (mu_hat_t[2] + np.pi) % (2*np.pi) - np.pi

    mu_t = mu_hat_t
    sigma_t = sigma_hat_t
    return mu_t, sigma_t, pzt
