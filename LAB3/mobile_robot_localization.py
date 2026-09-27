import numpy as np
import math


def EKF_localization_known_correspondences(mu_t_1, sigma_t_1, u_t, z_t, c_t, m, alpha, sigma, delta_t):
    mu_t_1 = np.asarray(mu_t_1, dtype=float)
    theta = mu_t_1[2]
    vt, wt = u_t
    a1, a2, a3, a4 = alpha
    sigma_r, sigma_phi, sigma_s = sigma

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



def g(ut, X_tu, X_tx_1, L, delta_t):
    X_t = []
    for i in range(2*L+1):
        vit = ut[0] + X_tu[0][i]
        wit = ut[1] + X_tu[1][i]
        thetait = X_tx_1[2][i]
        if abs(wit) > 1e-6:
            X_t.append([X_tx_1[0][i] - vit/wit*math.sin(thetait) + vit/wit*math.sin(thetait + wit*delta_t),
                       X_tx_1[1][i] + vit/wit*math.cos(thetait) - vit/wit*math.cos(thetait + wit*delta_t),
                       X_tx_1[2][i] + wit*delta_t])
        else:
            X_t.append([X_tx_1[0][i] + vit*delta_t*math.cos(thetait),
                       X_tx_1[1][i] + vit*delta_t*math.sin(thetait),
                       X_tx_1[2][i]])
    return np.array(X_t)

def h(Xt, X_tz, L, m_j):
    mx = m_j[0]
    my = m_j[1]
    Zt = []
    for i in range(2*L+1):
        Zt.append([math.sqrt((mx - Xt[i][0])**2 + (my - Xt[i][1])**2) + X_tz[0][i], 
                   math.atan2(my - Xt[i][1], mx - Xt[i][0]) - Xt[i][2] + X_tz[1][i]])

    return np.array(Zt)


def UKF_localization(mu_t_1, sigma_t_1, u_t, z_t, c_t, m, alpha, sigma, delta_t,):
    mu_t_1 = np.asarray(mu_t_1, dtype=float)
    a1, a2, a3, a4 = alpha
    sigma_r, sigma_phi = sigma[0], sigma[1]
    vt, wt = u_t

    M_t = np.array([[a1*vt**2 + a2*wt**2, 0],
                    [0, a3*vt**2 + a4*wt**2]])

    Q_t = np.array([[sigma_r**2, 0],
                    [0, sigma_phi**2]])

    mu_a_t_1 = np.concatenate([mu_t_1, np.zeros(2), np.zeros(2)])

    sigma_a_t_1 = np.zeros((7, 7))
    sigma_a_t_1[0:3, 0:3] = sigma_t_1
    sigma_a_t_1[3:5, 3:5] = M_t
    sigma_a_t_1[5:7, 5:7] = Q_t


    ukf_alpha=1.0
    beta=2.0
    kappa=0.0
    L = 7
    lam = ukf_alpha**2 * (L + kappa) - L
    gamma = np.sqrt(L + lam)

    wm = np.full(2*L + 1, 1 / (2*(L + lam)))
    wc = wm.copy()
    wm[0] = lam / (L + lam)
    wc[0] = wm[0] + (1 - ukf_alpha**2 + beta)
    
    
    #Generate sigma points
    sqrt_sigma = np.linalg.cholesky(sigma_a_t_1 + 1e-9*np.eye(L))
    X = np.column_stack([mu_a_t_1,
                         mu_a_t_1[:, None] + gamma*sqrt_sigma,
                         mu_a_t_1[:, None] - gamma*sqrt_sigma])

    #Pass sigma points through motion model and compute Gaussian statistics
    X_x = g(u_t, X[3:5], X[0:3], L, delta_t).T    

    mu_bar = X_x @ wm
    mu_bar[2] = np.arctan2(np.sin(X_x[2]) @ wm, np.cos(X_x[2]) @ wm)

    dX = X_x - mu_bar[:, None]
    dX[2] = (dX[2] + np.pi) % (2*np.pi) - np.pi
    sigma_bar = (wc * dX) @ dX.T

    #Predict observations at sigma points and compute Gaussian statistics
    Z_t = h(X_x.T, X[5:7], L, m[c_t]).T               
    Z_t[1] = (Z_t[1] + np.pi) % (2*np.pi) - np.pi

    z_hat = Z_t @ wm
    z_hat[1] = np.arctan2(np.sin(Z_t[1]) @ wm, np.cos(Z_t[1]) @ wm)

    dZ = Z_t - z_hat[:, None]
    dZ[1] = (dZ[1] + np.pi) % (2*np.pi) - np.pi
    S_t = (wc * dZ) @ dZ.T
    sigma_xz = (wc * dX) @ dZ.T


    #Update mean and covariance
    K_t = sigma_xz @ np.linalg.inv(S_t)
    dz = np.asarray(z_t, dtype=float)[:2] - z_hat
    dz[1] = (dz[1] + np.pi) % (2*np.pi) - np.pi

    mu_t = mu_bar + K_t @ dz
    mu_t[2] = (mu_t[2] + np.pi) % (2*np.pi) - np.pi
    sigma_t = sigma_bar - K_t @ S_t @ K_t.T
    pzt = np.linalg.det(2*np.pi*S_t)**(-1/2) * np.exp(-1/2 * dz @ np.linalg.inv(S_t) @ dz)

    return mu_t, sigma_t, pzt
