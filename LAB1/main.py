import math
import random



def motion_model_velocity(x_t, u_t, x_t_1, delta_t, alpha):
    """
    Calculates the probability of the robot moving from the previous pose x_t_1 to the current pose x_t, given the control u_t.

    Parameters:
        x_t: Current robot pose (x, y, theta).
        u_t: How the robot is told to move, with v being the forward speed and omega being how fast it turns.
        x_t_1: previous robot pose (x, y, theta).
        delta_t: Time between x_t_1 and x_t.
        alpha: Six parameters (alpha1, ...) that describe the amount of noise.


    The pseudocode does not include delta_t and alpha as function
    arguments, but they are passed in here so they can easily be changed
    for different simulations.

    used Normal / Gaussian, assingment doesnt specify wwhich one, but normal distrobution seems reasonable here. 
    """
    x, y, theta = x_t
    x_d, y_d, theta_d = x_t_1
    alpha1, alpha2, alpha3, alpha4, alpha5, alpha6 = alpha
    v, omega = u_t
    mu = 1 / 2 * ((x - x_d) * math.cos(theta) + (y - y_d) * math.sin(theta)) / ((y - y_d) * math.cos(theta) - (x - x_d) * math.sin(theta))

    x_star =  (x + x_d)/2 + mu*(y-y_d)
    y_star = (y + y_d)/2 + mu*(x_d-x)
    r_star = math.sqrt((x-x_star)**2+(y-y_star)**2)

    delta_theta = math.atan2(y_d-y_star, x_d-x_star) - math.atan2(y-y_star, x-x_star)

    omega_hat = delta_theta / delta_t
    v_hat = omega_hat * r_star
    gamma_hat = (theta_d - theta) / delta_t - omega_hat
    
    p1 = prob_normal_distribution(v - v_hat, alpha1 * v**2 + alpha2 * omega**2)
    p2 = prob_normal_distribution(omega - omega_hat, alpha3 * v**2 + alpha4 * omega**2)
    p3 = prob_normal_distribution(gamma_hat,alpha5 * v**2 + alpha6 * omega**2)

    return (p1*p2*p3)

def prob_normal_distribution(a, b_squared):
    return (1 / math.sqrt(2 * math.pi * b_squared)) * math.exp(-(a**2) / (2 * b_squared))


