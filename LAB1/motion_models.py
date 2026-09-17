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


def prob_triangular_distribution(a, b_squared):
    if abs(a) > math.sqrt(6 * b_squared):
        return 0
    else:
        return (1 / math.sqrt(6 * b_squared)) - (abs(a) / (6 * b_squared))





def sample_motion_model_velocity(u_t, x_t_1, delta_t, alpha):
    """

    """
    v, omega = u_t
    x, y, theta = x_t_1
    alpha1, alpha2, alpha3, alpha4, alpha5, alpha6 = alpha

    v_hat = v + sample_normal_distribution(alpha1 * v**2 + alpha2 * omega**2)
    omega_hat = omega + sample_normal_distribution(alpha3 * v**2 + alpha4 * omega**2)
    gamma_hat = sample_normal_distribution(alpha5 * v**2 + alpha6 * omega**2)

    r_hat = v_hat / omega_hat
    theta_hat = theta + omega_hat * delta_t

    x_d = x - r_hat * (math.sin(theta) - math.sin(theta_hat))
    y_d = y + r_hat * (math.cos(theta) - math.cos(theta_hat))
    theta_d = theta_hat + gamma_hat * delta_t

    return (x_d, y_d, theta_d)


def sample_normal_distribution(b_squared):
    b = math.sqrt(b_squared)
    total = 0
    for i in range(12):
        total += random.uniform(-b, b)

    return 0.5 * total


def sample_triangular_distribution(b_squared):
    b = math.sqrt(b_squared)
    return (math.sqrt(6) / 2) * (random.uniform(-b, b) + random.uniform(-b, b))


def motion_model_odometry(x, x_d, x_hat, x_hat_d, alpha):
    x_pos, y_pos, theta = x
    x_d_pos, y_d_pos, theta_d = x_d
    x_hat_pos, y_hat_pos, theta_hat = x_hat
    x_hat_d_pos, y_hat_d_pos, theta_hat_d = x_hat_d
    alpha1, alpha2, alpha3, alpha4, alpha5, alpha6 = alpha

    delta_rot_1 = math.atan2(y_hat_d_pos - y_hat_pos, x_hat_d_pos - x_hat_pos) - theta_hat
    delta_trans = math.sqrt((x_hat_d_pos - x_hat_pos)**2 + (y_hat_d_pos - y_hat_pos)**2)
    delta_rot_2 = theta_hat_d - theta_hat - delta_rot_1


    delta_hat_rot_1 = math.atan2(y_d_pos - y_pos, x_d_pos - x_pos) - theta
    delta_hat_trans = math.sqrt((x_d_pos - x_pos)**2 + (y_d_pos - y_pos)**2)
    delta_hat_rot_2 = theta_d - theta - delta_hat_rot_1

    p1 = prob_normal_distribution(delta_rot_1 - delta_hat_rot_1, alpha1 * abs(delta_rot_1) + alpha2 * delta_trans)
    p2 = prob_normal_distribution(delta_trans - delta_hat_trans, alpha3 * abs(delta_rot_1) + alpha4 * (abs(delta_rot_1) + abs(delta_rot_2)))
    p3 = prob_normal_distribution(delta_rot_2 - delta_hat_rot_2, alpha5 * abs(delta_rot_2) + alpha6 * delta_trans)

    return p1 * p2 * p3




