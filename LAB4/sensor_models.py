import math

def prob_normal_distribution(a, b_squared):
    return (1 / math.sqrt(2 * math.pi * b_squared)) * math.exp(-(a**2) / (2 * b_squared))


def ray_casting(x_t, beam_angle, m, z_max, sensor_offset=(0.0, 0.0)):
    x, y, theta = x_t
    x_sens, y_sens = sensor_offset
    x_0 = x + x_sens * math.cos(theta) - y_sens * math.sin(theta)
    y_0 = y + x_sens * math.sin(theta) + y_sens * math.cos(theta)
    d_x = math.cos(theta + beam_angle)
    d_y = math.sin(theta + beam_angle)

    closest = z_max
    for (x_1, y_1), (x_2, y_2) in m:
        s_x = x_2 - x_1
        s_y = y_2 - y_1
        denominator = d_x * s_y - d_y * s_x
        if abs(denominator) < 1e-12:
            continue  

        t = ((x_1 - x_0) * s_y - (y_1 - y_0) * s_x) / denominator
        u = ((x_1 - x_0) * d_y - (y_1 - y_0) * d_x) / denominator
        if t >= 0 and 0 <= u <= 1 and t < closest:
            closest = t

    return closest


def p_hit(z, z_star, z_max, sigma_hit):
    if not 0 <= z <= z_max:
        return 0.0
    eta = 1 / (0.5 * (math.erf((z_max - z_star) / (sigma_hit * math.sqrt(2)))
                      - math.erf((0 - z_star) / (sigma_hit * math.sqrt(2)))))
    return eta * prob_normal_distribution(z - z_star, sigma_hit**2)


def p_short(z, z_star, lambda_short):
    if not 0 <= z <= z_star:
        return 0.0
    eta = 1 / (1 - math.exp(-lambda_short * z_star))
    return eta * lambda_short * math.exp(-lambda_short * z)


def p_max(z, z_max):
    return 1.0 if z >= z_max else 0.0


def p_rand(z, z_max):
    if not 0 <= z < z_max:
        return 0.0
    return 1 / z_max


def beam_range_finder_model(z_t, x_t, m, theta_par, z_max, beam_angles, sensor_offset=(0.0, 0.0)):
    z_hit, z_short, z_max_weight, z_rand, sigma_hit, lambda_short = theta_par

    q = 1
    for k in range(len(z_t)):
        z_k = z_t[k]
        z_k_star = ray_casting(x_t, beam_angles[k], m, z_max, sensor_offset)

        p = (z_hit * p_hit(z_k, z_k_star, z_max, sigma_hit)
             + z_short * p_short(z_k, z_k_star, lambda_short)
             + z_max_weight * p_max(z_k, z_max)
             + z_rand * p_rand(z_k, z_max))

        q = q * p

    return q


