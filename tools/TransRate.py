# -*- coding: UTF-8 -*-

import math
import tools.Distance as distance

BANDWIDTH = 50000000  # 带宽50MHz
GN = 0.00000000001  # 高斯噪声-140dBm/Hz
PL_EXPONENT = 3  # 路径损耗指数
ENVIRONMENT_A = 11.61  # LOS环境参数A
ENVIRONMENT_B = 0.12  # LOS环境参数B
LOS_ATTENUATION = 1.7  # LOS格外衰减
NLOS_ATTENUATION = 24  # NLOS格外衰减
C = 3e8  # 光速 (m/s)


def v2r_rate(car, other_cars, rsu):
    """
    V2R通讯速率
    :param car: 当前车辆
    :param other_cars: 其他同样卸载到RSU的车辆
    :param rsu: 路边单元
    :return: 通讯速率bits/s
    """
    d = distance.calc_distance(car, rsu)
    other_noise = 0
    for other_car in other_cars:
        other_noise += (math.pow(distance.calc_distance(other_car, rsu), -PL_EXPONENT) * other_car.power)
    SINR = (math.pow(d, -PL_EXPONENT) * car.power) / (GN + other_noise)
    return BANDWIDTH * math.log(1 + SINR, 2)


def calc_los(car, uav):
    angle = math.asin(uav.z / distance.calc_distance(car, uav))
    return 1 / (1 + ENVIRONMENT_A * pow(math.e, (-ENVIRONMENT_B * (((180 / math.pi) * angle) - ENVIRONMENT_A))))


def calc_loss(car, uav):
    p_los = calc_los(car, uav)
    p_nlos = 1 - p_los
    free_space_loss = 20 * math.log((distance.calc_distance(car, uav) * car.frequency * 4 * math.pi) / C, 10)
    los_loss = free_space_loss + LOS_ATTENUATION
    nlos_loss = free_space_loss + NLOS_ATTENUATION
    loss = p_los * los_loss + p_nlos * nlos_loss

    return loss