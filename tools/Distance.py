# -*- coding: UTF-8 -*-

import math


def calc_distance(obj1, obj2):
    """
    计算不同平面两个设备之间的欧式距离
    :param obj1: 设备1
    :param obj2: 设备2
    :return: 欧式距离
    """
    distance = math.sqrt((obj1.x - obj2.x)**2 + (obj1.y - obj2.y)**2 + (obj1.z - obj2.z)**2)
    return distance
