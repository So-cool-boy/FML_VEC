# -*- coding: UTF-8 -*-

import math
import tools.Distance as distance

C = 3e8  # 光速 (m/s)


def calc_rssi(car, rsu):
    d = distance.calc_distance(car, rsu)

    rssi = car.power - 20 * math.log10(d * car.frequency * 4 * math.pi / C)

    return rssi
