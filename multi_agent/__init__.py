from collections import namedtuple

import runners.archs as archs


def tonamedtuple(dictionary):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            dictionary[key] = tonamedtuple(value)
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)


def get_arch(name):
    constructor = getattr(archs, name)
    return constructor


def comma_sep_ints(s):
    if s:
        return list(map(int, s.split(",")))
    else:
        return []