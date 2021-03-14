class Node():
    # TODO add support class or enum
    def __init__(self, name, x, y, z, force=(0, 0, 0), momentum=(0, 0, 0), support):
        self.name = name
        self.x = x
        self.y = y
        self.z = z
        self.force = force
        self.momentum = momentum
        self.support = support


class Bar():
    def __init__(self, name, origin, end):
        self.name
        self.origin
        self.end


class Structure():
    def __init__(self, bars):
        self.bars = bars