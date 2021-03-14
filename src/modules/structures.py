# TODO improve all calculations using numpy
# TODO add support class or enum

class Node:
    # TODO CHANGE SUPPORT BY ITS TYPE AND ADD A SETTER METHOD
    def __init__(self, name: str, x: float, y: float, z: float, force=(0, 0, 0), momentum=(0, 0, 0), support=0):
        self.name = name
        self.x = x
        self.y = y
        self.z = z
        self.force = force
        self.momentum = momentum
        self.support = support

    # TODO Write tests for the methods below
    def set_name(self, new_name):
        self.name = new_name

    def set_force(self, new_force):
        self.force = new_force

    def set_momentum(self, new_momentum):
        self.momentum = new_momentum

    def set_x(self, new_x):
        self.x = new_x

    def set_y(self, new_y):
        self.y = new_y

    def set_z(self, new_z):
        self.z = new_z

    def set_position(self, new_position):
        self.x = new_position.x
        self.y = new_position.y
        self.z = new_position.z


class Bar:
    def __init__(self, name: str, origin, end):
        # origin: Node
        # bar: Node

        self.name = name
        self.origin = origin
        self.end = end

    # TODO Write tests for the methods below
    def set_name(self, new_name):
        self.name = new_name

    def set_origin(self, new_origin):
        self.origin = new_origin

    def set_end(self, new_end):
        self.end = new_end


class Structure:
    # TODO When creating a structure check that there aren't multiple bars with the same name
    def __init__(self, name, bars):
        # TODO bars will be a list, or maybe better, a ditionary of bars
        self.bars = bars
