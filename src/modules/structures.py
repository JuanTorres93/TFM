# TODO improve all calculations using numpy
# TODO add support class or enum ¿tuple?
import math

class Node:
    # TODO CHANGE SUPPORT BY ITS TYPE AND ADD A SETTER METHOD
    def __init__(self, name: str, position=(0, 0, 0), force=(0, 0, 0), momentum=(0, 0, 0), support=0):
        if type(name) not in [str]:
            raise TypeError ("name must be of type str")

        self.name = name
        self.position = position
        self.force = force
        self.momentum = momentum
        self.support = support

    def set_name(self, new_name):
        self.name = new_name

    def set_force(self, new_force):
        self.force = new_force

    def set_momentum(self, new_momentum):
        self.momentum = new_momentum

    def set_position(self, new_position):
        self.position = new_position

    def x(self):
        return self.position[0]

    def y(self):
        return self.position[1]

    def z(self):
        return self.position[2]


class Bar:
    # TODO Add material and section
    def __init__(self, name: str, origin, end):
        # origin: Node
        # bar: Node
        if type(origin) not in [Node] or type(end) not in [Node]:
            raise TypeError ("Nodes must be of type 'Node'")

        if type(name) not in [str]:
            raise TypeError ("name must be of type str")

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

    def length(self):
        return math.sqrt((self.end.x() - self.origin.x())**2 +
                         (self.end.y() - self.origin.y())**2 +
                         (self.end.z() - self.origin.z())**2)

    def set_material(self):
        pass

    def set_section(self):
        pass


class Structure:
    # TODO When creating a structure check that there aren't multiple bars with the same name
    def __init__(self, name, bars):
        # TODO bars will be a list, or maybe better, a ditionary of bars
        self.bars = bars
