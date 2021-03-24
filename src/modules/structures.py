# TODO improve all calculations using numpy
# TODO add support class or enum ¿tuple?
import math
import numpy as np

import src.modules.database_utils as db

# Create the database if it does not exist
db.regenerate_initial_database()


class Node:
    # TODO CHANGE SUPPORT BY ITS TYPE AND ADD A SETTER METHOD
    def __init__(self, name: str, position=(0, 0, 0), force=(0, 0, 0), momentum=(0, 0, 0), support=0):
        if type(name) not in [str]:
            raise TypeError("name must be of type str")

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
            raise TypeError("Nodes must be of type 'Node'")

        if type(name) not in [str]:
            raise TypeError("name must be of type str")

        if origin == end:
            raise ValueError("Origin and end nodes must be different")

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

    def length(self) -> float:
        return math.sqrt((self.end.x() - self.origin.x()) ** 2 +
                         (self.end.y() - self.origin.y()) ** 2 +
                         (self.end.z() - self.origin.z()) ** 2)

    # TODO implement material and section as tables of a database
    def set_material(self, mat_name):
        """
        :param mat_name: str corresponds with a unique name in the materials table of the database
        :return:
        """
        pass

    def set_section(self):
        pass

    def local_rigidity_matrix_2d_rigid_nodes(self, e, a, i):
        # TODO Possible improvement, get e, a and i from material and section properties

        """e -> Young's modulus
        a -> cross section's area
        l -> beam length
        i -> modulus inertia"""

        l = self.length()

        return np.array([[e * a / l, 0, 0, -e * a / l, 0, 0],
                         [0, 12 * e * i / l ** 3, 6 * e * i / l ** 2, 0, -12 * e * i / l ** 3, 6 * e * i / l ** 2],
                         [0, 6 * e * i / l ** 2, 4 * e * i / l, 0, -6 * e * i / l ** 2, 2 * e * i / l],
                         [-e * a / l, 0, 0, e * a / l, 0, 0],
                         [0, -12 * e * i / l ** 3, -6 * e * i / l ** 2, 0, 12 * e * i / l ** 3, -6 * e * i / l ** 2],
                         [0, 6 * e * i / l ** 2, 2 * e * i / l, 0, -6 * e * i / l ** 2, 4 * e * i / l]])

    def system_change_matrix_2d_rigid_nodes(self, angle):
        """angle represents the angle that the global axis must be rotated in order to be the same
        than the local ones. angle is assumed in radians"""

        return np.array([
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle), math.cos(angle), 0],
            [0, 0, 1],
        ])


class Structure:
    # TODO When creating a structure check that there aren't multiple bars with the same name
    def __init__(self, name, bars):
        if type(name) not in [str]:
            raise TypeError("name must be of type str")
        if type(bars) not in [dict]:
            raise TypeError("bars must be of a dictionary of bars")

        # Check if every bar has a unique name
        bar_names = []

        for key, bar in bars.items():
            if bar.name not in bar_names:
                bar_names.append(bar.name)
            else:
                raise ValueError("More than one bar with name '" + bar.name + "'")

        del bar_names

        # TODO bars will be a dictionary of bars
        self.name = name
        self.bars = bars


class Material:
    def __init__(self, name):
        """
        :param name: must be the same than those of the table in the material database
        """
        conn = db.create_connection()

        # All parameters must be the same than those of the table in the material database
        self.generic_name, self.name, self.e = db.execute_read_query("""SELECT generic_name, name, e
        FROM materials:""")
