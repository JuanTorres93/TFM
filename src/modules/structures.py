import enum
import math
import numpy as np
import src.modules.databaseutils as db


@enum.unique
class Support(enum.Enum):
    """
    Enumeration for the different types of supports
    """
    NONE = 1
    ROLLER = 2  # Deslizadera
    PINNED = 3  # Fijo
    FIXED = 4   # Empotramiento


class Node:
    """
    Class that represents a node in a structure.
    """
    def __init__(self, name: str, position=(0, 0, 0), force=(0, 0, 0), momentum=(0, 0, 0), support=Support.NONE):
        """
        Constructor for Node class
        :param name: Name of the node
        :param position: Cartesian coordinates of the nodes (x, y, z)
        :param force: Force applied to the node (Fx, Fy, Fz)
        :param momentum: Momentum applied to the node (Mx, My, Mz)
        :param support: Support attached to the node
        """
        if type(name) not in [str]:
            raise TypeError("name must be of type str")

        if type(position) not in [tuple]:
            raise TypeError("Position must be a tuple")

        if type(force) not in [tuple]:
            raise TypeError("Force must be a tuple")

        if type(momentum) not in [tuple]:
            raise TypeError("Momentum must be a tuple")

        if type(support) not in [Support]:
            raise TypeError("Support must be of 'Support type'.")

        self.name = name
        self.position = np.array(position)
        self.force = np.array(force)
        self.momentum = np.array(momentum)
        self.support = support
        self.number_for_matrix = 0

    def set_name(self, new_name):
        """
        Sets the name of the node to the specified one
        :param new_name: new name of the node
        :return:
        """
        if type(new_name) not in [str]:
            raise TypeError("name must be of type str")

        self.name = new_name

    def set_force(self, new_force):
        """
        Sets the force of the node to the specified one
        :param new_force: new force applied to the node
        :return:
        """
        if type(new_force) not in [tuple]:
            raise TypeError("new_force must be a tuple.")

        self.force = np.array(new_force)

    def set_momentum(self, new_momentum):
        """
        Sets the momentum of the node to the specified one
        :param new_momentum: new momentum applied to the node
        :return:
        """
        if type(new_momentum) not in [tuple]:
            raise TypeError("new_momentum must be a tuple.")

        self.momentum = np.array(new_momentum)

    def set_position(self, new_position):
        """
        Sets the position of the node to the specified one
        :param new_position: new position applied to the node
        :return:
        """
        if type(new_position) not in [tuple]:
            raise TypeError("new_position must be a tuple.")

        self.position = np.array(new_position)

    def set_support(self, new_support):
        """
        Sets the support of the node to the specified one
        :param new_support: new support applied to the node
        :return:
        """
        if type(new_support) not in [Support]:
            raise TypeError("new_support must be a tuple.")

        self.support = new_support

    def x(self):
        """

        :return: x value of position
        """
        return self.position[0]

    def y(self):
        """

        :return: y value of position
        """
        return self.position[1]

    def z(self):
        """

        :return: z value of position
        """
        return self.position[2]


class Bar:
    """
    Class that represents a bar in a structure.
    """
    def __init__(self, name: str, origin, end, material="s275j", profile=("IPE", 300)):
        """
        Constructor for Bar class
        :param name: Name of the bar
        :param origin: Node that acts as the origin of the bar
        :param end: Node that acts as the end of the bar
        :param material: string that represents a material stored in the database
        :param profile: string that represents a beam profile stored in the database
        """
        if type(origin) not in [Node] or type(end) not in [Node]:
            raise TypeError("Nodes must be of type 'Node'")

        if type(name) not in [str] or type(material) not in [str]:
            raise TypeError("name and material must be of type str")

        if origin == end:
            raise ValueError("Origin and end nodes must be different")

        self.name = name
        self.origin = origin
        self.end = end
        self.material = Material(material)
        self.profile = Profile(profile[0], profile[1])

    def set_name(self, new_name):
        """
        Sets the name of the node to the specified one
        :param new_name: new name of the node
        :return:
        """
        if type(new_name) not in [str]:
            raise TypeError("name must be of type str")

        self.name = new_name

    def set_origin(self, new_origin):
        """
        Sets the origin node of the bar to the specified one
        :param new_origin: new origin node of the bar
        :return:
        """
        if type(new_origin) not in [Node]:
            raise TypeError("new_origin must be of type 'Node'")

        self.origin = new_origin

    def set_end(self, new_end):
        """
        Sets the end node of the bar to the specified one
        :param new_end: new end node of the bar
        :return:
        """
        if type(new_end) not in [Node]:
            raise TypeError("new_end must be of type 'Node'")

        self.end = new_end

    def length(self) -> float:
        """

        :return: Length of the bar
        """
        return np.linalg.norm(np.subtract(self.end.position, self.origin.position))

    def set_material(self, mat_name):
        """
        :param mat_name: str corresponds with a unique name in the materials table of the database
        :return:
        """
        # TODO Add more materials to database and write a test for this function
        self.material = Material(mat_name)

    def set_profile(self, profile_name, profile_number):
        """
        :param profile_name: str corresponds with a name in the profile table of the database. e.g. IPE
        :param profile_number: str corresponds with a name_number in the profile table of the database
        :return:
        """
        # TODO Add more profiles to database and write a test for this function
        self.profile = Profile(profile_name, profile_number)

    def local_rigidity_matrix_2d_rigid_nodes(self, use_inertia_x=True):
        """

        :param use_inertia_x: Specifies whether the inertia moment must be selected with respect to the
        x axis (default) or the y axis (f equals False)
        :return: local rigidity matrix for a 2D structure with rigid nodes
        """
        # TODO Check when it must be used inertia momentum x or y

        l = self.length()  # l -> beam length
        e = self.material.young_mod  # e -> Young's modulus
        a = self.profile.area  # a -> cross section's area
        # i -> modulus inertia
        if use_inertia_x:
            i = self.profile.inertia_moment_x
        else:
            i = self.profile.inertia_moment_y

        return np.array([[e * a / l, 0, 0, -e * a / l, 0, 0],
                         [0, 12 * e * i / l ** 3, 6 * e * i / l ** 2, 0, -12 * e * i / l ** 3, 6 * e * i / l ** 2],
                         [0, 6 * e * i / l ** 2, 4 * e * i / l, 0, -6 * e * i / l ** 2, 2 * e * i / l],
                         [-e * a / l, 0, 0, e * a / l, 0, 0],
                         [0, -12 * e * i / l ** 3, -6 * e * i / l ** 2, 0, 12 * e * i / l ** 3, -6 * e * i / l ** 2],
                         [0, 6 * e * i / l ** 2, 2 * e * i / l, 0, -6 * e * i / l ** 2, 4 * e * i / l]])

    def _angle_from_global_to_local(self):
        """

        :return: Needed angle for converting the local matrix to global one
        """
        reference = np.array((1, 0, 0))
        beam_line = np.subtract(self.end.position, self.origin.position)

        dot_product = reference @ beam_line

        cosine = dot_product / np.linalg.norm(reference) / np.linalg.norm(beam_line)
        return np.arccos(cosine)

    def system_change_matrix_2d_rigid_nodes(self):
        """

        :return: System change matrix
        """
        # Angle must be in radians
        angle = self._angle_from_global_to_local()

        return np.array([
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle), math.cos(angle), 0],
            [0, 0, 1],
        ])

    def global_rigidity_matrix_2d_rigid_nodes(self):
        """

        :return: Global rigidity matrix of the bar for a 2d structure with rigid nodes
        """
        g = self.system_change_matrix_2d_rigid_nodes()
        l = self.local_rigidity_matrix_2d_rigid_nodes()

        m_aux = g.dot(l[0:3, 0:3])
        k_ii = m_aux.dot(np.transpose(g))

        m_aux = g.dot(l[0:3, 3:6])
        k_ij = m_aux.dot(np.transpose(g))

        m_aux = g.dot(l[3:6, 0:3])
        k_ji = m_aux.dot(np.transpose(g))

        m_aux = g.dot(l[3:6, 3:6])
        k_jj = m_aux.dot(np.transpose(g))

        top = np.hstack((k_ii, k_ij))
        bottom = np.hstack((k_ji, k_jj))

        return np.vstack((top, bottom))


class Structure:
    def __init__(self, name, bars):
        """

        :param name: Unique name of the structure
        :param bars: dictionary of bars
        """
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

        self.name = name
        self.bars = bars

    # def check_validity_of_nodes(self):
    #     """
    #     Each node must belong to, at least, two bars, unless it has a support. If one node does not fulfill one of the
    #     two options, the node will not be valid and the structure cannot be solved.
    #     :return: True if all nodes are valid, False otherwise
    #     """
    #     # If think this is not needed for rigid structures
    #     def validate_nodes(first_list, second_list):
    #         nodes_validity = []
    #
    #         for node_f in first_list:
    #             if node_f.support is not Support.NONE:
    #                 nodes_validity.append(True)
    #             else:
    #                 if node_f in second_list:
    #                     nodes_validity.append(True)
    #                 elif first_list.count(node_f) > 1:
    #                     nodes_validity.append(True)
    #                 else:
    #                     nodes_validity.append(False)
    #
    #         return  nodes_validity
    #
    #     origin_nodes = []
    #     end_nodes = []
    #     for key, bar in self.bars.items():
    #         origin_nodes.append(bar.origin)
    #         end_nodes.append(bar.end)
    #
    #     origin_nodes_validity = validate_nodes(origin_nodes, end_nodes)
    #     end_nodes_validity = validate_nodes(end_nodes, origin_nodes)
    #
    #     print(origin_nodes_validity)
    #     print(list(map(lambda x: x.name, end_nodes)))
    #     print(end_nodes_validity)

    def rigidity_matrix(self):
        pass


class Material:
    def __init__(self, name):
        """
        :param name: must be the same than those of the table in the material database
        """
        if type(name) not in [str]:
            raise TypeError("name must be of type str")
        conn = db.create_connection()

        query = """SELECT generic_name, name, young_mod, rig_mod, poisson_coef,
        thermal_dil_coef, density
        FROM materials WHERE name = '""" + name + "';"

        # All parameters must be the same than those of the table in the material database
        result = db.execute_read_query(conn, query)

        if result:
            self.generic_name, self.name, self.young_mod, self.rig_mod, self.poisson_coef, \
            self.thermal_dil_coef, self.density = result[0]
        else:
            raise LookupError(
                "Error in the query: ''" + query + "''. Or maybe the material " + name + " is not defined in the database.")


class Profile:
    def __init__(self, name, name_number):
        """
        :param name: must be the same than those of the table in the profiles database
        """
        if type(name) not in [str]:
            raise TypeError("name must be of type str")
        if type(name_number) not in [str, int]:
            raise TypeError("name must be of type str")

        if type(name_number) in [int]:
            name_number = str(name_number)

        conn = db.create_connection()

        query = """SELECT name, name_number, area, weight, inertia_moment_x, res_mod_x,
        inertia_moment_y, res_mod_y
        FROM profiles WHERE name = '""" + name + "' AND name_number = " + \
                name_number + ";"

        # All parameters must be the same than those of the table in the profile database
        result = db.execute_read_query(conn, query)

        if result:
            self.name, self.name_number, self.area, self.weight, self.inertia_moment_x, self.res_mod_x, \
            self.inertia_moment_y, self.res_mod_y = result[0]
        else:
            raise LookupError("Error in the query: ''" + query + "''. Or maybe the profile " + name + " " +
                              name_number + " is not defined in the database.")
        pass



# TODO DELETE EVERYTHING DOWN HERE IT IS JUST FOR TESTING PURPOSES
n1 = Node("N1", position=(0, 0, 0), support=Support.PINNED)
n2 = Node("N2", position=(0, 4.2, 0))
n3 = Node("N3", position=(6.8, 5.25, 0))
n4 = Node("N4", position=(13.6, 4.2, 0))
n5 = Node("N5", position=(17.2, 3.5, 0))
n6 = Node("N6", position=(13.6, 0, 0), support=Support.PINNED)

b1 = Bar("B1", n1, n2)
b2 = Bar("B2", n2, n3)
b3 = Bar("B3", n3, n4)
b4 = Bar("B4", n4, n5)
b5 = Bar("B5", n4, n6)

bars = {
    b1.name: b1,
    b2.name: b2,
    b3.name: b3,
    b4.name: b4,
    b5.name: b5
}

st = Structure("S1", bars)
