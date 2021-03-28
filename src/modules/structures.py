# TODO add support class or enum ¿tuple?
import math
import numpy as np

import src.modules.databaseutils as db


class Node:
    # TODO CHANGE SUPPORT BY ITS TYPE AND ADD A SETTER METHOD
    def __init__(self, name: str, position=(0, 0, 0), force=(0, 0, 0), momentum=(0, 0, 0), support=0):
        if type(name) not in [str]:
            raise TypeError("name must be of type str")

        self.name = name
        self.position = np.array(position)
        self.force = np.array(force)
        self.momentum = np.array(momentum)
        self.support = support

    def set_name(self, new_name):
        self.name = new_name

    def set_force(self, new_force):
        self.force = np.array(new_force)

    def set_momentum(self, new_momentum):
        self.momentum = np.array(new_momentum)

    def set_position(self, new_position):
        self.position = np.array(new_position)

    def x(self):
        return self.position[0]

    def y(self):
        return self.position[1]

    def z(self):
        return self.position[2]


class Bar:
    def __init__(self, name: str, origin, end, material="s275j", profile=("IPE", 300)):
        # origin: Node
        # bar: Node
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
        self.name = new_name

    def set_origin(self, new_origin):
        self.origin = new_origin

    def set_end(self, new_end):
        self.end = new_end

    def length(self) -> float:
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
