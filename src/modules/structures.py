import enum
import math
import sys

import numpy as np
import src.modules.databaseutils as db
import src.modules.filesystemutils as fs

# For structures with rigid nodes, the size of the submatrixes is 3
submatrix_size = 3


@enum.unique
class Support(enum.Enum):
    """
    Enumeration for the different types of supports
    """
    NONE = 1
    ROLLER_X = 2  # Deslizadera
    ROLLER_Y = 5  # Deslizadera
    PINNED = 3  # Fijo
    FIXED = 4  # Empotramiento


@enum.unique
class DistributedChargeType(enum.Enum):
    """
    Enumeration for the different types of distributed charges
    """
    SQUARE = 1


class Node:
    """
    Class that represents a node in a structure.
    """

    def __init__(self, name: str, position=(0, 0, 0), forces_in_node=None, momentums_in_node=None,
                 support=Support.NONE):
        """
        Constructor for Node class
        :param name: Name of the node
        :param position: Cartesian coordinates of the nodes (x, y, z)
        :param forces_in_node: Dictionary representing forces applied directly to the node (Fx, Fy, Fz). Forces
        that appear as a result of refering charges to nodes are handled differently
        :param momentums_in_node: Dictionary representing momentums applied to the node (Mx, My, Mz). Momentums
        that appear as a result of refering charges to nodes are handled differently
        :param support: Support attached to the node
        """
        # forces and momentums are not directly assigned an empty dictionary because it does not work as expected
        if forces_in_node is None:
            forces_in_node = {}

        if momentums_in_node is None:
            momentums_in_node = {}

        if type(name) not in [str]:
            raise TypeError("name must be of type str")

        if type(position) not in [tuple]:
            raise TypeError("Position must be a tuple")

        if type(forces_in_node) not in [dict]:
            raise TypeError("forces must be a dict")

        if type(momentums_in_node) not in [dict]:
            raise TypeError("Momentum must be a dict")

        if type(support) not in [Support]:
            raise TypeError("Support must be of 'Support type'.")

        self.name = name
        self.position = position
        self.forces_in_node = forces_in_node
        self.referred_forces = {}
        self.momentums_in_node = momentums_in_node
        self.referred_momentums = {}
        self.support = support

        # Number assigned to construct the structure matrix (handled from structure class)
        self.solving_numeration = -1

        # Displacement of the node (handled from structure class)
        self.displacement = {}

        # Reactions in the node (handled from structure class)
        self.reactions = {}

    def equals(self, other_node):
        """
        Checks if this node is equal to other_node
        :param other_node: node instance to compare to
        :return: true if both nodes are equals, else otherwise
        """
        if self.position == other_node.position and \
                self.support == other_node.support:
            return True
        else:
            return False

    def set_name(self, new_name: str):
        """
        Sets the name of the node to the specified one
        :param new_name: new name of the node
        :return:
        """
        if type(new_name) not in [str]:
            raise TypeError("name must be of type str")

        self.name = new_name

    def set_position(self, new_position: (float, float, float)):
        """
        Sets the position of the node to the specified one
        :param new_position: new position applied to the node
        :return:
        """
        if type(new_position) not in [tuple]:
            raise TypeError("new_position must be a tuple.")

        self.position = np.array(new_position)

    def set_support(self, new_support: Support):
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

    def get_forces_in_node_dictionary(self):
        """

        :return: dictionary representing the forces applied to the node
        """
        return self.forces_in_node

    def get_referred_forces_dictionary(self):
        """

        :return: dictionary representing the forces applied to the node
        """
        return self.referred_forces

    def clear_referred_forces(self):
        self.referred_forces = {}

    def add_force(self, key, force, belongs_to_node):
        """
        Adds a force to the node
        :param key: key to store the force in the dictionary
        :param force: tuple representing the applied force (Fx, Fy, Fz)
        :param belongs_to_node: if true the force is applied directly to the node. If false, the force is a consecuence
        of a charge applied to the bar
        :return:
        """
        if type(force) in [tuple]:
            force = np.array(force)

        if type(force) not in [np.ndarray]:
            raise TypeError("Error. force must be of type numpy.array")

        if type(belongs_to_node) not in [bool]:
            raise TypeError("Error. belongs_to_node must be of type bool")

        if belongs_to_node:
            self.forces_in_node[key] = force
        else:
            self.referred_forces[key] = force

    def get_momentum_in_node_dictionary(self):
        """

        :return: dictionary representing the momentums applied to the node
        """
        return self.momentums_in_node

    def get_referred_momentum_dictionary(self):
        """

        :return: dictionary representing the momentums applied to the node
        """
        return self.referred_momentums

    def clear_referred_momentums(self):
        self.referred_momentums = {}

    def add_momentum(self, key, momentum, belongs_to_node):
        """
        Adds a momentum to the node
        :param key: key to store the momentum in the dictionary
        :param momentum: tuple representing the applied momentum (Mx, My, Mz)
        :param belongs_to_node: if true the momentum is applied directly to the node. If false, the momentum is a consecuence
        of a charge applied to the bar
        :return:
        """
        if type(momentum) in [tuple]:
            momentum = np.array(momentum)

        if type(momentum) not in [np.ndarray]:
            raise TypeError("Error. momentum must be of type numpy.array")

        if type(belongs_to_node) not in [bool]:
            raise TypeError("Error. belongs_to_node must be of type bool")

        if belongs_to_node:
            self.momentums_in_node[key] = momentum
        else:
            self.referred_momentums[key] = momentum

    def get_total_force_and_momentum(self):
        """

        :return: [x force, y force, momentum]
        """
        x_force = 0
        y_force = 0
        z_momentum = 0

        # Forces
        if len(self.forces_in_node) > 0:
            for key, force in self.forces_in_node.items():
                x_force += force[0]
                y_force += force[1]

        if len(self.referred_forces) > 0:
            for key, force in self.referred_forces.items():
                x_force += force[0]
                y_force += force[1]

        # Momentums
        if len(self.momentums_in_node) > 0:
            for key, momentum in self.momentums_in_node.items():
                z_momentum += momentum[2]

        if len(self.referred_momentums) > 0:
            for key, momentum in self.referred_momentums.items():
                z_momentum += momentum[2]

        return np.array([x_force, y_force, z_momentum])

    def set_displacement(self, new_displacement):
        """

        :param new_displacement: dictionary of x, y and angle displacements
        :return:
        """

        if type(new_displacement) not in [dict]:
            raise TypeError("new_displacement must be a dictionary")

        self.displacement = new_displacement

    def get_displacement(self):
        """

        :return: Node displacement
        """

        return self.displacement

    def set_reactions(self, new_reactions):
        """

        :param new_reactions: dictionary of x, y and moment reactions
        :return:
        """

        if type(new_reactions) not in [dict]:
            raise TypeError("new_reactions must be a dictionary")

        self.reactions = new_reactions

    def get_reactions(self):
        """

        :return: Node reactions
        """

        return self.reactions


class Bar:
    """
    Class that represents a bar in a structure.
    """

    def __init__(self, name: str, origin: Node, end: Node, material="s275j", profile=("IPE", 300)):
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

        # Number assigned to build the structure (assembled) matrix (handle from structure class)
        self.solving_numeration = -1

        # This submatrixes are here exposed for an easier way to assemble the assembled matrix from the structure class
        # Do not change their initialization
        self.k_ii = None
        self.k_ij = None
        self.k_ji = None
        self.k_jj = None

        # Distributed charges applied to the bar
        self.distributed_charges = {}

        # Punctual forces applied to the bar
        self.punctual_forces = {}

    def set_name(self, new_name: str):
        """
        Sets the name of the node to the specified one
        :param new_name: new name of the node
        :return:
        """
        if type(new_name) not in [str]:
            raise TypeError("name must be of type str")

        self.name = new_name

    def set_origin(self, new_origin: Node):
        """
        Sets the origin node of the bar to the specified one
        :param new_origin: new origin node of the bar
        :return:
        """
        if type(new_origin) not in [Node]:
            raise TypeError("new_origin must be of type 'Node'")

        self.origin = new_origin

    def set_end(self, new_end: Node):
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

    def set_material(self, mat_name: str):
        """
        :param mat_name: str corresponds with a unique name in the materials table of the database
        :return:
        """
        # TODO Add more materials to database and write a test for this function
        self.material = Material(mat_name)

    def set_profile(self, profile_name: str, profile_number: int):
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

    def angle_from_global_to_local(self):
        """

        :return: Needed angle for converting the local matrix to global one
        """
        reference = np.array((1, 0, 0))
        beam_line = np.subtract(self.end.position, self.origin.position)

        dot_product = reference @ beam_line

        cosine = dot_product / np.linalg.norm(reference) / np.linalg.norm(beam_line)
        angle = np.arccos(cosine)

        if self.end.x() >= self.origin.x() and self.end.y() >= self.origin.y():
            # angle = angle
            pass
        elif self.end.x() <= self.origin.x() and self.end.y() >= self.origin.y():
            angle = angle + math.pi / 2
        elif self.end.x() <= self.origin.x() and self.end.y() <= self.origin.y():
            angle = angle + math.pi
        else:
            angle = 2 * math.pi - angle

        return angle

    def system_change_matrix_2d_rigid_nodes(self):
        """

        :return: System change matrix
        """
        # Angle must be in radians
        angle = self.angle_from_global_to_local()

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
        self.k_ii = m_aux.dot(np.transpose(g))

        m_aux = g.dot(l[0:3, 3:6])
        self.k_ij = m_aux.dot(np.transpose(g))

        m_aux = g.dot(l[3:6, 0:3])
        self.k_ji = m_aux.dot(np.transpose(g))

        m_aux = g.dot(l[3:6, 3:6])
        self.k_jj = m_aux.dot(np.transpose(g))

        top = np.hstack((self.k_ii, self.k_ij))
        bottom = np.hstack((self.k_ji, self.k_jj))

        return np.vstack((top, bottom))

    def _add_object_to_instance_dictionary(self, dictionary, obj, obj_name, obj_type=None):
        """
        Function to include an object to an instance dictionary
        :param dictionary: dictorionary to include the new object into
        :param obj: object to include in the dicttionary
        :param obj_name: str key to assign to the object
        :param obj_type: if not None, then forces obj to be of type obj_type
        :return:
        """

        if obj_type is not None:
            if type(obj) not in [obj_type]:
                raise TypeError("new_distributed_charge must be of type " + str(obj_type))

        if obj_name is None:
            obj_name = fs.get_random_name("dc")

            while obj_name in dictionary.keys():
                obj_name = fs.get_random_name("dc")

        if type(obj_name) not in [str]:
            raise TypeError("obj_name must be of type str or None")

        dictionary[obj_name] = obj

    def add_distributed_charge(self, new_distributed_charge, dc_name=None):
        """
        Adds the specified distributed charge to the collection of distributed charges applied to the bar
        :param new_distributed_charge: DistributedCharge to be added
        :param dc_name: name to assign in dictionary
        :return:
        """

        self._add_object_to_instance_dictionary(self.distributed_charges, new_distributed_charge, dc_name,
                                                DistributedCharge)

    def get_distributed_charges(self):
        """

        :return: instance dictionary that contains all applied distributed charges in the bar
        """
        return self.distributed_charges

    def get_referred_distributed_charge_to_nodes(self, return_global_values=True):
        # TODO tener en cuenta todas las cargas y devolver solo un diccionatrio total. Ahora mismo solo sirve para ...
        # ... una carga aplicada, pues al procesar la primera, ya se encuentra el return

        # If there are distributed_charges applied to the bar
        if len(self.distributed_charges) > 0:
            for key, dc in self.distributed_charges.items():
                bar_length = self.length()
                forces = dc.max_value * np.array(dc.direction)
                if dc.dc_type == DistributedChargeType.SQUARE:
                    x_reaction = 0
                    y_reaction = - forces[1] * bar_length / 2
                    m_origin_reaction = - forces[1] * pow(bar_length, 2) / 12
                    m_end_reaction = forces[1] * pow(bar_length, 2) / 12

                    reaction = np.array([x_reaction, y_reaction, 0])
                    if not return_global_values:
                        pass
                    return {
                        "x": reaction[0],
                        "y": reaction[1],
                        "m_origin": m_origin_reaction,
                        "m_end": m_end_reaction
                    }

                    # Convert reactions to global coordinates
                    matrix_conversion_to_global = self.system_change_matrix_2d_rigid_nodes()
                    reaction_global = np.dot(matrix_conversion_to_global, reaction)

                    return {
                        "x": reaction_global[0],
                        "y": reaction_global[1],
                        "m_origin": m_origin_reaction,
                        "m_end": m_end_reaction
                    }

                # TODO Si se incluyen mas tipos de cargas distribuidas, agregarlos aqui con elifs y escribir sus tests

        elif len(self.distributed_charges) == 0:
            print("There are no distributed charges applied to bar " + self.name)

    def add_punctual_force(self, new_punctual_force, force_name=None):
        """
        Adds the specified punctual force to the collection of punctual forces applied to the bar
        :param new_punctual_force: PunctualForceInBar to be added
        :param force_name: name to assign in dictionary
        :return:
        """
        self._add_object_to_instance_dictionary(self.punctual_forces, new_punctual_force, force_name,
                                                PunctualForceInBar)

    def get_punctual_forces(self):
        return self.punctual_forces

    def get_referred_punctual_forces_in_bar_to_nodes(self, return_global_values=True):
        """

        :param return_global_values: boolean that specifies the return values must be in global coordinates. If it is
        not true, then the returned values are in local coordinates
        :return: Dictionary of forces and momentums of nodes in local coordinates
        """
        # TODO tener en cuenta todas las cargas y devolver solo un diccionatrio total. Ahora mismo solo sirve para ...
        # ... una carga aplicada, pues al procesar la primera, ya se encuentra el return
        # If there are punctual_forces applied to the bar
        if len(self.punctual_forces) > 0:
            for key, pf in self.punctual_forces.items():
                bar_length = self.length()
                distance_origin_force = bar_length * pf.origin_end_factor
                distance_end_force = bar_length * (1 - pf.origin_end_factor)

                # Decompose force according each axis
                forces_in_axis = tuple([pf.value * x for x in pf.direction])

                # Reactions in each axis
                y_reaction_origin = - forces_in_axis[1] * pow(distance_end_force, 2) * (bar_length + 2 * distance_origin_force) / pow(bar_length, 3)
                y_reaction_end = - forces_in_axis[1] * pow(distance_origin_force, 2) * (bar_length + 2 * distance_end_force) / pow(bar_length, 3)
                x_reaction = - forces_in_axis[0]

                reaction_origin = np.array([x_reaction, y_reaction_origin, 0])
                reaction_end = np.array([x_reaction, y_reaction_end, 0])

                # Flector momentums
                flector_origin = - forces_in_axis[1] * distance_origin_force * pow(distance_end_force, 2) / pow(bar_length, 2)
                flector_end = forces_in_axis[1] * pow(distance_origin_force, 2) * distance_end_force / pow(bar_length, 2)
                # TODO Repasar el signo en flector_force_point
                flector_force_point = 2 * forces_in_axis[1] * pow(distance_origin_force, 2) * pow(distance_end_force, 2) / pow(bar_length, 3)

                if not return_global_values:
                    return {
                        "x_origin": reaction_origin[0],
                        "x_end": reaction_end[0],
                        "y_origin": reaction_origin[1],
                        "y_end": reaction_end[1],
                        "m_origin": flector_origin,
                        "m_end": flector_end,
                        "m_force_point": flector_force_point
                    }

                # Convert reactions to global coordinates
                matrix_conversion_to_global = self.system_change_matrix_2d_rigid_nodes()
                reaction_origin_global = np.dot(matrix_conversion_to_global, reaction_origin)
                reaction_end_global = np.dot(matrix_conversion_to_global, reaction_end)

                return {
                    "x_origin": reaction_origin_global[0],
                    "x_end": reaction_end_global[0],
                    "y_origin": reaction_origin_global[1],
                    "y_end": reaction_end_global[1],
                    "m_origin": flector_origin,
                    "m_end": flector_end,
                    "m_force_point": flector_force_point
                }

        elif len(self.punctual_forces) == 0:
            print("There are no punctual forces applied to bar " + self.name)

    def has_distributed_charges(self):
        if len(self.distributed_charges) > 0:
            return True
        else:
            return False

    def has_punctual_forces(self):
        if len(self.punctual_forces) > 0:
            return True
        else:
            return False


class Structure:
    def __init__(self, name: str, bars: dict):
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

    def get_number_of_nodes(self):
        """

        :return: Number of nodes that belong to the structure
        """
        num_nodes = 0
        processed_nodes = []

        for key, bar in self.bars.items():
            origin = bar.origin
            end = bar.end

            if origin not in processed_nodes:
                num_nodes += 1
                processed_nodes.append(origin)

            if end not in processed_nodes:
                num_nodes += 1
                processed_nodes.append(end)

        return num_nodes

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

    def set_bars_and_nodes_numeration(self):
        """
        Assigns a number to each bar and each node of the structure

        :return:  number of nodes in the structure
        """
        # Initialize assignment of numbers to each bar and each node
        for key, bar in self.bars.items():
            bar.solving_numeration = -1
            bar.origin.solving_numeration = -1
            bar.end.solving_numeration = -1

        # Assign numeration for each bar and each node and check which bar each node belongs to
        bar_number = 1
        node_number = 1

        for key, bar in self.bars.items():
            origin_node = bar.origin
            end_node = bar.end

            # Bar solving number assignations
            if bar.solving_numeration <= 0:
                bar.solving_numeration = bar_number
                bar_number = bar_number + 1

            # Nodes solving number assignations
            if origin_node.solving_numeration <= 0:
                origin_node.solving_numeration = node_number
                node_number = node_number + 1

            if end_node.solving_numeration <= 0:
                end_node.solving_numeration = node_number
                node_number = node_number + 1

        return node_number - 1

    def find_submatrix(self, matrix, row, col):
        """

        :param matrix: matrix from which the submatrix is going to be extracted
        :param row: row index of the assembled matrix in the form of matrix of matrixes
        :param col: row index of the assembled matrix in the form of matrix of matrixes
        :return: returns submatrixes of the global matrix as global rigidity submatrixes
        """
        row = row - 1
        col = col - 1

        submatrix_row_start = int(row * submatrix_size)
        submatrix_row_end = int(row * submatrix_size + submatrix_size)
        submatrix_col_start = int(col * submatrix_size)
        submatrix_col_end = int(col * submatrix_size + submatrix_size)

        return {
            "matrix": matrix[submatrix_row_start:submatrix_row_end, submatrix_col_start:submatrix_col_end],
            "row_start": submatrix_row_start,
            "row_end": submatrix_row_end,
            "col_start": submatrix_col_start,
            "col_end": submatrix_col_end
        }

    def assembled_matrix(self):
        """
        This function returns the assembled matrix of the structure
        :return:
        """

        # Total number of nodes in structure
        num_nodes = self.set_bars_and_nodes_numeration()

        # Matrix to be returned as assembled matrix
        matrix = [[0] * num_nodes * submatrix_size] * num_nodes * submatrix_size
        matrix = np.array(matrix)

        for key, bar in self.bars.items():
            # Compute global rigidity matrix in order to get values for kii, kij, kji and kjj
            bar.global_rigidity_matrix_2d_rigid_nodes()

            origin_node = bar.origin
            end_node = bar.end

            # This list is used in a loop in order to use the different k_ij of the bar
            nodes_combination = [(origin_node.solving_numeration, origin_node.solving_numeration),
                                 (origin_node.solving_numeration, end_node.solving_numeration),
                                 (end_node.solving_numeration, origin_node.solving_numeration),
                                 (end_node.solving_numeration, end_node.solving_numeration)
                                 ]

            for i in range(4):
                submatrix_info = self.find_submatrix(matrix, nodes_combination[i][0], nodes_combination[i][1])
                submatrix = submatrix_info.get("matrix")
                row_start = submatrix_info.get("row_start")
                row_end = submatrix_info.get("row_end")
                col_start = submatrix_info.get("col_start")
                col_end = submatrix_info.get("col_end")

                if i == 0:
                    submatrix = np.add(submatrix, bar.k_ii)
                elif i == 1:
                    submatrix = np.add(submatrix, bar.k_ij)
                elif i == 2:
                    submatrix = np.add(submatrix, bar.k_ji)
                elif i == 3:
                    submatrix = np.add(submatrix, bar.k_jj)

                matrix[row_start:row_end, col_start:col_end] = submatrix

        return matrix

    def _get_zero_displacement_indexes(self):
        """

        :return: list of the indexes of rows and columns that must be deleted for the decoupled matrix
        """

        # Set numeration. Do not delete this line, other functions depend on numeration given here.
        self.set_bars_and_nodes_numeration()

        # List to be returned
        indexes_to_delete = []

        # Nodes already processed when searching for constraints
        processed_nodes = []

        def process_node(node, proc_nodes, index_to_delete):
            """
            Checks if a node has any kind of support and constrains the assembled matrix accordingly.

            :param node: Node to look for supports
            :param proc_nodes: list of already processed nodes
            :param index_to_delete: list of row and column indexes to delete
            :return:
            """
            if node.solving_numeration not in proc_nodes:
                # If the node has not yet been processed
                if node.support is not Support.NONE:
                    # If the support provides restrictions

                    # offset_index is used to cancel rows and columns in a more human-friendly way (using 0, 1 and 2)
                    # to cancel out the pertinent rows and columns
                    offset_index = (node.solving_numeration - 1) * submatrix_size

                    # range_start is used to restrain cancellations on rollers supports
                    range_start = 0

                    if node.support is Support.ROLLER_X:
                        # Remove x component
                        target_cancellations = 1
                    elif node.support is Support.ROLLER_Y:
                        # Remove y component
                        range_start = 1
                        target_cancellations = 2
                    elif node.support is Support.PINNED:
                        # Remove x and y components
                        target_cancellations = 2

                    # TODO Agregar más elifs si se introducen más tipos de apoyos
                    else:
                        # FIXED
                        # Remove x, y and angle components
                        target_cancellations = 3

                    # TODO cambiar el range si se implementan otros tipos de estructuras
                    for offset_component in range(range_start, target_cancellations):
                        # offset_component = 0, cancels out x component
                        # offset_component = 1, cancels out y component
                        # offset_component = 2, cancels out angle component

                        if (offset_index + offset_component) not in index_to_delete:
                            # Add index to the list in order to delete all rows and columns at once
                            indexes_to_delete.append(offset_index + offset_component)

                # Mark the current node as processed
                proc_nodes.append(node.solving_numeration)

        for key, bar in self.bars.items():
            origin_node = bar.origin
            end_node = bar.end

            process_node(origin_node, processed_nodes, indexes_to_delete)
            process_node(end_node, processed_nodes, indexes_to_delete)

        return indexes_to_delete

    def decoupled_matrix(self):
        """

        :return: Decoupled matrix of the structure
        """

        # The assembled matrix must be edited in order to obtain the decoupled one
        matrix = self.assembled_matrix()

        indexes_to_delete = self._get_zero_displacement_indexes()

        # Delete all rows at once and then all columns
        matrix = np.delete(matrix, indexes_to_delete, 0)
        matrix = np.delete(matrix, indexes_to_delete, 1)

        return matrix

    def inverse_decoupled_matrix(self):
        """

        :return: Inverse of decoupled matrix
        """
        return np.linalg.inv(self.decoupled_matrix())

    def inverse_assembled_matrix(self):
        """

        :return: Inverse of decoupled matrix
        """
        return np.linalg.inv(self.assembled_matrix())

    def forces_and_momentums_in_structure(self):
        """
        Collects all forces and momentums in the structure referred to the nodes

        :return: Array of x force, y force and z momentum applied to each node
        """

        # Assing numerations
        self.set_bars_and_nodes_numeration()

        # List to store the forces in an ordered way
        forces = []
        # Index to find the nodes in order
        current_search = 1

        def add_referred_force_and_momentum_to_node(node, values, original_charge, node_position):
            """
            Adds the referenced charges to a node
            :param node: Node to add the charges to
            :param values: Dictionary of force and momentum to add to node
            :param original_charge: dc if the charge is a distributed one or pf if it is a punctual force
            :param node_position: "origin" if is an origin node, "end" otherwise
            :return:
            """
            # If the charge applied to the bar is a distributed charge
            if original_charge == "dc":
                key_base = "dc"
                # TODO Si se añaden más tipos de carga, modificarlo igual que para la fuerza puntual
                global_y_force = values.get("y")
                force = np.array((0, global_y_force, 0))
                if node_position == "origin":
                    momentum = np.array([0, 0, values.get("m_origin")])
                else:
                    momentum = np.array([0, 0, values.get("m_end")])
            # If the charge applied to the bar is a punctual force
            elif original_charge == "pf":
                key_base = "pf"
                # If it is an origin node
                if node_position == "origin":
                    global_y_force = values.get("y_origin")
                    global_x_force = values.get("x_origin")
                    force = np.array((global_x_force, global_y_force, 0))
                    momentum = np.array([0, 0, values.get("m_origin")])
                # If it is an end node
                else:
                    global_y_force = values.get("y_end")
                    global_x_force = values.get("x_origin")
                    force = np.array((global_x_force, global_y_force, 0))
                    momentum = np.array([0, 0, values.get("m_end")])
            else:
                raise ValueError("The value " + str(original_charge) + " is not valid for parameter original_charge")

            real_key = key_base

            # Unique key assignation
            while (real_key in node.get_forces_in_node_dictionary()) or \
                    (real_key in node.get_referred_forces_dictionary()) or \
                    (real_key in node.get_momentum_in_node_dictionary()) or \
                (real_key in node.get_referred_momentum_dictionary()):
                real_key = fs.get_random_name(key_base)

            # Add force and momentum to node
            node.add_force(real_key, force, False)
            node.add_momentum(real_key, momentum, False)

        # Clear all previously referred forces and momentums added
        structure_nodes = self.get_nodes()
        for node in structure_nodes:
            node.clear_referred_forces()
            node.clear_referred_momentums()

        for key, bar in self.bars.items():
            if bar.has_distributed_charges():
                dc_nodes = bar.get_referred_distributed_charge_to_nodes()

                add_referred_force_and_momentum_to_node(bar.origin, dc_nodes, "dc", "origin")
                add_referred_force_and_momentum_to_node(bar.end, dc_nodes, "dc", "end")

            if bar.has_punctual_forces():
                pf_nodes = bar.get_referred_punctual_forces_in_bar_to_nodes()

                add_referred_force_and_momentum_to_node(bar.origin, pf_nodes, "pf", "origin")
                add_referred_force_and_momentum_to_node(bar.end, pf_nodes, "pf", "end")

        while current_search <= self.get_number_of_nodes():
            for key, bar in self.bars.items():
                origin = bar.origin
                end = bar.end

                if origin.solving_numeration == current_search or end.solving_numeration == current_search:
                    if origin.solving_numeration == current_search:
                        valid_node = origin
                    else:
                        valid_node = end

                    # X force
                    forces.append(valid_node.get_total_force_and_momentum()[0])
                    # Y force
                    forces.append(valid_node.get_total_force_and_momentum()[1])
                    # Z Momentum
                    forces.append(valid_node.get_total_force_and_momentum()[2])

                    # Look for the next node
                    current_search += 1

        return -1 * np.array(forces)

    def decoupled_forces_and_momentums_in_structure(self):
        """

        :return: array of not supported forces and momentums (array to muliply by the inverse of the deoupled matrix)
        """
        force_array = self.forces_and_momentums_in_structure()
        indexes_to_delete = self._get_zero_displacement_indexes()

        # Delete all rows at once
        return np.delete(force_array, indexes_to_delete, 0)

    def get_nodes_displacements(self):
        """
        Determines the displacement of each node, assigns its value to the node instances and returns the displacement
        vector

        :return: Array with the displacement of the nodes
        """

        # Constricted displacements on nodes
        zero_displacement_indexes = self._get_zero_displacement_indexes()
        # Sort them in order to build the matrix in a correct way
        zero_displacement_indexes.sort()
        # Calculate the displacement of the non-constricted nodes
        calculated_nodes_displacements = np.dot(self.inverse_decoupled_matrix(),
                                                self.decoupled_forces_and_momentums_in_structure())

        # List to store the displacement of ALL nodes
        nodes_displacements = []
        # Variable to set to zero the displacement of the constricted nodes
        included_zero_displacement_index = 0
        # Variable to travel trough calculated_node_displacements list
        included_non_zero_displacement_index = 0

        # Build the final displacement vector using the calculated and the known (zero value) ones
        for i in range(len(zero_displacement_indexes) + len(calculated_nodes_displacements)):
            if i == zero_displacement_indexes[included_zero_displacement_index]:
                nodes_displacements.append(0)
                included_zero_displacement_index += 1
            else:
                nodes_displacements.append(calculated_nodes_displacements[included_non_zero_displacement_index])
                included_non_zero_displacement_index += 1

        # Assign the displacement of each node to its instance
        assigned_nodes_numeration = []
        node_to_process = 1

        while node_to_process < self.get_number_of_nodes():
            for key, bar in self.bars.items():
                origin = bar.origin
                end = bar.end

                if origin.solving_numeration == node_to_process or \
                        end.solving_numeration == node_to_process:

                    if origin.solving_numeration == node_to_process:
                        valid_node = origin
                    else:
                        valid_node = end

                    index_offset = (node_to_process - 1) * submatrix_size

                    displacement = {
                        "x": nodes_displacements[index_offset],
                        "y": nodes_displacements[index_offset + 1],
                        "angle": nodes_displacements[index_offset + 2]
                    }

                    valid_node.set_displacement(displacement)

                    assigned_nodes_numeration.append(valid_node.solving_numeration)
                    node_to_process += 1

        return nodes_displacements

    def get_nodes(self):
        """

        :return: list containing all nodes in the structure, order by solving numeration
        """
        self.set_bars_and_nodes_numeration()

        got_nodes = []
        current_searched_node = 1

        while current_searched_node <= self.get_number_of_nodes():
            for key, bar in self.bars.items():
                origin = bar.origin
                end = bar.end

                if origin.solving_numeration == current_searched_node or \
                    end.solving_numeration == current_searched_node:
                    if origin.solving_numeration == current_searched_node:
                        valid_node = origin
                    else:
                        valid_node = end

                    got_nodes.append(valid_node)
                    current_searched_node += 1

        return got_nodes

    def get_nodes_reactions(self):
        """
        Once known the applied forces. determines the reactions of each node, assigns its value to the node instances
        and returns the reactions vector

        :return: Array with the reactions of the nodes
        """
        # Assign the displacement of each node to its instance
        assigned_nodes_numeration = []
        node_to_process = 1

        nodes_reactions = np.dot(self.assembled_matrix(), self.get_nodes_displacements())

        while node_to_process < self.get_number_of_nodes():
            for key, bar in self.bars.items():
                origin = bar.origin
                end = bar.end

                if origin.solving_numeration == node_to_process or \
                        end.solving_numeration == node_to_process:

                    if origin.solving_numeration == node_to_process:
                        valid_node = origin
                    else:
                        valid_node = end

                    index_offset = (node_to_process - 1) * submatrix_size

                    reactions = {
                        "x": nodes_reactions[index_offset],
                        "y": nodes_reactions[index_offset + 1],
                        "momentum": nodes_reactions[index_offset + 2]
                    }

                    valid_node.set_reactions(reactions)

                    assigned_nodes_numeration.append(valid_node.solving_numeration)
                    node_to_process += 1

        return nodes_reactions


class Material:
    def __init__(self, name: str):
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
    def __init__(self, name: str, name_number):
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


class DistributedCharge:
    """
    Class to represent a distributed charge applied to a single beam
    """
    # TODO incluir leyes de deformacion mirando un prontuario

    def __init__(self, dc_type, max_value, direction):
        """

        :param dc_type: type of distributed charge. Must be of type DistributedChargeType
        :param max_value: the maximum value of the distributed charge
        :param direction: unitary vector representing the direction of the force in local axis
        """
        # TODO completar el test con lo nuevo
        if type(dc_type) not in [DistributedChargeType]:
            raise TypeError("Error. dc_type must be of type structures.DistributedChargeType")

        self.dc_type = dc_type
        self.max_value = max_value
        self.direction = direction
        # If new parameters are included, they must be added to the equals function and to the test

    def equals(self, other_distributed_charge):
        if type(other_distributed_charge) not in [DistributedCharge]:
            raise TypeError("Error. The type of other_distributed_charge must be DistributedCharge")

        if self.dc_type == other_distributed_charge.dc_type and \
                self.max_value == other_distributed_charge.max_value:
            return True
        else:
            return False


class PunctualForceInBar:
    """
    Class that represents a punctual force applied in any point of a bar
    """

    # TODO incluir ley de deformacion mirando un prontuario

    def __init__(self, value, origin_end_factor, direction):
        # TODO Escribir test
        """

        :param value: the magnitud of the force
        :param origin_end_factor: value from 0 to 1 where 0 means origin node and 1 means end node
        :param direction: unitary vector representing the direction of the force in local axis
        """
        if origin_end_factor < 0 or origin_end_factor > 1:
            raise ValueError("Error. origin_end_factor must be between 0 and 1.")

        if type(direction) not in [tuple]:
            raise TypeError("Error. direction must be a tuple of representing the direction of the force")

        self.value = value
        self.origin_end_factor = origin_end_factor
        self.direction = direction
        # If new parameters are included, they must be added to the equals function and to the test

    def equals(self, other_punctual_force_in_bar):
        if type(other_punctual_force_in_bar) not in [PunctualForceInBar]:
            raise TypeError("Error. The type of other_punctual_force_in_bar must be PunctualForceInBar")

        if self.value == other_punctual_force_in_bar.value and \
                self.origin_end_factor == other_punctual_force_in_bar.origin_end_factor and \
                self.direction == other_punctual_force_in_bar.direction:
            return True
        else:
            return False


# TODO DELETE EVERYTHING DOWN HERE. IT IS JUST FOR TESTING PURPOSES
n1 = Node("N1", position=(0, 0, 0), support=Support.PINNED)
n2 = Node("N2", position=(0, 4.2, 0))
n3 = Node("N3", position=(6.8, 5.25, 0))
n4 = Node("N4", position=(13.6, 4.2, 0))
n5 = Node("N5", position=(17.2, 3.644117647, 0))
n6 = Node("N6", position=(13.6, 0, 0), support=Support.FIXED)

b1 = Bar("B1", n1, n2)
b2 = Bar("B2", n2, n3)
b3 = Bar("B3", n3, n4)
b4 = Bar("B4", n4, n5)
b5 = Bar("B5", n4, n6)

dc = DistributedCharge(DistributedChargeType.SQUARE, 10179.36, (0, -1, 0))
b2.add_distributed_charge(dc)
b3.add_distributed_charge(dc)
b4.add_distributed_charge(dc)

pf = PunctualForceInBar(-25000, 0.5, (0, 1, 0))
b5.add_punctual_force(pf, "pf")

bars = {
    b1.name: b1,
    b2.name: b2,
    b3.name: b3,
    b4.name: b4,
    b5.name: b5
}

st = Structure("S1", bars)
# st.assembled_matrix()
# disp = st.get_nodes_displacements()
# print("==========DISPLACEMENTS==========")
# for i in range(len(disp)):
#     if i % 3 == 0:
#         label = "x: "
#     elif i % 3 == 1:
#         label = "y: "
#     else:
#         label = "angle: "
#     print(label + str(disp[i]))


# Structure 2
n21 = Node("N21", position=(-1, 2, 0), support=Support.PINNED)
n22 = Node("N22", position=(-1, 5, 0))
n23 = Node("N23", position=(1, 5, 0))
n24 = Node("N24", position=(1, 4, 0))
n25 = Node("N25", position=(3, 4, 0))
n26 = Node("N26", position=(3, 5, 0))
n27 = Node("N27", position=(5, 5, 0))
n28 = Node("N28", position=(5, 2, 0), support=Support.FIXED)

b21 = Bar("B21", n21, n22)
b22 = Bar("B22", n22, n23)
b23 = Bar("B23", n23, n24)
b24 = Bar("B24", n24, n25)
b25 = Bar("B25", n25, n26)
b26 = Bar("B26", n26, n27)
b27 = Bar("B27", n27, n28)

dc2 = DistributedCharge(DistributedChargeType.SQUARE, 100000, (0, -1, 0))
b22.add_distributed_charge(dc2)
b26.add_distributed_charge(dc2)

pf21 = PunctualForceInBar(-40000, 0.5, (0, 1, 0))
pf24 = PunctualForceInBar(-25000, 0.5, (0, 1, 0))
pf27 = PunctualForceInBar(-30000, 0.5, (0, 1, 0))

b21.add_punctual_force(pf21)
b24.add_punctual_force(pf24)
b27.add_punctual_force(pf27)

bars2 = {
    b21.name: b21,
    b22.name: b22,
    b23.name: b23,
    b24.name: b24,
    b25.name: b25,
    b26.name: b26,
    b27.name: b27,
}

# st2 = Structure("st2", bars2)
# disp = st2.get_nodes_displacements()
# print("==========DISPLACEMENTS==========")
# for i in range(len(disp)):
#     if i % 3 == 0:
#         label = "x: "
#     elif i % 3 == 1:
#         label = "y: "
#     else:
#         label = "angle: "
#     print(label + str(disp[i]))
