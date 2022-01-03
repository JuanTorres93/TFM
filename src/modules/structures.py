import enum
import math

import numpy as np
import src.modules.databaseutils as db
import src.modules.filesystemutils as fs

# Create the database if it does not exist
db.regenerate_initial_database(force=False)

# For structures with rigid nodes, the size of the submatrixes is 3
submatrix_size = 3


@enum.unique
class Support(enum.Enum):
    """
    Enumeration for the different types of supports
    """
    NONE = 1
    ROLLER_X = 2  # Deslizadera
    ROLLER_Y = 3  # Deslizadera
    PINNED = 4  # Fijo
    FIXED = 5  # Empotramiento


@enum.unique
class DistributedChargeType(enum.Enum):
    """
    Enumeration for the different types of distributed charges
    """
    SQUARE = 1
    PARALLEL_TO_BAR = 2


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

        self.bars_belonging_to = {}

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
            raise TypeError("new_support must be of class Support.")

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

    def get_bars_belonging_to(self):
        """
        bars_belonging_to is a dictionary that contains all the bar that the node is part of
        :return: dictionary with the bar
        """
        return self.bars_belonging_to

    def add_bar_to_collection(self, bar):
        """
        This method appends the bar to the list of bars that the node is present in
        :param bar: bar to add to the dictionary
        """
        # The test of this function is carried along with the one of get_bars_belonging_to
        self.bars_belonging_to[bar.name] = bar

    def clear_referred_forces(self):
        """
        Deletes all referred forces stored in the node instance. This method is useful to not stack the same forces
        every time that they are calculated
        :return:
        """
        self.referred_forces = {}

    def add_force(self, key, force, belongs_to_node):
        """
        Adds a force to the node
        :param key: key to store the force in the dictionary
        :param force: tuple representing the applied force (Fx, Fy, Fz)
        :param belongs_to_node: if true the force is applied directly to the node. If false, the force is a consequence
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
        """
        Deletes all referred momentums stored in the node instance. This method is useful to not stack the same momentums
        every time that they are calculated
        :return:
        """
        self.referred_momentums = {}

    def add_momentum(self, key, momentum, belongs_to_node):
        """
        Adds a momentum to the node
        :param key: key to store the momentum in the dictionary
        :param momentum: tuple representing the applied momentum (Mx, My, Mz)
        :param belongs_to_node: if true the momentum is applied directly to the node. If false, the momentum is a
        consequence
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
        Stores the displacement of the node. Handled from Structure class
        :param new_displacement: dictionary of x, y and angle displacements.
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
        Stores the reaction of the node. Handled from Structure class
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

    def has_support(self):
        """

        :return: True if the node has support, else otherwise
        """
        if self.support != Support.NONE:
            return True
        else:
            return False


class Bar:
    """
    Class that represents a bar in a structure.
    """

    def __init__(self, name: str, origin: Node, end: Node, material, profile):
        """
        Constructor for Bar class
        :param name: Name of the bar
        :param origin: Node that acts as the origin of the bar
        :param end: Node that acts as the end of the bar
        :param material: string that represents a material stored in the database
        :param profile: tuple that represents a beam profile stored in the database
        """
        if type(origin) not in [Node] or type(end) not in [Node]:
            raise TypeError("Nodes must be of type 'Node'")

        if type(name) not in [str] or type(material) not in [str]:
            raise TypeError("name and material must be of type str")

        if origin == end:
            raise ValueError("Origin and end nodes must be different")

        self.name = name
        self.origin = origin
        # Store bar in origin node
        self.origin.add_bar_to_collection(self)
        self.end = end
        # Store bar in end node
        self.end.add_bar_to_collection(self)
        self.material = Material(material)
        self.profile = Profile(profile[0], profile[1])

        # Number assigned to build the structure (assembled) matrix (handle from structure class)
        self.solving_numeration = -1

        # This submatrixes are here exposed for an easier way to assemble the assembled matrix from the structure class
        # Do not change their initialization
        self.k_ii_global = None
        self.k_ij_global = None
        self.k_ji_global = None
        self.k_jj_global = None

        self.k_ii_local = None
        self.k_ij_local = None
        self.k_ji_local = None
        self.k_jj_local = None

        # Distributed charges applied to the bar
        self.distributed_charges = {}

        # Punctual forces applied to the bar
        self.punctual_forces = {}

        # Efforts of the bar
        self.efforts = {}

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

    def get_origin(self):
        """

        :return: origin node of the bar
        """
        # The test of this function is carried along with the one of get_bars_belonging_to
        return self.origin

    def set_end(self, new_end: Node):
        """
        Sets the end node of the bar to the specified one
        :param new_end: new end node of the bar
        :return:
        """
        if type(new_end) not in [Node]:
            raise TypeError("new_end must be of type 'Node'")

        self.end = new_end

    def get_end(self):
        """

        :return: end node of the bar
        """
        # The test of this function is carried along with the one of get_bars_belonging_to
        return self.end

    def get_nodes(self):
        """

        :return: List with origin and end nodes of the bar
        """
        # The test of this function is carried along with the one of get_bars_belonging_to
        return [self.origin, self.end]

    def swap_nodes(self):
        """
        Swaps origin and end nodes.
        """
        aux = self.origin
        self.origin = self.end
        self.end = aux

        for key, pf in self.get_punctual_forces().items():
            # Update distance to origin in punctual forces
            updated_origin_end_factor = 1 - pf.origin_end_factor
            pf.set_origin_end_factor(updated_origin_end_factor)
            # Update value
            pf.set_value(-pf.value)

        for key, dc in self.get_distributed_charges().items():
            # Update value
            dc.set_max_value(-dc.max_value)

    def length(self) -> float:
        """

        :return: Length of the bar
        """
        return np.linalg.norm(np.subtract(self.end.position, self.origin.position))

    def set_material(self, mat_name: str):
        """
        :param mat_name: str corresponds with a unique name in materials table of the database
        :return:
        """
        self.material = Material(mat_name)

    def get_material(self):
        """
        :return: current material of the bar
        """
        return self.material

    def set_profile(self, profile_name: str, profile_number: int):
        """
        :param profile_name: str corresponds with a name in profiles table of the database. e.g. IPE
        :param profile_number: str corresponds with a name_number in the profile table of the database
        :return:
        """
        self.profile = Profile(profile_name, profile_number)

    def get_profile(self):
        """
        :return: current profile of the bar
        """
        return self.profile

    def local_rigidity_matrix_2d_rigid_nodes(self, use_inertia_x=True):
        """

        :param use_inertia_x: Specifies whether the inertia moment must be selected with respect to the
        x-axis (default) or the y-axis (f equals False)
        :return: local rigidity matrix for a 2D structure with rigid nodes
        """

        l = self.length()  # l -> beam length
        e = self.material.young_mod  # e -> Young's modulus
        a = self.profile.area  # a -> cross section's area
        # i -> modulus inertia
        if use_inertia_x:
            i = self.profile.inertia_moment_x
        else:
            i = self.profile.inertia_moment_y

        local_rigidity_matrix = np.array([[e * a / l, 0, 0, -e * a / l, 0, 0],
                                          [0, 12 * e * i / l ** 3, 6 * e * i / l ** 2, 0, -12 * e * i / l ** 3,
                                           6 * e * i / l ** 2],
                                          [0, 6 * e * i / l ** 2, 4 * e * i / l, 0, -6 * e * i / l ** 2, 2 * e * i / l],
                                          [-e * a / l, 0, 0, e * a / l, 0, 0],
                                          [0, -12 * e * i / l ** 3, -6 * e * i / l ** 2, 0, 12 * e * i / l ** 3,
                                           -6 * e * i / l ** 2],
                                          [0, 6 * e * i / l ** 2, 2 * e * i / l, 0, -6 * e * i / l ** 2,
                                           4 * e * i / l]])

        m_aux = local_rigidity_matrix[0:3, 0:3]
        self.k_ii_local = m_aux

        m_aux = local_rigidity_matrix[0:3, 3:6]
        self.k_ij_local = m_aux

        m_aux = local_rigidity_matrix[3:6, 0:3]
        self.k_ji_local = m_aux

        m_aux = local_rigidity_matrix[3:6, 3:6]
        self.k_jj_local = m_aux

        return local_rigidity_matrix

    def angle_from_global_to_local(self):
        """

        :return: Needed angle (in radians) for converting the local matrix to global one
        """
        # x-axis as a reference to compute the rotated angle
        reference = np.array((1, 0, 0))
        # Vector with origin in the origin node of the bar (beam) and with end in the end node of the bar
        beam_line = np.subtract(self.get_end().position, self.get_origin().position)

        # Calculate the dot product of the previous vectors
        dot_product = reference @ beam_line

        # Compute the angle between the vector using the definition of the dot product
        cosine = dot_product / np.linalg.norm(reference) / np.linalg.norm(beam_line)
        angle = np.arccos(cosine)

        # Adjust the angle to the pertinent quadrant
        if self.get_end().x() >= self.get_origin().x() and self.get_end().y() >= self.get_origin().y():
            # angle = angle
            pass
        elif self.get_end().x() <= self.get_origin().x() and self.get_end().y() >= self.get_origin().y():
            angle = angle + math.pi / 2
        elif self.get_end().x() <= self.get_origin().x() and self.get_end().y() <= self.get_origin().y():
            angle = angle + math.pi
        else:
            angle = 2 * math.pi - angle

        return angle

    def system_change_matrix_2d_rigid_nodes(self):
        """

        :return: System change matrix from local axis to local ones
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
        system_change_matrix = self.system_change_matrix_2d_rigid_nodes()
        local_matrix = self.local_rigidity_matrix_2d_rigid_nodes()

        m_aux = system_change_matrix.dot(local_matrix[0:3, 0:3])
        self.k_ii_global = m_aux.dot(np.transpose(system_change_matrix))

        m_aux = system_change_matrix.dot(local_matrix[0:3, 3:6])
        self.k_ij_global = m_aux.dot(np.transpose(system_change_matrix))

        m_aux = system_change_matrix.dot(local_matrix[3:6, 0:3])
        self.k_ji_global = m_aux.dot(np.transpose(system_change_matrix))

        m_aux = system_change_matrix.dot(local_matrix[3:6, 3:6])
        self.k_jj_global = m_aux.dot(np.transpose(system_change_matrix))

        top = np.hstack((self.k_ii_global, self.k_ij_global))
        bottom = np.hstack((self.k_ji_global, self.k_jj_global))

        return np.vstack((top, bottom))

    def _add_object_to_instance_dictionary(self, dictionary, obj, obj_name, obj_type=None):
        """
        Function to include an object to an instance dictionary
        :param dictionary: dictionary to include the new object into
        :param obj: object to include in the dictionary
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
        """

        :param return_global_values: Specifies whether the return value is in global or local coordinates.
        :return: reactions produced by charges that are applied directly to the Bar and not in a node.
        """
        # Dictionary of reactions for every found distributed charge
        referred_charges = {}

        # If there are distributed_charges applied to the bar
        if len(self.distributed_charges) > 0:
            for key, dc in self.distributed_charges.items():
                referred = dc.refer_to_nodes(self, return_global_values)
                referred_charges[str(len(referred_charges))] = {
                    "x": referred.get("x"),
                    "y": referred.get("y"),
                    "m_origin": referred.get("m_origin"),
                    "m_end": referred.get("m_end")
                }
            return referred_charges

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
        """

        :return: punctual forces applied to the bar instance
        """
        return self.punctual_forces

    def get_referred_punctual_forces_in_bar_to_nodes(self, return_global_values=True):
        """

        :param return_global_values: boolean that specifies the return values must be in global coordinates. If it is
        not true, then the returned values are in local coordinates
        :return: Dictionary of forces and momentums of nodes in local coordinates
        """
        # Dictionary of reactions for every found punctual force
        referred_punctual_forces = {}

        # If there are punctual_forces applied to the bar
        if len(self.punctual_forces) > 0:
            for key, pf in self.punctual_forces.items():
                referred = pf.refer_to_nodes(self, return_global_values)
                referred_punctual_forces[str(len(referred_punctual_forces))] = {
                    "x_origin": referred.get("x_origin"),
                    "x_end": referred.get("x_end"),
                    "y_origin": referred.get("y_origin"),
                    "y_end": referred.get("y_end"),
                    "m_origin": referred.get("m_origin"),
                    "m_end": referred.get("m_end"),
                    "m_force_point": referred.get("m_force_point")
                }

            return referred_punctual_forces

        elif len(self.punctual_forces) == 0:
            print("There are no punctual forces applied to bar " + self.name)

    def has_distributed_charges(self):
        """

        :return: True is the bar has any distributed charge applied to it
        """
        if len(self.distributed_charges) > 0:
            return True
        else:
            return False

    def has_punctual_forces(self):
        """

        :return: True is the bar has any punctual force applied to it
        """
        if len(self.punctual_forces) > 0:
            return True
        else:
            return False

    def has_support(self):
        """

        :return: True if, at least, one node of the bar has a support
        """
        if self.origin.has_support() or self.end.has_support():
            return True
        else:
            return False

    def calculate_efforts(self):
        """
        Computes the efforts to which the bar is subjected to and store them in instance variables
        :return:
        """
        # TEST IS WRITTEN IN STRUCTURE TESTS

        # Determine the local matrixes of the origin and end nodes if they aren't available
        if (self.k_ii_local is None) or (self.k_ij_local is None) or \
                (self.k_ji_local is None) or (self.k_jj_local is None):
            self.local_rigidity_matrix_2d_rigid_nodes()

        # FREE STATE CONTRIBUTION
        # Equations say that, in order to compute the efforts, it is needed to multiply each submatrix of the local
        # rigidity matrix of the bar by the transpose of the system change matrix (G). Then this resultant matrix must
        # be multiplied by the displacements of the nodes.
        # Here, it is calculated the multiplication of the local rigidity matrix by the transpose of G
        # Top row of the final matrix
        transpose_system_change_matrix = np.transpose(self.system_change_matrix_2d_rigid_nodes())

        rigidity_matrix_by_g_trans_top_row = np.hstack(
            (np.dot(self.k_ii_local, transpose_system_change_matrix),
             np.dot(self.k_ij_local, transpose_system_change_matrix)))
        # Bottom row of the final matrix
        rigidity_matrix_by_g_trans_bottom_row = np.hstack(
            (np.dot(self.k_ji_local, transpose_system_change_matrix),
             np.dot(self.k_jj_local, transpose_system_change_matrix)))
        # Final matrix
        rigidity_matrix_by_g_trans = np.vstack(
            (rigidity_matrix_by_g_trans_top_row,
             rigidity_matrix_by_g_trans_bottom_row))

        # Get displacements of the origin nodes
        displacements_i = self.get_origin().get_displacement()
        displacements_i = np.array([displacements_i.get("x"),
                                    displacements_i.get("y"),
                                    displacements_i.get("angle")])
        # Arrange the vector, so it can be multiplied by the matrix
        displacements_i = np.vstack(displacements_i)

        # Get displacements of the end nodes
        displacements_j = self.get_end().get_displacement()
        displacements_j = np.array([displacements_j.get("x"),
                                    displacements_j.get("y"),
                                    displacements_j.get("angle")])
        # Arrange the vector, so it can be multiplied by the matrix
        displacements_j = np.vstack(displacements_j)

        # Compose the final displacement vector
        displacements = np.vstack(
            (displacements_i,
             displacements_j)
        )

        # Calculate the efforts in the bar
        efforts = np.dot(rigidity_matrix_by_g_trans, displacements)

        # In order to get the correct results, it is needed to sum to the efforts the reactions obtained in the locked
        # state
        # LOCKED STATE CONTRIBUTION
        # Distributed charges
        # Dictionary of dictionaries containing all distributed charges in the bar
        distributed_charges_contribution = self.get_referred_distributed_charge_to_nodes(return_global_values=False)
        if distributed_charges_contribution is not None:
            # Iterate over every found distributed charge
            for key, dcc in distributed_charges_contribution.items():
                dc_x = dcc.get("x")
                dc_y = dcc.get("y")
                dc_momentum_i = dcc.get("m_origin")
                dc_momentum_j = dcc.get("m_end")

                # Get contribution of the distributed charge in origin and end nodes
                ri = np.array([dc_x, dc_y, dc_momentum_i])
                rj = np.array([dc_x, dc_y, dc_momentum_j])

                # Transpose the vectors in order to be able to multiply its values
                ri = np.vstack(ri)
                rj = np.vstack(rj)

                # Compose final reaction vector
                dcc = np.vstack(
                    (ri,
                     rj)
                )

                # Add the reactions to the efforts
                efforts += dcc

        # Punctual forces
        # Dictionary of dictionaries containing all punctual forces in the bar
        punctual_forces_contribution = self.get_referred_punctual_forces_in_bar_to_nodes(return_global_values=True)
        if punctual_forces_contribution is not None:
            # Iterate over every found punctual forces
            for key, pfc in punctual_forces_contribution.items():
                dc_x_i = pfc.get("x_origin")
                dc_x_j = pfc.get("x_end")
                dc_y_i = pfc.get("y_origin")
                dc_y_j = pfc.get("y_end")
                dc_momentum_i = pfc.get("m_origin")
                dc_momentum_j = pfc.get("m_end")

                # Get contribution of the punctual force in origin and end nodes
                ri = np.array([dc_x_i, dc_y_i, dc_momentum_i])
                rj = np.array([dc_x_j, dc_y_j, dc_momentum_j])

                # Change reference system
                ri = np.dot(np.transpose(self.system_change_matrix_2d_rigid_nodes()),
                            ri)
                rj = np.dot(np.transpose(self.system_change_matrix_2d_rigid_nodes()),
                            rj)

                # Transpose the vectors in order to be able to multiply its values
                ri = np.vstack(ri)
                rj = np.vstack(rj)

                # Compose final reaction vector
                pfc = np.vstack(
                    (ri,
                     rj)
                )

                # Add the reactions to the efforts
                efforts += pfc

        # Clear all previously stored efforts
        self.efforts.clear()
        # And store the new calculated ones
        self._add_object_to_instance_dictionary(self.efforts, np.array([efforts[0], efforts[1], efforts[2]]),
                                                "p_ij", np.ndarray)
        self._add_object_to_instance_dictionary(self.efforts, np.array([efforts[3], efforts[4], efforts[5]]),
                                                "p_ji", np.ndarray)

        print("Efforts for bar " + self.name + " are determined. Available using method get_efforts")

    def get_efforts(self):
        """
        In order to be able to get the efforts it is first needed to calculate them using calculate_efforts
        :return: efforts of the bar as a dictionary {N, Q, M}
        """
        return self.efforts

    def axial_force_law(self, origin_end_factor):
        """
        :param origin_end_factor: point in percentage of length from origin to end in which the value
        of the axial force is desired to be known
        :return: Value of the axial force point corresponding to the point origin_end_factor
        """
        if origin_end_factor < 0 or origin_end_factor > 1:
            raise ValueError("Error. origin_end_factor must be between 0 and 1, both included.")

        origin = self.get_origin()

        if origin.has_support():
            origin_reactions = origin.get_reactions()

            x_reaction = origin_reactions.get("x")
            y_reaction = origin_reactions.get("y")

            bar_angle = self.angle_from_global_to_local()

            n = - (y_reaction * math.sin(bar_angle) + x_reaction * math.cos(bar_angle))
        else:
            bar_efforts = self.get_efforts()
            n = - bar_efforts.get("p_ij")[0]

        if self.has_distributed_charges():
            for key, dc in self.get_distributed_charges().items():
                n += dc.axial_force_law(self, origin_end_factor)

        if self.has_punctual_forces():
            for key, pf in self.get_punctual_forces().items():
                n += pf.axial_force_law(self, origin_end_factor)

        return n

    def shear_strength_law(self, origin_end_factor):
        """

        :param origin_end_factor: point in percentage of length from origin to end in which the value
        of the shear strength is desired to be known
        :return: Value of the shear strength corresponding to the point origin_end_factor
        """
        if origin_end_factor < 0 or origin_end_factor > 1:
            raise ValueError("Error. origin_end_factor must be between 0 and 1, both included.")

        origin = self.get_origin()

        if origin.support != Support.NONE:
            origin_reactions = origin.get_reactions()

            x_reaction = origin_reactions.get("x")
            y_reaction = origin_reactions.get("y")

            bar_angle = self.angle_from_global_to_local()

            v = - y_reaction * math.cos(bar_angle) + x_reaction * math.sin(bar_angle)
        else:
            bar_efforts = self.get_efforts()
            v = - bar_efforts.get("p_ij")[1]

        if self.has_distributed_charges():
            for key, dc in self.get_distributed_charges().items():
                v += dc.shear_strength_law(self, origin_end_factor)

        if self.has_punctual_forces():
            for key, pf in self.get_punctual_forces().items():
                v += pf.shear_strength_law(self, origin_end_factor)

        return v

    def bending_moment_law(self, origin_end_factor):
        """

        :param origin_end_factor: point in percentage of length from origin to end in which the value
        of the bending moment is desired to be known
        :return: Value of the bending moment corresponding to the point origin_end_factor
        """
        if origin_end_factor < 0 or origin_end_factor > 1:
            raise ValueError("Error. origin_end_factor must be between 0 and 1, both included.")

        origin = self.get_origin()
        x = origin_end_factor * self.length()

        if origin.support != Support.NONE:
            origin_reactions = origin.get_reactions()
            bar_angle = self.angle_from_global_to_local()

            x_reaction = origin_reactions.get("x")
            y_reaction = origin_reactions.get("y")

            shear_force_contribution = - x_reaction * math.sin(bar_angle) * x + y_reaction * math.cos(bar_angle) * x

            m_contribution = - origin_reactions.get("momentum")

            m = shear_force_contribution + m_contribution
        else:
            bar_efforts = self.get_efforts()
            shear_force_contribution = bar_efforts.get("p_ij")[1] * x
            m_contribution = - bar_efforts.get("p_ij")[2]
            m = shear_force_contribution + m_contribution

        if self.has_distributed_charges():
            for key, dc in self.get_distributed_charges().items():
                m += dc.bending_moment_law(self, origin_end_factor)

        if self.has_punctual_forces():
            for key, pf in self.get_punctual_forces().items():
                m += pf.bending_moment_law(self, origin_end_factor)

        return m


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

    def get_bars(self, return_as_list=False):
        """

        :param return_as_list:
        :return: bars of the structure
        """
        if not return_as_list:
            return self.bars
        else:
            return list(map(lambda x: x[1],
                            self.get_bars().items()))

    def get_number_of_nodes(self):
        """

        :return: Number of nodes that belong to the structure
        """
        num_nodes = 0
        processed_nodes = []

        for key, bar in self.bars.items():
            origin = bar.get_origin()
            end = bar.get_end()

            if origin not in processed_nodes:
                num_nodes += 1
                processed_nodes.append(origin)

            if end not in processed_nodes:
                num_nodes += 1
                processed_nodes.append(end)

        return num_nodes

    def set_bars_and_nodes_numeration(self):
        """
        Assigns a number to each bar and each node of the structure

        :return:  number of nodes in the structure
        """
        nodes = self.get_nodes(ordered_by_solving_number=False)
        nodes_with_support = list(filter(lambda x: x.has_support() is True,
                                         nodes))

        # Initialize assignment of numbers to each bar and each node
        for key, bar in self.bars.items():
            bar.solving_numeration = -1
            bar.get_origin().solving_numeration = -1
            bar.get_end().solving_numeration = -1

        left_node = None
        for node in nodes_with_support:
            if left_node is None:
                left_node = node
            else:
                if node.x() < left_node.x():
                    left_node = node
                elif node.x() == left_node.x():
                    if node.y() < left_node.y():
                        left_node = node

        next_number = 1
        numbered_nodes = []

        def get_nodes_recursively(node):
            """
            This function finds recursively all nodes in structure, according bars.
            :param node: node to find other nodes
            """
            nonlocal next_number
            nonlocal numbered_nodes
            nonlocal self

            # Assign numeration if it not has already been done
            if node.solving_numeration == -1:
                node.solving_numeration = next_number
                next_number += 1
                # Mark the node as numbered
                if node not in numbered_nodes:
                    numbered_nodes.append(node)

            # Stop when all nodes have been numbered
            if len(numbered_nodes) == self.get_number_of_nodes():
                return

            # Get bars attached to the node
            bars = list(
                map(lambda x: x[1],
                    list(node.get_bars_belonging_to().items())
                    )
            )

            for bar in bars:
                # Get nodes in bar
                nodes_in_bar = bar.get_nodes()
                # Assign numeration to node
                for n in nodes_in_bar:
                    if n.solving_numeration < 1:
                        get_nodes_recursively(n)

        # Assign nodes numeration starting from the first one
        get_nodes_recursively(left_node)

        return self.get_number_of_nodes()

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

        :return: assembled matrix of the structure
        """

        # Total number of nodes in structure
        num_nodes = self.set_bars_and_nodes_numeration()

        # Matrix to be returned as assembled matrix
        matrix = [[0] * num_nodes * submatrix_size] * num_nodes * submatrix_size
        matrix = np.array(matrix)

        for key, bar in self.bars.items():
            origin_node = bar.get_origin()
            end_node = bar.get_end()

            # Compute global rigidity matrix in order to get values for kii, kij, kji and kjj
            bar.global_rigidity_matrix_2d_rigid_nodes()

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
                    submatrix = np.add(submatrix, bar.k_ii_global)
                elif i == 1:
                    submatrix = np.add(submatrix, bar.k_ij_global)
                elif i == 2:
                    submatrix = np.add(submatrix, bar.k_ji_global)
                elif i == 3:
                    submatrix = np.add(submatrix, bar.k_jj_global)

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

                    if node.support is Support.ROLLER_Y:
                        # Remove x component
                        target_cancellations = 1
                    elif node.support is Support.ROLLER_X:
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
            origin_node = bar.get_origin()
            end_node = bar.get_end()

            if origin_node.solving_numeration > end_node.solving_numeration:
                bar.swap_nodes()
                origin_node = bar.get_origin()
                end_node = bar.get_end()

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
        try:
            inverse = np.linalg.inv(self.decoupled_matrix())
        except np.linalg.LinAlgError:
            return np.linalg.LinAlgError

        return inverse

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

        # Assign numerations
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
                global_x_force = values.get("x")
                global_y_force = values.get("y")
                force = np.array((global_x_force, global_y_force, 0))
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
                    global_x_force = values.get("x_end")
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

        # Determine and assign node-referred forces and momentums for each node in each bar
        for key, bar in self.bars.items():
            if bar.has_distributed_charges():
                dc_nodes = bar.get_referred_distributed_charge_to_nodes(return_global_values=True)

                for key_dc, distributed_charge in dc_nodes.items():
                    add_referred_force_and_momentum_to_node(bar.get_origin(), distributed_charge, "dc", "origin")
                    add_referred_force_and_momentum_to_node(bar.get_end(), distributed_charge, "dc", "end")

            if bar.has_punctual_forces():
                pf_nodes = bar.get_referred_punctual_forces_in_bar_to_nodes(return_global_values=True)

                for key_pf, punctual_force in pf_nodes.items():
                    add_referred_force_and_momentum_to_node(bar.get_origin(), punctual_force, "pf", "origin")
                    add_referred_force_and_momentum_to_node(bar.get_end(), punctual_force, "pf", "end")

        # Compose the force vector
        while current_search <= self.get_number_of_nodes():
            for key, bar in self.bars.items():
                origin = bar.get_origin()
                end = bar.get_end()

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

        # Return the opposite of the found forces because the needed values are the reactions, not the forces themselves
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
            if included_zero_displacement_index >= len(zero_displacement_indexes) or \
                    i != zero_displacement_indexes[included_zero_displacement_index]:
                nodes_displacements.append(calculated_nodes_displacements[included_non_zero_displacement_index])
                included_non_zero_displacement_index += 1
            else:
                nodes_displacements.append(0)
                included_zero_displacement_index += 1

        nodes_in_structure = self.get_nodes()

        for i in range(self.get_number_of_nodes()):
            current_node = nodes_in_structure[i]
            index_offset = i * submatrix_size

            displacement = {
                "x": nodes_displacements[index_offset],
                "y": nodes_displacements[index_offset + 1],
                "angle": nodes_displacements[index_offset + 2]
            }

            current_node.set_displacement(displacement)

        return np.array(nodes_displacements)

    def get_nodes(self, ordered_by_solving_number=True):
        """

        :param ordered_by_solving_number:
        :return: list containing all nodes in the structure, order by solving numeration
        """
        got_nodes = []

        if ordered_by_solving_number:
            self.set_bars_and_nodes_numeration()

            current_searched_node = 1

            while current_searched_node <= self.get_number_of_nodes():
                for key, bar in self.bars.items():
                    origin = bar.get_origin()
                    end = bar.get_end()

                    if origin.solving_numeration == current_searched_node or \
                            end.solving_numeration == current_searched_node:
                        if origin.solving_numeration == current_searched_node:
                            valid_node = origin
                        else:
                            valid_node = end

                        got_nodes.append(valid_node)
                        current_searched_node += 1
        else:
            for key, bar in self.bars.items():
                origin = bar.get_origin()
                end = bar.get_end()

                if origin not in got_nodes:
                    got_nodes.append(origin)

                if end not in got_nodes:
                    got_nodes.append(end)

        return got_nodes

    def get_nodes_reactions(self):
        """
        Once known the applied forces. determines the reactions of each node, assigns its value to the node instances
        and returns the reactions vector

        :return: Array with the reactions of the nodes
        """
        # Compute the forces and momentums referred to the nodes
        self.forces_and_momentums_in_structure()

        # Compute node reactions according the equation
        nodes_reactions = np.dot(self.assembled_matrix(), self.get_nodes_displacements())

        # Get all nodes in the structure in solving_nueeeeeeen order
        nodes_in_structure = self.get_nodes()

        # Assign each node its reactions
        for i in range(self.get_number_of_nodes()):
            # Since i starts at 0, the current solving_numeration = i + 1
            current_node = nodes_in_structure[i]
            # Index to retrieve the reactions more easily
            index_offset = i * submatrix_size

            x_reaction = nodes_reactions[index_offset]
            y_reaction = nodes_reactions[index_offset + 1]
            m_reaction = nodes_reactions[index_offset + 2]

            if current_node.has_support():
                node_forces = current_node.get_total_force_and_momentum()
                # x reaction
                x_reaction += node_forces[0]
                # y reaction
                y_reaction += node_forces[1]
                # m reaction
                m_reaction += node_forces[2]

                # Update nodes_reactions with the new values
                nodes_reactions[index_offset] = x_reaction
                nodes_reactions[index_offset + 1] = y_reaction
                nodes_reactions[index_offset + 2] = m_reaction

            reactions = {
                "x": x_reaction,
                "y": y_reaction,
                "momentum": m_reaction
            }

            current_node.set_reactions(reactions)

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
                f"Error in the query: ''{query}''. Or maybe the material {name} is not defined in the database."
            )

    def equals(self, other_material):
        if self.generic_name == other_material.generic_name and \
                self.name == other_material.name and \
                self.young_mod == other_material.young_mod and \
                self.rig_mod == other_material.rig_mod and \
                self.poisson_coef == other_material.poisson_coef and \
                self.thermal_dil_coef == other_material.thermal_dil_coef and \
                self.density == other_material.density:
            return True
        else:
            return False


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

        query = """SELECT name, name_number, area, inertia_moment_x, res_mod_x,
        inertia_moment_y, res_mod_y
        FROM profiles WHERE name = '""" + name + "' AND name_number = " + \
                name_number + ";"

        # All parameters must be the same than those of the table in the profile database
        result = db.execute_read_query(conn, query)

        if result:
            self.name, self.name_number, self.area, self.inertia_moment_x, self.res_mod_x, \
            self.inertia_moment_y, self.res_mod_y = result[0]
        else:
            raise LookupError("Error in the query: ''" + query + "''. Or maybe the profile " + name + " " +
                              name_number + " is not defined in the database.")

    def equals(self, other_profile):
        if self.name == other_profile.name and \
                self.name_number == other_profile.name_number and \
                self.area == other_profile.area and \
                self.inertia_moment_x == other_profile.inertia_moment_x and \
                self.res_mod_x == other_profile.res_mod_x and \
                self.inertia_moment_y == other_profile.inertia_moment_y and \
                self.res_mod_y == other_profile.res_mod_y:
            return True
        else:
            return False


class DistributedCharge:
    """
    Class to represent a distributed charge applied to a single beam
    """

    def __init__(self, dc_type, max_value, direction):
        """

        :param dc_type: type of distributed charge. Must be of type DistributedChargeType
        :param max_value: the maximum value of the distributed charge
        :param direction: unitary vector representing the direction of the force in local axis. If dc_type is
        PARALLEL_TO_BAR, the direction must be (0, 1, 0) or (0, -1, 0)
        """
        if type(dc_type) not in [DistributedChargeType]:
            raise TypeError("Error. dc_type must be of type structures.DistributedChargeType")

        self.dc_type = dc_type
        self.max_value = float(max_value)
        # self.direction = direction
        # Normalize given direction
        self.direction = tuple(map(
            lambda x: x / math.sqrt(direction[0] ** 2 +
                                    direction[1] ** 2 +
                                    direction[2] ** 2),
            direction
        ))

        # If new parameters are included, they must be added to the equals function and to the test

    def set_dc_type(self, new_type):
        """
        Modifies the distributed charge type to the specified one
        :param new_type: desired type to be changed to
        """
        if type(new_type) not in [DistributedChargeType]:
            raise TypeError("Error: new_type must be of type DistributedChargeType")

        self.dc_type = new_type

    def set_max_value(self, new_max_value):
        """
        Modifies the distributed charge maximum value to the specified one
        :param new_max_value: desired maximum value to be channged to
        """
        self.max_value = float(new_max_value)

    def set_direction(self, new_direction):
        """
        Modifies the distributed direction value to the specified one
        :param new_direction: desired direction be channged to
        """
        self.direction = new_direction

    def _redefine_direction(self, bar):
        bar_angle = bar.angle_from_global_to_local()

        if self.dc_type == DistributedChargeType.PARALLEL_TO_BAR:
            x_direction = abs(math.sin(bar_angle))
            y_direction = abs(math.cos(bar_angle))

            if self.direction[1] < 0:
                y_direction *= -1

            if (bar_angle < math.pi / 2 and bar_angle < 0) or \
                    (bar_angle < 3 * math.pi / 2 and bar_angle < math.pi):
                x_direction *= -1

            direction = (x_direction, y_direction, 0)
        else:
            direction = self.direction

        return direction

    def equals(self, other_distributed_charge):
        if type(other_distributed_charge) not in [DistributedCharge]:
            raise TypeError("Error. The type of other_distributed_charge must be DistributedCharge")

        if self.dc_type == other_distributed_charge.dc_type and \
                self.max_value == other_distributed_charge.max_value and \
                self.direction == other_distributed_charge.direction:
            return True
        else:
            return False

    def refer_to_nodes(self, bar, return_global_values):
        """
        Refer the distributed charge to the nodes of the bar provided
        :param bar: Bar over which the charge is applied to
        :param return_global_values: True if global values are wanted to be returned, False if local values are wanted
        :return: dictionary containing the referred values
        """
        bar_length = bar.length()
        direction = self._redefine_direction(bar)

        forces = self.max_value * np.array(direction)

        if self.dc_type in [DistributedChargeType.SQUARE, DistributedChargeType.PARALLEL_TO_BAR]:
            x_reaction = - forces[0] * bar_length / 2
            if self.max_value < 0 and self.dc_type == DistributedChargeType.PARALLEL_TO_BAR:
                x_reaction = - x_reaction

            y_reaction = - forces[1] * bar_length / 2

            momentum_abs = forces[1] * pow(bar_length, 2) / 12

            m_origin_reaction = - momentum_abs
            m_end_reaction = - m_origin_reaction
        else:
            raise Exception(f"Charge {self.dc_type} not implemented.")

        # TODO Si se incluyen mas tipos de cargas distribuidas, agregarlos aqui con elifs y escribir sus tests

        if not return_global_values:
            referred_charge = {
                "x": x_reaction,
                "y": y_reaction,
                "m_origin": m_origin_reaction,
                "m_end": m_end_reaction
            }
        else:
            # Convert reactions to global coordinates
            reaction_forces = np.array([x_reaction, y_reaction, 0])
            matrix_conversion_to_global = bar.system_change_matrix_2d_rigid_nodes()
            reaction_forces_global = np.dot(matrix_conversion_to_global, reaction_forces)

            referred_charge = {
                "x": reaction_forces_global[0],
                "y": reaction_forces_global[1],
                "m_origin": m_origin_reaction,
                "m_end": m_end_reaction
            }

        return referred_charge

    def axial_force_law(self, bar, origin_to_end_factor):
        """
        Determines the point x in the axial force law
        :param bar: bar in which the distributed charge is applied to
        :param origin_to_end_factor: Point to calculate. Can't be lesser than zero nor greater than 1
        :return:
        """
        if origin_to_end_factor < 0 or origin_to_end_factor > 1:
            raise ValueError("Error. x must be between 0 and 1, both inclusive.")

        direction = self._redefine_direction(bar)

        q = self.max_value
        x = bar.length() * origin_to_end_factor
        n = - q * direction[0] * x

        if self.max_value < 0 and self.dc_type == DistributedChargeType.PARALLEL_TO_BAR:
            n = -n

        return n

    def bending_moment_law(self, bar, origin_to_end_factor):
        """
        Determines the point x in the flector law
        :param bar: Bar in which the distributed charge is applied to
        :param origin_to_end_factor: Point to calculate. Can't be lesser than zero nor greater than 1
        :return:
        """
        if origin_to_end_factor < 0 or origin_to_end_factor > 1:
            raise ValueError("Error. x must be between 0 and 1, both inclusive.")

        direction = self._redefine_direction(bar)

        q = self.max_value
        x = origin_to_end_factor * bar.length()

        m = q * pow(x, 2) / 2 * direction[1]

        return m

    def shear_strength_law(self, bar, origin_to_end_factor):
        """
        Determines the point x in the shear strength
        :param bar: Bar in which the distributed charge is applied to
        :param origin_to_end_factor: Point to calculate. Can't be lesser than zero nor greater than 1
        :return:
        """
        if origin_to_end_factor < 0 or origin_to_end_factor > 1:
            raise ValueError("Error. x must be between 0 and 1, both inclusive.")

        direction = self._redefine_direction(bar)

        q = self.max_value
        bar_length = bar.length()
        x = origin_to_end_factor * bar_length

        v_general = - q * x * direction[1]

        v = v_general

        return v


class PunctualForceInBar:
    """
    Class that represents a punctual force applied in any point of a bar
    """

    def __init__(self, value, origin_end_factor, direction):
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

    def set_value(self, new_value):
        """
        Changes the current value of the force to the specified one.
        :param new_value: New value of the force.
        """
        self.value = new_value

    def set_origin_end_factor(self, new_origin_end_factor):
        """
        Changes the current value of origin_end_factor to the specified one.
        :param new_origin_end_factor: New value of origin_end_factor
        """
        self.origin_end_factor = new_origin_end_factor

    def set_direction(self, new_direction):
        """
        Changes the current value of direction to the specified one.
        :param new_direction: New value of direction
        """
        self.direction = new_direction

    def refer_to_nodes(self, bar, return_global_values):
        """
        Refer the punctual force to the nodes of the bar provided
        :param bar: Bar over which the charge is applied to
        :param return_global_values: True if global values are wanted to be returned, False if local values are wanted
        :return: dictionary containing the referred values
        """
        bar_length = bar.length()
        distance_origin_force = bar_length * self.origin_end_factor
        distance_end_force = bar_length * (1 - self.origin_end_factor)

        # Decompose force according each axis
        forces_in_axis = tuple([self.value * x for x in self.direction])

        # Reactions in each axis
        y_reaction_origin = - forces_in_axis[1] * pow(distance_end_force, 2) * (
                bar_length + 2 * distance_origin_force) / pow(bar_length, 3)
        y_reaction_end = - forces_in_axis[1] * pow(distance_origin_force, 2) * (
                bar_length + 2 * distance_end_force) / pow(bar_length, 3)
        x_reaction = - forces_in_axis[0] / 2

        reaction_origin = np.array([x_reaction, y_reaction_origin, 0])
        reaction_end = np.array([x_reaction, y_reaction_end, 0])

        # Bending moments
        flector_origin = - forces_in_axis[1] * distance_origin_force * pow(distance_end_force, 2) / pow(
            bar_length, 2)
        flector_end = forces_in_axis[1] * pow(distance_origin_force, 2) * distance_end_force / pow(bar_length,
                                                                                                   2)
        # TODO Repasar el signo en flector_force_point
        flector_force_point = 2 * forces_in_axis[1] * pow(distance_origin_force, 2) * pow(distance_end_force,
                                                                                          2) / pow(bar_length,
                                                                                                   3)

        if not return_global_values:
            referred_punctual_force = {
                "x_origin": reaction_origin[0],
                "x_end": reaction_end[0],
                "y_origin": reaction_origin[1],
                "y_end": reaction_end[1],
                "m_origin": flector_origin,
                "m_end": flector_end,
                "m_force_point": flector_force_point
            }
        else:
            # Convert reactions to global coordinates
            matrix_conversion_to_global = bar.system_change_matrix_2d_rigid_nodes()
            reaction_origin_global = np.dot(matrix_conversion_to_global, reaction_origin)
            reaction_end_global = np.dot(matrix_conversion_to_global, reaction_end)

            referred_punctual_force = {
                "x_origin": reaction_origin_global[0],
                "x_end": reaction_end_global[0],
                "y_origin": reaction_origin_global[1],
                "y_end": reaction_end_global[1],
                "m_origin": flector_origin,
                "m_end": flector_end,
                "m_force_point": flector_force_point
            }

        return referred_punctual_force

    def bending_moment_law(self, bar, origin_to_end_factor):
        """
        Determines the point x in the flector law
        :param bar: Bar in which the punctual force is applied to.
        :param origin_to_end_factor: Point to calculate. Can't be lesser than zero nor greater than 1
        :return:
        """

        if origin_to_end_factor < 0 or origin_to_end_factor > 1:
            raise ValueError("Error. x must be between 0 and 1, both inclusive.")

        p = self.value
        bar_length = bar.length()
        x = origin_to_end_factor * bar_length
        distance_origin_to_force = bar_length * self.origin_end_factor

        if origin_to_end_factor <= self.origin_end_factor:
            m_origin_to_force_point = 0
            return m_origin_to_force_point
        else:
            m_force_point_to_end = p * self.direction[1] * (x - distance_origin_to_force)
            return m_force_point_to_end

    def shear_strength_law(self, bar, origin_to_end_factor):
        """
        Determines the point x in the shear strength law
        :param bar: Bar in which the punctual force is applied to.
        :param origin_to_end_factor: Point to calculate. Can't be lesser than zero nor greater than 1
        :return:
        """
        if origin_to_end_factor < 0 or origin_to_end_factor > 1:
            raise ValueError("Error. x must be between 0 and 1, both included.")

        p = self.value * self.direction[1]

        # If the point is situated before the one in which the force is applied...
        if origin_to_end_factor <= self.origin_end_factor:
            return 0
        else:
            return - p

    def axial_force_law(self, bar, origin_to_end_factor):
        """
        Determines the point x in the axial force law
        :param bar: Bar in which the punctual force is applied to.
        :param origin_to_end_factor: Point to calculate. Can't be lesser than zero nor greater than 1
        :return:
        """
        if origin_to_end_factor < 0 or origin_to_end_factor > 1:
            raise ValueError("Error. x must be between 0 and 1, both included.")

        p = self.value * self.direction[0]

        # If the point is situated before the one in which the force is applied...
        if origin_to_end_factor <= self.origin_end_factor:
            return 0
        else:
            return - p
