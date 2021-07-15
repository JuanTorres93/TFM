# Run this file with python3 -m unittest
# Otherwise it will fail due to imports
import math

import numpy as np
import unittest
try:
    import context
except ModuleNotFoundError:
    import src.tests.context

from src.modules import structures as st



class TestNode(unittest.TestCase):
    def test_constructor(self):
        node = st.Node("N1", position=(1, 2, 3), forces_in_node={"F1": (4, 5, 6)}, momentums_in_node={"M1": (7, 8, 9)},
                       support=st.Support.NONE)

        self.assertEqual(node.name, "N1")
        np.testing.assert_array_equal(node.position, np.array((1, 2, 3)))
        np.testing.assert_array_equal(node.forces_in_node.get("F1"), np.array((4, 5, 6)))
        np.testing.assert_array_equal(node.momentums_in_node.get("M1"), np.array((7, 8, 9)))
        self.assertEqual(node.support, st.Support.NONE)
        self.assertRaises(TypeError, st.Node, 3)

        self.assertRaises(TypeError, st.Node, "N1", position=[1, 2, 3])
        self.assertRaises(TypeError, st.Node, "N1", force=[1, 2, 3])
        self.assertRaises(TypeError, st.Node, "N1", momentum=[1, 2, 3])

    def test_equals(self):
        n1 = st.Node("N1", position=(1, 2, 3), forces_in_node={"F1": (4, 5, 6)}, momentums_in_node={"M1": (7, 8, 9)},
                     support=st.Support.PINNED)
        n2 = st.Node("N2", position=(2, 2, 3), forces_in_node={"F1": (4, 5, 6)}, momentums_in_node={"M1": (7, 8, 9)})
        n3 = st.Node("N3", position=(1, 2, 3), forces_in_node={"F1": (4, 5, 6)}, momentums_in_node={"M1": (7, 8, 9)})
        n4 = st.Node("N4", position=(2, 2, 3), forces_in_node={"F1": (4, 5, 6)}, momentums_in_node={"M1": (7, 8, 9)},
                     support=st.Support.PINNED)

        self.assertTrue(n1.equals(n1))
        self.assertTrue(not n1.equals(n2))
        self.assertTrue(not n1.equals(n3))
        self.assertTrue(not n1.equals(n4))

    def test_set_name(self):
        node = st.Node("N1")

        node.set_name("node")
        self.assertEqual("node", node.name)

        self.assertRaises(TypeError, node.set_name, 0)

    def test_set_position(self):
        node = st.Node("N1", position=(1, 1, 1))

        node.set_position((2, 2, 2))

        np.testing.assert_array_equal(node.position, (2, 2, 2))

        self.assertRaises(TypeError, node.set_position, 0)
        self.assertRaises(TypeError, node.set_position, [0, 0, 0])

    def test_set_support(self):
        node = st.Node("N1", momentums_in_node={"M1": (1, 1, 1)})
        self.assertEqual(node.support, st.Support.NONE)

        node.set_support(st.Support.FIXED)
        self.assertEqual(node.support, st.Support.FIXED)

        self.assertRaises(TypeError, node.set_support, 0)
        self.assertRaises(TypeError, node.set_support, [0, 0, 0])

    def test_x(self):
        node = st.Node("N1", position=(1, 2, 3))
        self.assertEqual(1, node.x())

        node.set_position((2, 2, 2))
        self.assertEqual(2, node.x())

    def test_y(self):
        node = st.Node("N1", position=(1, 2, 3))
        self.assertEqual(2, node.y())

        node.set_position((2, 2, 2))
        self.assertEqual(2, node.y())

    def test_z(self):
        node = st.Node("N1", position=(1, 2, 3))
        self.assertEqual(3, node.z())

        node.set_position((2, 2, 2))
        self.assertEqual(2, node.z())

    def test_add_force(self):
        node = st.Node("N1", position=(1, 2, 3))

        self.assertRaises(TypeError, node.add_force, "n2", 2, True)
        self.assertRaises(TypeError, node.add_force, "n2", (2, 2, 2), "j")

        self.assertEqual(len(node.get_forces_in_node_dictionary()), 0)

        node.add_force("n1", (0, 0, 0), True)
        self.assertEqual(len(node.get_forces_in_node_dictionary()), 1)

        node2 = st.Node("N1", position=(1, 2, 3), forces_in_node={"n2": (3, 4, 5)})
        self.assertEqual(len(node2.get_forces_in_node_dictionary()), 1)

        # Momentum does not affect force
        node2 = st.Node("N1", position=(1, 2, 3), momentums_in_node={"n2": (3, 4, 5)})
        self.assertEqual(len(node2.get_forces_in_node_dictionary()), 0)

        node2.add_force("f1", (1, 1, 1), belongs_to_node=True)
        self.assertEqual(len(node2.get_forces_in_node_dictionary()), 1)

        node2.add_force("f2", (1, 1, 1), belongs_to_node=False)
        self.assertEqual(len(node2.get_referred_forces_dictionary()), 1)

    def test_add_momentum(self):
        node = st.Node("N1", position=(1, 2, 3))

        self.assertRaises(TypeError, node.add_momentum, "n2", 2, True)
        self.assertRaises(TypeError, node.add_momentum, "n2", (2, 2, 2), "j")

        self.assertEqual(len(node.get_momentum_in_node_dictionary()), 0)

        node.add_momentum("n1", (0, 0, 0), True)
        self.assertEqual(len(node.get_momentum_in_node_dictionary()), 1)

        node2 = st.Node("N2", position=(1, 2, 3), momentums_in_node={"n2": (3, 4, 5)})
        self.assertEqual(len(node2.get_momentum_in_node_dictionary()), 1)

        # Force does not affect momentum
        node2 = st.Node("N2", position=(1, 2, 3), forces_in_node={"n2": (3, 4, 5)})
        self.assertEqual(len(node2.get_momentum_in_node_dictionary()), 0)

        node2.add_momentum("f1", (1, 1, 1), belongs_to_node=True)
        self.assertEqual(len(node2.get_momentum_in_node_dictionary()), 1)

        node2.add_momentum("f2", (1, 1, 1), belongs_to_node=False)
        self.assertEqual(len(node2.get_referred_momentum_dictionary()), 1)

    def test_get_total_force_and_momentum(self):
        # Test 1
        force_1 = {"F1": (1, 2, 3)}
        n1 = st.Node("N1", position=(1, 2, 3), forces_in_node=force_1)

        # expected_result = np.array([Fx, Fy, Mz])
        expected_result = np.array([1, 2, 0])
        np.testing.assert_array_equal(expected_result, n1.get_total_force_and_momentum())

        # Test 2
        force_1 = {"F1": (1, 2, 3)}
        momentum_1 = {"M1": (1, 2, 3)}
        n1 = st.Node("N1", position=(1, 2, 3), forces_in_node=force_1, momentums_in_node=momentum_1)

        # expected_result = np.array([Fx, Fy, Mz])
        expected_result = np.array([1, 2, 3])
        np.testing.assert_array_equal(expected_result, n1.get_total_force_and_momentum())

        # Test 2
        force_1 = {"F1": (1, 2, 3)}
        momentum_1 = {"M1": (1, 2, 3)}
        n1 = st.Node("N1", position=(1, 2, 3), forces_in_node=force_1, momentums_in_node=momentum_1)

        n1.add_force("F2", (3, 2, 1), False)

        # expected_result = np.array([Fx, Fy, Mz])
        expected_result = np.array([4, 4, 3])
        np.testing.assert_array_equal(expected_result, n1.get_total_force_and_momentum())

        n1.add_momentum("M2", (3, 3, 1), False)
        expected_result = np.array([4, 4, 4])
        np.testing.assert_array_equal(expected_result, n1.get_total_force_and_momentum())

        # TODO Incluir mas tests dentro de este metodo cuando las fuerzas se especifiquen como listas y haya metodos
        # para agregaar y quitar fuerzas aplciadas

    def test_set_and_get_displacement(self):
        node = st.Node("N1")

        new_displacement = {
            "x": 1,
            "y": 2,
            "angle": 3
        }

        node.set_displacement(new_displacement)
        self.assertEqual(node.get_displacement(), new_displacement)

        self.assertRaises(TypeError, node.set_displacement, 0)

    def test_set_and_get_reactions(self):
        node = st.Node("N1")

        new_reactions = {
            "x": 1,
            "y": 2,
            "momentum": 3
        }

        node.set_reactions(new_reactions)

        self.assertEqual(node.get_reactions(), new_reactions)
        self.assertRaises(TypeError, node.set_reactions, 0)

    def test_has_support(self):
        n1 = st.Node("N1")

        self.assertFalse(n1.has_support())

        n2 = st.Node("N2", support=st.Support.PINNED)
        self.assertTrue(n2.has_support())


class TestBar(unittest.TestCase):
    def test_constructor(self):
        n_ori = st.Node("N1")
        n_end = st.Node("N2", position=(1, 2, 3))

        bar = st.Bar("B1", n_ori, n_end)

        self.assertEqual(bar.name, "B1")
        self.assertEqual(bar.origin, n_ori)
        self.assertEqual(bar.end, n_end)
        self.assertIsInstance(bar.material, st.Material)
        self.assertRaises(TypeError, st.Bar, "name", n_ori, (0, 0, 0))
        self.assertRaises(TypeError, st.Bar, "name", (0, 0, 0), n_end)
        self.assertRaises(TypeError, st.Bar, 3, n_ori, n_end)
        self.assertRaises(ValueError, st.Bar, "name", n_ori, n_ori)
        self.assertRaises(TypeError, st.Bar, "name", n_ori, n_end, material=2)

    def test_set_name(self):
        n_ori = st.Node("N1")
        n_end = st.Node("N2")

        bar = st.Bar("B1", n_ori, n_end)
        bar.set_name("New name")

        self.assertEqual("New name", bar.name)

        self.assertRaises(TypeError, bar.set_name, 0)

    def test_set_origin(self):
        n_ori = st.Node("N1")
        n_end = st.Node("N2")

        bar = st.Bar("B1", n_ori, n_end)
        self.assertEqual(bar.origin, n_ori)

        new_origin = st.Node("N3", position=(1, 2, 3))
        bar.set_origin(new_origin)

        self.assertEqual(bar.origin, new_origin)

        self.assertRaises(TypeError, bar.set_origin, 0)

    def test_set_end(self):
        n_ori = st.Node("N1")
        n_end = st.Node("N2")

        bar = st.Bar("B1", n_ori, n_end)
        self.assertEqual(bar.end, n_end)

        new_end = st.Node("N3", (1, 2, 3))
        bar.set_end(new_end)

        self.assertEqual(bar.end, new_end)
        self.assertRaises(TypeError, bar.set_end, 0)
        self.assertRaises(TypeError, bar.set_end, (0, 0, 0))

    def test_length(self):
        n_ori = st.Node("N1")
        n_end = st.Node("N2", position=(1, 2, 3))

        bar = st.Bar("B1", n_ori, n_end)

        self.assertAlmostEqual(bar.length(), 3.741657387)

    def test_local_rigidity_matrix_2d_rigid_nodes(self):
        n_ori = st.Node("N1")
        n_end = st.Node("N2", position=(0, 4.2, 0))

        bar = st.Bar("B1", n_ori, n_end)

        calculated_matrix = bar.local_rigidity_matrix_2d_rigid_nodes()

        expected_matrix = np.array([
            [263798885, 0, 0, -263798885, 0, 0],
            [0, 2787223.38, 5853169.1, 0, -2787223.38, 5853169.10],
            [0, 5853169.1, 16388873.48, 0, -5853169.1, 8194436.74],
            [-263798885, 0, 0, 263798885, 0, 0],
            [0, -2787223.38, -5853169.1, 0, 2787223.38, -5853169.10],
            [0, 5853169.1, 8194436.74, 0, -5853169.1, 16388873.48],
        ])

        np.testing.assert_allclose(calculated_matrix, expected_matrix)


    def test_system_change_matrix_2d_rigid_nodes(self):
        n_ori = st.Node("N1")
        n_end = st.Node("N2", position=(0, 4.2, 0))

        bar = st.Bar("B1", n_ori, n_end)

        calculated_matrix = bar.system_change_matrix_2d_rigid_nodes()

        expected_matrix = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ])

        np.testing.assert_allclose(calculated_matrix, expected_matrix, rtol=1e-5, atol=1e-7)

    def test_angle_from_global_to_local(self):
        n1 = st.Node("N1", position=(0, 0, 0), support=st.Support.PINNED)
        n2 = st.Node("N2", position=(0, 4.2, 0))
        n3 = st.Node("N3", position=(6.8, 5.25, 0))
        n4 = st.Node("N4", position=(13.6, 4.2, 0))
        n5 = st.Node("N5", position=(17.2, 3.5, 0))
        n6 = st.Node("N6", position=(13.6, 0, 0), support=st.Support.PINNED)

        b1 = st.Bar("B1", n1, n2)
        b2 = st.Bar("B2", n2, n3)
        b3 = st.Bar("B3", n3, n4)
        b4 = st.Bar("B4", n4, n5)
        b5 = st.Bar("B5", n4, n6)

        b1 = st.Bar("B1", n1, n2)

        np.testing.assert_equal(b1.angle_from_global_to_local(), np.pi / 2)
        np.testing.assert_almost_equal(b2.angle_from_global_to_local(), math.radians(8.778), 3)
        np.testing.assert_almost_equal(b3.angle_from_global_to_local(), math.radians(351.222), 3)
        # Differs in 0.03 radians due to trigonometry used to calculate node position
        np.testing.assert_almost_equal(b4.angle_from_global_to_local(), math.radians(351.222), 0)
        np.testing.assert_equal(b5.angle_from_global_to_local(), math.radians(270))

    def test_global_rigidity_matrix_2d_rigid_nodes(self):
        n_ori = st.Node("N1")
        n_end = st.Node("N2", position=(0, 4.2, 0))

        bar = st.Bar("B1", n_ori, n_end)

        calculated_matrix = bar.global_rigidity_matrix_2d_rigid_nodes()

        expected_matrix = np.array([
            [2787223.38, 0, -5853169.10, -2787223.38, 0, -5853169.10],
            [0, 263798885, 0, 0, -263798885, 0],
            [-5853169.1, 0, 16388873.48, 5853169.10, 0, 8194436.74],
            [-2787223.38, 0, 5853169.10, 2787223.38, 0, 5853169.10],
            [0, -263798885, 0, 0, 263798885, 0],
            [-5853169.1, 0, 8194436.74, 5853169.10, 0, 16388873.48],
        ])

        np.testing.assert_allclose(calculated_matrix, expected_matrix, rtol=1e-6, atol=1)

    def test_add_and_get_distributed_charge(self):
        n2 = st.Node("N2", position=(0, 4.2, 0))
        n3 = st.Node("N3", position=(6.8, 5.25, 0))

        b2 = st.Bar("B2", n2, n3)

        self.assertRaises(TypeError, b2.add_distributed_charge, 3)

        new_distributed_charge = st.DistributedCharge(st.DistributedChargeType.SQUARE, max_value=7,
                                                      direction=(0, -1, 0))

        dc_name = "dc"

        b2.add_distributed_charge(new_distributed_charge, dc_name)
        are_equals = new_distributed_charge.equals(b2.get_distributed_charges().get(dc_name))
        self.assertTrue(are_equals)

    def test_get_referred_distributed_charge_to_nodes(self):
        # Single bar test
        n2 = st.Node("N2", position=(0, 4.2, 0))
        n3 = st.Node("N3", position=(6.8, 5.25, 0))

        b2 = st.Bar("B2", n2, n3)

        dc = st.DistributedCharge(dc_type=st.DistributedChargeType.SQUARE, max_value=10179.36, direction=(0, -1, 0))
        b2.add_distributed_charge(dc, "test1")

        calculated_values = b2.get_referred_distributed_charge_to_nodes(return_global_values=False)
        expected_values = {
            "x": 0,
            "y": 35020.05,
            "m_origin": 40159.83,
            "m_end": - 40159.83
        }

        self.assertAlmostEqual(calculated_values.get("x"), expected_values.get("x"), places=0)
        self.assertAlmostEqual(calculated_values.get("y"), expected_values.get("y"), places=0)
        self.assertAlmostEqual(calculated_values.get("m_origin"), expected_values.get("m_origin"), places=0)
        self.assertAlmostEqual(calculated_values.get("m_end"), expected_values.get("m_end"), places=0)


    def test_get_referred_punctual_force_to_nodes(self):
        n1 = st.Node("N1", position=(0, 0, 0), support=st.Support.PINNED)
        n2 = st.Node("N2", position=(0, 4.2, 0))
        n3 = st.Node("N3", position=(6.8, 5.25, 0))
        n4 = st.Node("N4", position=(13.6, 4.2, 0))
        n5 = st.Node("N5", position=(17.2, 3.644117647, 0))
        n6 = st.Node("N6", position=(13.6, 0, 0), support=st.Support.FIXED)

        b1 = st.Bar("B1", n1, n2)
        b2 = st.Bar("B2", n2, n3)
        b3 = st.Bar("B3", n3, n4)
        b4 = st.Bar("B4", n4, n5)
        b5 = st.Bar("B5", n4, n6)

        dc = st.DistributedCharge(st.DistributedChargeType.SQUARE, -10179.36, (0, -1, 0))
        b2.add_distributed_charge(dc)
        b3.add_distributed_charge(dc)
        b4.add_distributed_charge(dc)

        pf = st.PunctualForceInBar(-25000, 0.5, (0, 1, 0))
        b5.add_punctual_force(pf, "pf")

        bars = {
            b1.name: b1,
            b2.name: b2,
            b3.name: b3,
            b4.name: b4,
            b5.name: b5
        }

        structure = st.Structure("S1", bars)

        # Local coordinates
        calculated_values = b5.get_referred_punctual_forces_in_bar_to_nodes(return_global_values=False)
        expected_values = {
            "x_origin": 0,
            "x_end": 0,
            "y_origin": 12500,
            "y_end": 12500,
            "m_origin": 13125,
            "m_end": -13125
        }

        self.assertAlmostEqual(calculated_values.get("y_origin"), expected_values.get("y_origin"), places=0)
        self.assertAlmostEqual(calculated_values.get("y_end"), expected_values.get("y_end"), places=0)
        self.assertAlmostEqual(calculated_values.get("x_origin"), expected_values.get("x_origin"), places=0)
        self.assertAlmostEqual(calculated_values.get("x_end"), expected_values.get("x_end"), places=0)
        self.assertAlmostEqual(calculated_values.get("m_origin"), expected_values.get("m_origin"), places=0)
        self.assertAlmostEqual(calculated_values.get("m_end"), expected_values.get("m_end"), places=0)

        # Global coordinates
        calculated_values = b5.get_referred_punctual_forces_in_bar_to_nodes(return_global_values=True)
        expected_values = {
            "x_origin": 12500,
            "x_end": 12500,
            "y_origin": 0,
            "y_end": 0,
            "m_origin": 13125,
            "m_end": -13125
        }

        self.assertAlmostEqual(calculated_values.get("y_origin"), expected_values.get("y_origin"), places=0)
        self.assertAlmostEqual(calculated_values.get("y_end"), expected_values.get("y_end"), places=0)
        self.assertAlmostEqual(calculated_values.get("x_origin"), expected_values.get("x_origin"), places=0)
        self.assertAlmostEqual(calculated_values.get("x_end"), expected_values.get("x_end"), places=0)
        self.assertAlmostEqual(calculated_values.get("m_origin"), expected_values.get("m_origin"), places=0)
        self.assertAlmostEqual(calculated_values.get("m_end"), expected_values.get("m_end"), places=0)

    def test__add_object_to_instance_dictionary(self):
        n2 = st.Node("N2", position=(0, 4.2, 0))
        n3 = st.Node("N3", position=(6.8, 5.25, 0))

        b2 = st.Bar("B2", n2, n3)

        self.assertEqual(len(b2.get_distributed_charges()), 0)
        self.assertEqual(len(b2.get_punctual_forces()), 0)

        # Add a distributed charge
        dc = st.DistributedCharge(dc_type=st.DistributedChargeType.SQUARE, max_value=10179.36, direction=(0, -1, 0))
        b2.add_distributed_charge(dc, "test1")

        self.assertEqual(len(b2.get_distributed_charges()), 1)

        # Add a punctual force
        pf = st.PunctualForceInBar(40, 0.8, (-1, 0, 0))
        b2.add_punctual_force(pf, "test2")
        self.assertEqual(len(b2.get_punctual_forces()), 1)

    def test_add_and_get_punctual_forces(self):
        n2 = st.Node("N2", position=(0, 4.2, 0))
        n3 = st.Node("N3", position=(6.8, 5.25, 0))

        b2 = st.Bar("B2", n2, n3)

        self.assertRaises(TypeError, b2.add_punctual_force, 3)

        new_punctual_force = st.PunctualForceInBar(459, 0.8, (-1, 0, 0))

        pf_name = "pf"

        b2.add_punctual_force(new_punctual_force, pf_name)
        are_equals = new_punctual_force.equals(b2.get_punctual_forces().get(pf_name))
        self.assertTrue(are_equals)

    def test_get_referred_punctual_forces_to_nodes(self):
        n4 = st.Node("N4", position=(13.6, 4.2, 0))
        n6 = st.Node("N6", position=(13.6, 0, 0))

        b5 = st.Bar("B5", n4, n6)

        pf = st.PunctualForceInBar(-25000, 0.5, (0, 1, 0))
        b5.add_punctual_force(pf, "pf")

        calculated_values = b5.get_referred_punctual_forces_in_bar_to_nodes(False)
        expected_values = {
            "y_origin": 12500,
            "y_end": 12500,
            # TODO check that both signs are correct
            "m_origin": 13125,
            "m_end": - 13125
        }

        self.assertAlmostEqual(calculated_values.get("y_origin"), expected_values.get("y_origin"), places=0)
        self.assertAlmostEqual(calculated_values.get("y_end"), expected_values.get("y_end"), places=0)
        self.assertAlmostEqual(calculated_values.get("m_origin"), expected_values.get("m_origin"), places=0)
        self.assertAlmostEqual(calculated_values.get("m_end"), expected_values.get("m_end"), places=0)

    def test_has_distributed_charges(self):
        n1 = st.Node("N1", position=(0, 0, 0), support=st.Support.PINNED)
        n2 = st.Node("N2", position=(0, 4.2, 0))

        b1 = st.Bar("B1", n1, n2)
        self.assertFalse(b1.has_distributed_charges())

        dc = st.DistributedCharge(st.DistributedChargeType.SQUARE, -10179.36, (0, -1, 0))
        b1.add_distributed_charge(dc)
        self.assertTrue(b1.has_distributed_charges())

    def test_has_punctual_force(self):
        n4 = st.Node("N4", position=(13.6, 4.2, 0))
        n6 = st.Node("N6", position=(13.6, 0, 0))

        b5 = st.Bar("B5", n4, n6)
        self.assertFalse(b5.has_punctual_forces())

        pf = st.PunctualForceInBar(-25000, 0.5, (0, 1, 0))
        b5.add_punctual_force(pf, "pf")
        self.assertTrue(b5.has_punctual_forces())

class TestStructure(unittest.TestCase):
    def test_constructor(self):
        n1 = st.Node("N1")
        n2 = st.Node("N2", position=(0, 2, 3))
        n3 = st.Node("N3", position=(1, 2, 3))

        bar_1 = st.Bar("B1", n1, n2)
        bar_2 = st.Bar("B2", n2, n3)

        bar_list = [bar_2, bar_1]

        self.assertRaises(TypeError, st.Structure, "name", bar_1)
        self.assertRaises(TypeError, st.Structure, "name", bar_list)

        bar_dict = {
            'b1': bar_1,
            'b2': bar_2
        }

        structure = st.Structure("S1", bar_dict)

        self.assertEqual(structure.name, "S1")
        self.assertEqual(structure.bars.get('b1'), bar_1)
        self.assertEqual(structure.bars.get('b2'), bar_2)

        bar_3 = st.Bar("B1", n2, n3)
        bar_dict = {
            'b1': bar_1,
            'b2': bar_3
        }

        self.assertRaises(ValueError, st.Structure, "name", bar_dict)

    def test_assembled_matrix(self):
        n1 = st.Node("N1", position=(0, 0, 0), support=st.Support.PINNED)
        n2 = st.Node("N2", position=(0, 4.2, 0))
        n3 = st.Node("N3", position=(6.8, 5.25, 0))
        n4 = st.Node("N4", position=(13.6, 4.2, 0))
        n5 = st.Node("N5", position=(17.2, 3.644117647, 0))
        n6 = st.Node("N6", position=(13.6, 0, 0), support=st.Support.FIXED)

        b1 = st.Bar("B1", n1, n2)
        b2 = st.Bar("B2", n2, n3)
        b3 = st.Bar("B3", n3, n4)
        b4 = st.Bar("B4", n4, n5)
        b5 = st.Bar("B5", n4, n6)

        bars = {
            b1.name: b1,
            b2.name: b2,
            b3.name: b3,
            b4.name: b4,
            b5.name: b5
        }

        structure = st.Structure("S1", bars)

        calculated_matrix = structure.assembled_matrix()

        expected_matrix = 10000 * np.array([
            [278, 0, -585, -278, 0, -585, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 26379, 0, 0, -26379, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-585, 0, 1638, 585, 0, 819, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-278, 0, 585, 16007, 2418, 552, -15729, -2428, -33, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, -26379, 0, 2418, 26816, 215, -2418, -436, 215, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-585, 0, 819, 552, 215, 2639, 33, -215, 500, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, -15729, -2418, 33, 31458, 0, 66, -15729, 2418, 33, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, -2418, -436, -215, 0, 873, 0, 2418, -436, 215, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, -33, 215, 500, 66, 0, 2000, -33, -215, 500, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, -15729, 2418, -33, 45725, -6941, 670, -29717, 4522, 118, -278, 0, 585],
            [0, 0, 0, 0, 0, 0, 2418, -436, -215, -6941, 27942, 553, 4522, -1125, 769, 0, -26379, 0],
            [0, 0, 0, 0, 0, 0, 33, 215, 500, 670, 553, 4528, -118, -769, 944, -585, 0, 819],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, -29717, 4522, -118, 29717, -4522, -118, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 4522, -1125, -769, -4522, 1125, -769, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 118, 769, 944, -118, -769, 1889, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, -278, 0, -585, 0, 0, 0, 278, 0, -585],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -26379, 0, 0, 0, 0, 0, 26379, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 585, 0, 819, 0, 0, 0, -585, 0, 1638]
        ])

        np.testing.assert_allclose(calculated_matrix, expected_matrix, atol=90300)

    def test_decoupled_matrix(self):
        n1 = st.Node("N1", position=(0, 0, 0), support=st.Support.PINNED)
        n2 = st.Node("N2", position=(0, 4.2, 0))
        n3 = st.Node("N3", position=(6.8, 5.25, 0))
        n4 = st.Node("N4", position=(13.6, 4.2, 0))
        n5 = st.Node("N5", position=(17.2, 3.644117647, 0))
        n6 = st.Node("N6", position=(13.6, 0, 0), support=st.Support.FIXED)

        b1 = st.Bar("B1", n1, n2)
        b2 = st.Bar("B2", n2, n3)
        b3 = st.Bar("B3", n3, n4)
        b4 = st.Bar("B4", n4, n5)
        b5 = st.Bar("B5", n4, n6)

        bars = {
            b1.name: b1,
            b2.name: b2,
            b3.name: b3,
            b4.name: b4,
            b5.name: b5
        }

        structure = st.Structure("S1", bars)

        calculated_matrix = structure.decoupled_matrix()

        expected_matrix = 10000 * np.array([
            [1638, 585, 0, 819, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [585, 16007, 2418, 552, -15729, -2428, -33, 0, 0, 0, 0, 0, 0],
            [0, 2418, 26816, 215, -2418, -436, 215, 0, 0, 0, 0, 0, 0],
            [819, 552, 215, 2639, 33, -215, 500, 0, 0, 0, 0, 0, 0],
            [0, -15729, -2418, 33, 31458, 0, 66, -15729, 2418, 33, 0, 0, 0],
            [0, -2418, -436, -215, 0, 873, 0, 2418, -436, 215, 0, 0, 0],
            [0, -33, 215, 500, 66, 0, 2000, -33, -215, 500, 0, 0, 0],
            [0, 0, 0, 0, -15729, 2418, -33, 45725, -6941, 670, -29717, 4522, 118],
            [0, 0, 0, 0, 2418, -436, -215, -6941, 27942, 553, 4522, -1125, 769],
            [0, 0, 0, 0, 33, 215, 500, 670, 553, 4528, -118, -769, 944],
            [0, 0, 0, 0, 0, 0, 0, -29717, 4522, -118, 29717, -4522, -118],
            [0, 0, 0, 0, 0, 0, 0, 4522, -1125, -769, -4522, 1125, -769],
            [0, 0, 0, 0, 0, 0, 0, 118, 769, 944, -118, -769, 1889],
        ])

        np.testing.assert_allclose(calculated_matrix, expected_matrix, atol=90300)

    def test_get_number_of_nodes(self):
        n1 = st.Node("N1", position=(0, 0, 0), support=st.Support.PINNED)
        n2 = st.Node("N2", position=(0, 4.2, 0))
        n3 = st.Node("N3", position=(6.8, 5.25, 0))
        n4 = st.Node("N4", position=(13.6, 4.2, 0))
        n5 = st.Node("N5", position=(17.2, 3.644117647, 0))
        n6 = st.Node("N6", position=(13.6, 0, 0), support=st.Support.FIXED)

        b1 = st.Bar("B1", n1, n2)
        b2 = st.Bar("B2", n2, n3)
        b3 = st.Bar("B3", n3, n4)
        b4 = st.Bar("B4", n4, n5)
        b5 = st.Bar("B5", n4, n6)

        bars = {
            b1.name: b1,
            b2.name: b2,
            b3.name: b3,
            b4.name: b4,
            b5.name: b5
        }

        structure = st.Structure("S1", bars)

        self.assertEqual(structure.get_number_of_nodes(), 6)

    def test_forces_and_momentums_in_structure(self):
        # TODO no se deben añadir las fuerzas ni los momentos más de una vez, para cualquier número ...
        # ... de llamadas a esta u otroas funciones
        n1 = st.Node("N1", position=(0, 0, 0), support=st.Support.PINNED)
        n2 = st.Node("N2", position=(0, 4.2, 0))
        n3 = st.Node("N3", position=(6.8, 5.25, 0))
        n4 = st.Node("N4", position=(13.6, 4.2, 0))
        n5 = st.Node("N5", position=(17.2, 3.644117647, 0))
        n6 = st.Node("N6", position=(13.6, 0, 0), support=st.Support.FIXED)

        b1 = st.Bar("B1", n1, n2)
        b2 = st.Bar("B2", n2, n3)
        b3 = st.Bar("B3", n3, n4)
        b4 = st.Bar("B4", n4, n5)
        b5 = st.Bar("B5", n4, n6)

        dc = st.DistributedCharge(st.DistributedChargeType.SQUARE, 10179.36, (0, -1, 0))
        b2.add_distributed_charge(dc)
        b3.add_distributed_charge(dc)
        b4.add_distributed_charge(dc)

        pf = st.PunctualForceInBar(-25000, 0.5, (0, 1, 0))
        b5.add_punctual_force(pf, "pf")

        bars = {
            b1.name: b1,
            b2.name: b2,
            b3.name: b3,
            b4.name: b4,
            b5.name: b5
        }

        structure = st.Structure("S1", bars)

        calculated_forces = structure.forces_and_momentums_in_structure()
        expected_forces = np.array([0, 0, 0,
                                    0, -35020.05, -40159.83,
                                    0, -70040.1, 0,
                                    -12500, -53560.23, 15778.78,
                                    0, -18540.18, 11256.05,
                                    -12500, 0, 13125])

        np.testing.assert_allclose(calculated_forces, expected_forces, atol=0.3)

        # This is done to assure that referred forces and momentums are not stacked
        calculated_forces = structure.forces_and_momentums_in_structure()
        calculated_forces = structure.forces_and_momentums_in_structure()
        calculated_forces = structure.forces_and_momentums_in_structure()
        np.testing.assert_allclose(calculated_forces, expected_forces, atol=0.3)

    def test_get_indexes_to_delete(self):
        n1 = st.Node("N1", position=(0, 0, 0), forces_in_node={"F1": (0, 0, 0)}, momentums_in_node={"M1": (0, 0, 0)},
                     support=st.Support.PINNED)
        n2 = st.Node("N2", position=(0, 4.2, 0), forces_in_node={"F1": (0, -35020.05, 0)},
                     momentums_in_node={"M1": (0, 0, -40159.83)})
        n3 = st.Node("N3", position=(6.8, 5.25, 0), forces_in_node={"F1": (0, -70040.1, 0)},
                     momentums_in_node={"M1": (0, 0, 0)})
        n4 = st.Node("N4", position=(13.6, 4.2, 0), forces_in_node={"F1": (-12500, -53560.23, 0)},
                     momentums_in_node={"M1": (0, 0, 15778.78)})
        n5 = st.Node("N5", position=(17.2, 3.644117647, 0), forces_in_node={"F1": (0, -18540.18, 0)},
                     momentums_in_node={"M1": (0, 0, 11256.05)})
        n6 = st.Node("N6", position=(13.6, 0, 0), forces_in_node={"F1": (-12500, 0, 0)},
                     momentums_in_node={"M1": (0, 0, 13125)}, support=st.Support.FIXED)

        b1 = st.Bar("B1", n1, n2)
        b2 = st.Bar("B2", n2, n3)
        b3 = st.Bar("B3", n3, n4)
        b4 = st.Bar("B4", n4, n5)
        b5 = st.Bar("B5", n4, n6)

        bars = {
            b1.name: b1,
            b2.name: b2,
            b3.name: b3,
            b4.name: b4,
            b5.name: b5
        }

        structure = st.Structure("S1", bars)

        calculated_indexes = structure._get_zero_displacement_indexes()
        expected_indexes = [0, 1, 15, 16, 17]

        self.assertEqual(calculated_indexes, expected_indexes)

    def test_decoupled_forces_and_momentums_in_structure(self):
        n1 = st.Node("N1", position=(0, 0, 0), support=st.Support.PINNED)
        n2 = st.Node("N2", position=(0, 4.2, 0))
        n3 = st.Node("N3", position=(6.8, 5.25, 0))
        n4 = st.Node("N4", position=(13.6, 4.2, 0))
        n5 = st.Node("N5", position=(17.2, 3.644117647, 0))
        n6 = st.Node("N6", position=(13.6, 0, 0), support=st.Support.FIXED)

        b1 = st.Bar("B1", n1, n2)
        b2 = st.Bar("B2", n2, n3)
        b3 = st.Bar("B3", n3, n4)
        b4 = st.Bar("B4", n4, n5)
        b5 = st.Bar("B5", n4, n6)

        dc = st.DistributedCharge(st.DistributedChargeType.SQUARE, 10179.36, (0, -1, 0))
        b2.add_distributed_charge(dc)
        b3.add_distributed_charge(dc)
        b4.add_distributed_charge(dc)

        pf = st.PunctualForceInBar(-25000, 0.5, (0, 1, 0))
        b5.add_punctual_force(pf, "pf")

        bars = {
            b1.name: b1,
            b2.name: b2,
            b3.name: b3,
            b4.name: b4,
            b5.name: b5
        }

        structure = st.Structure("S1", bars)

        calculated_forces = structure.decoupled_forces_and_momentums_in_structure()
        expected_forces = np.array([0,
                                    0, -35020.05, -40159.83,
                                    0, -70040.1, 0,
                                    -12500, -53560.23, 15778.78,
                                    0, -18540.18, 11256.05])

        np.testing.assert_allclose(calculated_forces, expected_forces, atol=0.3)

    def test_nodes_displacements(self):
        # Test structure 1
        n1 = st.Node("N1", position=(0, 0, 0), support=st.Support.PINNED)
        n2 = st.Node("N2", position=(0, 4.2, 0))
        n3 = st.Node("N3", position=(6.8, 5.25, 0))
        n4 = st.Node("N4", position=(13.6, 4.2, 0))
        n5 = st.Node("N5", position=(17.2, 3.644117647, 0))
        n6 = st.Node("N6", position=(13.6, 0, 0), support=st.Support.FIXED)

        b1 = st.Bar("B1", n1, n2)
        b2 = st.Bar("B2", n2, n3)
        b3 = st.Bar("B3", n3, n4)
        b4 = st.Bar("B4", n4, n5)
        b5 = st.Bar("B5", n4, n6)

        dc = st.DistributedCharge(st.DistributedChargeType.SQUARE, 10179.36, (0, -1, 0))
        b2.add_distributed_charge(dc)
        b3.add_distributed_charge(dc)
        b4.add_distributed_charge(dc)

        pf = st.PunctualForceInBar(-25000, 0.5, (0, 1, 0))
        b5.add_punctual_force(pf, "pf")

        bars = {
            b1.name: b1,
            b2.name: b2,
            b3.name: b3,
            b4.name: b4,
            b5.name: b5
        }

        structure = st.Structure("S1", bars)

        calculated = structure.get_nodes_displacements()
        expected = np.array([0,
                             0,
                             0.01032450,
                             -0.02154358,
                             -0.00026348,
                             -0.00526072,
                             -0.01203257,
                             -0.06329572,
                             0.00023138,
                             -0.00254491,
                             -0.00040809,
                             0.00427134,
                             -0.00211703,
                             0.00230199,
                             -0.00041022,
                             0,
                             0,
                             0])

        np.testing.assert_allclose(calculated, expected, atol=10**-6)

        # Test structure 2
        n21 = st.Node("N21", position=(-1, 2, 0), support=st.Support.PINNED)
        n22 = st.Node("N22", position=(-1, 5, 0))
        n23 = st.Node("N23", position=(1, 5, 0))
        n24 = st.Node("N24", position=(1, 4, 0))
        n25 = st.Node("N25", position=(3, 4, 0))
        n26 = st.Node("N26", position=(3, 5, 0))
        n27 = st.Node("N27", position=(5, 5, 0))
        n28 = st.Node("N28", position=(5, 2, 0), support=st.Support.FIXED)

        b21 = st.Bar("B21", n21, n22)
        b22 = st.Bar("B22", n22, n23)
        b23 = st.Bar("B23", n23, n24)
        b24 = st.Bar("B24", n24, n25)
        b25 = st.Bar("B25", n25, n26)
        b26 = st.Bar("B26", n26, n27)
        b27 = st.Bar("B27", n27, n28)

        dc2 = st.DistributedCharge(st.DistributedChargeType.SQUARE, 100000, (0, -1, 0))
        b22.add_distributed_charge(dc2)
        b26.add_distributed_charge(dc2)

        pf21 = st.PunctualForceInBar(-40000, 0.5, (0, 1, 0))
        pf24 = st.PunctualForceInBar(-25000, 0.5, (0, 1, 0))
        pf27 = st.PunctualForceInBar(-30000, 0.5, (0, 1, 0))

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

        structure2 = st.Structure("st2", bars2)

        calculated = structure2.get_nodes_displacements()
        # These results are the exact same than Cespla provides
        expected = np.array([0,
                             0,
                             -0.00107992,
                             0.0131575,
                             -0.000572115,
                             -0.0123052,
                             0.0130259,
                             -0.031139,
                             -0.0139501,
                             0.00163605,
                             -0.0311492,
                             -0.00812327,
                             0.00150441,
                             -0.0308714,
                             0.00835424,
                             -0.0100462,
                             -0.030859,
                             0.0140407,
                             -0.0101779,
                             -0.000578654,
                             0.0118345,
                             0,
                             0,
                             0])

        np.testing.assert_allclose(calculated, expected, atol=5**-8)

    def test_get_nodes(self):
        n1 = st.Node("N1", position=(0, 0, 0), forces_in_node={}, momentums_in_node={"M1": (0, 0, 0)},
                     support=st.Support.PINNED)
        n2 = st.Node("N2", position=(0, 4.2, 0), forces_in_node={"F1": (0, -35020.05, 0)},
                     momentums_in_node={"M1": (0, 0, -40159.83)})
        n3 = st.Node("N3", position=(6.8, 5.25, 0), forces_in_node={"F1": (0, -70040.1, 0)},
                     momentums_in_node={"M1": (0, 0, 0)})
        n4 = st.Node("N4", position=(13.6, 4.2, 0), forces_in_node={"F1": (-12500, -53560.23, 0)},
                     momentums_in_node={"M1": (0, 0, 15778.78)})
        n5 = st.Node("N5", position=(17.2, 3.644117647, 0), forces_in_node={"F1": (0, -18540.18, 0)},
                     momentums_in_node={"M1": (0, 0, 11256.05)})
        n6 = st.Node("N6", position=(13.6, 0, 0), forces_in_node={"F1": (-12500, 0, 0)},
                     momentums_in_node={"M1": (0, 0, 13125)}, support=st.Support.FIXED)

        b1 = st.Bar("B1", n1, n2)
        b2 = st.Bar("B2", n2, n3)
        b3 = st.Bar("B3", n3, n4)
        b4 = st.Bar("B4", n4, n5)
        b5 = st.Bar("B5", n4, n6)

        bars = {
            b1.name: b1,
            b2.name: b2,
            b3.name: b3,
            b4.name: b4,
            b5.name: b5
        }

        structure = st.Structure("S1", bars)

        self.assertEqual(len(structure.get_nodes()), 6)
        self.assertTrue(structure.get_nodes()[0].equals(n1))
        self.assertTrue(structure.get_nodes()[1].equals(n2))
        self.assertTrue(structure.get_nodes()[2].equals(n3))
        self.assertTrue(structure.get_nodes()[3].equals(n4))
        self.assertTrue(structure.get_nodes()[4].equals(n5))
        self.assertTrue(structure.get_nodes()[5].equals(n6))

    def test_get_nodes_reactions(self):
        # Test structure 1
        n1 = st.Node("N1", position=(0, 0, 0), support=st.Support.PINNED)
        n2 = st.Node("N2", position=(0, 4.2, 0))
        n3 = st.Node("N3", position=(6.8, 5.25, 0))
        n4 = st.Node("N4", position=(13.6, 4.2, 0))
        n5 = st.Node("N5", position=(17.2, 3.644117647, 0))
        n6 = st.Node("N6", position=(13.6, 0, 0), support=st.Support.FIXED)

        b1 = st.Bar("B1", n1, n2)
        b2 = st.Bar("B2", n2, n3)
        b3 = st.Bar("B3", n3, n4)
        b4 = st.Bar("B4", n4, n5)
        b5 = st.Bar("B5", n4, n6)

        dc = st.DistributedCharge(st.DistributedChargeType.SQUARE, 10179.36, (0, -1, 0))
        b2.add_distributed_charge(dc)
        b3.add_distributed_charge(dc)
        b4.add_distributed_charge(dc)

        pf = st.PunctualForceInBar(-25000, 0.5, (0, 1, 0))
        b5.add_punctual_force(pf, "pf")

        bars = {
            b1.name: b1,
            b2.name: b2,
            b3.name: b3,
            b4.name: b4,
            b5.name: b5
        }

        structure = st.Structure("S1", bars)
        calculated = structure.get_nodes_reactions()
        expected = np.array([30408,
                             69506,
                             0,
                             0,
                             -35020.05,
                             -40159.83,
                             0,
                             -70040.1,
                             0,
                             -12500,
                             -53560.23,
                             15778.78,
                             0,
                             -18540.18,
                             11256.05,
                             -5407.6,
                             107650,
                             6980.4])

        np.testing.assert_allclose(calculated, expected, atol=5)

        # Test structure 2
        n21 = st.Node("N21", position=(-1, 2, 0), support=st.Support.PINNED)
        n22 = st.Node("N22", position=(-1, 5, 0))
        n23 = st.Node("N23", position=(1, 5, 0))
        n24 = st.Node("N24", position=(1, 4, 0))
        n25 = st.Node("N25", position=(3, 4, 0))
        n26 = st.Node("N26", position=(3, 5, 0))
        n27 = st.Node("N27", position=(5, 5, 0))
        n28 = st.Node("N28", position=(5, 2, 0), support=st.Support.FIXED)

        b21 = st.Bar("B21", n21, n22)
        b22 = st.Bar("B22", n22, n23)
        b23 = st.Bar("B23", n23, n24)
        b24 = st.Bar("B24", n24, n25)
        b25 = st.Bar("B25", n25, n26)
        b26 = st.Bar("B26", n26, n27)
        b27 = st.Bar("B27", n27, n28)

        dc2 = st.DistributedCharge(st.DistributedChargeType.SQUARE, 100000, (0, -1, 0))
        b22.add_distributed_charge(dc2)
        b26.add_distributed_charge(dc2)

        pf21 = st.PunctualForceInBar(-40000, 0.5, (0, 1, 0))
        pf24 = st.PunctualForceInBar(-25000, 0.5, (0, 1, 0))
        pf27 = st.PunctualForceInBar(-30000, 0.5, (0, 1, 0))

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

        structure2 = st.Structure("st2", bars2)

        calculated = structure2.get_nodes_reactions()
        # These results are the exact same than Cespla provides
        expected = np.array([32926.2,
                             211293,
                             0,
                             20000,
                             -100000,
                             -18333.33,
                             0,
                             -100000,
                             33333.33,
                             0,
                             -12500,
                             -6250,
                             0,
                             -12500,
                             6250,
                             0,
                             -100000,
                             -33333.33,
                             -15000,
                             -100000,
                             22083.33,
                             -42926.2,
                             213707,
                             7755.35])

        np.testing.assert_allclose(calculated, expected, atol=.5)


class TestMaterial(unittest.TestCase):
    def test_constructor(self):
        mat = st.Material("s275j")

        self.assertEqual(mat.generic_name, "steel")
        self.assertEqual(mat.name, "s275j")
        self.assertEqual(mat.young_mod, 205939650000)
        self.assertEqual(mat.rig_mod, 81000000000)
        self.assertEqual(mat.poisson_coef, 0.3)
        self.assertEqual(mat.thermal_dil_coef, 0.000012)
        self.assertEqual(mat.density, 7.85)

        self.assertRaises(TypeError, st.Material, 4)
        self.assertRaises(LookupError, st.Material, "foobar")


class TestProfile(unittest.TestCase):
    def test_constructor(self):
        pro = st.Profile("IPE", 80)

        self.assertEqual(pro.name, "IPE")
        self.assertEqual(pro.name_number, 80)
        self.assertEqual(pro.area, 0.000764)
        self.assertEqual(pro.weight, 6)
        self.assertEqual(pro.inertia_moment_x, 0.000000801)
        self.assertEqual(pro.res_mod_x, 0.000020)
        self.assertEqual(pro.inertia_moment_y, 0.0000000849)
        self.assertEqual(pro.res_mod_y, 0.00000369)

        self.assertRaises(TypeError, st.Profile, 4, 3)
        self.assertRaises(LookupError, st.Profile, "h", "foobar")


class TestDistributedCharge(unittest.TestCase):
    def test_constructor(self):
        dc = st.DistributedCharge(st.DistributedChargeType.SQUARE, 10, (0, -1, 0))

        self.assertEqual(dc.dc_type, st.DistributedChargeType.SQUARE)
        self.assertEqual(dc.max_value, 10)

        self.assertRaises(TypeError, st.DistributedCharge, 2, 20)

    def test_equals(self):
        # TODO añadir más tests cuando se tengan distintos tipos de carga distribuida
        dc1 = st.DistributedCharge(st.DistributedChargeType.SQUARE, 10, (0, -1, 0))
        dc2 = st.DistributedCharge(st.DistributedChargeType.SQUARE, 20, (0, -1, 0))

        self.assertRaises(TypeError, dc1.equals, 3)
        self.assertTrue(dc1.equals(dc1))
        self.assertFalse(dc1.equals(dc2))


class TestPunctualForce(unittest.TestCase):
    def test_constructor(self):
        pf = st.PunctualForceInBar(10, 0.5, (1, 0, 0))

        self.assertEqual(pf.value, 10)
        self.assertEqual(pf.origin_end_factor, 0.5)
        self.assertEqual(pf.direction, (1, 0, 0))

        self.assertRaises(ValueError, st.PunctualForceInBar, 10, -3, (1, 0, 0))
        self.assertRaises(ValueError, st.PunctualForceInBar, 10, 3, (1, 0, 0))
        self.assertRaises(TypeError, st.PunctualForceInBar, 10, .3, [1, 0, 0])


    def test_equals(self):
        # TODO añadir más tests cuando se tengan distintos tipos de carga distribuida
        pf1 = st.PunctualForceInBar(10, 0.5, (1, 0, 0))
        pf2 = st.PunctualForceInBar(9, 0.5, (1, 0, 0))
        pf3 = st.PunctualForceInBar(10, 0.3, (1, 0, 0))
        pf4 = st.PunctualForceInBar(10, 0.5, (2, 0, 0))

        self.assertRaises(TypeError, pf1.equals, 3)
        self.assertTrue(pf1.equals(pf1))
        self.assertFalse(pf1.equals(pf2))
        self.assertFalse(pf1.equals(pf3))
        self.assertFalse(pf1.equals(pf4))



if __name__ == '__main__':
    unittest.main()
