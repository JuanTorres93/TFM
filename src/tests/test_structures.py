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

        n1.add_force("F2", (3, 2, 1), belongs_to_node=False)

        # expected_result = np.array([Fx, Fy, Mz])
        expected_result = np.array([4, 4, 3])
        np.testing.assert_array_equal(expected_result, n1.get_total_force_and_momentum())

        n1.add_momentum("M2", (3, 3, 1), belongs_to_node=False)
        expected_result = np.array([4, 4, 4])
        np.testing.assert_array_equal(expected_result, n1.get_total_force_and_momentum())

        n1.add_force("F3", (1, 2, 0), belongs_to_node=True)
        expected_result = np.array([5, 6, 4])
        np.testing.assert_array_equal(expected_result, n1.get_total_force_and_momentum())

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

        bar = st.Bar("B1", n_ori, n_end, "s275j", ("IPE", 300))

        self.assertEqual(bar.name, "B1")
        self.assertEqual(bar.origin, n_ori)
        self.assertEqual(bar.end, n_end)
        self.assertIsInstance(bar.material, st.Material)
        self.assertRaises(TypeError, st.Bar, "name", n_ori, (0, 0, 0))
        self.assertRaises(TypeError, st.Bar, "name", (0, 0, 0), n_end)
        self.assertRaises(TypeError, st.Bar, 3, n_ori, n_end)
        self.assertRaises(ValueError, st.Bar, "name", n_ori, n_ori, "s275j", ("IPE", 300))
        self.assertRaises(TypeError, st.Bar, "name", n_ori, n_end, material=2)

    def test_set_name(self):
        n_ori = st.Node("N1")
        n_end = st.Node("N2")

        bar = st.Bar("B1", n_ori, n_end, "s275j", ("IPE", 300))
        bar.set_name("New name")

        self.assertEqual("New name", bar.name)

        self.assertRaises(TypeError, bar.set_name, 0)

    def test_set_origin(self):
        n_ori = st.Node("N1")
        n_end = st.Node("N2")

        bar = st.Bar("B1", n_ori, n_end, "s275j", ("IPE", 300))
        self.assertEqual(bar.origin, n_ori)

        new_origin = st.Node("N3", position=(1, 2, 3))
        bar.set_origin(new_origin)

        self.assertEqual(bar.origin, new_origin)

        self.assertRaises(TypeError, bar.set_origin, 0)

    def test_set_end(self):
        n_ori = st.Node("N1")
        n_end = st.Node("N2")

        bar = st.Bar("B1", n_ori, n_end, "s275j", ("IPE", 300))
        self.assertEqual(bar.end, n_end)

        new_end = st.Node("N3", (1, 2, 3))
        bar.set_end(new_end)

        self.assertEqual(bar.end, new_end)
        self.assertRaises(TypeError, bar.set_end, 0)
        self.assertRaises(TypeError, bar.set_end, (0, 0, 0))

    def test_swap(self):
        n_ori = st.Node("N1")
        n_end = st.Node("N2")

        bar = st.Bar("B1", n_ori, n_end, "s275j", ("IPE", 300))
        self.assertEqual(bar.end, n_end)
        self.assertEqual(bar.origin, n_ori)

        bar.swap_nodes()
        self.assertEqual(bar.end, n_ori)
        self.assertEqual(bar.origin, n_end)

    def test_set_material(self):
        n1 = st.Node("N1")
        n2 = st.Node("N2", position=(1, 2, 3))

        bar = st.Bar("B1", n1, n2, "s275j", ("IPE", 300))
        self.assertTrue(bar.get_material().equals(st.Material("s275j")))

        bar.set_material("s280")
        self.assertTrue(bar.get_material().equals(st.Material("s280")))
        self.assertFalse(bar.get_material().equals(st.Material("s275j")))

    def test_set_profile(self):
        n1 = st.Node("N1")
        n2 = st.Node("N2", position=(1, 2, 3))

        bar = st.Bar("B1", n1, n2, "s275j", ("IPE", 300))
        self.assertTrue(bar.get_profile().equals(st.Profile("IPE", 300)))

        bar.set_profile("IPE", 80)
        self.assertTrue(bar.get_profile().equals(st.Profile("IPE", 80)))
        self.assertFalse(bar.get_profile().equals(st.Profile("IPE", 300)))

    def test_length(self):
        n_ori = st.Node("N1")
        n_end = st.Node("N2", position=(1, 2, 3))

        bar = st.Bar("B1", n_ori, n_end, "s275j", ("IPE", 300))

        self.assertAlmostEqual(bar.length(), 3.741657387)

    def test_local_rigidity_matrix_2d_rigid_nodes(self):
        n_ori = st.Node("N1")
        n_end = st.Node("N2", position=(0, 4.2, 0))

        bar = st.Bar("B1", n_ori, n_end, "s275j", ("IPE", 300))

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

        bar = st.Bar("B1", n_ori, n_end, "s275j", ("IPE", 300))

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

        b1 = st.Bar("B1", n1, n2, "s275j", ("IPE", 300))
        b2 = st.Bar("B2", n2, n3, "s275j", ("IPE", 300))
        b3 = st.Bar("B3", n3, n4, "s275j", ("IPE", 300))
        b4 = st.Bar("B4", n4, n5, "s275j", ("IPE", 300))
        b5 = st.Bar("B5", n4, n6, "s275j", ("IPE", 300))

        b1 = st.Bar("B1", n1, n2, "s275j", ("IPE", 300))

        np.testing.assert_equal(b1.angle_from_global_to_local(), np.pi / 2)
        np.testing.assert_almost_equal(b2.angle_from_global_to_local(), math.radians(8.778), 3)
        np.testing.assert_almost_equal(b3.angle_from_global_to_local(), math.radians(351.222), 3)
        # Differs in 0.03 radians due to trigonometry used to calculate node position
        np.testing.assert_almost_equal(b4.angle_from_global_to_local(), math.radians(351.222), 0)
        np.testing.assert_equal(b5.angle_from_global_to_local(), math.radians(270))

    def test_global_rigidity_matrix_2d_rigid_nodes(self):
        n_ori = st.Node("N1")
        n_end = st.Node("N2", position=(0, 4.2, 0))

        bar = st.Bar("B1", n_ori, n_end, "s275j", ("IPE", 300))

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

        b2 = st.Bar("B2", n2, n3, "s275j", ("IPE", 300))

        self.assertRaises(TypeError, b2.add_distributed_charge, 3)

        new_distributed_charge = st.DistributedCharge(st.DistributedChargeType.SQUARE, max_value=7,
                                                      direction=(0, -1, 0))

        dc_name = "dc"

        b2.add_distributed_charge(new_distributed_charge, dc_name)
        are_equals = new_distributed_charge.equals(b2.get_distributed_charges().get(dc_name))
        self.assertTrue(are_equals)

    def test_get_referred_distributed_charge_to_nodes(self):
        structure = get_test_structure(1)

        b2 = structure.get_bars().get("B2")

        calculated_values = b2.get_referred_distributed_charge_to_nodes(return_global_values=False).get("0")
        expected_values = {
            "x": 5232.97,
            "y": 31995.89,
            "m_origin": 34690.47,
            "m_end": - 34690.47
        }

        np.testing.assert_allclose(calculated_values.get("x"), expected_values.get("x"), rtol=0.02)
        np.testing.assert_allclose(calculated_values.get("y"), expected_values.get("y"), rtol=0.02)
        np.testing.assert_allclose(calculated_values.get("m_origin"), expected_values.get("m_origin"), rtol=0.02)
        np.testing.assert_allclose(calculated_values.get("m_end"), expected_values.get("m_end"), rtol=0.02)

    def test_get_referred_punctual_force_to_nodes(self):
        n1 = st.Node("N1", position=(0, 0, 0), support=st.Support.PINNED)
        n2 = st.Node("N2", position=(0, 4.2, 0))
        n3 = st.Node("N3", position=(6.8, 5.25, 0))
        n4 = st.Node("N4", position=(13.6, 4.2, 0))
        n5 = st.Node("N5", position=(17.2, 3.644117647, 0))
        n6 = st.Node("N6", position=(13.6, 0, 0), support=st.Support.FIXED)

        b1 = st.Bar("B1", n1, n2, "s275j", ("IPE", 300))
        b2 = st.Bar("B2", n2, n3, "s275j", ("IPE", 300))
        b3 = st.Bar("B3", n3, n4, "s275j", ("IPE", 300))
        b4 = st.Bar("B4", n4, n5, "s275j", ("IPE", 300))
        b5 = st.Bar("B5", n4, n6, "s275j", ("IPE", 300))

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
        calculated_values = b5.get_referred_punctual_forces_in_bar_to_nodes(return_global_values=False).get("0")
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
        calculated_values = b5.get_referred_punctual_forces_in_bar_to_nodes(return_global_values=True).get("0")
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

        b2 = st.Bar("B2", n2, n3, "s275j", ("IPE", 300))

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

        b2 = st.Bar("B2", n2, n3, "s275j", ("IPE", 300))

        self.assertRaises(TypeError, b2.add_punctual_force, 3)

        new_punctual_force = st.PunctualForceInBar(459, 0.8, (-1, 0, 0))

        pf_name = "pf"

        b2.add_punctual_force(new_punctual_force, pf_name)
        are_equals = new_punctual_force.equals(b2.get_punctual_forces().get(pf_name))
        self.assertTrue(are_equals)

    def test_get_referred_punctual_forces_to_nodes(self):
        n4 = st.Node("N4", position=(13.6, 4.2, 0))
        n6 = st.Node("N6", position=(13.6, 0, 0))

        b5 = st.Bar("B5", n4, n6, "s275j", ("IPE", 300))

        pf = st.PunctualForceInBar(-25000, 0.5, (0, 1, 0))
        b5.add_punctual_force(pf, "pf")

        calculated_values = b5.get_referred_punctual_forces_in_bar_to_nodes(False).get("0")
        expected_values = {
            "y_origin": 12500,
            "y_end": 12500,
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

        b1 = st.Bar("B1", n1, n2, "s275j", ("IPE", 300))
        self.assertFalse(b1.has_distributed_charges())

        dc = st.DistributedCharge(st.DistributedChargeType.SQUARE, -10179.36, (0, -1, 0))
        b1.add_distributed_charge(dc)
        self.assertTrue(b1.has_distributed_charges())

    def test_has_punctual_force(self):
        n4 = st.Node("N4", position=(13.6, 4.2, 0))
        n6 = st.Node("N6", position=(13.6, 0, 0))

        b5 = st.Bar("B5", n4, n6, "s275j", ("IPE", 300))
        self.assertFalse(b5.has_punctual_forces())

        pf = st.PunctualForceInBar(-25000, 0.5, (0, 1, 0))
        b5.add_punctual_force(pf, "pf")
        self.assertTrue(b5.has_punctual_forces())


def get_test_structure(num_test_st):
    if num_test_st == 1:
        # Structure solved in the fifth individual work
        n1 = st.Node("N1", position=(0, 0, 0), support=st.Support.PINNED)
        n2 = st.Node("N2", position=(0, 4.2, 0))
        n3 = st.Node("N3", position=(6.42, 5.25, 0))
        n4 = st.Node("N4", position=(12.84, 4.2, 0))
        n5 = st.Node("N5", position=(15.68, 3.735514019, 0))
        n6 = st.Node("N6", position=(12.84, 0, 0), support=st.Support.FIXED)

        b1 = st.Bar("B1", n1, n2, "s275j", ("IPE", 300))
        b2 = st.Bar("B2", n2, n3, "s275j", ("IPE", 300))
        b3 = st.Bar("B3", n3, n4, "s275j", ("IPE", 300))
        b4 = st.Bar("B4", n4, n5, "s275j", ("IPE", 300))
        b5 = st.Bar("B5", n4, n6, "s275j", ("IPE", 300))

        b2.add_distributed_charge(st.DistributedCharge(
            st.DistributedChargeType.PARALLEL_TO_BAR, -9967.568, (0, 1, 0)
        ))
        b3.add_distributed_charge(st.DistributedCharge(
            st.DistributedChargeType.PARALLEL_TO_BAR, 9967.568, (0, -1, 0)
        ))
        b4.add_distributed_charge(st.DistributedCharge(
            st.DistributedChargeType.PARALLEL_TO_BAR, 9967.568, (0, -1, 0)
        ))

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

    if num_test_st == 1.1:
        # Structure solved in the fifth individual work
        n1 = st.Node("N1", position=(0, 0, 0), support=st.Support.PINNED)
        n2 = st.Node("N2", position=(0, 4.2, 0))
        n3 = st.Node("N3", position=(6.8, 5.25, 0))
        n4 = st.Node("N4", position=(13.6, 4.2, 0))
        n5 = st.Node("N5", position=(17.2, 3.644117647, 0))
        n6 = st.Node("N6", position=(13.6, 0, 0), support=st.Support.FIXED)

        b1 = st.Bar("B1", n1, n2, "s275j", ("IPE", 300))
        b2 = st.Bar("B2", n2, n3, "s275j", ("IPE", 300))
        b3 = st.Bar("B3", n3, n4, "s275j", ("IPE", 300))
        b4 = st.Bar("B4", n4, n5, "s275j", ("IPE", 300))
        b5 = st.Bar("B5", n4, n6, "s275j", ("IPE", 300))

        bars = {
            b1.name: b1,
            b2.name: b2,
            b3.name: b3,
            b4.name: b4,
            b5.name: b5
        }

        structure = st.Structure("S1", bars)

    elif num_test_st == 2:
        # Invented structure
        n1 = st.Node("N1", position=(-1, 2, 0), support=st.Support.PINNED)
        n2 = st.Node("N2", position=(-1, 5, 0))
        n3 = st.Node("N3", position=(1, 5, 0))
        n4 = st.Node("N4", position=(1, 4, 0))
        n5 = st.Node("N5", position=(3, 4, 0))
        n6 = st.Node("N6", position=(3, 5, 0))
        n7 = st.Node("N7", position=(5, 5, 0))
        n8 = st.Node("N8", position=(5, 2, 0), support=st.Support.FIXED)

        b1 = st.Bar("B1", n1, n2, "s275j", ("IPE", 300))
        b2 = st.Bar("B2", n2, n3, "s275j", ("IPE", 300))
        b3 = st.Bar("B3", n3, n4, "s275j", ("IPE", 300))
        b4 = st.Bar("B4", n4, n5, "s275j", ("IPE", 300))
        b5 = st.Bar("B5", n5, n6, "s275j", ("IPE", 300))
        b6 = st.Bar("B6", n6, n7, "s275j", ("IPE", 300))
        b7 = st.Bar("B7", n7, n8, "s275j", ("IPE", 300))

        dc2 = st.DistributedCharge(st.DistributedChargeType.SQUARE, 100000, (0, -1, 0))
        b2.add_distributed_charge(dc2)
        b6.add_distributed_charge(dc2)

        pf1 = st.PunctualForceInBar(-40000, 0.5, (0, 1, 0))
        pf4 = st.PunctualForceInBar(-25000, 0.5, (0, 1, 0))
        pf7 = st.PunctualForceInBar(-30000, 0.5, (0, 1, 0))

        b1.add_punctual_force(pf1)
        b4.add_punctual_force(pf4)
        b7.add_punctual_force(pf7)

        bars2 = {
            b1.name: b1,
            b2.name: b2,
            b3.name: b3,
            b4.name: b4,
            b5.name: b5,
            b6.name: b6,
            b7.name: b7,
        }

        structure = st.Structure("st2", bars2)

    elif num_test_st == 3:
        # Horizontal bar with two applied charges to it
        n1 = st.Node("N1", position=(0, 0, 0), support=st.Support.PINNED)
        n2 = st.Node("N2", position=(2, 0, 0), support=st.Support.PINNED)

        b1 = st.Bar("B1", n1, n2, "s275j", ("IPE", 300))

        dc1 = st.DistributedCharge(st.DistributedChargeType.SQUARE, 100000, (0, -1, 0))
        dc2 = st.DistributedCharge(st.DistributedChargeType.SQUARE, 200000, (0, -1, 0))

        b1.add_distributed_charge(dc1)
        b1.add_distributed_charge(dc2)

        bars = {
            b1.name: b1
        }

        structure = st.Structure("S1", bars)

    elif num_test_st == 4:
        # Horizontal bar with two applied forces to it
        n1 = st.Node("N1", position=(0, 0, 0), support=st.Support.PINNED)
        n2 = st.Node("N2", position=(3, 0, 0), support=st.Support.PINNED)

        b1 = st.Bar("B1", n1, n2, "s275j", ("IPE", 300))

        pf1 = st.PunctualForceInBar(100000, 1 / b1.length(), (0, -1, 0))
        pf2 = st.PunctualForceInBar(200000, 2 / b1.length(), (0, -1, 0))

        b1.add_punctual_force(pf1)
        b1.add_punctual_force(pf2)

        bars = {
            b1.name: b1
        }

        structure = st.Structure("S1", bars)

    elif num_test_st == 5:
        # Horizontal bar with one distributed charge and one punctual force
        n1 = st.Node("N1", position=(0, 0, 0), support=st.Support.PINNED)
        n2 = st.Node("N2", position=(10, 0, 0), support=st.Support.FIXED)

        b1 = st.Bar("B1", n1, n2, "s275j", ("IPE", 300))

        pf1 = st.PunctualForceInBar(200000, 0.7, (0, 1, 0))
        dc1 = st.DistributedCharge(st.DistributedChargeType.SQUARE, 150000, (0, -1, 0))

        b1.add_punctual_force(pf1)
        b1.add_distributed_charge(dc1)

        bars = {
            b1.name: b1
        }

        structure = st.Structure("S1", bars)

    elif num_test_st == 6:
        # Horizontal bar with one inclined punctual force
        n1 = st.Node("N1", position=(0, 0, 0), support=st.Support.FIXED)
        n2 = st.Node("N2", position=(2, 0, 0), support=st.Support.PINNED)

        b1 = st.Bar("B1", n1, n2, "s275j", ("IPE", 300))

        angle = math.radians(30)
        pf1 = st.PunctualForceInBar(100000, 0.5, (- math.cos(angle), - math.sin(angle), 0))

        b1.add_punctual_force(pf1)

        bars = {
            b1.name: b1
        }

        structure = st.Structure("S1", bars)

    elif num_test_st == 7:
        # Horizontal bar with one punctual force and roller_x support
        n1 = st.Node("N1", position=(0, 0, 0), support=st.Support.PINNED)
        n2 = st.Node("N2", position=(2, 0, 0), support=st.Support.ROLLER_X)

        b1 = st.Bar("B1", n1, n2, "s275j", ("IPE", 300))

        pf1 = st.PunctualForceInBar(100000, 0.5, (0, -1, 0))

        b1.add_punctual_force(pf1)

        bars = {
            b1.name: b1
        }

        structure = st.Structure("S1", bars)

    elif num_test_st == 8:
        # Horizontal bar with one punctual force and roller_x support
        n1 = st.Node("N1", position=(0, 0, 0), support=st.Support.FIXED)
        n2 = st.Node("N2", position=(2, 0, 0), support=st.Support.ROLLER_Y)

        b1 = st.Bar("B1", n1, n2, "s275j", ("IPE", 300))

        pf1 = st.PunctualForceInBar(100000, 0.5, (0, -1, 0))

        b1.add_punctual_force(pf1)

        bars = {
            b1.name: b1
        }

        structure = st.Structure("S1", bars)

    elif num_test_st == 9:
        # 3-bar structure with perpendicular distributed charge in bars 1 and 3 and a punctual force in bar 2
        n1 = st.Node("N1", position=(0, 0, 0), support=st.Support.FIXED)
        n2 = st.Node("N2", position=(2, 3, 0))
        n3 = st.Node("N3", position=(5, 3, 0))
        n4 = st.Node("N4", position=(6, 0, 0), support=st.Support.PINNED)

        b1 = st.Bar("B1", n1, n2, "s235j", ("IPE", 80))
        b2 = st.Bar("B2", n2, n3, "s235j", ("IPE", 80))
        b3 = st.Bar("B3", n3, n4, "s235j", ("IPE", 80))

        b1.add_distributed_charge(st.DistributedCharge(
            st.DistributedChargeType.SQUARE, 50000, (0, -1, 0))
        )

        b2.add_punctual_force(st.PunctualForceInBar(
            120000, 0.33333333333, (0, 1, 0))
        )

        b3.add_distributed_charge(st.DistributedCharge(
            st.DistributedChargeType.SQUARE, 30000, (0, 1, 0))
        )

        bars = {
            b1.name: b1,
            b2.name: b2,
            b3.name: b3
        }

        structure = st.Structure("S1", bars)

    elif num_test_st == 10:
        # Inclined bar with perpendicular distributed charge
        n1 = st.Node("N1", position=(0, 0, 0), support=st.Support.FIXED)
        n2 = st.Node("N2", position=(2, 3, 0), support=st.Support.PINNED)

        b1 = st.Bar("B1", n1, n2, "s275j", ("IPE", 300))

        b1.add_distributed_charge(st.DistributedCharge(
            st.DistributedChargeType.SQUARE, 50000, (0, -1, 0)
        ))

        bars = {
            b1.name: b1
        }

        structure = st.Structure("S1", bars)

    return structure


class TestStructure(unittest.TestCase):

    def test_constructor(self):
        n1 = st.Node("N1")
        n2 = st.Node("N2", position=(0, 2, 3))
        n3 = st.Node("N3", position=(1, 2, 3))

        bar_1 = st.Bar("B1", n1, n2, "s275j", ("IPE", 300))
        bar_2 = st.Bar("B2", n2, n3, "s275j", ("IPE", 300))

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

        bar_3 = st.Bar("B1", n2, n3, "s275j", ("IPE", 300))
        bar_dict = {
            'b1': bar_1,
            'b2': bar_3
        }

        self.assertRaises(ValueError, st.Structure, "name", bar_dict)

    def test_assembled_matrix(self):
        structure = get_test_structure(1.1)

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
        structure = get_test_structure(1.1)

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
        structure = get_test_structure(1)

        self.assertEqual(structure.get_number_of_nodes(), 6)

    def test_forces_and_momentums_in_structure(self):
        # Test structure 1
        structure = get_test_structure(1)

        calculated_forces = structure.forces_and_momentums_in_structure()
        expected_forces = np.array([0, 0, 0,
                                    0, -32421, -34690.47,
                                    0, -64842, 0,
                                    -12500, -46763, 14776.92,
                                    0, -14342, 6788.55,
                                    -12500, 0, 13125])

        np.testing.assert_allclose(calculated_forces, expected_forces, rtol=0.02, atol=821)

        # This is done to assure that referred forces and momentums are not stacked
        calculated_forces = structure.forces_and_momentums_in_structure()
        calculated_forces = structure.forces_and_momentums_in_structure()
        calculated_forces = structure.forces_and_momentums_in_structure()
        np.testing.assert_allclose(calculated_forces, expected_forces, rtol=0.02, atol=821)

    def test_get_indexes_to_delete(self):
        structure = get_test_structure(1)

        calculated_indexes = structure._get_zero_displacement_indexes()
        expected_indexes = [0, 1, 15, 16, 17]

        self.assertEqual(calculated_indexes, expected_indexes)

    def test_decoupled_forces_and_momentums_in_structure(self):
        structure = get_test_structure(1)

        calculated_forces = structure.decoupled_forces_and_momentums_in_structure()

        # expected_forces = np.array([0,
        #                             5344, -34609, -40159.83,
        #                             0, -69219, 0,
        #                             -20673, -52931, 15779,
        #                             -2829, -18322, 11255])
        expected_forces = np.array([0,
                                    0, -32421, -34690.47,
                                    0, -64842, 0,
                                    -12500, -46763, 14776.92,
                                    0, -14342, 6788.55])

        np.testing.assert_allclose(calculated_forces, expected_forces, rtol=0.02, atol=2)

    def test_nodes_displacements(self):
        # Test structure 1
        structure = get_test_structure(1)

        calculated = structure.get_nodes_displacements()

        expected = np.array([0,
                             0,
                             0.009985658,

                             -0.022098846,
                             -0.000248405,
                             -0.004186425,

                             -0.01386954,
                             -0.051751769,
                             -0.000285887,

                             -0.005655847,
                             -0.000351931,
                             0.005281693,

                             -0.003987598,
                             0.009810969,
                             0.003011205,

                             0,
                             0,
                             0])

        np.testing.assert_allclose(calculated, expected, rtol=0.02)

        # Test structure 2
        structure2 = get_test_structure(2)

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

        np.testing.assert_allclose(calculated, expected, atol=5 ** -8)

        # Test structure 3
        structure = get_test_structure(3)

        calculated = structure.get_nodes_displacements()
        expected = np.array([0,
                             0,
                             -0.005698,
                             0,
                             0,
                             0.005698])

        np.testing.assert_allclose(calculated, expected, rtol=0.02)

        # Test structure 4
        structure = get_test_structure(4)

        calculated = structure.get_nodes_displacements()
        expected = np.array([0,
                             0,
                             -0.008231,
                             0,
                             0,
                             0.008864])

        np.testing.assert_allclose(calculated, expected, rtol=0.02)

        # Test structure 5
        structure = get_test_structure(5)

        calculated = structure.get_nodes_displacements()
        expected = np.array([0,
                             0,
                             -0.1601,
                             0,
                             0,
                             0])

        np.testing.assert_allclose(calculated, expected, rtol=0.02)

        # Test structure 6
        structure = get_test_structure(6)

        calculated = structure.get_nodes_displacements()
        expected = np.array([0,
                             0,
                             0,
                             0,
                             0,
                             0.00035616])

        np.testing.assert_allclose(calculated, expected, rtol=0.02)

        # Test structure 7
        structure = get_test_structure(7)

        calculated = structure.get_nodes_displacements()
        expected = np.array([0,
                             0,
                             -0.001425,
                             0,
                             0,
                             0.001425])

        np.testing.assert_allclose(calculated, expected, rtol=0.02)

        # Test structure 8
        structure = get_test_structure(8)

        calculated = structure.get_nodes_displacements()
        expected = np.array([0,
                             0,
                             0,
                             0,
                             -0.004749,
                             -0.002849])

        np.testing.assert_allclose(calculated, expected, rtol=0.02)

        # Test structure 9
        structure = get_test_structure(9)

        calculated = structure.get_nodes_displacements()
        expected = np.array([0,
                             0,
                             0,
                             0.870064,
                             -0.576764,
                             0.284958,
                             0.870936,
                             0.290768,
                             0.0782637,
                             0,
                             0,
                             -0.5827])

        np.testing.assert_allclose(calculated, expected, rtol=0.021)

        # Test structure 10
        structure = get_test_structure(10)

        calculated = structure.get_nodes_displacements()
        expected = np.array([0,
                             0,
                             0,
                             0,
                             0,
                             0.0028373
                             ])

        np.testing.assert_allclose(calculated, expected, rtol=0.02)

    def test_get_nodes(self):
        structure = get_test_structure(1)

        nodes = structure.get_nodes()
        n1 = nodes[0]
        n2 = nodes[1]
        n3 = nodes[2]
        n4 = nodes[3]
        n5 = nodes[4]
        n6 = nodes[5]

        self.assertEqual(len(structure.get_nodes()), 6)
        self.assertTrue(structure.get_nodes()[0].equals(n1))
        self.assertTrue(structure.get_nodes()[1].equals(n2))
        self.assertTrue(structure.get_nodes()[2].equals(n3))
        self.assertTrue(structure.get_nodes()[3].equals(n4))
        self.assertTrue(structure.get_nodes()[4].equals(n5))
        self.assertTrue(structure.get_nodes()[5].equals(n6))

    def test_get_nodes_reactions(self):
        # Test structure 1
        structure = get_test_structure(1)

        calculated = structure.get_nodes_reactions()
        expected = np.array([27650,
                             65528.88,
                             0,

                             0,
                             -32421,
                             -34690.47,

                             0,
                             -64842,
                             0,

                             -12500,
                             -46763,
                             14776.92,

                             0,
                             -14342,
                             6788.55,

                             -2650.53,
                             92839.124,
                             -2949.13])

        np.testing.assert_allclose(calculated, expected, rtol=0.02, atol=1)

        # Test structure 2
        structure2 = get_test_structure(2)

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

        # Test structure 3
        structure2 = get_test_structure(3)

        calculated = structure2.get_nodes_reactions()
        # These results are the exact same than ustatic provides
        expected = np.array([0,
                             300000,
                             0,
                             0,
                             300000,
                             0])

        np.testing.assert_allclose(calculated, expected, atol=.5)

        # Test structure 4
        structure2 = get_test_structure(4)

        calculated = structure2.get_nodes_reactions()
        # These results are the exact same than ustatic provides
        expected = np.array([0,
                             133300,
                             0,
                             0,
                             166700,
                             0])

        np.testing.assert_allclose(calculated, expected, atol=40)

        # Test structure 5
        structure2 = get_test_structure(5)

        calculated = structure2.get_nodes_reactions()
        # These results are the exact same than ustatic provides
        expected = np.array([0,
                             538200,
                             0,
                             0,
                             761800,
                             -1518000])

        np.testing.assert_allclose(calculated, expected, atol=40)

        # Test structure 6
        structure2 = get_test_structure(6)

        calculated = structure2.get_nodes_reactions()
        # These results are the exact same than ustatic provides
        expected = np.array([43300,
                             34370,
                             18750,
                             43300,
                             15630,
                             0])

        np.testing.assert_allclose(calculated, expected, atol=40)

        # Test structure 7
        structure2 = get_test_structure(7)

        calculated = structure2.get_nodes_reactions()
        # These results are the exact same than ustatic provides
        expected = np.array([0,
                             50000,
                             0,
                             0,
                             50000,
                             0])

        np.testing.assert_allclose(calculated, expected, atol=40)

        # Test structure 8
        structure2 = get_test_structure(8)

        calculated = structure2.get_nodes_reactions()
        # These results are the exact same than ustatic provides
        expected = np.array([0,
                             100000,
                             100000,
                             0,
                             0,
                             0])

        np.testing.assert_allclose(calculated, expected, atol=40)

        # Test structure 9
        structure2 = get_test_structure(9)

        calculated = structure2.get_nodes_reactions()
        # These results are the exact same than ustatic provides
        expected = np.array([-195755,
                             -12547,
                             159715,

                             75000,
                             38888,
                             107499,

                             45000,
                             46111,
                             -1666,

                             -44244,
                             -37452,
                             0])

        np.testing.assert_allclose(calculated, expected, atol=40)

        # Test structure 10
        structure2 = get_test_structure(10)

        calculated = structure2.get_nodes_reactions()
        # These results are the exact same than ustatic provides
        expected = np.array([-93750,
                             62500,
                             81250,
                             -56250,
                             37500,
                             0])

        np.testing.assert_allclose(calculated, expected, atol=40)

    def test_calculate_efforts(self):
        # Test structure 1
        structure = get_test_structure(1)
        structure.assembled_matrix()
        structure.get_nodes_displacements()
        structure.get_nodes_reactions()

        for key, bar in structure.bars.items():
            bar.calculate_efforts()

        bars = structure.get_bars()
        b1 = bars.get("B1")
        b2 = bars.get("B2")
        b3 = bars.get("B3")
        b4 = bars.get("B4")
        b5 = bars.get("B5")

        ## Bar 1 efforts
        efforts = b1.get_efforts()
        calculated_pij_values = efforts.get("p_ij")
        calculated_pji_values = efforts.get("p_ji")

        expected_pij_values = np.vstack(np.array([65528.89,
                                                  -27650.54,
                                                  0]))
        expected_pji_values = np.vstack(np.array([-65528.89,
                                                  27650.54,
                                                  -116132.259]))

        np.testing.assert_allclose(calculated_pij_values, expected_pij_values, rtol=0.02, atol=800)
        np.testing.assert_allclose(calculated_pji_values, expected_pji_values, rtol=0.02, atol=800)

        ## Bar 2 efforts
        efforts = b2.get_efforts()
        calculated_pij_values = efforts.get("p_ij")
        calculated_pji_values = efforts.get("p_ji")

        expected_pij_values = np.vstack(np.array([37864.8,
                                                  60206.69,
                                                  116132.259]))
        expected_pji_values = np.vstack(np.array([-27398.85,
                                                  3785.11,
                                                  67387.322]))

        np.testing.assert_allclose(calculated_pij_values, expected_pij_values, rtol=0.02)
        np.testing.assert_allclose(calculated_pji_values, expected_pji_values, rtol=0.02)

        ## Bar 3 efforts
        efforts = b3.get_efforts()
        calculated_pij_values = efforts.get("p_ij")
        calculated_pji_values = efforts.get("p_ji")

        expected_pij_values = np.vstack(np.array([27177.11,
                                                  5140.86,
                                                  -67387.322]))
        expected_pji_values = np.vstack(np.array([-37643.06,
                                                  58850.94,
                                                  -107321.681]))

        np.testing.assert_allclose(calculated_pij_values, expected_pij_values, rtol=0.02)
        np.testing.assert_allclose(calculated_pji_values, expected_pji_values, rtol=0.02)

        ## Bar 4 efforts
        efforts = b4.get_efforts()
        calculated_pij_values = efforts.get("p_ij")
        calculated_pji_values = efforts.get("p_ji")

        expected_pij_values = np.vstack(np.array([-4629.84,
                                                  28307.9,
                                                  40731.296]))
        expected_pji_values = np.vstack(np.array([0,
                                                  0,
                                                  0]))

        np.testing.assert_allclose(calculated_pij_values, expected_pij_values, atol=0.06)
        np.testing.assert_allclose(calculated_pji_values, expected_pji_values, atol=0.06)

        ## Bar 5 efforts
        efforts = b5.get_efforts()
        calculated_pij_values = efforts.get("p_ij")
        calculated_pji_values = efforts.get("p_ji")

        expected_pij_values = np.vstack(np.array([92839.14,
                                                  27650.54,
                                                  66581.385]))
        expected_pji_values = np.vstack(np.array([-92839.14,
                                                  -2650.54,
                                                  -2949.126]))

        np.testing.assert_allclose(calculated_pij_values, expected_pij_values, rtol=0.02)
        np.testing.assert_allclose(calculated_pji_values, expected_pji_values, rtol=0.02)

        # Test structure 2
        structure2 = get_test_structure(2)

        structure2.assembled_matrix()
        structure2.get_nodes_reactions()
        structure2.get_nodes_displacements()

        for key, bar in structure2.bars.items():
            bar.calculate_efforts()

        bars = structure2.get_bars()
        b1 = bars.get("B1")
        b2 = bars.get("B2")
        b3 = bars.get("B3")
        b4 = bars.get("B4")
        b5 = bars.get("B5")
        b6 = bars.get("B6")
        b7 = bars.get("B7")

        ## Bar 1 efforts
        efforts = b1.get_efforts()
        calculated_pij_values = efforts.get("p_ij")
        calculated_pji_values = efforts.get("p_ji")

        expected_pij_values = np.vstack(np.array([211292.56,
                                                  -32926.24,
                                                  0]))
        expected_pji_values = np.vstack(np.array([-211292.56,
                                                  72926.24,
                                                  -158778.72]))

        np.testing.assert_allclose(calculated_pij_values, expected_pij_values, atol=1)
        np.testing.assert_allclose(calculated_pji_values, expected_pji_values, atol=1)

        ## Bar 2 efforts
        efforts = b2.get_efforts()
        calculated_pij_values = efforts.get("p_ij")
        calculated_pji_values = efforts.get("p_ji")

        expected_pij_values = np.vstack(np.array([72926.24,
                                                  211292.56,
                                                  158778.72]))
        expected_pji_values = np.vstack(np.array([-72926.24,
                                                  -11292.56,
                                                  63806.40]))

        np.testing.assert_allclose(calculated_pij_values, expected_pij_values, atol=1)
        np.testing.assert_allclose(calculated_pji_values, expected_pji_values, atol=1)

        ## Bar 3 efforts
        efforts = b3.get_efforts()
        calculated_pij_values = efforts.get("p_ij")
        calculated_pji_values = efforts.get("p_ji")

        expected_pij_values = np.vstack(np.array([-11292.56,
                                                  72926.24,
                                                  -63806.4]))
        expected_pji_values = np.vstack(np.array([11292.56,
                                                  -72926.24,
                                                  136732.64]))

        np.testing.assert_allclose(calculated_pij_values, expected_pij_values, atol=1)
        np.testing.assert_allclose(calculated_pji_values, expected_pji_values, atol=1)

        ## Bar 4 efforts
        efforts = b4.get_efforts()
        calculated_pij_values = efforts.get("p_ij")
        calculated_pji_values = efforts.get("p_ji")

        expected_pij_values = np.vstack(np.array([72926.24,
                                                  11292.56,
                                                  -136732.64]))
        expected_pji_values = np.vstack(np.array([-72926.24,
                                                  13707.44,
                                                  134317.75]))

        np.testing.assert_allclose(calculated_pij_values, expected_pij_values, atol=1)
        np.testing.assert_allclose(calculated_pji_values, expected_pji_values, atol=1)

        ## Bar 5 efforts
        efforts = b5.get_efforts()
        calculated_pij_values = efforts.get("p_ij")
        calculated_pji_values = efforts.get("p_ji")

        expected_pij_values = np.vstack(np.array([-13707.44,
                                                  -72926.24,
                                                  -134317.75]))
        expected_pji_values = np.vstack(np.array([13707.44,
                                                  72926.24,
                                                  61391.51]))

        np.testing.assert_allclose(calculated_pij_values, expected_pij_values, atol=1)
        np.testing.assert_allclose(calculated_pji_values, expected_pji_values, atol=1)

        ## Bar 6 efforts
        efforts = b6.get_efforts()
        calculated_pij_values = efforts.get("p_ij")
        calculated_pji_values = efforts.get("p_ji")

        expected_pij_values = np.vstack(np.array([72926.24,
                                                  -13707.44,
                                                  -61391.51]))
        expected_pji_values = np.vstack(np.array([-72926.24,
                                                  213707.44,
                                                  -166023.37]))

        np.testing.assert_allclose(calculated_pij_values, expected_pij_values, atol=1)
        np.testing.assert_allclose(calculated_pji_values, expected_pji_values, atol=1)

        ## Bar 7 efforts
        efforts = b7.get_efforts()
        calculated_pij_values = efforts.get("p_ij")
        calculated_pji_values = efforts.get("p_ji")

        expected_pij_values = np.vstack(np.array([213707,
                                                  72926.24,
                                                  166023.37]))
        expected_pji_values = np.vstack(np.array([-213707.44,
                                                  -42926.24,
                                                  7755.35]))

        np.testing.assert_allclose(calculated_pij_values, expected_pij_values, atol=1)
        np.testing.assert_allclose(calculated_pji_values, expected_pji_values, atol=1)

        # Test structure 9
        structure = get_test_structure(9)
        structure.assembled_matrix()
        structure.get_nodes_displacements()
        structure.get_nodes_reactions()

        for key, bar in structure.bars.items():
            bar.calculate_efforts()

        bars = structure.get_bars()
        b1 = bars.get("B1")
        b2 = bars.get("B2")
        b3 = bars.get("B3")

        ## Bar 1 efforts
        efforts = b1.get_efforts()
        calculated_pij_values = efforts.get("p_ij")
        calculated_pji_values = efforts.get("p_ji")

        expected_pij_values = np.vstack(np.array([-119025,
                                                  155918,
                                                  159715]))
        expected_pji_values = np.vstack(np.array([119025,
                                                  24359,
                                                  77455]))

        np.testing.assert_allclose(calculated_pij_values, expected_pij_values, rtol=0.02, atol=800)
        np.testing.assert_allclose(calculated_pji_values, expected_pji_values, rtol=0.02, atol=800)

        # Bar 2 efforts
        efforts = b2.get_efforts()
        calculated_pij_values = efforts.get("p_ij")
        calculated_pji_values = efforts.get("p_ji")

        expected_pij_values = np.vstack(np.array([-45755,
                                                  -112547,
                                                  -77455]))
        expected_pji_values = np.vstack(np.array([45755,
                                                  -7452,
                                                  -20186]))

        np.testing.assert_allclose(calculated_pij_values, expected_pij_values, rtol=0.02)
        np.testing.assert_allclose(calculated_pji_values, expected_pji_values, rtol=0.02)

        ## Bar 3 efforts
        efforts = b3.get_efforts()
        calculated_pij_values = efforts.get("p_ij")
        calculated_pji_values = efforts.get("p_ji")

        expected_pij_values = np.vstack(np.array([-21539,
                                                  -41050,
                                                  20186]))
        expected_pji_values = np.vstack(np.array([21539,
                                                  -53817,
                                                  0]))

        np.testing.assert_allclose(calculated_pij_values, expected_pij_values, rtol=0.02)
        np.testing.assert_allclose(calculated_pji_values, expected_pji_values, rtol=0.02, atol=1)

    def test_axial_force_law(self):
        # Test structure 1
        structure = get_test_structure(1)
        structure.assembled_matrix()
        structure.get_nodes_reactions()
        structure.get_nodes_displacements()

        for key, bar in structure.bars.items():
            bar.calculate_efforts()

        bars = structure.get_bars()
        b1 = bars.get("B1")
        b2 = bars.get("B2")
        b3 = bars.get("B3")
        b4 = bars.get("B4")
        b5 = bars.get("B5")

        self.assertRaises(ValueError, b1.axial_force_law, -2)
        self.assertRaises(ValueError, b1.axial_force_law, 2)

        # Bar 1
        np.testing.assert_allclose(b1.axial_force_law(.3), -65528, rtol=0.02)
        np.testing.assert_allclose(b1.axial_force_law(.5), -65528, rtol=0.02)
        np.testing.assert_allclose(b1.axial_force_law(.7), -65528, rtol=0.02)

        # Bar 2
        bar = b2
        x = 1.7 / bar.length()
        np.testing.assert_allclose(bar.axial_force_law(x), -35110, rtol=0.02)
        x = 3.4 / bar.length()
        np.testing.assert_allclose(bar.axial_force_law(x), -32356, rtol=0.02)
        x = 5.8 / bar.length()
        np.testing.assert_allclose(bar.axial_force_law(x), -28500, rtol=0.02)

        # Bar 3
        bar = b3
        x = 1.7 / bar.length()
        np.testing.assert_allclose(bar.axial_force_law(x), -29931, rtol=0.02)
        x = 3.4 / bar.length()
        np.testing.assert_allclose(bar.axial_force_law(x), -32685, rtol=0.02)
        x = 5.8 / bar.length()
        np.testing.assert_allclose(bar.axial_force_law(x), -36541, rtol=0.02)

        # Bar 4
        bar = b4
        x = 0.3 / bar.length()
        np.testing.assert_allclose(bar.axial_force_law(x), 4142, rtol=0.02)
        x = 1.5 / bar.length()
        np.testing.assert_allclose(bar.axial_force_law(x), 2192, rtol=0.02)
        x = 2.7 / bar.length()
        np.testing.assert_allclose(bar.axial_force_law(x), 243, rtol=0.02, atol=45)

        # Bar 5
        bar = b5
        x = 0.7 / bar.length()
        np.testing.assert_allclose(bar.axial_force_law(x), -92839, rtol=0.02)
        x = 1.5 / bar.length()
        np.testing.assert_allclose(bar.axial_force_law(x), -92839, rtol=0.02)
        x = 3.5 / bar.length()
        np.testing.assert_allclose(bar.axial_force_law(x), -92839, rtol=0.02)

        # Test structure 2
        structure = get_test_structure(2)
        structure.assembled_matrix()
        structure.get_nodes_reactions()
        structure.get_nodes_displacements()

        for key, bar in structure.bars.items():
            bar.calculate_efforts()

        bars = structure.get_bars()
        b1 = bars.get("B1")
        b2 = bars.get("B2")
        b3 = bars.get("B3")
        b4 = bars.get("B4")
        b5 = bars.get("B5")
        b6 = bars.get("B6")
        b7 = bars.get("B7")

        self.assertRaises(ValueError, b1.axial_force_law, -2)
        self.assertRaises(ValueError, b1.axial_force_law, 2)

        # LOS DATOS DE LOS ESFUERZOS PARA COMPARAR SE HAN OBTENIDO EN CESPLA

        # Bar 1
        bar = b1
        np.testing.assert_allclose(bar.axial_force_law(.3), -211292.56, rtol=0.02)
        np.testing.assert_allclose(bar.axial_force_law(.5), -211292.56, rtol=0.02)
        np.testing.assert_allclose(bar.axial_force_law(.7), -211292.56, rtol=0.02)

        # Bar 2
        bar = b2
        np.testing.assert_allclose(bar.axial_force_law(.3), -72926.24, rtol=0.02)
        np.testing.assert_allclose(bar.axial_force_law(.5), -72926.24, rtol=0.02)
        np.testing.assert_allclose(bar.axial_force_law(.7), -72926.24, rtol=0.02)

        # Bar 3
        bar = b3
        np.testing.assert_allclose(bar.axial_force_law(.3), 11292.56, rtol=0.02)
        np.testing.assert_allclose(bar.axial_force_law(.5), 11292.56, rtol=0.02)
        np.testing.assert_allclose(bar.axial_force_law(.7), 11292.56, rtol=0.02)

        # Bar 4
        bar = b4
        np.testing.assert_allclose(bar.axial_force_law(.3), -72926.24, rtol=0.02)
        np.testing.assert_allclose(bar.axial_force_law(.5), -72926.24, rtol=0.02)
        np.testing.assert_allclose(bar.axial_force_law(.7), -72926.24, rtol=0.02)

        # Bar 5
        bar = b5
        np.testing.assert_allclose(bar.axial_force_law(.3), 13707.44, rtol=0.02)
        np.testing.assert_allclose(bar.axial_force_law(.5), 13707.44, rtol=0.02)
        np.testing.assert_allclose(bar.axial_force_law(.7), 13707.44, rtol=0.02)

        # Bar 6
        bar = b6
        np.testing.assert_allclose(bar.axial_force_law(.3), -72926.24, rtol=0.02)
        np.testing.assert_allclose(bar.axial_force_law(.5), -72926.24, rtol=0.02)
        np.testing.assert_allclose(bar.axial_force_law(.7), -72926.24, rtol=0.02)

        # Bar 57
        bar = b7
        np.testing.assert_allclose(bar.axial_force_law(.3), -213707.44, rtol=0.02)
        np.testing.assert_allclose(bar.axial_force_law(.5), -213707.44, rtol=0.02)
        np.testing.assert_allclose(bar.axial_force_law(.7), -213707.44, rtol=0.02)

        # Test structure 3
        # LOS DATOS DE LOS ESFUERZOS PARA COMPARAR SE HAN OBTENIDO EN USTATIC
        structure = get_test_structure(3)
        structure.assembled_matrix()
        structure.get_nodes_reactions()
        structure.get_nodes_displacements()

        for key, bar in structure.bars.items():
            bar.calculate_efforts()

        bars = structure.get_bars()
        b1 = bars.get("B1")

        self.assertRaises(ValueError, b1.axial_force_law, -2)
        self.assertRaises(ValueError, b1.axial_force_law, 2)

        # Bar 1
        np.testing.assert_allclose(b1.axial_force_law(.3), 0, rtol=0.02)
        np.testing.assert_allclose(b1.axial_force_law(.5), 0, rtol=0.02)
        np.testing.assert_allclose(b1.axial_force_law(.7), 0, rtol=0.02)

        # Test structure 4
        # LOS DATOS DE LOS ESFUERZOS PARA COMPARAR SE HAN OBTENIDO EN USTATIC
        structure = get_test_structure(4)
        structure.assembled_matrix()
        structure.get_nodes_reactions()
        structure.get_nodes_displacements()

        for key, bar in structure.bars.items():
            bar.calculate_efforts()

        bars = structure.get_bars()
        b1 = bars.get("B1")

        self.assertRaises(ValueError, b1.axial_force_law, -2)
        self.assertRaises(ValueError, b1.axial_force_law, 2)

        # Bar 1
        np.testing.assert_allclose(b1.axial_force_law(.3), 0, rtol=0.02)
        np.testing.assert_allclose(b1.axial_force_law(.5), 0, rtol=0.02)
        np.testing.assert_allclose(b1.axial_force_law(.7), 0, rtol=0.02)

        # Test structure 5
        # LOS DATOS DE LOS ESFUERZOS PARA COMPARAR SE HAN OBTENIDO EN USTATIC
        structure = get_test_structure(5)
        structure.assembled_matrix()
        structure.get_nodes_reactions()
        structure.get_nodes_displacements()

        for key, bar in structure.bars.items():
            bar.calculate_efforts()

        bars = structure.get_bars()
        b1 = bars.get("B1")

        self.assertRaises(ValueError, b1.axial_force_law, -2)
        self.assertRaises(ValueError, b1.axial_force_law, 2)

        # Bar 1
        np.testing.assert_allclose(b1.axial_force_law(.3), 0, rtol=0.02)
        np.testing.assert_allclose(b1.axial_force_law(.5), 0, rtol=0.02)
        np.testing.assert_allclose(b1.axial_force_law(.7), 0, rtol=0.02)

        # Test structure 6
        # LOS DATOS DE LOS ESFUERZOS PARA COMPARAR SE HAN OBTENIDO EN USTATIC
        structure = get_test_structure(6)
        structure.assembled_matrix()
        structure.get_nodes_reactions()
        structure.get_nodes_displacements()

        for key, bar in structure.bars.items():
            bar.calculate_efforts()

        bars = structure.get_bars()
        b1 = bars.get("B1")

        self.assertRaises(ValueError, b1.axial_force_law, -2)
        self.assertRaises(ValueError, b1.axial_force_law, 2)

        # Bar 1
        bar = b1
        x = 0.3349 / bar.length()
        np.testing.assert_allclose(bar.axial_force_law(x), -43300, rtol=0.01)
        x = 0.7324 / bar.length()
        np.testing.assert_allclose(bar.axial_force_law(x), -43300, rtol=0.01)
        x = 1.247 / bar.length()
        np.testing.assert_allclose(bar.axial_force_law(x), 43300, rtol=0.01)
        x = 1.643 / bar.length()
        np.testing.assert_allclose(bar.axial_force_law(x), 43300, rtol=0.01)

        # Test structure 7
        # LOS DATOS DE LOS ESFUERZOS PARA COMPARAR SE HAN OBTENIDO EN USTATIC
        structure = get_test_structure(7)
        structure.assembled_matrix()
        structure.get_nodes_reactions()
        structure.get_nodes_displacements()

        for key, bar in structure.bars.items():
            bar.calculate_efforts()

        bars = structure.get_bars()
        b1 = bars.get("B1")

        self.assertRaises(ValueError, b1.axial_force_law, -2)
        self.assertRaises(ValueError, b1.axial_force_law, 2)

        # Bar 1
        bar = b1
        x = 0.3349 / bar.length()
        np.testing.assert_allclose(bar.axial_force_law(x), 0, rtol=0.01)
        x = 0.7324 / bar.length()
        np.testing.assert_allclose(bar.axial_force_law(x), 0, rtol=0.01)
        x = 1.247 / bar.length()
        np.testing.assert_allclose(bar.axial_force_law(x), 0, rtol=0.01)
        x = 1.643 / bar.length()
        np.testing.assert_allclose(bar.axial_force_law(x), 0, rtol=0.01)

        # Test structure 8
        # LOS DATOS DE LOS ESFUERZOS PARA COMPARAR SE HAN OBTENIDO EN USTATIC
        structure = get_test_structure(8)
        structure.assembled_matrix()
        structure.get_nodes_reactions()
        structure.get_nodes_displacements()

        for key, bar in structure.bars.items():
            bar.calculate_efforts()

        bars = structure.get_bars()
        b1 = bars.get("B1")

        self.assertRaises(ValueError, b1.axial_force_law, -2)
        self.assertRaises(ValueError, b1.axial_force_law, 2)

        # Bar 1
        bar = b1
        x = 0.3349 / bar.length()
        np.testing.assert_allclose(bar.axial_force_law(x), 0, rtol=0.01)
        x = 0.7324 / bar.length()
        np.testing.assert_allclose(bar.axial_force_law(x), 0, rtol=0.01)
        x = 1.247 / bar.length()
        np.testing.assert_allclose(bar.axial_force_law(x), 0, rtol=0.01)
        x = 1.643 / bar.length()
        np.testing.assert_allclose(bar.axial_force_law(x), 0, rtol=0.01)

        # Test structure 9
        # LOS DATOS DE LOS ESFUERZOS PARA COMPARAR SE HAN OBTENIDO EN USTATIC
        structure = get_test_structure(9)
        structure.assembled_matrix()
        structure.get_nodes_reactions()
        structure.get_nodes_displacements()

        for key, bar in structure.bars.items():
            bar.calculate_efforts()

        bars = structure.get_bars()
        b1 = bars.get("B1")

        self.assertRaises(ValueError, b1.axial_force_law, -2)
        self.assertRaises(ValueError, b1.axial_force_law, 2)

        # Bar 1
        bar = b1
        np.testing.assert_allclose(bar.axial_force_law(.2), 119000, rtol=0.01)
        np.testing.assert_allclose(bar.axial_force_law(.4), 119000, rtol=0.01)
        np.testing.assert_allclose(bar.axial_force_law(.6), 119000, rtol=0.01)
        np.testing.assert_allclose(bar.axial_force_law(.8), 119000, rtol=0.01)

        # Bar 2
        bar = bars.get("B2")
        np.testing.assert_allclose(bar.axial_force_law(.2), 45760, rtol=0.01)
        np.testing.assert_allclose(bar.axial_force_law(.4), 45760, rtol=0.01)
        np.testing.assert_allclose(bar.axial_force_law(.6), 45760, rtol=0.01)
        np.testing.assert_allclose(bar.axial_force_law(.8), 45760, rtol=0.01)

        # Bar 3
        bar = bars.get("B3")
        np.testing.assert_allclose(bar.axial_force_law(.2), 21540, rtol=0.01)
        np.testing.assert_allclose(bar.axial_force_law(.4), 21540, rtol=0.01)
        np.testing.assert_allclose(bar.axial_force_law(.6), 21540, rtol=0.01)
        np.testing.assert_allclose(bar.axial_force_law(.8), 21540, rtol=0.01)

        # Test structure 10
        # LOS DATOS DE LOS ESFUERZOS PARA COMPARAR SE HAN OBTENIDO EN USTATIC
        structure = get_test_structure(10)
        structure.assembled_matrix()
        structure.get_nodes_reactions()
        structure.get_nodes_displacements()

        for key, bar in structure.bars.items():
            bar.calculate_efforts()

        bars = structure.get_bars()
        b1 = bars.get("B1")

        self.assertRaises(ValueError, b1.axial_force_law, -2)
        self.assertRaises(ValueError, b1.axial_force_law, 2)

        # Bar 1
        bar = b1
        np.testing.assert_allclose(bar.axial_force_law(0.2), 0, atol=0.01)
        np.testing.assert_allclose(bar.axial_force_law(0.4), 0, atol=0.01)
        np.testing.assert_allclose(bar.axial_force_law(0.6), 0, atol=0.01)
        np.testing.assert_allclose(bar.axial_force_law(0.8), 0, atol=0.01)

    def test_shear_strength_law(self):
        # Test structure 1
        structure = get_test_structure(1)
        structure.assembled_matrix()
        structure.get_nodes_reactions()
        structure.get_nodes_displacements()

        for key, bar in structure.bars.items():
            bar.calculate_efforts()

        bars = structure.get_bars()
        b1 = bars.get("B1")
        b2 = bars.get("B2")
        b3 = bars.get("B3")
        b4 = bars.get("B4")
        b5 = bars.get("B5")

        self.assertRaises(ValueError, b1.shear_strength_law, -2)
        self.assertRaises(ValueError, b1.shear_strength_law, 2)

        # Bar 1
        bar = b1
        np.testing.assert_allclose(bar.shear_strength_law(.3), 27650, rtol=0.02)
        np.testing.assert_allclose(bar.shear_strength_law(.5), 27650, rtol=0.02)
        np.testing.assert_allclose(bar.shear_strength_law(.7), 27650, rtol=0.02)

        # Bar 2
        bar = b2
        # x = 1.7 / bar.length()
        # np.testing.assert_allclose(bar.shear_strength_law(x), -43366, rtol=0.02)
        # x = 3.4 / bar.length()
        # np.testing.assert_allclose(bar.shear_strength_law(x), -26526, rtol=0.02)
        # x = 5.8 / bar.length()
        # np.testing.assert_allclose(bar.shear_strength_law(x), -2950, rtol=0.02, atol=205)
        x = 1.2 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), -48400, rtol=0.02)
        x = 2.88 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), -31870, rtol=0.02)
        x = 5.115 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), -9888, rtol=0.02)
        x = 6.402 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), 2765, rtol=0.02)

        np.testing.assert_allclose(bar.shear_strength_law(1), 3785.11, rtol=0.02)

        # Bar 3
        bar = b3
        x = 1.7 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), 11699, rtol=0.03)
        x = 3.4 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), 28539, rtol=0.02)
        x = 5.8 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), 52114, rtol=0.02)

        # Bar 4
        bar = b4
        x = 0.3 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), -25328, rtol=0.02)
        x = 1.5 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), -13409, rtol=0.02)
        x = bar.length() / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), 0, rtol=0.02, atol=1)

        # Bar 5
        bar = b5
        x = 0.7 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), -27650, rtol=0.02)
        x = 1.5 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), -27650, rtol=0.02)
        x = 3.5 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), -2650, rtol=0.02)

        # Test structure 2
        structure = get_test_structure(2)
        structure.assembled_matrix()
        structure.get_nodes_reactions()
        structure.get_nodes_displacements()

        for key, bar in structure.bars.items():
            bar.calculate_efforts()

        bars = structure.get_bars()
        b1 = bars.get("B1")
        b2 = bars.get("B2")
        b3 = bars.get("B3")
        b4 = bars.get("B4")
        b5 = bars.get("B5")
        b6 = bars.get("B6")
        b7 = bars.get("B7")

        self.assertRaises(ValueError, b1.shear_strength_law, -2)
        self.assertRaises(ValueError, b1.shear_strength_law, 2)

        # Bar 1
        bar = b1
        np.testing.assert_allclose(bar.shear_strength_law(.3), 32926.24, rtol=0.02)
        np.testing.assert_allclose(bar.shear_strength_law(.5), 32926.24, rtol=0.02)
        np.testing.assert_allclose(bar.shear_strength_law(.7), 72926.24, rtol=0.02)

        # Bar 2
        bar = b2
        x = 0.6 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), -148134.66, rtol=0.05)
        x = 0.7 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), -137608.35, rtol=0.05)
        x = 0.8 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), -127082.03, rtol=0.05)
        x = 0.9 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), -116555.72, rtol=0.05)
        x = 1.1 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), -106029.40, rtol=0.05)
        x = 1.2 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), -95503.28, rtol=0.05)
        x = 1.3 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), -84976.77, rtol=0.05)
        x = 1.4 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), -74450.45, rtol=0.05)
        x = 1.5 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), -63924.14, rtol=0.05)

        # Bar 3
        bar = b3
        np.testing.assert_allclose(bar.shear_strength_law(0.3), -72926.24, rtol=0.05)
        np.testing.assert_allclose(bar.shear_strength_law(0.5), -72926.24, rtol=0.05)
        np.testing.assert_allclose(bar.shear_strength_law(0.7), -72926.24, rtol=0.05)

        # Bar 4
        bar = b4
        np.testing.assert_allclose(bar.shear_strength_law(0.3), -11292.56, rtol=0.05)
        np.testing.assert_allclose(bar.shear_strength_law(0.5), -11292.56, rtol=0.05)
        np.testing.assert_allclose(bar.shear_strength_law(0.7), 13707.44, rtol=0.05)

        # Bar 5
        bar = b5
        np.testing.assert_allclose(bar.shear_strength_law(0.3), 72926.24, rtol=0.05)
        np.testing.assert_allclose(bar.shear_strength_law(0.5), 72926.24, rtol=0.05)
        np.testing.assert_allclose(bar.shear_strength_law(0.7), 72926.24, rtol=0.05)

        # Bar 6
        bar = b6
        x = 0.5 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), 66339.02, rtol=0.05)
        x = 0.6 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), 76865.34, rtol=0.05)
        x = 0.7 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), 87391.65, rtol=0.05)
        x = 0.8 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), 97917.97, rtol=0.05)
        x = 0.9 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), 108444.28, rtol=0.05)
        x = 1.1 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), 118970.6, rtol=0.05)
        x = 1.2 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), 129496.92, rtol=0.05)
        x = 1.3 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), 140023.23, rtol=0.05)
        x = 1.4 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), 150549.55, rtol=0.05)

        # Bar 7
        bar = b7
        np.testing.assert_allclose(bar.shear_strength_law(0.3), -72926.24, rtol=0.05)
        np.testing.assert_allclose(bar.shear_strength_law(0.5), -72926.24, rtol=0.05)
        np.testing.assert_allclose(bar.shear_strength_law(1), -42926.24, rtol=0.05)

        # Test structure 3
        structure = get_test_structure(3)
        structure.assembled_matrix()
        structure.get_nodes_reactions()
        structure.get_nodes_displacements()

        for key, bar in structure.bars.items():
            bar.calculate_efforts()

        bars = structure.get_bars()
        b1 = bars.get("B1")

        self.assertRaises(ValueError, b1.shear_strength_law, -2)
        self.assertRaises(ValueError, b1.shear_strength_law, 2)

        # Bar 1
        bar = b1
        x = .3979 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), -180600, rtol=0.01)
        x = .8003 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), -59920, rtol=0.01)
        x = 1.198 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), 59250, rtol=0.01)
        x = 1.629 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), 188600, rtol=0.01)

        # Test structure 4
        structure = get_test_structure(4)
        structure.assembled_matrix()
        structure.get_nodes_reactions()
        structure.get_nodes_displacements()

        for key, bar in structure.bars.items():
            bar.calculate_efforts()

        bars = structure.get_bars()
        b1 = bars.get("B1")

        self.assertRaises(ValueError, b1.shear_strength_law, -2)
        self.assertRaises(ValueError, b1.shear_strength_law, 2)

        # Bar 1
        bar = b1
        x = 0.5472 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), -133300, rtol=0.01)
        x = 1.493 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), -33333, rtol=0.01)
        x = 2.261 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), 166700, rtol=0.01)

        # Test structure 5
        structure = get_test_structure(5)
        structure.assembled_matrix()
        structure.get_nodes_reactions()
        structure.get_nodes_displacements()

        for key, bar in structure.bars.items():
            bar.calculate_efforts()

        bars = structure.get_bars()
        b1 = bars.get("B1")

        self.assertRaises(ValueError, b1.shear_strength_law, -2)
        self.assertRaises(ValueError, b1.shear_strength_law, 2)

        # Bar 1
        bar = b1
        x = 0.7494 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), -425800, rtol=0.01)
        x = 2.36 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), -184200, rtol=0.01)
        x = 4.755 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), 175000, rtol=0.01)
        x = 6.171 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), 387500, rtol=0.01)
        x = 7.803 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), 432300, rtol=0.01)
        x = 9.299 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), 656600, rtol=0.01)

        # Test structure 6
        structure = get_test_structure(6)
        structure.assembled_matrix()
        structure.get_nodes_reactions()
        structure.get_nodes_displacements()

        for key, bar in structure.bars.items():
            bar.calculate_efforts()

        bars = structure.get_bars()
        b1 = bars.get("B1")

        self.assertRaises(ValueError, b1.shear_strength_law, -2)
        self.assertRaises(ValueError, b1.shear_strength_law, 2)

        # Bar 1
        bar = b1
        x = 0.7494 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), -34370, rtol=0.01)
        x = 0.6086 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), -34370, rtol=0.01)
        x = 1.159 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), 15630, rtol=0.01)
        x = 1.679 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), 15630, rtol=0.01)

        # Test structure 7
        structure = get_test_structure(7)
        structure.assembled_matrix()
        structure.get_nodes_reactions()
        structure.get_nodes_displacements()

        for key, bar in structure.bars.items():
            bar.calculate_efforts()

        bars = structure.get_bars()
        b1 = bars.get("B1")

        self.assertRaises(ValueError, b1.shear_strength_law, -2)
        self.assertRaises(ValueError, b1.shear_strength_law, 2)

        # Bar 1
        bar = b1
        x = 0.44 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), -50000, rtol=0.01)
        x = 1.552 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), 50000, rtol=0.01)

        # Test structure 8
        structure = get_test_structure(8)
        structure.assembled_matrix()
        structure.get_nodes_reactions()
        structure.get_nodes_displacements()

        for key, bar in structure.bars.items():
            bar.calculate_efforts()

        bars = structure.get_bars()
        b1 = bars.get("B1")

        self.assertRaises(ValueError, b1.shear_strength_law, -2)
        self.assertRaises(ValueError, b1.shear_strength_law, 2)

        # Bar 1
        bar = b1
        x = 0.44 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), -100000, rtol=0.01)
        x = 1.552 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), 0, rtol=0.01)

        # Test structure 9
        structure = get_test_structure(9)
        structure.assembled_matrix()
        structure.get_nodes_reactions()
        structure.get_nodes_displacements()

        for key, bar in structure.bars.items():
            bar.calculate_efforts()

        bars = structure.get_bars()
        b1 = bars.get("B1")
        b2 = bars.get("B2")
        b3 = bars.get("B3")

        self.assertRaises(ValueError, b1.shear_strength_law, -2)
        self.assertRaises(ValueError, b1.shear_strength_law, 2)

        # Bar 1
        bar = b1
        x = 0.64 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), -123900, rtol=0.01)
        x = 1.507 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), -80550, rtol=0.01)
        x = 2.507 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), -30580, rtol=0.01)
        x = 3.409 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), 14550, rtol=0.01)

        # Bar 2
        bar = b2
        x = 0.34 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), 112500, rtol=0.01)
        x = 0.68 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), 112500, rtol=0.01)
        x = 1.4 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), -7453, rtol=0.01)
        x = bar.length() / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), -7453, rtol=0.01)

        # Bar 3
        bar = b3
        x = 0.2126 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), 34670, rtol=0.01)
        x = 1.076 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), 8782, rtol=0.01)
        x = 2.132 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), -22910, rtol=0.01)

        # Test structure 10
        structure = get_test_structure(10)
        structure.assembled_matrix()
        structure.get_nodes_reactions()
        structure.get_nodes_displacements()

        for key, bar in structure.bars.items():
            bar.calculate_efforts()

        bars = structure.get_bars()
        b1 = bars.get("B1")

        self.assertRaises(ValueError, b1.shear_strength_law, -2)
        self.assertRaises(ValueError, b1.shear_strength_law, 2)

        # Bar 1
        bar = b1
        x = 0.4223 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), -91560, rtol=0.01)
        x = 1.47 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), -39160, rtol=0.01)
        x = 2.727 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), 23670, rtol=0.01)
        x = 3.368 / bar.length()
        np.testing.assert_allclose(bar.shear_strength_law(x), 55720, rtol=0.01)

    def test_bending_moment_law(self):
        # Test structure 1
        structure = get_test_structure(1)
        structure.assembled_matrix()
        structure.get_nodes_reactions()
        structure.get_nodes_displacements()

        for key, bar in structure.bars.items():
            bar.calculate_efforts()

        bars = structure.get_bars()
        b1 = bars.get("B1")
        b2 = bars.get("B2")
        b3 = bars.get("B3")
        b4 = bars.get("B4")
        b5 = bars.get("B5")

        self.assertRaises(ValueError, b1.bending_moment_law, -2)
        self.assertRaises(ValueError, b1.bending_moment_law, 2)

        # Bar 1
        bar = b1
        np.testing.assert_allclose(bar.bending_moment_law(0), 0, rtol=0.02, atol=1)
        np.testing.assert_allclose(bar.bending_moment_law(1), -116132.24, rtol=0.02, atol=1)
        x = 0.7 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), -19350, rtol=0.02)
        x = 1.5 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), -41570, rtol=0.02)
        x = 3.5 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), -97795, rtol=0.02)

        # Bar 2
        bar = b2
        x = 1.541 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), -35030, rtol=0.02)
        x = 1.7 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), -27477, rtol=0.02)
        x = 3.761 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), 40730, rtol=0.02)
        x = 5.548 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), 66500, rtol=0.02)

        # Bar 3
        bar = b3
        x = 1.7 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), 61773, rtol=0.02)
        x = 3.4 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), 27331, rtol=0.03)
        x = 5.8 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), -69319, rtol=0.03)

        # Bar 4
        bar = b4
        x = 0.3 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), -32607, rtol=0.02)
        x = 1.5 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), -9139, rtol=0.022)
        x = 2.7 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), -112, rtol=0.02, atol=44)

        # Bar 5
        bar = b5
        x = 0.7 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), -48244, rtol=0.022)
        x = 0.7722 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), -45230, rtol=0.02)
        x = 1.684 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), -20030, rtol=0.02)
        x = 2.561 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), -7292, rtol=0.02)
        x = 3.586 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), -4578, rtol=0.02)

        # Test structure 2
        structure = get_test_structure(2)
        structure.assembled_matrix()
        structure.get_nodes_reactions()
        structure.get_nodes_displacements()

        for key, bar in structure.bars.items():
            bar.calculate_efforts()

        bars = structure.get_bars()
        b1 = bars.get("B1")
        b2 = bars.get("B2")
        b3 = bars.get("B3")
        b4 = bars.get("B4")
        b5 = bars.get("B5")
        b6 = bars.get("B6")
        b7 = bars.get("B7")

        self.assertRaises(ValueError, b1.bending_moment_law, -2)
        self.assertRaises(ValueError, b1.bending_moment_law, 2)

        # Bar 1
        bar = b1
        x = 0.8 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), -25994.40, rtol=0.05)
        x = 0.9 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), -31193.28, rtol=0.05)
        x = 1.1 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), -36392.16, rtol=0.05)
        x = 1.3 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), -41591.04, rtol=0.05)
        x = 1.4 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), -46789.92, rtol=0.05)
        x = 1.6 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), -55146.7, rtol=0.05)
        x = 1.7 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), -66661.37, rtol=0.05)
        x = 1.9 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), -78176.03, rtol=0.05)
        x = 2.1 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), -89690.7, rtol=0.05)

        # Bar 2
        bar = b2
        x = 0.1582 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), -126600, rtol=0.05)
        x = 0.717 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), -32990, rtol=0.05)
        x = 1.286 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), 30260, rtol=0.05)
        x = 1.751 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), 57900, rtol=0.05)

        # Bar 3
        bar = b3
        x = 0.3 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), 86835.74, rtol=0.05)
        x = 0.4 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), 94512.18, rtol=0.05)
        x = 0.1832 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), 77160, rtol=0.05)
        x = 0.4061 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), 93420, rtol=0.05)
        x = 0.6442 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), 110800, rtol=0.05)
        x = 0.8305 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), 124400, rtol=0.05)

        # Bar 4
        bar = b4
        x = 0.2894 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), 140000, rtol=0.05)
        x = 0.7168 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), 144800, rtol=0.05)
        x = 1.239 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), 144700, rtol=0.05)
        x = 1.724 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), 138100, rtol=0.05)

        # Bar 5
        bar = b5
        x = 0.2074 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), 119200, rtol=0.05)
        x = 0.4251 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), 103300, rtol=0.05)
        x = 0.6358 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), 87950, rtol=0.05)
        x = 0.8183 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), 74640, rtol=0.05)

        # Bar 6
        bar = b6
        x = 0.223 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), 55850, rtol=0.05)
        x = 0.7005 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), 27250, rtol=0.05)
        x = 1.248 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), -33630, rtol=0.05)
        x = 1.684 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), -103400, rtol=0.05)

        # Bar 7
        bar = b7
        x = 0.3081 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), -143600, rtol=0.05)
        x = 1.003 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), -92860, rtol=0.05)
        x = 1.79 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), -44190, rtol=0.05)
        x = 2.527 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), -12540, rtol=0.05)
        x = 2.92 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), 4338, rtol=0.05)

        # Test structure 3
        structure = get_test_structure(3)
        structure.assembled_matrix()
        structure.get_nodes_reactions()
        structure.get_nodes_displacements()

        for key, bar in structure.bars.items():
            bar.calculate_efforts()

        bars = structure.get_bars()
        b1 = bars.get("B1")

        self.assertRaises(ValueError, b1.bending_moment_law, -2)
        self.assertRaises(ValueError, b1.bending_moment_law, 2)

        # Bar 1
        bar = b1
        x = 0.3979 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), 95630, rtol=0.01)
        x = 0.7715 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), 142200, rtol=0.01)
        x = 1.269 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), 139200, rtol=0.01)
        x = 1.71 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), 74450, rtol=0.01)

        # Test structure 4
        structure = get_test_structure(4)
        structure.assembled_matrix()
        structure.get_nodes_reactions()
        structure.get_nodes_displacements()

        for key, bar in structure.bars.items():
            bar.calculate_efforts()

        bars = structure.get_bars()
        b1 = bars.get("B1")

        self.assertRaises(ValueError, b1.bending_moment_law, -2)
        self.assertRaises(ValueError, b1.bending_moment_law, 2)

        # Bar 1
        bar = b1
        x = 0.2847 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), 37960, rtol=0.01)
        x = 0.7347 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), 97960, rtol=0.01)
        x = 1.163 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), 138800, rtol=0.01)
        x = 1.699 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), 156600, rtol=0.01)
        x = 2.288 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), 118600, rtol=0.01)
        x = 2.706 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), 48980, rtol=0.01)

        # Test structure 5
        structure = get_test_structure(5)
        structure.assembled_matrix()
        structure.get_nodes_reactions()
        structure.get_nodes_displacements()

        for key, bar in structure.bars.items():
            bar.calculate_efforts()

        bars = structure.get_bars()
        b1 = bars.get("B1")

        self.assertRaises(ValueError, b1.bending_moment_law, -2)
        self.assertRaises(ValueError, b1.bending_moment_law, 2)

        # Bar 1
        bar = b1
        x = 1.983 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), 772400, rtol=0.01)
        x = 4.004 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), 952500, rtol=0.01)
        x = 5.998 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), 529800, rtol=0.01)
        x = 8.108 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), -345000, rtol=0.01)

        # Test structure 6
        structure = get_test_structure(6)
        structure.assembled_matrix()
        structure.get_nodes_reactions()
        structure.get_nodes_displacements()

        for key, bar in structure.bars.items():
            bar.calculate_efforts()

        bars = structure.get_bars()
        b1 = bars.get("B1")

        self.assertRaises(ValueError, b1.bending_moment_law, -2)
        self.assertRaises(ValueError, b1.bending_moment_law, 2)

        # Bar 1
        bar = b1
        x = 0.2839 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), -8992, rtol=0.01)
        x = 0.7339 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), 6478, rtol=0.01)
        x = 1.178 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), 12850, rtol=0.01)
        x = 1.634 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), 5720, rtol=0.01)

        # Test structure 7
        structure = get_test_structure(7)
        structure.assembled_matrix()
        structure.get_nodes_reactions()
        structure.get_nodes_displacements()

        for key, bar in structure.bars.items():
            bar.calculate_efforts()

        bars = structure.get_bars()
        b1 = bars.get("B1")

        self.assertRaises(ValueError, b1.bending_moment_law, -2)
        self.assertRaises(ValueError, b1.bending_moment_law, 2)

        # Bar 1
        bar = b1
        x = 0.2254 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), 11270, rtol=0.01)
        x = 0.8391 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), 41960, rtol=0.01)
        x = 1.337 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), 33150, rtol=0.01)
        x = 1.839 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), 8041, rtol=0.01)

        # Test structure 8
        structure = get_test_structure(8)
        structure.assembled_matrix()
        structure.get_nodes_reactions()
        structure.get_nodes_displacements()

        for key, bar in structure.bars.items():
            bar.calculate_efforts()

        bars = structure.get_bars()
        b1 = bars.get("B1")

        self.assertRaises(ValueError, b1.bending_moment_law, -2)
        self.assertRaises(ValueError, b1.bending_moment_law, 2)

        # Bar 1
        bar = b1
        x = 0.1388 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), -86120, rtol=0.01)
        x = 0.562 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), -43800, rtol=0.01)
        x = 1.453 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), 0, atol=0.003)
        x = 1.839 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), 0, atol=0.003)

        # Test structure 9
        structure = get_test_structure(9)
        structure.assembled_matrix()
        structure.get_nodes_reactions()
        structure.get_nodes_displacements()

        for key, bar in structure.bars.items():
            bar.calculate_efforts()

        bars = structure.get_bars()
        b1 = bars.get("B1")
        b2 = bars.get("B2")
        b3 = bars.get("B3")

        self.assertRaises(ValueError, b1.bending_moment_law, -2)
        self.assertRaises(ValueError, b1.bending_moment_law, 2)

        # Bar 1
        bar = b1
        x = 0.4054 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), -100600, rtol=0.01)
        x = 1.103 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), -18130, rtol=0.01)
        x = 2.093 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), 57090, rtol=0.01)
        x = 2.907 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), 82270, rtol=0.003)

        # Bar 2
        bar = b2
        x = 0.2161 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), 53130, rtol=0.01)
        x = 0.8415 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), -17260, rtol=0.01)
        x = 1.36 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), -32410, rtol=0.003)
        x = 2.32 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), -25260, rtol=0.003)

        # Bar 3
        bar = b3
        x = 0.4171 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), -34700, rtol=0.01)
        x = 1.231 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), -47990, rtol=0.01)
        x = 1.95 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), -43200, rtol=0.003)
        x = 2.665 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), -23070, rtol=0.003)

        # Test structure 10
        structure = get_test_structure(10)
        structure.assembled_matrix()
        structure.get_nodes_reactions()
        structure.get_nodes_displacements()

        for key, bar in structure.bars.items():
            bar.calculate_efforts()

        bars = structure.get_bars()
        b1 = bars.get("B1")

        self.assertRaises(ValueError, b1.bending_moment_law, -2)
        self.assertRaises(ValueError, b1.bending_moment_law, 2)

        # Bar 1
        bar = b1
        x = 0.3612 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), -43810, rtol=0.02)
        x = 1.348 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), 25210, rtol=0.02)
        x = 2.137 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), 45360, rtol=0.02)
        x = 3.022 / bar.length()
        np.testing.assert_allclose(bar.bending_moment_law(x), 30940, rtol=0.02)


class TestMaterial(unittest.TestCase):
    def test_constructor(self):
        mat = st.Material("s275j")

        self.assertEqual(mat.generic_name, "steel")
        self.assertEqual(mat.name, "s275j")
        self.assertEqual(mat.young_mod, 205939650000)
        self.assertEqual(mat.rig_mod, 81000000000)
        self.assertEqual(mat.poisson_coef, 0.3)
        self.assertEqual(mat.thermal_dil_coef, 0.000012)
        self.assertEqual(mat.density, 7850)

        self.assertRaises(TypeError, st.Material, 4)
        self.assertRaises(LookupError, st.Material, "foobar")


class TestProfile(unittest.TestCase):
    def test_constructor(self):
        pro = st.Profile("IPE", 80)

        self.assertEqual(pro.name, "IPE")
        self.assertEqual(pro.name_number, 80)
        self.assertEqual(pro.area, 0.000764)
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
        self.assertEqual(dc.direction, (0, -1, 0))

        self.assertRaises(TypeError, st.DistributedCharge, 2, 20)

    def test_equals(self):
        # TODO añadir más tests cuando se tengan distintos tipos de carga distribuida
        dc1 = st.DistributedCharge(st.DistributedChargeType.SQUARE, 10, (0, -1, 0))
        dc2 = st.DistributedCharge(st.DistributedChargeType.SQUARE, 20, (0, -1, 0))
        dc3 = st.DistributedCharge(st.DistributedChargeType.SQUARE, 10, (0, 1, 0))

        self.assertRaises(TypeError, dc1.equals, 3)
        self.assertTrue(dc1.equals(dc1))
        self.assertFalse(dc1.equals(dc2))
        self.assertFalse(dc1.equals(dc3))

    def test_set_dc_type(self):
        dc = st.DistributedCharge(
            st.DistributedChargeType.SQUARE, 10, (0, 1, 0)
        )

        self.assertEqual(dc.dc_type, st.DistributedChargeType.SQUARE)
        self.assertNotEqual(dc.dc_type, st.DistributedChargeType.PARALLEL_TO_BAR)

        dc.set_dc_type(st.DistributedChargeType.PARALLEL_TO_BAR)

        self.assertEqual(dc.dc_type, st.DistributedChargeType.PARALLEL_TO_BAR)
        self.assertNotEqual(dc.dc_type, st.DistributedChargeType.SQUARE)

    def test_set_max_value(self):
        dc = st.DistributedCharge(
            st.DistributedChargeType.SQUARE, 10, (0, 1, 0)
        )

        self.assertEqual(dc.max_value, 10)
        self.assertNotEqual(dc.max_value, 2)

        dc.set_max_value(20)

        self.assertEqual(dc.max_value, 20)
        self.assertNotEqual(dc.max_value, 2)

    def test_set_direction(self):
        dc = st.DistributedCharge(
            st.DistributedChargeType.SQUARE, 10, (0, 1, 0)
        )

        self.assertEqual(dc.direction, (0, 1, 0))
        self.assertNotEqual(dc.direction, (0, 1, 1))

        dc.set_direction((1, 0, 0))

        self.assertEqual(dc.direction, (1, 0, 0))
        self.assertNotEqual(dc.direction, (1, 1, 0))

    # def test_axial_force_law(self):
    #     structure = get_test_structure(1)
    #
    #     # b2 length = 6.8806
    #     bar = structure.get_bars().get("B2")
    #     distributed_charges = bar.get_distributed_charges()
    #
    #     for key, item in distributed_charges.items():
    #         dc = item
    #
    #     np.testing.assert_allclose(dc.axial_force_law(bar, 0.5), 0, rtol=0.02)
    #
    #     # b3 length = 6.8806
    #     bar = structure.get_bars().get("B3")
    #     distributed_charges = bar.get_distributed_charges()
    #
    #     for key, item in distributed_charges.items():
    #         dc = item
    #
    #     np.testing.assert_allclose(dc.axial_force_law(bar, 0.5), 0, rtol=0.02)
    #
    #     # b4 length = 3.6427
    #     bar = structure.get_bars().get("B4")
    #     distributed_charges = bar.get_distributed_charges()
    #
    #     for key, item in distributed_charges.items():
    #         dc = item
    #
    #     np.testing.assert_allclose(dc.axial_force_law(bar, 0.5), 0, rtol=0.02)
    #
    # def test_bending_moment_law(self):
    #     structure = get_test_structure(1)
    #
    #     structure.assembled_matrix()
    #     structure.get_nodes_reactions()
    #     structure.get_nodes_displacements()
    #     for key, bar in structure.get_bars().items():
    #         bar.calculate_efforts()
    #
    #     # Bar 2
    #     bar = structure.get_bars().get("B2")
    #     distributed_charges = bar.get_distributed_charges()
    #
    #     for key, item in distributed_charges.items():
    #         dc = item
    #
    #     np.testing.assert_allclose(dc.bending_moment_law(bar, 0), 0, rtol=0.02)
    #     np.testing.assert_allclose(dc.bending_moment_law(bar, 0.1), -2381.28, rtol=0.02)
    #     np.testing.assert_allclose(dc.bending_moment_law(bar, 0.3), -21432.45, rtol=0.02)
    #     np.testing.assert_allclose(dc.bending_moment_law(bar, 0.4), -38102.13, rtol=0.02)
    #     np.testing.assert_allclose(dc.bending_moment_law(bar, 0.5), -59534.57, rtol=0.02)
    #     np.testing.assert_allclose(dc.bending_moment_law(bar, 0.7), -116687.76, rtol=0.02)
    #     np.testing.assert_allclose(dc.bending_moment_law(bar, 0.9), -192890.02, rtol=0.02)
    #     np.testing.assert_allclose(dc.bending_moment_law(bar, 1), -238135.3, rtol=0.02)
    #
    #     self.assertRaises(ValueError, dc.bending_moment_law, bar, -1)
    #     self.assertRaises(ValueError, dc.bending_moment_law, bar, 2)
    #
    #     # TODO Añadir más barras al test
    #
    # def test_shear_strength_law(self):
    #     structure = get_test_structure(1)
    #
    #     # b2 length = 6.8806
    #     bar = structure.get_bars().get("B2")
    #     distributed_charges = bar.get_distributed_charges()
    #
    #     for key, item in distributed_charges.items():
    #         dc = item
    #
    #     np.testing.assert_allclose(dc.shear_strength_law(bar, 0.5), 34609.76, rtol=0.02)
    #
    #     # b3 length = 6.8806
    #     bar = structure.get_bars().get("B3")
    #     distributed_charges = bar.get_distributed_charges()
    #
    #     for key, item in distributed_charges.items():
    #         dc = item
    #
    #     np.testing.assert_allclose(dc.shear_strength_law(bar, 0.5), 34609.76, rtol=0.02)
    #
    #     # b4 length = 3.6427
    #     bar = structure.get_bars().get("B4")
    #     distributed_charges = bar.get_distributed_charges()
    #
    #     for key, item in distributed_charges.items():
    #         dc = item
    #
    #     np.testing.assert_allclose(dc.shear_strength_law(bar, 0.5), 18322.96, rtol=0.02)


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

    def test_set_value(self):
        pf = st.PunctualForceInBar(10, .5, (0, 1, 0))

        self.assertEqual(pf.value, 10)
        self.assertNotEqual(pf.value, 11)

        pf.set_value(99)
        self.assertEqual(pf.value, 99)
        self.assertNotEqual(pf.value, 0)

    def test_set_origin_end_factor(self):
        pf = st.PunctualForceInBar(10, .5, (0, 1, 0))

        self.assertEqual(pf.origin_end_factor, .5)
        self.assertNotEqual(pf.origin_end_factor, .3)

        pf.set_origin_end_factor(.3)
        self.assertEqual(pf.origin_end_factor, .3)
        self.assertNotEqual(pf.origin_end_factor, .4)

    def test_set_direction(self):
        pf = st.PunctualForceInBar(10, .5, (0, 1, 0))

        self.assertEqual(pf.direction, (0, 1, 0))
        self.assertNotEqual(pf.direction, (1, 1, 0))

        pf.set_direction((1, 0, 0))
        self.assertEqual(pf.direction, (1, 0, 0))
        self.assertNotEqual(pf.direction, (1, 1, 0))

    # def test_bending_moment_law(self):
    #     structure = get_test_structure(1)
    #
    #     structure.assembled_matrix()
    #     structure.get_nodes_reactions()
    #     structure.get_nodes_displacements()
    #     for key, bar in structure.get_bars().items():
    #         bar.calculate_efforts()
    #
    #     # Bar 5
    #     bar = structure.get_bars().get("B5")
    #     punctual_forces = bar.get_punctual_forces()
    #
    #     for key, item in punctual_forces.items():
    #         pf = item
    #
    #     np.testing.assert_allclose(pf.bending_moment_law(bar, 0), 0, rtol=0.02)
    #     np.testing.assert_allclose(pf.bending_moment_law(bar, 0.1), 0, rtol=0.02)
    #     np.testing.assert_allclose(pf.bending_moment_law(bar, 0.3), 0, rtol=0.02)
    #     np.testing.assert_allclose(pf.bending_moment_law(bar, 0.4), 0, rtol=0.02)
    #     np.testing.assert_allclose(pf.bending_moment_law(bar, 0.5), 0, rtol=0.02)
    #     np.testing.assert_allclose(pf.bending_moment_law(bar, 0.7), -21000, rtol=0.02)
    #     np.testing.assert_allclose(pf.bending_moment_law(bar, 0.9), -42000, rtol=0.02)
    #     np.testing.assert_allclose(pf.bending_moment_law(bar, 1), -52500, rtol=0.02)
    #
    #     self.assertRaises(ValueError, pf.bending_moment_law, bar, -1)
    #     self.assertRaises(ValueError, pf.bending_moment_law, bar, 2)
    #
    #     # TODO Añadir más barras al test

    # def test_shear_strength_law(self):
    #     n1 = st.Node("N1", (0, 0, 0))
    #     n2 = st.Node("N2", (2, 0, 0))
    #
    #     b1 = st.Bar("B1", n1, n2, "s275j", ("IPE", 300))
    #     self.assertEqual(b1.length(), 2)
    #
    #     # Test 1
    #     origin_force_distance = 0.4
    #     pf = st.PunctualForceInBar(100, origin_force_distance, (0, -1, 0))
    #
    #     np.testing.assert_allclose(pf.shear_strength_law(b1, 0), 0, rtol=0.02)
    #     np.testing.assert_allclose(pf.shear_strength_law(b1, 0.2), 0, rtol=0.02)
    #     np.testing.assert_allclose(pf.shear_strength_law(b1, 0.4), 0, rtol=0.02)
    #     np.testing.assert_allclose(pf.shear_strength_law(b1, 0.6), 100, rtol=0.02)
    #     np.testing.assert_allclose(pf.shear_strength_law(b1, 0.8), 100, rtol=0.02)
    #     np.testing.assert_allclose(pf.shear_strength_law(b1, 1), 100, rtol=0.02)
    #
    #     # Test 2
    #     origin_force_distance = 0.7
    #     pf = st.PunctualForceInBar(100, origin_force_distance, (0, -1, 0))
    #
    #     np.testing.assert_allclose(pf.shear_strength_law(b1, 0), 0, rtol=0.02)
    #     np.testing.assert_allclose(pf.shear_strength_law(b1, 0.2), 0, rtol=0.02)
    #     np.testing.assert_allclose(pf.shear_strength_law(b1, 0.4), 0, rtol=0.02)
    #     np.testing.assert_allclose(pf.shear_strength_law(b1, 0.6), 0, rtol=0.02)
    #     np.testing.assert_allclose(pf.shear_strength_law(b1, 0.8), 100, rtol=0.02)
    #     np.testing.assert_allclose(pf.shear_strength_law(b1, 1), 100, rtol=0.02)
    #
    #     self.assertRaises(ValueError, pf.bending_moment_law, b1.length(), -1)
    #     self.assertRaises(ValueError, pf.bending_moment_law, b1.length(), 2)


if __name__ == '__main__':
    unittest.main()
