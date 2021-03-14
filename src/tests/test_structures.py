# Run this file with python3 -m unittest
# Otherwise it will fail due to imports

import unittest
from modules import structures as st


class TestNode(unittest.TestCase):
    # TODO name should be an str
    def test_set_name(self):
        node = st.Node("N1", 0, 0, 0)

        name = node.set_name("node")
        self.assertEqual("node", node.name)
        self.assertNotEqual("not_node", node.name)

    # TODO force should be a tuple of int or float
    def test_set_force(self):
        node = st.Node("N1", 0, 0, 0, force=(1,1,1))

        node.set_force((2, 2, 2))
        self.assertEqual((2, 2, 2), node.force)
        self.assertNotEqual((1, 1, 1), node.force)


if __name__ == '__main__':
    unittest.main()
