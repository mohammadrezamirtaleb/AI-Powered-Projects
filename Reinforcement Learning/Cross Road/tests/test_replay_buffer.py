import unittest
import numpy as np
from src.ai.dqn_agent import SumTree, PrioritizedReplayBuffer

class TestSumTreeAndPER(unittest.TestCase):
    def test_sumtree_basic_add_and_total(self):
        tree = SumTree(capacity=4)
        tree.add(1.0, "data1")
        tree.add(2.0, "data2")
        tree.add(3.0, "data3")
        self.assertAlmostEqual(tree.total_priority, 6.0, places=5)
        self.assertEqual(tree.size, 3)

    def test_sumtree_sampling(self):
        tree = SumTree(capacity=4)
        tree.add(1.0, "data1")
        tree.add(10.0, "data2")
        idx, priority, data = tree.get_leaf(5.0)
        self.assertEqual(data, "data2")

    def test_per_push_and_sample(self):
        buf = PrioritizedReplayBuffer(capacity=100)
        state_dummy = np.zeros(87, dtype=np.float32)
        for i in range(50):
            buf.push(state_dummy, 0, 1.0, state_dummy, False)

        self.assertEqual(len(buf), 50)
        batch = buf.sample(batch_size=16)
        self.assertIsNotNone(batch)
        states, actions, rewards, next_states, dones, indices, weights = batch
        self.assertEqual(len(states), 16)
        self.assertEqual(len(weights), 16)

        # Test update priorities with clipping
        errors = np.full(16, 50.0) # extreme error to test clipping
        buf.update_priorities(indices, errors)
        self.assertLessEqual(buf.max_priority, 10.0)

if __name__ == '__main__':
    unittest.main()
