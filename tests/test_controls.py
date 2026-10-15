import unittest
from controls import ACTIONS, Feedback

class ControlsTests(unittest.TestCase):
    def test_feedback_is_consumed_once(self):
        feedback = Feedback()
        feedback.add(1)
        feedback.add(-0.25)
        feedback.done = True
        self.assertEqual(feedback.consume(), (0.75, True))
        self.assertEqual(feedback.consume(), (0.0, False))

    def test_actions_never_hold_opposite_directions(self):
        self.assertTrue(any('jump' in action for action in ACTIONS))
        self.assertTrue(any('light' in action for action in ACTIONS))
        for action in ACTIONS:
            self.assertFalse({'left', 'right'} <= set(action))

if __name__ == '__main__':
    unittest.main()
