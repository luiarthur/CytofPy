import unittest

class Test_Something(unittest.TestCase):
    def approx(self, a, b, eps=1E-8):
        self.assertTrue(abs(a - b) < eps)

    def test_something(self):
        self.assertTrue(1 + 1 == 2)

