#!/usr/bin/env python
# encoding: utf-8
###############################################################################
#                                                                             #
#    kESI                                                                     #
#                                                                             #
#    Copyright (C) 2019 Jakub M. Dzik (Laboratory of Neuroinformatics;        #
#    Nencki Institute of Experimental Biology of Polish Academy of Sciences)  #
#                                                                             #
#    This software is free software: you can redistribute it and/or modify    #
#    it under the terms of the GNU General Public License as published by     #
#    the Free Software Foundation, either version 3 of the License, or        #
#    (at your option) any later version.                                      #
#                                                                             #
#    This software is distributed in the hope that it will be useful,         #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of           #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            #
#    GNU General Public License for more details.                             #
#                                                                             #
#    You should have received a copy of the GNU General Public License        #
#    along with this software.  If not, see http://www.gnu.org/licenses/.     #
#                                                                             #
###############################################################################

import unittest

from kesi._engine import LinearMixture

try:
    from ._common import Stub
    # When run as script raises:
    #  - `ModuleNotFoundError(ImportError)` (Python 3.6-7), or
    #  - `SystemError` (Python 3.3-5), or
    #  - `ValueError` (Python 2.7).

except (ImportError, SystemError, ValueError):
    from _common import Stub

import kesi as kesi

class _MixtureTestBase(unittest.TestCase):
    def setUp(self):
        try:
            init = self.INIT
        except AttributeError:
            self.skipTest('Method of abstract class called.')
        else:
            self.mixture = LinearMixture(*init)

    @property
    def INIT(self):
        return [(Stub(**ss)
                 if isinstance(ss, dict)
                 else [(Stub(**s), w)
                       for w, s in ss])
                for ss in self._INIT]

    def testNotEqualsOne(self):
        self.assertFalse(self.mixture == 1)
        self.assertTrue(self.mixture != 1)

    def testIsReflexive(self):
        self.assertTrue(self.mixture == self.mixture)
        self.assertFalse(self.mixture != self.mixture)

    def testNonexistingAttributeAccessRaisesAttributeError(self):
        mixture = self.mixture
        with self.assertRaises(AttributeError):
            mixture.nonexisting_attribute

    def testMultipliedByOneIsSelf(self):
        self.assertIs(self.mixture, self.mixture * 1)
        self.assertIs(self.mixture * 1, self.mixture)

    def testDividedByOneIsSelf(self):
        self.assertIs(self.mixture, self.mixture / 1)

    def testDividedByZeroRaisesZeroDivisionError(self):
        with self.assertRaises(ZeroDivisionError):
            self.mixture / 0

    def testSummedWithZeroIsSelf(self):
        self.assertIs(self.mixture, self.mixture + 0)
        self.assertIs(self.mixture, 0 + self.mixture)

    def testSubtractedZeroIsSelf(self):
        self.assertIs(self.mixture, self.mixture - 0)

    def testDir(self):
        for attr in self.DIR:
            self.assertIn(attr, dir(self.mixture))

    def testHasAttr(self):
        for attr in self.DIR:
            self.assertTrue(hasattr(self.mixture, attr))


class _GivenEmptyMixture(_MixtureTestBase):
    DIR = []

    def testEqualsZero(self):
        self.assertTrue(self.mixture == 0)
        self.assertFalse(self.mixture != 0)

    def testIsFalse(self):
        self.assertFalse(bool(self.mixture))

    def testMultipliedIsSelf(self):
        a = 1337
        self.assertIs(self.mixture,
                      self.mixture * a)
        self.assertIs(self.mixture,
                      a * self.mixture)


class GivenNoComponents(_GivenEmptyMixture):
    _INIT = []


class GivenEmptyComponentSequence(_GivenEmptyMixture):
    _INIT = [[]]


class _GivenFilledMixture(_MixtureTestBase):
    def testNotEqualsZero(self):
        self.assertFalse(self.mixture == 0)
        self.assertTrue(self.mixture != 0)

    def testIsTrue(self):
        self.assertTrue(bool(self.mixture))


class GivenSingleComponentWithoutMethods(_GivenFilledMixture):
    _INIT = [{}]
    DIR = []


class _GivenSingleComponentWithMethods(_GivenFilledMixture):
    METHODS = {'f': lambda x: 1,
               'g': lambda x: x,
               }
    DIR = METHODS

    def setUp(self):
        super(_GivenSingleComponentWithMethods, self).setUp()
        if not hasattr(self, 'SCALE_FACTOR'):
            self.skipTest('Abstract class')

    def testMethodsAreScaled(self):
        self.checkScalesProperly(self.mixture,
                                 self.SCALE_FACTOR)

    def testScalesWithMultiplication(self):
        a = 2
        self.checkScalesProperly(a * self.mixture,
                                 a * self.SCALE_FACTOR)
        self.checkScalesProperly(self.mixture * a,
                                 a * self.SCALE_FACTOR)

    def testIsAdditive(self):
        self.checkScalesProperly(self.mixture + 2 * self.mixture,
                                 3 * self.SCALE_FACTOR)

    def testCanBeDivided(self):
        self.checkScalesProperly((self.mixture + self.mixture) / 2,
                                 self.SCALE_FACTOR)

    def checkScalesProperly(self, mixture, scale_factor):
        for name, f in self.METHODS.items():
            for x in [0, 1]:
                self.assertEqual(scale_factor * f(x),
                                 getattr(mixture, name)(x))


class _GivenIdentityMixture(_GivenSingleComponentWithMethods):
    SCALE_FACTOR = 1


class GivenSingleComponentWithoutScaling(_GivenIdentityMixture):
    _INIT = [_GivenIdentityMixture.METHODS]


class GivenSingleComponentWithIdentityScaling(_GivenIdentityMixture):
    _INIT = [[(1, _GivenIdentityMixture.METHODS)]]


class GivenSimpleScaling(_GivenSingleComponentWithMethods):
    SCALE_FACTOR = 2

    _INIT = [[(SCALE_FACTOR,
               _GivenSingleComponentWithMethods.METHODS)]]


class GivenMixtureOfTwoSources(_GivenFilledMixture):
    METHODS_A = {'f': lambda x, y: x,
                 'a_only': lambda x: 1,
                 }
    METHODS_B = {'f': lambda x, y: y,
                 'b_only': lambda x: -1,
                 }

    DIR = ['f']
    SCALE_FACTOR_A = 2
    SCALE_FACTOR_B = 1
    _INIT = [[(SCALE_FACTOR_A, METHODS_A),
              (SCALE_FACTOR_B, METHODS_B)]]

    def setUp(self):
        super(GivenMixtureOfTwoSources, self).setUp()
        self.mixture_a = LinearMixture([(Stub(**self.METHODS_A),
                                         self.SCALE_FACTOR_A)])
        self.mixture_b = LinearMixture([(Stub(**self.METHODS_B),
                                         self.SCALE_FACTOR_B)])

    def testRaisesAttributeErrorWhenAccessedAttributeNotPresentInAllSources(self):
        for attr in ['a_only', 'b_only']:
            with self.assertRaises(AttributeError):
                getattr(self.mixture, attr)

    def testCombinesSourcesProperly(self):
        for x in [0, 1]:
            for y in [0, 1]:
                self.assertEqual((self.mixture_a.f(x, y) + self.mixture_b.f(x, y)),
                                 self.mixture.f(x, y=y))

    def testIsAdditive(self):
        mixture_ab = self.mixture_a + self.mixture_b
        for x in [0, 1]:
            for y in [0, 1]:
                self.assertEqual(self.mixture.f(x, y=y),
                                 mixture_ab.f(x, y))

    def testHasAdditiveInverse(self):
        mixture_a = self.mixture + -self.mixture_b
        for x in [0, 1]:
            for y in [0, 1]:
                self.assertEqual(self.mixture_a.f(x, y=y),
                                 mixture_a.f(x, y))

    def testIsSubtractive(self):
        mixture_a = self.mixture - self.mixture_b
        for x in [0, 1]:
            for y in [0, 1]:
                self.assertEqual(self.mixture_a.f(x, y=y),
                                 mixture_a.f(x, y))


class TestPerformance(_MixtureTestBase):
    DIR = ['f']

    def setUp(self):
        self.call_counter = 0

        def f():
            self.call_counter += 1
            return 0

        self._INIT = [[(1, {'f': f})]]
        super(TestPerformance, self).setUp()

    def testSumCallsMethodOnce(self):
        mixture = self.mixture + self.mixture
        mixture.f()
        self.assertEqual(1, self.call_counter)

    def testMixtureOfMixturesCallsMethodOnce(self):
        mixture = LinearMixture([(self.mixture, 1),
                                 (self.mixture, 1)])
        mixture.f()
        self.assertEqual(1, self.call_counter)


if __name__ == '__main__':
    unittest.main()
