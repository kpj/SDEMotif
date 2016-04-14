from unittest import TestCase

import io

from reaction_finder import *


class TestMatcher(TestCase):
    def test_simple_match(self):
        cgroups = {'foo': 3, 'bar': 1}
        rspec = {'foo': 2, 'bar': 1}

        self.assertTrue(match(cgroups, rspec))

    def test_simple_mismatch(self):
        cgroups = {'foo': 1, 'bar': 1}
        rspec = {'foo': 2, 'bar': 1}

        self.assertFalse(match(cgroups, rspec))

class TestPairFinder(TestCase):
    def test_simple_finder(self):
        cdata = {
            'fooC': {'H': 3, 'O': 2},
            'barC': {'H': 4, 'O': 1},
            'bazC': {'H': 0, 'O': 3}
        }
        rdata = {
            'rea1': {
                'c1': {'H': 3, 'O': 2},
                'c2': {'H': 4, 'O': 0},
                'res': {'H': -1, 'O': 2}
            },
            'rea2': {
                'c1': {'H': 0, 'O': 1},
                'c2': {'H': 1, 'O': 1},
                'res': {'H': 1, 'O': 0}
            },
            'rea3': {
                'c1': {'H': 4, 'O': 0},
                'c2': None,
                'res': {'H': -2, 'O': 1}
            }
        }

        res = check_pair('fooC', 'barC', cdata, rdata)
        self.assertEqual(sorted(res), ['rea1', 'rea2'])

        res = check_pair('barC', 'fooC', cdata, rdata)
        self.assertEqual(res, ['rea2'])

        res = check_pair('fooC', 'bazC', cdata, rdata)
        self.assertEqual(res, [])

        res = check_pair('bazC', 'fooC', cdata, rdata)
        self.assertEqual(res, ['rea2'])

        res = check_pair('barC', 'bazC', cdata, rdata)
        self.assertEqual(res, [])

        res = check_pair('bazC', 'barC', cdata, rdata)
        self.assertEqual(res, ['rea2'])

        res = check_pair('barC', None, cdata, rdata)
        self.assertEqual(res, ['rea3'])

class TestCompoundGuesser(TestCase):
    def test_simple_generation(self):
        cdata = {
            'fooC': {'H': 3, 'O': 2},
            'barC': {'H': 4, 'O': 1},
            'bazC': {'H': 0, 'O': 3}
        }
        rdata = {
            'rea1': {
                'c1': {'H': 3, 'O': 2},
                'c2': {'H': 4, 'O': 0},
                'res': {'H': -1, 'O': 2}
            },
            'rea2': {
                'c1': {'H': 0, 'O': 1},
                'c2': {'H': 1, 'O': 1},
                'res': {'H': 1, 'O': 0}
            }
        }
        combs = {
            'rea1': [('fooC', 'barC')]
        }

        res = guess_new_compounds(combs, cdata, rdata)

        self.assertTrue(len(res), 1)
        self.assertEqual(res['(fooC) rea1 (barC)'], {'H': 6, 'O': 5})

    def test_guess_with_none(self):
        cdata = {
            'fooC': {'H': 3, 'O': 2},
        }
        rdata = {
            'rea1': {
                'c1': {'H': 3, 'O': 2},
                'c2': None,
                'res': {'H': -2, 'O': 0}
            }
        }
        combs = {
            'rea1': [('fooC', None)]
        }

        res = guess_new_compounds(combs, cdata, rdata)

        self.assertTrue(len(res), 1)
        self.assertEqual(res['(fooC) rea1 (None)'], {'H': 1, 'O': 2})

    def test_negative_group_number(self):
        cdata = {
            'fooC': {'H': 1, 'O': 1},
            'barC': {'H': 1, 'O': 1},
        }
        rdata = {
            'rea1': {
                'c1': {'H': 0, 'O': 0},
                'c2': {'H': 0, 'O': 0},
                'res': {'H': -4, 'O': -3}
            }
        }
        combs = {
            'rea1': [('fooC', 'barC')]
        }

        res = guess_new_compounds(combs, cdata, rdata)

        self.assertTrue(len(res), 1)
        self.assertEqual(res['(fooC) rea1 (barC)'], {'H': -2, 'O': -1})

class TestFileInput(TestCase):
    def test_compound_reader(self):
        data = read_compounds_file('./tests/data/compounds.csv')

        self.assertEqual(len(data), 3)
        self.assertEqual(
            data['foo'], {
                '-H': 1,
                '-O': 4
            })
        self.assertEqual(
            data['bar'], {
                '-H': 5,
                '-O': 0
            })
        self.assertEqual(
            data['baz'], {
                '-H': 3,
                '-O': 3
            })

    def test_reaction_reader(self):
        data = read_reactions_file('./tests/data/reactions.csv')

        self.assertEqual(len(data), 3)
        self.assertEqual(
            data['rea1'], {
                'c1': {'-H': 3, '-O': 2},
                'c2': {'-H': 4, '-O': 0},
                'res': {'-H': -1, '-O': 2}
            })
        self.assertEqual(
            data['rea2'], {
                'c1': {'-H': 0, '-O': 1},
                'c2': {'-H': 1, '-O': 1},
                'res': {'-H': 1, '-O': 0}
            })
        self.assertEqual(
            data['rea3'], {
                'c1': {'-H': 2, '-O': 3},
                'c2': None,
                'res': {'-H': -2, '-O': -1}
            })

class IntegrationTest(TestCase):
    def setUp(self):
        self.compounds = io.StringIO("""Name,-H,-O,-N
c1,1,2,3
c2,2,1,3
""")
        self.reactions = io.StringIO("""Reaction,Requirement Matrix - Compound 1,,,Requirement Matrix - Compound 2,,,Result Matrix,,,Transformation
  ,-H,-O,-N,-H,-O,-N,-H,-O,-N,,
r1, 1, 2, 3, 2, 1, 3, 1, 1,-6,,
r2, 4, 4, 0, 1, 2, 0, 1, 0,-1,,
r3, 4, 4, 0, X,  ,  , 0, 0, 1,,
""")

    def test_interactions(self):
        comps = read_compounds_file(self.compounds)
        reacts = read_reactions_file(self.reactions)

        # first iteration
        res = iterate_once(comps, reacts)
        self.assertEqual(len(res), 1)
        self.assertIn('(c1) r1 (c2)', res)
        self.assertEqual(res['(c1) r1 (c2)'], {'-H': 4, '-O': 4, '-N': 0})

        # second iteration
        comps.update(res)
        nres = iterate_once(comps, reacts)

        self.assertEqual(len(nres), 4)
        self.assertIn('(c1) r1 (c2)', nres)
        self.assertIn('((c1) r1 (c2)) r2 (c1)', nres)
        self.assertIn('((c1) r1 (c2)) r3 (None)', nres)
        self.assertIn('((c1) r1 (c2)) r2 ((c1) r1 (c2))', nres)
        self.assertEqual(nres['(c1) r1 (c2)'], {'-H': 4, '-O': 4, '-N': 0})
        self.assertEqual(nres['((c1) r1 (c2)) r2 (c1)'], {'-H': 6, '-O': 6, '-N': 2})
        self.assertEqual(nres['((c1) r1 (c2)) r3 (None)'], {'-H': 4, '-O': 4, '-N': 1})
        self.assertEqual(nres['((c1) r1 (c2)) r2 ((c1) r1 (c2))'], {'-N': -1, '-O': 8, '-H': 9})
