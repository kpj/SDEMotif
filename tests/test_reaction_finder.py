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
            'fooC': {'groups': {'H': 3, 'O': 2}},
            'barC': {'groups': {'H': 4, 'O': 1}},
            'bazC': {'groups': {'H': 0, 'O': 3}}
        }
        rdata = {
            'rea1': {
                'c1': {'H': 3, 'O': 2},
                'c2': {'H': 4, 'O': 0},
                'group_trans': {'H': -1, 'O': 2},
                'mass_trans': 1
            },
            'rea2': {
                'c1': {'H': 0, 'O': 1},
                'c2': {'H': 1, 'O': 1},
                'group_trans': {'H': 1, 'O': 0},
                'mass_trans': 1
            },
            'rea3': {
                'c1': {'H': 4, 'O': 0},
                'c2': None,
                'group_trans': {'H': -2, 'O': 1},
                'mass_trans': 1
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
            'fooC': {'groups': {'H': 3, 'O': 2}, 'mass': 1, 'atoms': {'N': 1, 'C': 1}},
            'barC': {'groups': {'H': 4, 'O': 1}, 'mass': 1, 'atoms': {'N': 3, 'C': 0}},
            'bazC': {'groups': {'H': 0, 'O': 3}, 'mass': 1, 'atoms': {'N': 1, 'C': 2}}
        }
        rdata = {
            'rea1': {
                'c1': {'H': 3, 'O': 2},
                'c2': {'H': 4, 'O': 0},
                'group_trans': {'H': -1, 'O': 2},
                'mass_trans': 1,
                'atom_trans': {'c1': True, 'c2': True, 'N': 1, 'C': -2}
            },
            'rea2': {
                'c1': {'H': 0, 'O': 1},
                'c2': {'H': 1, 'O': 1},
                'group_trans': {'H': 1, 'O': 0},
                'mass_trans': 1,
                'atom_trans': {'c1': True, 'c2': True, 'N': 0, 'C': 0}
            }
        }
        combs = {
            'rea1': [('fooC', 'barC')]
        }

        res = guess_new_compounds(combs, cdata, rdata)

        self.assertTrue(len(res), 1)
        self.assertEqual(
            res['(fooC) {rea1} (barC)']['groups'],
            {'H': 6, 'O': 5})
        self.assertEqual(res['(fooC) {rea1} (barC)']['mass'], 3)
        self.assertEqual(
            res['(fooC) {rea1} (barC)']['atoms'],
            {'N': 5, 'C': -1})

    def test_guess_with_none(self):
        cdata = {
            'fooC': {
                'groups': {'H': 3, 'O': 2},
                'mass': 2,
                'atoms': {'N': 4, 'C': 3}
            },
        }
        rdata = {
            'rea1': {
                'c1': {'H': 3, 'O': 2},
                'c2': None,
                'group_trans': {'H': -2, 'O': 0},
                'mass_trans': 1,
                'atom_trans': {'c1': True, 'c2': False, 'N': -2, 'C': 1}
            }
        }
        combs = {
            'rea1': [('fooC', None)]
        }

        res = guess_new_compounds(combs, cdata, rdata)

        self.assertTrue(len(res), 1)
        self.assertEqual(
            res['(fooC) {rea1} (None)']['groups'],
            {'H': 1, 'O': 2})
        self.assertEqual(res['(fooC) {rea1} (None)']['mass'], 3)
        self.assertEqual(
            res['(fooC) {rea1} (None)']['atoms'],
            {'N': 2, 'C': 4})

    def test_negative_group_number(self):
        cdata = {
            'fooC': {'groups': {'H': 1, 'O': 1}, 'mass': -4},
            'barC': {'groups': {'H': 1, 'O': 1}, 'mass': -4},
        }
        rdata = {
            'rea1': {
                'c1': {'H': 0, 'O': 0},
                'c2': {'H': 0, 'O': 0},
                'group_trans': {'H': -4, 'O': -3},
                'mass_trans': 6,
                'atom_trans': {'c1': False, 'c2': False}
            }
        }
        combs = {
            'rea1': [('fooC', 'barC')]
        }

        res = guess_new_compounds(combs, cdata, rdata)

        self.assertTrue(len(res), 1)
        self.assertEqual(
            res['(fooC) {rea1} (barC)']['groups'],
            {'H': -2, 'O': -1})
        self.assertEqual(res['(fooC) {rea1} (barC)']['mass'], -2)

class TestFileInput(TestCase):
    def test_compound_reader(self):
        data = read_compounds_file('./tests/data/compounds.csv')

        self.assertEqual(len(data), 3)

        self.assertEqual(
            data['foo']['groups'], {
                '-H': 1,
                '-O': 4
            })
        self.assertEqual(
            data['bar']['groups'], {
                '-H': 5,
                '-O': 0
            })
        self.assertEqual(
            data['baz']['groups'], {
                '-H': 3,
                '-O': 3
            })

        self.assertEqual(
            data['foo']['atoms'], {
                'N': 0,
                'C': 5
            })
        self.assertEqual(
            data['bar']['atoms'], {
                'N': 2,
                'C': 3
            })
        self.assertEqual(
            data['baz']['atoms'], {
                'N': 1,
                'C': 1
            })

        self.assertEqual(data['foo']['mass'], 1.5)
        self.assertEqual(data['bar']['mass'], 1.7)
        self.assertEqual(data['baz']['mass'], 1.1)

    def test_reaction_reader(self):
        data = read_reactions_file('./tests/data/reactions.csv')

        self.assertEqual(len(data), 3)
        self.assertEqual(
            data['rea1'], {
                'c1': {'-H': 3, '-O': 2},
                'c2': {'-H': 4, '-O': 0},
                'group_trans': {'-H': -1, '-O': 2},
                'mass_trans': -1.1,
                'atom_trans': {'c1': True, 'c2': True, 'H': -2, 'O': -1}
            })
        self.assertEqual(
            data['rea2'], {
                'c1': {'-H': 0, '-O': 1},
                'c2': {'-H': 1, '-O': 1},
                'group_trans': {'-H': 1, '-O': 0},
                'mass_trans': 2.2,
                'atom_trans': {'c1': True, 'c2': False, 'O': 1}
            })
        self.assertEqual(
            data['rea3'], {
                'c1': {'-H': 2, '-O': 3},
                'c2': None,
                'group_trans': {'-H': -2, '-O': -1},
                'mass_trans': -3.3,
                'atom_trans': {'c1': True, 'c2': False, 'H': -2, 'O': -1}
            })

class IntegrationTest(TestCase):
    def setUp(self):
        self.compounds = io.StringIO("""Name,-H,-O,-N,M-H
c1,1,2,3,1.2
c2,2,1,3,2.3
""")
        self.reactions = io.StringIO("""Reaction,Requirement Matrix - Compound 1,,,Requirement Matrix - Compound 2,,,Result Matrix,,,Transformation,Mass Addendum
  ,-H,-O,-N,-H,-O,-N,-H,-O,-N,,
r1, 1, 2, 3, 2, 1, 3, 1, 1,-6,,1.1
r2, 4, 4, 0, 1, 2, 0, 1, 0,-1,,-2.2
r3, 4, 4, 0, X,  ,  , 0, 0, 1,,3.3
""")

    def test_interactions(self):
        comps = read_compounds_file(self.compounds)
        reacts = read_reactions_file(self.reactions)

        # first iteration
        res = iterate_once(comps, reacts)
        self.assertEqual(len(res), 1)
        self.assertIn('(c1) {r1} (c2)', res)
        self.assertEqual(
            res['(c1) {r1} (c2)']['groups'],
            {'-H': 4, '-O': 4, '-N': 0})
        self.assertEqual(res['(c1) {r1} (c2)']['mass'], 4.6)

        # second iteration
        comps.update(res)
        nres = iterate_once(comps, reacts)

        self.assertEqual(len(nres), 4)
        self.assertIn('(c1) {r1} (c2)', nres)
        self.assertIn('((c1) {r1} (c2)) {r2} (c1)', nres)
        self.assertIn('((c1) {r1} (c2)) {r3} (None)', nres)
        self.assertIn('((c1) {r1} (c2)) {r2} ((c1) {r1} (c2))', nres)
        self.assertEqual(
            nres['(c1) {r1} (c2)']['groups'],
            {'-H': 4, '-O': 4, '-N': 0})
        self.assertEqual(
            nres['((c1) {r1} (c2)) {r2} (c1)']['groups'],
            {'-H': 6, '-O': 6, '-N': 2})
        self.assertEqual(
            nres['((c1) {r1} (c2)) {r3} (None)']['groups'],
            {'-H': 4, '-O': 4, '-N': 1})
        self.assertEqual(
            nres['((c1) {r1} (c2)) {r2} ((c1) {r1} (c2))']['groups'],
            {'-N': -1, '-O': 8, '-H': 9})
        self.assertEqual(nres['(c1) {r1} (c2)']['mass'], 4.6)
        self.assertAlmostEqual(nres['((c1) {r1} (c2)) {r2} (c1)']['mass'], 3.6)

class TestNameParser(TestCase):
    def test_basic_name(self):
        name = '(Caffeic acid) {C-C linkage} (Rhamnazin)'
        c1, r, c2 = parse_compound_name(name)

        self.assertEqual(c1, 'Caffeic acid')
        self.assertEqual(r, 'C-C linkage')
        self.assertEqual(c2, 'Rhamnazin')

    def test_nested_name1(self):
        name = '(Phloretin) {C-C linkage} ((Abscisic acid) {C-C linkage} (Aurantinidin))'
        c1, r, c2 = parse_compound_name(name)

        self.assertEqual(c1, 'Phloretin')
        self.assertEqual(r, 'C-C linkage')
        self.assertEqual(c2, '(Abscisic acid) {C-C linkage} (Aurantinidin)')

    def test_nested_name2(self):
        name = '((Abscisic acid) {C-C linkage} (Aurantinidin)) {Condensation - Ester Formation} (Eudesmic acid)'
        c1, r, c2 = parse_compound_name(name)

        self.assertEqual(c1, '(Abscisic acid) {C-C linkage} (Aurantinidin)')
        self.assertEqual(r, 'Condensation - Ester Formation')
        self.assertEqual(c2, 'Eudesmic acid')

    def test_rwb_name(self):
        name = '(Ononitol) {Oxidation - (O) Addition} (None)'
        c1, r, c2 = parse_compound_name(name)

        self.assertEqual(c1, 'Ononitol')
        self.assertEqual(r, 'Oxidation - (O) Addition')
        self.assertEqual(c2, 'None')

class TestAtomTransformationParser(TestCase):
    def test_easy_case(self):
        inp = 'M1 + M2 - H2O'
        res = parse_atom_transformation(inp)

        self.assertEqual(res, {
            'c1': True,
            'c2': True,
            'H': -2,
            'O': -1
        })

    def test_no_c2(self):
        inp = 'M1 + H2'
        res = parse_atom_transformation(inp)

        self.assertEqual(res, {
            'c1': True,
            'c2': False,
            'H': 2
        })
