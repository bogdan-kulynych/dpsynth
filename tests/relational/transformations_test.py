# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for dpsynth.relational.transformations."""

from absl.testing import absltest
from dpsynth.relational import domain as rel_domain
from dpsynth.relational import transformations
import numpy as np
import pandas as pd


class TransformationsTest(absltest.TestCase):

  def test_transformations_import_and_callable(self):
    self.assertTrue(callable(transformations.compute_hierarchical_weights))
    self.assertTrue(
        callable(transformations.build_permuted_exploration_dataset)
    )
    self.assertTrue(
        callable(transformations.create_slot_linear_chain_constraints)
    )
    self.assertTrue(callable(transformations.symmetrize_to_wide_domain))
    self.assertTrue(callable(transformations.quantile_copula_coupling))
    self.assertTrue(callable(transformations.unstack_wide_family_records))

  def test_compute_row_root_mappings_single_table(self):
    households = pd.DataFrame({
        'household_id': ['h1', 'h2', 'h3'],
        'income': [50000.0, 75000.0, 100000.0],
    })
    hierarchy = [(0, 'households', None)]
    mapping = transformations._compute_row_root_mappings(
        {'households': households}, hierarchy
    )
    self.assertIsInstance(mapping['households'], pd.Series)
    self.assertEqual(
        mapping['households'].tolist(),
        [0, 1, 2],
    )

  def test_compute_row_root_mappings_2tier(self):
    households = pd.DataFrame({'hid': ['h1', 'h2']})
    persons = pd.DataFrame({
        'pid': ['p1', 'p2', 'p3'],
        'hid': ['h1', 'h1', 'h2'],
    })
    fk = rel_domain.ForeignKeyRelation(
        parent_table='households',
        parent_primary_key='hid',
        child_table='persons',
        child_foreign_key='hid',
        max_children_per_parent=3,
    )
    hierarchy = [(0, 'households', None), (1, 'persons', fk)]
    mapping = transformations._compute_row_root_mappings(
        {'households': households, 'persons': persons}, hierarchy
    )
    self.assertIsInstance(mapping['persons'], pd.Series)
    self.assertEqual(
        mapping['persons'].tolist(),
        [0, 0, 1],
    )

  def test_compute_row_root_mappings_truncation_and_cascading(self):
    households = pd.DataFrame({'hid': ['h1']})
    # h1 has 3 persons, but max_children_per_parent is 2 -> exactly 2 chosen
    persons = pd.DataFrame({
        'pid': ['p1', 'p2', 'p3'],
        'hid': ['h1', 'h1', 'h1'],
    })
    # activities for each person
    activities = pd.DataFrame({
        'aid': ['a1', 'a2', 'a3'],
        'pid': ['p1', 'p2', 'p3'],
    })
    fk1 = rel_domain.ForeignKeyRelation(
        parent_table='households',
        parent_primary_key='hid',
        child_table='persons',
        child_foreign_key='hid',
        max_children_per_parent=2,
    )
    fk2 = rel_domain.ForeignKeyRelation(
        parent_table='persons',
        parent_primary_key='pid',
        child_table='activities',
        child_foreign_key='pid',
        max_children_per_parent=2,
    )
    hierarchy = [
        (0, 'households', None),
        (1, 'persons', fk1),
        (2, 'activities', fk2),
    ]

    rng = np.random.default_rng(42)
    mapping = transformations._compute_row_root_mappings(
        {
            'households': households,
            'persons': persons,
            'activities': activities,
        },
        hierarchy,
        rng=rng,
    )
    # Exactly 2 persons are active (non-None), 1 is truncated (None)
    active_persons = mapping['persons'].dropna().tolist()
    self.assertLen(active_persons, 2)
    self.assertEqual(mapping['persons'].isna().sum(), 1)

    # Subchildren of active persons active, subchild of truncated person is None
    active_activities = mapping['activities'].dropna().tolist()
    self.assertLen(active_activities, 2)
    self.assertEqual(mapping['activities'].isna().sum(), 1)

  def test_compute_row_root_mappings_branching_tree(self):
    households = pd.DataFrame({'hid': ['h1', 'h2']})
    persons = pd.DataFrame({'pid': ['p1', 'p2'], 'hid': ['h1', 'h2']})
    vehicles = pd.DataFrame(
        {'vid': ['v1', 'v2', 'v3'], 'hid': ['h1', 'h1', 'h2']}
    )

    fk_p = rel_domain.ForeignKeyRelation(
        'households', 'hid', 'persons', 'hid', 2
    )
    fk_v = rel_domain.ForeignKeyRelation(
        'households', 'hid', 'vehicles', 'hid', 5
    )
    hierarchy = [
        (0, 'households', None),
        (1, 'persons', fk_p),
        (1, 'vehicles', fk_v),
    ]
    mapping = transformations._compute_row_root_mappings(
        {'households': households, 'persons': persons, 'vehicles': vehicles},
        hierarchy,
    )
    self.assertEqual(
        mapping['persons'].tolist(),
        [0, 1],
    )
    self.assertEqual(
        mapping['vehicles'].tolist(),
        [0, 0, 1],
    )

  def test_compute_row_root_mappings_multi_tree_forest(self):
    households = pd.DataFrame({'hid': ['h1']})
    persons = pd.DataFrame({'pid': ['p1'], 'hid': ['h1']})
    companies = pd.DataFrame({'cid': ['c1']})
    departments = pd.DataFrame({'did': ['d1'], 'cid': ['c1']})

    fk_h = rel_domain.ForeignKeyRelation(
        'households', 'hid', 'persons', 'hid', 2
    )
    fk_c = rel_domain.ForeignKeyRelation(
        'companies', 'cid', 'departments', 'cid', 5
    )
    hierarchy = [
        (0, 'households', None),
        (0, 'companies', None),
        (1, 'persons', fk_h),
        (1, 'departments', fk_c),
    ]
    mapping = transformations._compute_row_root_mappings(
        {
            'households': households,
            'persons': persons,
            'companies': companies,
            'departments': departments,
        },
        hierarchy,
    )
    self.assertEqual(mapping['households'].tolist(), [0])
    self.assertEqual(mapping['companies'].tolist(), [0])
    self.assertEqual(mapping['persons'].tolist(), [0])
    self.assertEqual(mapping['departments'].tolist(), [0])

  def test_compute_row_root_mappings_custom_index_alignment(self):
    # Non-standard indices (strings, custom obj) can't break positional mapping
    households = pd.DataFrame(
        {'hid': ['h1', 'h2']}, index=['custom_a', 'custom_b']
    )
    persons = pd.DataFrame(
        {'pid': ['p1', 'p2', 'p3'], 'hid': ['h1', 'h2', 'h1']},
        index=[100, 200, 300],
    )
    fk = rel_domain.ForeignKeyRelation('households', 'hid', 'persons', 'hid', 5)
    mapping = transformations._compute_row_root_mappings(
        {'households': households, 'persons': persons},
        [(0, 'households', None), (1, 'persons', fk)],
    )
    # Positions are strictly 0 and 1 in households DataFrame
    self.assertEqual(
        mapping['persons'].tolist(),
        [0, 1, 0],
    )

  def test_compute_row_root_mappings_orphans_and_validation(self):
    households = pd.DataFrame({'hid': ['h1']})
    persons = pd.DataFrame({
        'pid': ['p1', 'p2'],
        'hid': ['h1', 'orphan_h'],
    })
    fk = rel_domain.ForeignKeyRelation(
        parent_table='households',
        parent_primary_key='hid',
        child_table='persons',
        child_foreign_key='hid',
        max_children_per_parent=2,
    )
    mapping = transformations._compute_row_root_mappings(
        {'households': households, 'persons': persons},
        [(0, 'households', None), (1, 'persons', fk)],
    )
    self.assertEqual(mapping['persons'].tolist(), [0, None])

    # Missing parent primary key
    bad_fk1 = rel_domain.ForeignKeyRelation(
        parent_table='households',
        parent_primary_key='missing_id',
        child_table='persons',
        child_foreign_key='hid',
        max_children_per_parent=2,
    )
    with self.assertRaisesRegex(ValueError, 'Parent primary key'):
      transformations._compute_row_root_mappings(
          {'households': households, 'persons': persons},
          [(0, 'households', None), (1, 'persons', bad_fk1)],
      )

    # Missing child foreign key
    bad_fk2 = rel_domain.ForeignKeyRelation(
        parent_table='households',
        parent_primary_key='hid',
        child_table='persons',
        child_foreign_key='missing_hid',
        max_children_per_parent=2,
    )
    with self.assertRaisesRegex(ValueError, 'Child foreign key'):
      transformations._compute_row_root_mappings(
          {'households': households, 'persons': persons},
          [(0, 'households', None), (1, 'persons', bad_fk2)],
      )

  def test_compute_row_root_mappings_nan_and_corrupt_data(self):
    households = pd.DataFrame({'hid': ['h1', np.nan, 'h2', 'h1']})
    persons = pd.DataFrame({
        'pid': ['p1', 'p2', 'p3', 'p4', 'p5'],
        'hid': ['h1', np.nan, 'h2', None, 'orphan'],
    })
    fk = rel_domain.ForeignKeyRelation(
        parent_table='households',
        parent_primary_key='hid',
        child_table='persons',
        child_foreign_key='hid',
        max_children_per_parent=2,
    )
    hierarchy = [(0, 'households', None), (1, 'persons', fk)]
    mapping = transformations._compute_row_root_mappings(
        {'households': households, 'persons': persons},
        hierarchy,
    )
    self.assertEqual(
        mapping['persons'].tolist(),
        [0, None, 2, None, None],
    )

  def test_compute_row_root_mappings_empty_tables(self):
    empty_h = pd.DataFrame({'hid': []})
    persons = pd.DataFrame({'pid': ['p1'], 'hid': ['h1']})
    fk = rel_domain.ForeignKeyRelation(
        parent_table='households',
        parent_primary_key='hid',
        child_table='persons',
        child_foreign_key='hid',
        max_children_per_parent=2,
    )
    mapping = transformations._compute_row_root_mappings(
        {'households': empty_h, 'persons': persons},
        [(0, 'households', None), (1, 'persons', fk)],
    )
    self.assertEqual(mapping['households'].tolist(), [])
    self.assertEqual(mapping['persons'].tolist(), [None])

    empty_p = pd.DataFrame({'pid': [], 'hid': []})
    mapping2 = transformations._compute_row_root_mappings(
        {'households': empty_h, 'persons': empty_p},
        [(0, 'households', None), (1, 'persons', fk)],
    )
    self.assertEqual(mapping2['households'].tolist(), [])
    self.assertEqual(mapping2['persons'].tolist(), [])

  def test_compute_row_root_mappings_random_subsampling_reproducibility(self):
    # A single household with 10 persons, capacity s = 3
    households = pd.DataFrame({'hid': ['h1']})
    persons = pd.DataFrame({
        'pid': [f'p{i}' for i in range(10)],
        'hid': ['h1'] * 10,
    })
    fk = rel_domain.ForeignKeyRelation('households', 'hid', 'persons', 'hid', 3)
    hierarchy = [(0, 'households', None), (1, 'persons', fk)]

    # Same seed must yield identical active row selections
    rng1 = np.random.default_rng(123)
    mapping1 = transformations._compute_row_root_mappings(
        {'households': households, 'persons': persons}, hierarchy, rng=rng1
    )
    rng2 = np.random.default_rng(123)
    mapping2 = transformations._compute_row_root_mappings(
        {'households': households, 'persons': persons}, hierarchy, rng=rng2
    )
    self.assertEqual(mapping1['persons'].tolist(), mapping2['persons'].tolist())
    self.assertEqual(mapping1['persons'].dropna().count(), 3)
    self.assertEqual(mapping1['persons'].isna().sum(), 7)

    # Different seeds must produce valid selections of size exactly 3
    rng3 = np.random.default_rng(999)
    mapping3 = transformations._compute_row_root_mappings(
        {'households': households, 'persons': persons}, hierarchy, rng=rng3
    )
    self.assertEqual(mapping3['persons'].dropna().count(), 3)
    self.assertEqual(mapping3['persons'].isna().sum(), 7)

  def test_compute_hierarchical_weights_single_table(self):
    households = pd.DataFrame({
        'household_id': ['h1', 'h2', 'h3'],
        'income': [50000.0, 75000.0, 100000.0],
    })
    hierarchy = [(0, 'households', None)]
    weights = transformations.compute_hierarchical_weights(
        {'households': households}, hierarchy=hierarchy
    )
    self.assertIn('households', weights)
    self.assertEqual(weights['households'].shape, (3,))
    np.testing.assert_allclose(weights['households'], np.array([1.0, 1.0, 1.0]))
    self.assertAlmostEqual(weights['households'].sum(), 3.0)

  def test_compute_hierarchical_weights_2tier(self):
    # H1 has 2 persons (P1, P2) -> w = 0.5 each
    # H2 has 1 person (P3) -> w = 1.0
    households = pd.DataFrame({'hid': ['h1', 'h2']})
    persons = pd.DataFrame({
        'pid': ['p1', 'p2', 'p3'],
        'hid': ['h1', 'h1', 'h2'],
    })
    fk = rel_domain.ForeignKeyRelation(
        parent_table='households',
        parent_primary_key='hid',
        child_table='persons',
        child_foreign_key='hid',
        max_children_per_parent=3,
    )
    hierarchy = [(0, 'households', None), (1, 'persons', fk)]
    weights = transformations.compute_hierarchical_weights(
        {'households': households, 'persons': persons}, hierarchy=hierarchy
    )
    np.testing.assert_allclose(weights['households'], np.array([1.0, 1.0]))
    np.testing.assert_allclose(weights['persons'], np.array([0.5, 0.5, 1.0]))
    # Total sum of weights in every table matches number of households (2.0)
    self.assertAlmostEqual(weights['households'].sum(), 2.0)
    self.assertAlmostEqual(weights['persons'].sum(), 2.0)

  def test_compute_hierarchical_weights_3tier_with_truncation(self):
    # H1 has 3 persons, but s1 = 2
    # -> 2 active (w = 0.5 each), 1 truncated (w = 0.0)
    # H1's active persons have 2 activities each (4 total) -> w = 0.25 each
    # H1's truncated person has 2 activities -> both cascade to w = 0.0
    households = pd.DataFrame({'hid': ['h1']})
    persons = pd.DataFrame({
        'pid': ['p1', 'p2', 'p3'],
        'hid': ['h1', 'h1', 'h1'],
    })
    activities = pd.DataFrame({
        'aid': ['a1', 'a2', 'a3', 'a4', 'a5', 'a6'],
        'pid': ['p1', 'p1', 'p2', 'p2', 'p3', 'p3'],
    })
    fk1 = rel_domain.ForeignKeyRelation(
        parent_table='households',
        parent_primary_key='hid',
        child_table='persons',
        child_foreign_key='hid',
        max_children_per_parent=2,
    )
    fk2 = rel_domain.ForeignKeyRelation(
        parent_table='persons',
        parent_primary_key='pid',
        child_table='activities',
        child_foreign_key='pid',
        max_children_per_parent=2,
    )
    hierarchy = [
        (0, 'households', None),
        (1, 'persons', fk1),
        (2, 'activities', fk2),
    ]

    rng = np.random.default_rng(42)
    weights = transformations.compute_hierarchical_weights(
        {
            'households': households,
            'persons': persons,
            'activities': activities,
        },
        hierarchy=hierarchy,
        rng=rng,
    )

    # Household sum = 1.0
    self.assertAlmostEqual(weights['households'].sum(), 1.0)
    # Person sum = 1.0 (2 active persons with 0.5, 1 truncated with 0.0)
    self.assertAlmostEqual(weights['persons'].sum(), 1.0)
    self.assertEqual((weights['persons'] == 0.0).sum(), 1)
    self.assertEqual((weights['persons'] == 0.5).sum(), 2)

    # Activity sum = 1.0 (4 active activities with 0.25, 2 truncated with 0.0)
    self.assertAlmostEqual(weights['activities'].sum(), 1.0)
    self.assertEqual((weights['activities'] == 0.0).sum(), 2)
    self.assertEqual((weights['activities'] == 0.25).sum(), 4)

  def test_compute_hierarchical_weights_empty_table(self):
    empty_h = pd.DataFrame({'hid': []})
    empty_p = pd.DataFrame({'pid': [], 'hid': []})
    fk = rel_domain.ForeignKeyRelation(
        parent_table='households',
        parent_primary_key='hid',
        child_table='persons',
        child_foreign_key='hid',
        max_children_per_parent=2,
    )
    hierarchy = [(0, 'households', None), (1, 'persons', fk)]
    weights = transformations.compute_hierarchical_weights(
        {'households': empty_h, 'persons': empty_p}, hierarchy=hierarchy
    )
    self.assertEqual(weights['households'].shape, (0,))
    self.assertEqual(weights['persons'].shape, (0,))

  def test_compute_hierarchical_weights_multi_tree_forest(self):
    households = pd.DataFrame({'hid': ['h1', 'h2']})
    persons = pd.DataFrame({
        'pid': ['p1', 'p2', 'p3', 'p4'],
        'hid': ['h1', 'h1', 'h2', 'h2'],
    })
    companies = pd.DataFrame({'cid': ['c1']})
    departments = pd.DataFrame({
        'did': ['d1', 'd2'],
        'cid': ['c1', 'c1'],
    })

    fk_h = rel_domain.ForeignKeyRelation(
        'households', 'hid', 'persons', 'hid', 5
    )
    fk_c = rel_domain.ForeignKeyRelation(
        'companies', 'cid', 'departments', 'cid', 5
    )
    hierarchy = [
        (0, 'households', None),
        (0, 'companies', None),
        (1, 'persons', fk_h),
        (1, 'departments', fk_c),
    ]
    weights = transformations.compute_hierarchical_weights(
        {
            'households': households,
            'persons': persons,
            'companies': companies,
            'departments': departments,
        },
        hierarchy=hierarchy,
    )
    self.assertAlmostEqual(weights['households'].sum(), 2.0)
    self.assertAlmostEqual(weights['persons'].sum(), 2.0)
    self.assertAlmostEqual(weights['companies'].sum(), 1.0)
    self.assertAlmostEqual(weights['departments'].sum(), 1.0)

  def test_dp_adversarial_data_dependent_robustness(self):
    """Stress tests DP safety: mechanism must never crash on adversarial data."""
    # Dataset with special values, duplicate keys, and missing references.
    households = pd.DataFrame({
        'hid': [
            'h1',
            np.nan,
            None,
            float('inf'),
            float('-inf'),
            'h1',  # duplicate
            'h12345',
            'h_true',
        ],
        'val': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    })
    persons = pd.DataFrame({
        'pid': [f'p{i}' for i in range(12)],
        'hid': [
            'h1',  # matches first h1
            'orphan_key',  # orphan
            np.nan,  # NaN
            None,  # None
            pd.NA,  # pd.NA
            float('inf'),  # matches inf
            float('-inf'),  # matches -inf
            'h12345',  # matches h12345
            'h_true',  # matches h_true
            'orphan_2',  # another orphan
            'orphan_3',  # another orphan
            'h1',  # another match to h1
        ],
    })
    activities = pd.DataFrame({
        'aid': [f'a{i}' for i in range(5)],
        'pid': ['p0', 'p1', 'p9', 'missing_person', 'p11'],
    })

    fk1 = rel_domain.ForeignKeyRelation(
        'households', 'hid', 'persons', 'hid', 1
    )
    fk2 = rel_domain.ForeignKeyRelation(
        'persons', 'pid', 'activities', 'pid', 2
    )
    hierarchy = [
        (0, 'households', None),
        (1, 'persons', fk1),
        (2, 'activities', fk2),
    ]

    rng = np.random.default_rng(100)
    # Must execute cleanly (no exceptions) on this corrupted dataset
    weights = transformations.compute_hierarchical_weights(
        {
            'households': households,
            'persons': persons,
            'activities': activities,
        },
        hierarchy=hierarchy,
        rng=rng,
    )

    # Output shapes must be strictly aligned with input DataFrame row counts
    self.assertEqual(weights['households'].shape, (len(households),))
    self.assertEqual(weights['persons'].shape, (len(persons),))
    self.assertEqual(weights['activities'].shape, (len(activities),))

    # All weights must be finite non-negative floats
    self.assertTrue(np.all(np.isfinite(weights['households'])))
    self.assertTrue(np.all(weights['households'] >= 0.0))
    self.assertTrue(np.all(np.isfinite(weights['persons'])))
    self.assertTrue(np.all(weights['persons'] >= 0.0))
    self.assertTrue(np.all(np.isfinite(weights['activities'])))
    self.assertTrue(np.all(weights['activities'] >= 0.0))

    # Sensitivity invariant: sum of weights for any single household <= 1.0
    mapping = transformations._compute_row_root_mappings(
        {
            'households': households,
            'persons': persons,
            'activities': activities,
        },
        hierarchy=hierarchy,
        rng=rng,
    )
    for table_name in ['households', 'persons', 'activities']:
      t_roots = mapping[table_name]
      t_weights = weights[table_name]
      for root in t_roots.dropna().unique():
        root_mask = (t_roots == root).values
        root_weight_sum = t_weights[root_mask].sum()
        self.assertAlmostEqual(root_weight_sum, 1.0, places=5)


if __name__ == '__main__':
  absltest.main()
