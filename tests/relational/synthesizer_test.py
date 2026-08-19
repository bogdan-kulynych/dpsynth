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

"""Unit tests for dpsynth.relational.synthesizer."""

import math
from absl.testing import absltest
import dp_accounting
from dpsynth import api
from dpsynth import discrete_mechanisms
from dpsynth import domain
from dpsynth.local_mode import initialization
from dpsynth.relational import domain as rel_domain
from dpsynth.relational import post_processing
from dpsynth.relational import synthesizer
from dpsynth.relational import transformations
import mbi
import numpy as np
import pandas as pd


class SynthesizerTest(absltest.TestCase):

  def test_classes_inherit_correct_api_abstractions(self):
    self.assertTrue(
        issubclass(synthesizer.MultiTableConfig, api.MechanismConfig)
    )
    self.assertTrue(
        issubclass(synthesizer.MultiTableMechanism, api.CalibratedMechanism)
    )

  def test_create_table_initializers_success(self):
    domains = {
        'Household': {
            'income': domain.NumericalAttribute(min_value=0.0, max_value=100.0),
            'region': domain.CategoricalAttribute(possible_values=['U', 'R']),
            'tags': domain.OpenSetCategoricalAttribute(),
        },
        'Person': {
            'age': domain.NumericalAttribute(min_value=0, max_value=100),
            'gender': domain.CategoricalAttribute(possible_values=['M', 'F']),
        },
    }
    inits = synthesizer._create_table_initializers(domains, numerical_bins=16)

    self.assertIn('Household', inits)
    self.assertIn('Person', inits)
    self.assertIsInstance(
        inits['Household']['income'], initialization.NumericalInitializerConfig
    )
    self.assertEqual(inits['Household']['income'].num_partitions, 16)
    self.assertIsInstance(
        inits['Household']['region'],
        initialization.CategoricalInitializerConfig,
    )
    self.assertIsInstance(
        inits['Household']['tags'], initialization.OpenSetInitializerConfig
    )
    self.assertIsInstance(
        inits['Person']['age'], initialization.NumericalInitializerConfig
    )
    self.assertIsInstance(
        inits['Person']['gender'], initialization.CategoricalInitializerConfig
    )

  def test_create_table_initializers_unsupported_type_raises(self):
    domains = {
        'Household': {
            'text': domain.FreeFormTextAttribute(),
        }
    }
    with self.assertRaisesRegex(ValueError, 'Unsupported attribute type'):
      synthesizer._create_table_initializers(domains, numerical_bins=16)

  def test_compute_table_col_deltas_with_open_set(self):
    domains = {
        'Household': {
            'income': domain.NumericalAttribute(min_value=0.0, max_value=100.0),
            'tags': domain.OpenSetCategoricalAttribute(),
        },
        'Person': {
            'hobbies': domain.OpenSetCategoricalAttribute(),
            'gender': domain.CategoricalAttribute(possible_values=['M', 'F']),
        },
    }
    deltas = synthesizer._compute_table_col_deltas(
        domains, delta=1e-4, init_budget_fraction=0.2
    )
    # Total thresholding delta = 0.2 * 1e-4 = 2e-5.
    # 2 open-set columns -> each gets 1e-5.
    self.assertAlmostEqual(deltas['Household']['tags'], 1e-5)
    self.assertEqual(deltas['Household']['income'], 0.0)
    self.assertAlmostEqual(deltas['Person']['hobbies'], 1e-5)
    self.assertEqual(deltas['Person']['gender'], 0.0)

  def test_compute_table_col_deltas_no_open_set(self):
    domains = {
        'Household': {
            'income': domain.NumericalAttribute(min_value=0.0, max_value=100.0),
            'region': domain.CategoricalAttribute(possible_values=['U', 'R']),
        },
    }
    deltas = synthesizer._compute_table_col_deltas(
        domains, delta=0.0, init_budget_fraction=0.1
    )
    self.assertEqual(deltas['Household']['income'], 0.0)
    self.assertEqual(deltas['Household']['region'], 0.0)

  def test_compute_table_col_deltas_missing_delta_raises(self):
    domains = {
        'Household': {
            'tags': domain.OpenSetCategoricalAttribute(),
        },
    }
    with self.assertRaisesRegex(ValueError, 'delta must be positive'):
      synthesizer._compute_table_col_deltas(
          domains, delta=0.0, init_budget_fraction=0.1
      )

  def test_dp_event_composition(self):
    # Setup calibrated initializers for 2 tables.
    household_inits = {
        'income': (
            initialization.NumericalInitializerConfig(
                name='income',
                num_partitions=16,
                attribute=domain.NumericalAttribute(min_value=0, max_value=100),
            ).configure(zcdp_rho=0.01)
        ),
        'region': (
            initialization.CategoricalInitializerConfig(
                name='region',
                attribute=domain.CategoricalAttribute(
                    possible_values=['U', 'R']
                ),
            ).configure(zcdp_rho=0.01)
        ),
    }
    person_inits = {
        'age': (
            initialization.NumericalInitializerConfig(
                name='age',
                num_partitions=16,
                attribute=domain.NumericalAttribute(min_value=0, max_value=100),
            ).configure(zcdp_rho=0.01)
        ),
    }
    calibrated_initializers = {
        'Household': household_inits,
        'Person': person_inits,
    }

    # Setup calibrated discrete mechanism for link Household -> Person.
    calibrated_discrete = discrete_mechanisms.AIMConfig().configure(
        zcdp_rho=0.1
    )
    calibrated_discrete_mechanisms = {
        'Household->Person': calibrated_discrete,
    }

    mech = synthesizer.MultiTableMechanism(
        domains={},
        foreign_keys=(),
        calibrated_discrete_mechanisms=calibrated_discrete_mechanisms,
        calibrated_initializers=calibrated_initializers,
        total_count_sigma=5.0,
    )

    event = mech.dp_event
    self.assertIsInstance(event, dp_accounting.ComposedDpEvent)
    # Expected: 3 column initializers
    # + 1 root Gaussian count + 1 discrete mechanism = 5 events.
    self.assertLen(event.events, 5)
    # Verify root total count event is Gaussian with noise_multiplier=5.0.
    gaussian_events = [
        e
        for e in event.events
        if isinstance(e, dp_accounting.GaussianDpEvent)
        and e.noise_multiplier == 5.0
    ]
    self.assertLen(gaussian_events, 1)

  def test_compute_link_sensitivities_3_tier(self):
    tables = ['Household', 'Person', 'Activity']
    foreign_keys = [
        rel_domain.ForeignKeyRelation(
            parent_table='Household',
            parent_primary_key='hid',
            child_table='Person',
            child_foreign_key='hid',
            max_children_per_parent=3,  # s_1 = 3
        ),
        rel_domain.ForeignKeyRelation(
            parent_table='Person',
            parent_primary_key='pid',
            child_table='Activity',
            child_foreign_key='pid',
            max_children_per_parent=2,  # s_2 = 2
        ),
    ]
    hierarchy = rel_domain.topological_sort_hierarchy(tables, foreign_keys)

    # Unit sensitivity (max_records_per_user = 1)
    sensitivities = synthesizer._compute_link_sensitivities(
        hierarchy, max_records_per_user=1
    )
    # Link 1 (Household -> Person): Delta_1 = 1
    self.assertEqual(sensitivities['Household->Person'], 1)
    # Link 2 (Person -> Activity): Delta_2 = s_1 = 3
    self.assertEqual(sensitivities['Person->Activity'], 3)

  def test_compute_link_sensitivities_branching_and_scaled(self):
    tables = ['Household', 'Person', 'Vehicle']
    foreign_keys = [
        rel_domain.ForeignKeyRelation(
            parent_table='Household',
            parent_primary_key='hid',
            child_table='Person',
            child_foreign_key='hid',
            max_children_per_parent=3,
        ),
        rel_domain.ForeignKeyRelation(
            parent_table='Household',
            parent_primary_key='hid',
            child_table='Vehicle',
            child_foreign_key='hid',
            max_children_per_parent=2,
        ),
    ]
    hierarchy = rel_domain.topological_sort_hierarchy(tables, foreign_keys)

    sensitivities = synthesizer._compute_link_sensitivities(
        hierarchy, max_records_per_user=4
    )
    # Both direct children scale by max_records_per_user * 1 = 4
    self.assertEqual(sensitivities['Household->Person'], 4)
    self.assertEqual(sensitivities['Household->Vehicle'], 4)

  def test_compute_link_sensitivities_no_links(self):
    tables = ['Household', 'Logs']
    hierarchy = rel_domain.topological_sort_hierarchy(tables, foreign_keys=())
    sensitivities = synthesizer._compute_link_sensitivities(
        hierarchy, max_records_per_user=1
    )
    self.assertEmpty(sensitivities)

  def test_configure_single_table_raises(self):
    with self.assertRaisesRegex(
        ValueError, 'requires at least two tables in domains'
    ):
      synthesizer.MultiTableConfig(
          domains={'Household': {'income': domain.NumericalAttribute(0, 100)}},
          foreign_keys=(),
      )

  def test_configure_empty_foreign_keys_raises(self):
    with self.assertRaisesRegex(
        ValueError, 'requires at least one foreign key relationship'
    ):
      synthesizer.MultiTableConfig(
          domains={
              'Household': {'income': domain.NumericalAttribute(0, 100)},
              'Person': {'age': domain.NumericalAttribute(0, 100)},
          },
          foreign_keys=(),
      )

  def test_configure_end_to_end_3_tier(self):
    domains = {
        'Household': {
            'income': domain.NumericalAttribute(0, 100),
            'region': domain.CategoricalAttribute(['U', 'R']),
        },
        'Person': {
            'age': domain.NumericalAttribute(0, 100),
            'gender': domain.CategoricalAttribute(['M', 'F']),
        },
        'Activity': {
            'amount': domain.NumericalAttribute(0, 50),
        },
    }
    foreign_keys = [
        rel_domain.ForeignKeyRelation(
            parent_table='Household',
            parent_primary_key='hid',
            child_table='Person',
            child_foreign_key='hid',
            max_children_per_parent=3,
        ),
        rel_domain.ForeignKeyRelation(
            parent_table='Person',
            parent_primary_key='pid',
            child_table='Activity',
            child_foreign_key='pid',
            max_children_per_parent=2,
        ),
    ]
    config = synthesizer.MultiTableConfig(
        domains=domains,
        foreign_keys=foreign_keys,
        init_budget_fraction=0.1,
    )

    # 5 total cols + 1 root count = 6 initializers.
    # init_rho = 0.1 * 0.6 = 0.06 => per_col_rho = 0.01.
    # total_count_sigma = sqrt(0.5 / 0.01) = sqrt(50).
    # discrete_rho = 0.6 - 0.06 = 0.54 => per_link_rho = 0.27 across 2 links.
    mech = config.configure(zcdp_rho=0.6, max_records_per_user=1)

    self.assertIsInstance(mech, synthesizer.MultiTableMechanism)
    self.assertAlmostEqual(mech.total_count_sigma, math.sqrt(50.0))
    self.assertLen(mech.calibrated_initializers, 3)
    self.assertLen(mech.calibrated_discrete_mechanisms, 2)
    self.assertIn('Household->Person', mech.calibrated_discrete_mechanisms)
    self.assertIn('Person->Activity', mech.calibrated_discrete_mechanisms)

    # Verify composed DpEvent has
    # 5 col inits + 1 root Gaussian + 2 discrete links = 8 events.
    event = mech.dp_event
    self.assertIsInstance(event, dp_accounting.ComposedDpEvent)
    self.assertLen(event.events, 8)

  def test_calibrate_end_to_end(self):
    domains = {
        'Household': {
            'income': domain.NumericalAttribute(0, 100),
            'region': domain.CategoricalAttribute(['U', 'R']),
        },
        'Person': {
            'age': domain.NumericalAttribute(0, 100),
        },
    }
    foreign_keys = [
        rel_domain.ForeignKeyRelation(
            parent_table='Household',
            parent_primary_key='hid',
            child_table='Person',
            child_foreign_key='hid',
            max_children_per_parent=3,
        ),
    ]
    config = synthesizer.MultiTableConfig(
        domains=domains,
        foreign_keys=foreign_keys,
    )
    # Calibrate solves for optimal zcdp_rho using PLD / RDP accountant.
    mech = config.calibrate(epsilon=1.0, delta=1e-5)
    self.assertIsInstance(mech, synthesizer.MultiTableMechanism)
    self.assertGreater(mech.total_count_sigma, 0.0)
    self.assertLen(mech.calibrated_discrete_mechanisms, 1)

  def test_configure_hyperparameter_validations(self):
    domains = {
        'Household': {'income': domain.NumericalAttribute(0, 100)},
        'Person': {'age': domain.NumericalAttribute(0, 100)},
    }
    foreign_keys = [
        rel_domain.ForeignKeyRelation(
            parent_table='Household',
            parent_primary_key='hid',
            child_table='Person',
            child_foreign_key='hid',
            max_children_per_parent=3,
        ),
    ]

    with self.subTest('negative_zcdp_rho'):
      config = synthesizer.MultiTableConfig(
          domains=domains, foreign_keys=foreign_keys
      )
      with self.assertRaisesRegex(ValueError, 'zcdp_rho must be positive'):
        config.configure(zcdp_rho=-0.1)

    with self.subTest('invalid_init_budget_fraction'):
      with self.assertRaisesRegex(
          ValueError, 'init_budget_fraction must be in'
      ):
        synthesizer.MultiTableConfig(
            domains=domains,
            foreign_keys=foreign_keys,
            init_budget_fraction=1.5,
        )

    with self.subTest('invalid_numerical_bins'):
      with self.assertRaisesRegex(ValueError, 'numerical_bins must be >= 1'):
        synthesizer.MultiTableConfig(
            domains=domains,
            foreign_keys=foreign_keys,
            numerical_bins=0,
        )

    with self.subTest('invalid_num_permutation_slots'):
      with self.assertRaisesRegex(
          ValueError, 'num_permutation_slots must be >= 1'
      ):
        synthesizer.MultiTableConfig(
            domains=domains,
            foreign_keys=foreign_keys,
            num_permutation_slots=0,
        )

    with self.subTest('invalid_exploration_strategy'):
      with self.assertRaisesRegex(
          ValueError, 'Unsupported exploration_strategy'
      ):
        synthesizer.MultiTableConfig(
            domains=domains,
            foreign_keys=foreign_keys,
            exploration_strategy='unsupported_strategy',
        )

  def test_validate_input_table_columns_success(self):
    domains = {
        'Household': {
            'income': domain.NumericalAttribute(0, 100),
            'region': domain.CategoricalAttribute(['U', 'R']),
        },
        'Person': {
            'age': domain.NumericalAttribute(0, 100),
        },
    }
    foreign_keys = [
        rel_domain.ForeignKeyRelation(
            parent_table='Household',
            parent_primary_key='hid',
            child_table='Person',
            child_foreign_key='hid',
            max_children_per_parent=3,
        ),
    ]
    table_columns = {
        'Household': ['hid', 'income', 'region'],
        'Person': ['pid', 'hid', 'age'],
    }
    # Should complete without error.
    synthesizer._validate_input_table_columns(
        domains, foreign_keys, table_columns
    )

  def test_validate_input_table_columns_raises(self):
    domains = {
        'Household': {'income': domain.NumericalAttribute(0, 100)},
        'Person': {'age': domain.NumericalAttribute(0, 100)},
    }
    foreign_keys = [
        rel_domain.ForeignKeyRelation(
            parent_table='Household',
            parent_primary_key='hid',
            child_table='Person',
            child_foreign_key='hid',
            max_children_per_parent=3,
        ),
    ]

    with self.subTest('missing_table'):
      table_columns = {'Household': ['hid', 'income']}
      with self.assertRaisesRegex(ValueError, 'Table .* not found'):
        synthesizer._validate_input_table_columns(
            domains, foreign_keys, table_columns
        )

    with self.subTest('missing_schema_column'):
      table_columns = {
          'Household': ['hid'],  # missing 'income'
          'Person': ['hid', 'age'],
      }
      with self.assertRaisesRegex(ValueError, 'Column .* not found in table'):
        synthesizer._validate_input_table_columns(
            domains, foreign_keys, table_columns
        )

    with self.subTest('missing_parent_primary_key'):
      table_columns = {
          'Household': ['income'],  # missing 'hid'
          'Person': ['hid', 'age'],
      }
      with self.assertRaisesRegex(
          ValueError, 'Parent primary key column .* not found'
      ):
        synthesizer._validate_input_table_columns(
            domains, foreign_keys, table_columns
        )

    with self.subTest('missing_child_foreign_key'):
      table_columns = {
          'Household': ['hid', 'income'],
          'Person': ['age'],  # missing 'hid'
      }
      with self.assertRaisesRegex(
          ValueError, 'Child foreign key column .* not found'
      ):
        synthesizer._validate_input_table_columns(
            domains, foreign_keys, table_columns
        )

  def test_preprocess_weighted_tables(self):
    tables = {
        'Household': pd.DataFrame({
            'hid': ['H1', 'H2'],
            'income': [50.0, 75.0],
        }),
        'Person': pd.DataFrame({
            'pid': ['P1', 'P2', 'P3', 'P4', 'P5'],
            'hid': ['H1', 'H1', 'H1', 'H2', 'Orphan'],
            'age': [10, 20, 30, 40, 50],
        }),
    }
    foreign_keys = [
        rel_domain.ForeignKeyRelation(
            parent_table='Household',
            parent_primary_key='hid',
            child_table='Person',
            child_foreign_key='hid',
            max_children_per_parent=2,  # H1 has 3 persons -> 1 truncated
        ),
    ]
    hierarchy = rel_domain.topological_sort_hierarchy(
        list(tables.keys()), foreign_keys
    )
    rng = np.random.default_rng(42)

    filtered_tables, filtered_weights = synthesizer._preprocess_weighted_tables(
        tables, hierarchy, rng=rng
    )

    # Root table retains all 2 rows with weight 1.0 each.
    self.assertLen(filtered_tables['Household'], 2)
    np.testing.assert_allclose(filtered_weights['Household'], [1.0, 1.0])

    # Person table had 5 rows:
    # 2 kept from H1 (weight 0.5 each), 1 kept from H2 (weight 1.0),
    # 1 truncated from H1 (w=0.0 filtered out), 1 orphan (w=0.0 filtered out).
    self.assertLen(filtered_tables['Person'], 3)
    np.testing.assert_allclose(filtered_weights['Person'], [0.5, 0.5, 1.0])
    self.assertAlmostEqual(filtered_weights['Person'].sum(), 2.0)

  def test_measure_root_total_count(self):
    rng = np.random.default_rng(123)
    # Zero noise test
    total, measurement = synthesizer._measure_root_total_count(
        rng,
        root_record_count=100,
        total_count_sigma=0.0,
        max_records_per_user=1,
    )
    self.assertEqual(total, 100.0)
    self.assertEqual(measurement.clique, ())
    self.assertEqual(measurement.stddev, 0.0)
    np.testing.assert_allclose(measurement.noisy_measurement, [100.0])

    # Noisy test with sensitivity scaling
    total_noisy, measurement_noisy = synthesizer._measure_root_total_count(
        rng,
        root_record_count=100,
        total_count_sigma=5.0,
        max_records_per_user=2,
    )
    self.assertGreaterEqual(total_noisy, 1.0)
    self.assertEqual(measurement_noisy.clique, ())
    self.assertEqual(measurement_noisy.stddev, 10.0)

    # Negative noisy value clips to 1.0
    total_clipped, measurement_clipped = synthesizer._measure_root_total_count(
        rng,
        root_record_count=0,
        total_count_sigma=0.0,
        max_records_per_user=1,
    )
    self.assertEqual(total_clipped, 1.0)
    np.testing.assert_allclose(measurement_clipped.noisy_measurement, [1.0])

  def test_run_single_col_initializer(self):
    rng = np.random.default_rng(42)

    # 1. Numerical initializer on weighted data
    num_init = initialization.NumericalInitializerConfig(
        name='income',
        num_partitions=8,
        attribute=domain.NumericalAttribute(min_value=0.0, max_value=100.0),
    ).configure(zcdp_rho=0.5)
    data_num = np.array([10.0, 20.0, 30.0, 80.0, 90.0])
    weights_num = np.array([0.5, 0.5, 1.0, 0.5, 0.5])
    res_num = synthesizer._run_single_col_initializer(
        num_init, rng, data_num, weights_num, estimated_total=100.0
    )
    self.assertIsNotNone(res_num.bin_edges)
    self.assertIsNotNone(res_num.categorical_attribute)
    self.assertIsNotNone(res_num.measurement)
    self.assertEqual(res_num.measurement.clique, ('income',))

    # 2. Categorical initializer on weighted data
    cat_init = initialization.CategoricalInitializerConfig(
        name='gender',
        attribute=domain.CategoricalAttribute(possible_values=['M', 'F']),
    ).configure(zcdp_rho=0.5)
    data_cat = np.array(['M', 'M', 'F', 'F'])
    weights_cat = np.array([0.5, 0.5, 1.0, 1.0])
    res_cat = synthesizer._run_single_col_initializer(
        cat_init, rng, data_cat, weights_cat
    )
    self.assertIsNone(res_cat.bin_edges)
    self.assertEqual(res_cat.categorical_attribute.size, 2)
    self.assertEqual(res_cat.measurement.clique, ('gender',))

    # 3. Open-set initializer on weighted strings (including mixed-type data)
    open_init = initialization.OpenSetInitializerConfig(
        name='tags',
        attribute=domain.OpenSetCategoricalAttribute(),
        min_count=1,
    ).configure(zcdp_rho=0.5, delta=1e-3)
    data_open = np.array(['sport', 123, 'music'] * 50, dtype=object)
    weights_open = np.ones(len(data_open))
    res_open = synthesizer._run_single_col_initializer(
        open_init, rng, data_open, weights_open
    )
    self.assertIn('sport', res_open.categorical_attribute.possible_values)
    self.assertIn('123', res_open.categorical_attribute.possible_values)

    # 4. Unsupported initializer raises
    with self.assertRaisesRegex(ValueError, 'Unsupported initializer type'):
      synthesizer._run_single_col_initializer(
          object(), rng, data_num, weights_num  # pyrefly: ignore[bad-argument-type]
      )

  def test_run_table_initializers(self):
    rng = np.random.default_rng(42)
    calibrated_inits = {
        'Household': {
            'income': (
                initialization.NumericalInitializerConfig(
                    name='income',
                    num_partitions=8,
                    attribute=domain.NumericalAttribute(
                        min_value=0.0, max_value=100.0
                    ),
                ).configure(zcdp_rho=0.1)
            ),
            'region': (
                initialization.CategoricalInitializerConfig(
                    name='region',
                    attribute=domain.CategoricalAttribute(
                        possible_values=['U', 'R']
                    ),
                ).configure(zcdp_rho=0.1)
            ),
        },
        'Person': {
            'age': (
                initialization.NumericalInitializerConfig(
                    name='age',
                    num_partitions=8,
                    attribute=domain.NumericalAttribute(
                        min_value=0, max_value=100
                    ),
                ).configure(zcdp_rho=0.1)
            ),
        },
    }
    tables = {
        'Household': pd.DataFrame({
            'income': [20.0, 80.0],
            'region': ['U', 'R'],
        }),
        'Person': pd.DataFrame({
            'age': [15, 30, 45],
        }),
    }
    weights = {
        'Household': np.array([1.0, 1.0]),
        'Person': np.array([0.5, 0.5, 1.0]),
    }

    results = synthesizer._run_table_initializers(
        calibrated_inits,
        rng=rng,
        tables=tables,
        weights=weights,
        estimated_total=2.0,
    )

    self.assertIn('Household', results)
    self.assertIn('Person', results)
    self.assertIn('income', results['Household'])
    self.assertIn('region', results['Household'])
    self.assertIn('age', results['Person'])

    self.assertIsInstance(
        results['Household']['income'], initialization.ColumnMeasurement
    )
    self.assertIsInstance(
        results['Household']['region'], initialization.ColumnMeasurement
    )
    self.assertIsInstance(
        results['Person']['age'], initialization.ColumnMeasurement
    )

  def test_encode_and_compress_tables(self):
    domains = {
        'Household': {
            'income': domain.NumericalAttribute(0, 100),
            'region': domain.CategoricalAttribute(['U', 'R']),
        },
        'Person': {
            'age': domain.NumericalAttribute(0, 100),
        },
    }
    table_measurements = {
        'Household': {
            'income': initialization.ColumnMeasurement(
                categorical_attribute=domain.CategoricalAttribute([
                    '0',
                    '1',
                    '2',
                    '3',
                ]),
                bin_edges=np.array([25.0, 50.0, 75.0]),
                measurement=mbi.LinearMeasurement(
                    np.array([5.0, 5.0, 5.0, 5.0]), ('income',), stddev=1.0
                ),
            ),
            'region': initialization.ColumnMeasurement(
                categorical_attribute=domain.CategoricalAttribute(['U', 'R']),
                measurement=mbi.LinearMeasurement(
                    np.array([10.0, 10.0]), ('region',), stddev=1.0
                ),
            ),
        },
        'Person': {
            'age': initialization.ColumnMeasurement(
                categorical_attribute=domain.CategoricalAttribute(['0', '1']),
                bin_edges=np.array([50.0]),
                measurement=mbi.LinearMeasurement(
                    np.array([10.0, 10.0]), ('age',), stddev=1.0
                ),
            ),
        },
    }
    tables = {
        'Household': pd.DataFrame({
            'income': [20.0, 80.0],
            'region': ['U', 'R'],
        }),
        'Person': pd.DataFrame({
            'age': [30, 60],
        }),
    }
    weights = {
        'Household': np.array([1.0, 1.0]),
        'Person': np.array([0.5, 0.5]),
    }

    codecs, datasets, mappings, one_ways = (
        synthesizer._encode_and_compress_tables(
            domains,
            table_measurements,
            tables,
            weights,
            compress_columns=True,
        )
    )

    self.assertIn('Household', codecs)
    self.assertIn('Person', codecs)
    self.assertIn('Household', datasets)
    self.assertIn('Person', datasets)
    self.assertIn('Household', mappings)
    self.assertIn('Person', mappings)

    self.assertIsInstance(datasets['Household'], mbi.Dataset)
    self.assertIsInstance(datasets['Person'], mbi.Dataset)
    np.testing.assert_allclose(datasets['Household'].weights, [1.0, 1.0])
    np.testing.assert_allclose(datasets['Person'].weights, [0.5, 0.5])

    self.assertIn('Household', one_ways)
    self.assertIn('Person', one_ways)
    self.assertNotEmpty(one_ways['Household'])
    self.assertNotEmpty(one_ways['Person'])

  def test_run_table_preprocessing_end_to_end(self):
    domains = {
        'Household': {
            'income': domain.NumericalAttribute(0, 100),
            'region': domain.CategoricalAttribute(['U', 'R']),
        },
        'Person': {
            'age': domain.NumericalAttribute(0, 100),
            'gender': domain.CategoricalAttribute(['M', 'F']),
        },
    }
    foreign_keys = [
        rel_domain.ForeignKeyRelation(
            parent_table='Household',
            parent_primary_key='hid',
            child_table='Person',
            child_foreign_key='hid',
            max_children_per_parent=3,
        ),
    ]
    config = synthesizer.MultiTableConfig(
        domains=domains,
        foreign_keys=foreign_keys,
        init_budget_fraction=0.2,
    )
    mech = config.configure(zcdp_rho=0.5, max_records_per_user=1)
    rng = np.random.default_rng(42)

    data = {
        'Household': pd.DataFrame({
            'hid': ['H1', 'H2'],
            'income': [30.0, 80.0],
            'region': ['U', 'R'],
        }),
        'Person': pd.DataFrame({
            'pid': ['P1', 'P2', 'P3', 'P4'],
            'hid': ['H1', 'H1', 'H2', 'Orphan'],
            'age': [10, 20, 40, 90],
            'gender': ['M', 'F', 'F', 'M'],
        }),
    }

    preprocessed = synthesizer._run_table_preprocessing(
        mech, rng=rng, data=data
    )

    self.assertIsInstance(preprocessed, synthesizer.PreprocessedTables)
    self.assertIn('Household', preprocessed.compressed_datasets)
    self.assertIn('Person', preprocessed.compressed_datasets)

    # Active records: 2 households, 3 valid persons (1 orphan stripped).
    self.assertEqual(preprocessed.compressed_datasets['Household'].records, 2)
    self.assertEqual(preprocessed.compressed_datasets['Person'].records, 3)

    # Verify keys are captured in table_keys.
    self.assertIn('hid', preprocessed.table_keys['Household'])
    self.assertIn('hid', preprocessed.table_keys['Person'])
    self.assertIn('pid', preprocessed.table_keys['Person'])
    self.assertLen(preprocessed.table_keys['Household']['hid'], 2)
    self.assertLen(preprocessed.table_keys['Person']['hid'], 3)

    # Verify root total count and measurement.
    self.assertGreaterEqual(preprocessed.noisy_root_total, 1.0)
    self.assertEqual(preprocessed.root_total_measurement.clique, ())

    # Verify codecs and one-ways.
    self.assertIn('Household', preprocessed.column_codecs)
    self.assertIn('Person', preprocessed.column_codecs)
    self.assertIn('Household', preprocessed.one_way_measurements)
    self.assertIn('Person', preprocessed.one_way_measurements)

  def test_synthesized_link_result_dataclass(self):
    parent_dom = mbi.Domain(('income',), (2,))
    child_dom = mbi.Domain(('amt',), (2,))
    parent_ds = mbi.Dataset({'income': np.array([0])}, parent_dom)
    child_ds = mbi.Dataset({'amt': np.array([1])}, child_dom)
    parent_idx = np.array([0], dtype=np.int64)

    link_res = synthesizer.SynthesizedLinkResult(
        unstacked_child_dataset=child_ds,
        parent_row_indices=parent_idx,
        synth_parent_dataset=parent_ds,
    )

    self.assertEqual(link_res.unstacked_child_dataset.records, 1)
    self.assertIsNotNone(link_res.synth_parent_dataset)
    self.assertEqual(link_res.synth_parent_dataset.records, 1)
    np.testing.assert_array_equal(link_res.parent_row_indices, [0])
    self.assertIsNone(link_res.discrete_mechanism_result)

  def test_fit_and_sample_wide_link_mrf_zero_rows(self):
    wide_domain = mbi.Domain(('income', 'slot_1.age'), (3, 4))
    res = synthesizer._fit_and_sample_wide_link_mrf(
        wide_domain=wide_domain,
        wide_measurements=(),
        wide_constraints=(),
        num_rows=0,
    )
    self.assertEqual(res.records, 0)
    self.assertEqual(res.domain.attributes, ('income', 'slot_1.age'))

  def test_fit_and_sample_wide_link_mrf_with_constraints(self):
    # Wide domain: 1 parent attribute, 1 slot with 2 attributes
    #   (age [3 bins], gender [2 bins])
    child_domain = mbi.Domain(('age', 'gender'), (3, 2))
    parent_domain = mbi.Domain(('income',), (2,))
    wide_domain = transformations.build_exploration_domain(
        parent_domain=parent_domain,
        child_domain=child_domain,
        max_group_size=1,
        num_permutation_slots=1,
        strategy='empty_token',
    )
    # Constraints: monolithic locking on slot 1
    constraints = post_processing.create_slot_linear_chain_constraints(
        child_domain=child_domain,
        num_permutation_slots=1,
    )
    # Measurements: 1-way for parent
    # & child features + 2-way clique for constraints
    measurements = [
        mbi.LinearMeasurement(np.array([10.0, 10.0]), ('income',), stddev=1.0),
        mbi.LinearMeasurement(
            np.array([5.0, 5.0, 5.0, 5.0]), ('slot_1.age',), stddev=1.0
        ),
        mbi.LinearMeasurement(
            np.array([8.0, 7.0, 5.0]), ('slot_1.gender',), stddev=1.0
        ),
        mbi.LinearMeasurement(
            np.full(12, 2.0), ('slot_1.age', 'slot_1.gender'), stddev=1.0
        ),
    ]
    sampled = synthesizer._fit_and_sample_wide_link_mrf(
        wide_domain=wide_domain,
        wide_measurements=measurements,
        wide_constraints=constraints,
        num_rows=20,
        iters=20,
    )
    self.assertEqual(sampled.records, 20)
    # Check that mixed states are never sampled: age==3 iff gender==2 (<EMPTY>)
    ages = sampled.data['slot_1.age']
    genders = sampled.data['slot_1.gender']
    for a, g in zip(ages, genders):
      if a == 3:
        self.assertEqual(g, 2)
      if g == 2:
        self.assertEqual(a, 3)

  def test_synthesize_relational_link_level_1(self):
    rng = np.random.default_rng(42)
    parent_dom = mbi.Domain(('income',), (2,))
    child_dom = mbi.Domain(('age',), (3,))
    parent_ds = mbi.Dataset({'income': np.array([0, 1] * 10)}, parent_dom)
    child_ds = mbi.Dataset({'age': np.array([0, 1, 2, 0] * 10)}, child_dom)
    parent_pks = [f'H{i}' for i in range(20)]
    child_fks = [f'H{i // 2}' for i in range(40)]
    fk = rel_domain.ForeignKeyRelation(
        parent_table='Household',
        parent_primary_key='hid',
        child_table='Person',
        child_foreign_key='hid',
        max_children_per_parent=2,
    )
    discrete_mech = discrete_mechanisms.AIMConfig(
        pgm_iters=10, max_rounds=2
    ).configure(zcdp_rho=0.5)

    res = synthesizer._synthesize_relational_link(
        parent_dataset=parent_ds,
        child_dataset=child_ds,
        parent_primary_keys=parent_pks,
        child_foreign_keys=child_fks,
        fk_relation=fk,
        discrete_mechanism=discrete_mech,
        num_permutation_slots=2,
        strategy='empty_token',
        rng=rng,
        synth_parents=None,
        noisy_root_total=20.0,
    )

    self.assertIsInstance(res, synthesizer.SynthesizedLinkResult)
    self.assertIsNotNone(res.synth_parent_dataset)
    self.assertEqual(res.synth_parent_dataset.records, 20)
    self.assertEqual(res.unstacked_child_dataset.domain.attributes, ('age',))
    self.assertIsNotNone(res.discrete_mechanism_result)
    self.assertGreaterEqual(len(res.parent_row_indices), 0)
    if len(res.parent_row_indices) > 0:
      self.assertTrue(np.all(res.parent_row_indices < 20))

  def test_synthesize_relational_link_downstream_copula(self):
    rng = np.random.default_rng(42)
    parent_dom = mbi.Domain(('age',), (3,))
    child_dom = mbi.Domain(('amt',), (2,))
    parent_ds = mbi.Dataset({'age': np.array([0, 1, 2])}, parent_dom)
    child_ds = mbi.Dataset({'amt': np.array([0, 1, 1])}, child_dom)
    parent_pks = ['P1', 'P2', 'P3']
    child_fks = ['P1', 'P2', 'P2']
    fk = rel_domain.ForeignKeyRelation(
        parent_table='Person',
        parent_primary_key='pid',
        child_table='Activity',
        child_foreign_key='pid',
        max_children_per_parent=2,
    )
    discrete_mech = discrete_mechanisms.AIMConfig(
        pgm_iters=10, max_rounds=2
    ).configure(zcdp_rho=0.5)

    # Simulated synth_parents from Level 1
    synth_parents = mbi.Dataset({'age': np.array([2, 0, 1])}, parent_dom)

    res = synthesizer._synthesize_relational_link(
        parent_dataset=parent_ds,
        child_dataset=child_ds,
        parent_primary_keys=parent_pks,
        child_foreign_keys=child_fks,
        fk_relation=fk,
        discrete_mechanism=discrete_mech,
        num_permutation_slots=2,
        strategy='empty_token',
        rng=rng,
        synth_parents=synth_parents,
    )

    self.assertIsInstance(res, synthesizer.SynthesizedLinkResult)
    self.assertIsNone(res.synth_parent_dataset)
    self.assertEqual(res.unstacked_child_dataset.domain.attributes, ('amt',))
    self.assertIsNotNone(res.discrete_mechanism_result)
    self.assertTrue(np.all(res.parent_row_indices < 3))

  def test_synthesize_relational_hierarchy_3_tier(self):
    rng = np.random.default_rng(42)
    data = {
        'Household': pd.DataFrame(
            {'hid': ['H1', 'H2'], 'income': [30.0, 80.0]}
        ),
        'Person': pd.DataFrame({
            'pid': ['P1', 'P2', 'P3'],
            'hid': ['H1', 'H1', 'H2'],
            'age': [25, 30, 45],
        }),
        'Activity': pd.DataFrame({
            'aid': ['A1', 'A2', 'A3', 'A4'],
            'pid': ['P1', 'P2', 'P2', 'P3'],
            'amt': [100.0, 200.0, 150.0, 300.0],
        }),
    }
    domains = {
        'Household': {
            'income': domain.NumericalAttribute(0, 100),
        },
        'Person': {
            'age': domain.NumericalAttribute(0, 100),
        },
        'Activity': {
            'amt': domain.NumericalAttribute(0, 500),
        },
    }
    fks = [
        rel_domain.ForeignKeyRelation(
            parent_table='Household',
            parent_primary_key='hid',
            child_table='Person',
            child_foreign_key='hid',
            max_children_per_parent=2,
        ),
        rel_domain.ForeignKeyRelation(
            parent_table='Person',
            parent_primary_key='pid',
            child_table='Activity',
            child_foreign_key='pid',
            max_children_per_parent=2,
        ),
    ]
    hierarchy = rel_domain.topological_sort_hierarchy(
        tables=list(domains.keys()), foreign_keys=fks
    )
    cfg = synthesizer.MultiTableConfig(
        domains=domains,
        foreign_keys=fks,
        discrete_mechanism=discrete_mechanisms.AIMConfig(
            pgm_iters=10, max_rounds=2
        ),
        num_permutation_slots=2,
    )
    mech = cfg.configure(zcdp_rho=0.5, max_records_per_user=1)

    preprocessed = synthesizer._run_table_preprocessing(
        mechanism=mech,
        rng=rng,
        data=data,
    )

    synth_datasets, parent_mappings, discrete_results = (
        synthesizer._synthesize_relational_hierarchy(
            mechanism=mech,
            preprocessed=preprocessed,
            hierarchy=hierarchy,
            rng=rng,
        )
    )

    self.assertIn('Household', synth_datasets)
    self.assertIn('Person', synth_datasets)
    self.assertIn('Activity', synth_datasets)
    self.assertGreater(synth_datasets['Household'].records, 0)
    self.assertIn('Person', parent_mappings)
    self.assertIn('Activity', parent_mappings)
    if synth_datasets['Person'].records > 0:
      self.assertTrue(
          np.all(
              parent_mappings['Person'] < synth_datasets['Household'].records
          )
      )
    if synth_datasets['Activity'].records > 0:
      self.assertTrue(
          np.all(parent_mappings['Activity'] < synth_datasets['Person'].records)
      )
    self.assertIn('Household->Person', discrete_results)
    self.assertIn('Person->Activity', discrete_results)

  def test_synthesize_relational_hierarchy_branching(self):
    rng = np.random.default_rng(42)
    data = {
        'Household': pd.DataFrame(
            {'hid': ['H1', 'H2'], 'income': [30.0, 80.0]}
        ),
        'Person': pd.DataFrame({
            'pid': ['P1', 'P2', 'P3'],
            'hid': ['H1', 'H1', 'H2'],
            'age': [25, 30, 45],
        }),
        'Vehicle': pd.DataFrame({
            'vid': ['V1', 'V2'],
            'hid': ['H1', 'H2'],
            'type': ['sedan', 'suv'],
        }),
    }
    domains = {
        'Household': {
            'income': domain.NumericalAttribute(0, 100),
        },
        'Person': {
            'age': domain.NumericalAttribute(0, 100),
        },
        'Vehicle': {
            'type': domain.CategoricalAttribute(['sedan', 'suv']),
        },
    }
    fks = [
        rel_domain.ForeignKeyRelation(
            parent_table='Household',
            parent_primary_key='hid',
            child_table='Person',
            child_foreign_key='hid',
            max_children_per_parent=2,
        ),
        rel_domain.ForeignKeyRelation(
            parent_table='Household',
            parent_primary_key='hid',
            child_table='Vehicle',
            child_foreign_key='hid',
            max_children_per_parent=2,
        ),
    ]
    hierarchy = rel_domain.topological_sort_hierarchy(
        tables=list(domains.keys()), foreign_keys=fks
    )
    cfg = synthesizer.MultiTableConfig(
        domains=domains,
        foreign_keys=fks,
        discrete_mechanism=discrete_mechanisms.AIMConfig(
            pgm_iters=10, max_rounds=2
        ),
        num_permutation_slots=2,
    )
    mech = cfg.configure(zcdp_rho=0.5, max_records_per_user=1)

    preprocessed = synthesizer._run_table_preprocessing(
        mechanism=mech,
        rng=rng,
        data=data,
    )

    synth_datasets, parent_mappings, discrete_results = (
        synthesizer._synthesize_relational_hierarchy(
            mechanism=mech,
            preprocessed=preprocessed,
            hierarchy=hierarchy,
            rng=rng,
        )
    )

    self.assertIn('Household', synth_datasets)
    self.assertIn('Person', synth_datasets)
    self.assertIn('Vehicle', synth_datasets)
    if synth_datasets['Person'].records > 0:
      self.assertTrue(
          np.all(
              parent_mappings['Person'] < synth_datasets['Household'].records
          )
      )
    if synth_datasets['Vehicle'].records > 0:
      self.assertTrue(
          np.all(
              parent_mappings['Vehicle'] < synth_datasets['Household'].records
          )
      )
    self.assertIn('Household->Person', discrete_results)
    self.assertIn('Household->Vehicle', discrete_results)


if __name__ == '__main__':
  absltest.main()
