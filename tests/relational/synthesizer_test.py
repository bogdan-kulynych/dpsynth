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
from dpsynth.relational import synthesizer


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


if __name__ == '__main__':
  absltest.main()
