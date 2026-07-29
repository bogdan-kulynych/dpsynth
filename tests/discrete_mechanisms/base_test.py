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

"""Unit tests for the shared ``DiscreteMechanism`` base-class machinery.

These tests exercise the base class in isolation via minimal concrete
subclasses, rather than relying on inherited coverage from child integration
tests. This lets us cover edge cases in the shared boilerplate (budget
splitting, non-fatal precompile failures, and the ``_select`` contract).
"""

import dataclasses
from unittest import mock

from absl.testing import absltest
import dp_accounting
from dpsynth.discrete_mechanisms import base
from dpsynth.discrete_mechanisms import common
import mbi
import mbi.estimation
import numpy as np


def _dataset(n: int = 200) -> mbi.Dataset:
  return mbi.Dataset.synthetic(mbi.Domain(['a', 'b', 'c'], [3, 4, 5]), N=n)


@dataclasses.dataclass
class _NoOpMechanism(base.DiscreteMechanism):
  """Minimal concrete mechanism that selects no additional marginals."""

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    self._check_calibration()
    return dp_accounting.GaussianDpEvent(noise_multiplier=1.0)

  def supporting_cliques(self, domain: mbi.Domain) -> list[mbi.Clique]:
    return [(a,) for a in domain.attributes]

  def _select(self, rng, data, measurements, phase_times):
    return []


@dataclasses.dataclass
class _NoSelectMechanism(base.DiscreteMechanism):
  """Concrete mechanism that (incorrectly) does not override ``_select``."""

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    self._check_calibration()
    return dp_accounting.GaussianDpEvent(noise_multiplier=1.0)

  def supporting_cliques(self, domain: mbi.Domain) -> list[mbi.Clique]:
    return [(a,) for a in domain.attributes]


class ConfigureTest(absltest.TestCase):
  """Tests for the shared ``configure`` / ``_allocate_budget`` budgeting."""

  def test_default_fraction_splits_one_way_budget(self):
    configured = _NoOpMechanism(one_way_budget_fraction=0.25).configure(
        zcdp_rho=100.0
    )
    self.assertEqual(configured.one_way_rho, 25.0)
    self.assertEqual(configured.remaining_rho, 75.0)

  def test_zero_one_way_budget_fraction_skips_one_way(self):
    configured = _NoOpMechanism(one_way_budget_fraction=0.0).configure(
        zcdp_rho=100.0
    )
    self.assertIsNone(configured.one_way_rho)
    self.assertEqual(configured.remaining_rho, 100.0)

  def test_default_allocate_budget_leaves_measurement_rho_unset(self):
    # The base ``_allocate_budget`` hook returns an empty mapping, so no
    # mechanism-specific budget fields are populated.
    configured = _NoOpMechanism().configure(zcdp_rho=100.0)
    self.assertIsNone(configured.measurement_rho)

  def test_initial_measurements_skip_one_way(self):
    configured = _NoOpMechanism(one_way_budget_fraction=0.5).configure(
        zcdp_rho=100.0, initial_measurements=[mock.sentinel.measurement]
    )
    self.assertIsNone(configured.one_way_rho)
    self.assertEqual(configured.remaining_rho, 100.0)


class CalibrationGuardTest(absltest.TestCase):
  """Tests that using an unconfigured mechanism fails fast."""

  def test_call_without_configure_raises(self):
    mechanism = _NoOpMechanism()
    with self.assertRaisesRegex(ValueError, 'calibrate'):
      mechanism(np.random.default_rng(0), _dataset())

  def test_dp_event_without_configure_raises(self):
    with self.assertRaisesRegex(ValueError, 'calibrate'):
      _ = _NoOpMechanism().dp_event


class RunMachineryTest(absltest.TestCase):
  """Tests for the shared ``_run`` pipeline."""

  def test_precompile_failure_is_non_fatal(self):
    mechanism = _NoOpMechanism(pgm_iters=100).configure(zcdp_rho=1000.0)
    with mock.patch.object(
        mbi.estimation.MirrorDescent,
        'precompile',
        side_effect=RuntimeError('simulated precompile failure'),
    ) as mocked_precompile:
      result = mechanism(np.random.default_rng(0), _dataset())
    mocked_precompile.assert_called_once()
    self.assertIsInstance(result, common.DiscreteMechanismResult)
    self.assertIsNotNone(result.model)

  def test_missing_select_raises_not_implemented(self):
    mechanism = _NoSelectMechanism(pgm_iters=100).configure(zcdp_rho=1000.0)
    with self.assertRaises(NotImplementedError):
      mechanism(np.random.default_rng(0), _dataset())


if __name__ == '__main__':
  absltest.main()
