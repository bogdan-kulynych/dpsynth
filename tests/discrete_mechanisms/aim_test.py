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

from absl.testing import absltest
from dpsynth.discrete_mechanisms import accounting
from dpsynth.discrete_mechanisms import aim
from dpsynth.discrete_mechanisms import aim_gdp
from dpsynth.discrete_mechanisms import aim_gdp_acc
from dpsynth.discrete_mechanisms import common
from dpsynth.discrete_mechanisms import independent
import mbi
import numpy as np
import scipy.stats


def _make_correlated_dataset(rng, n=1000):
  domain = mbi.Domain(["a", "b", "c"], [3, 3, 3])
  a = rng.integers(0, 3, size=n)
  b = np.where(rng.random(n) < 0.75, a, rng.integers(0, 3, size=n))
  c = (a + b + rng.integers(0, 2, size=n)) % 3
  return mbi.Dataset({"a": a, "b": b, "c": c}, domain)


def _normalized_l1(data, model, clique):
  expected = data.project(clique).datavector()
  actual = model.project(clique).datavector()
  expected /= expected.sum()
  actual /= actual.sum()
  return np.abs(expected - actual).sum() / 2.0


def _correlated_workload_mechanism_baseline_errors(
    mechanism, baseline_config, workload, zcdp_rho=5.0
):
  """Compares an already-calibrated mechanism against the baseline at zcdp_rho.

  The mechanism is passed in calibrated because not every mechanism under test
  takes its budget as a zCDP rho; the baseline always does.
  """
  rng = np.random.default_rng(0)
  data = _make_correlated_dataset(rng)

  mechanism_result = mechanism(rng, data)
  baseline_result = baseline_config.configure(zcdp_rho=zcdp_rho)(rng, data)

  mechanism_error = np.mean([
      _normalized_l1(data, mechanism_result.model, clique)
      for clique in workload
  ])
  baseline_error = np.mean([
      _normalized_l1(data, baseline_result.model, clique) for clique in workload
  ])
  return mechanism_error, baseline_error


class AIMTest(absltest.TestCase):

  def test_fits_one_way_marginals_with_aim(self):
    data = mbi.Dataset.synthetic(mbi.Domain(["a", "b", "c"], [3, 4, 5]), N=1000)
    workload = [("a",), ("b",), ("c",)]
    config = aim.AIMConfig(workload=workload, max_rounds=4, pgm_iters=500)

    calibrated = config.configure(zcdp_rho=10000)
    result = calibrated(np.random.default_rng(0), data)

    self.assertIsInstance(result, common.DiscreteMechanismResult)
    self.assertNotEmpty(result.measurements)
    for col in data.domain:
      expected = data.project([col]).datavector()
      actual = result.model.project([col]).datavector()
      np.testing.assert_allclose(actual, expected, atol=1)

  def test_fits_one_way_marginals_with_aim_gdp(self):
    data = mbi.Dataset.synthetic(mbi.Domain(["a", "b", "c"], [3, 4, 5]), N=1000)
    workload = [("a",), ("b",), ("c",)]

    config = aim_gdp.AIMGDPConfig(
        workload=workload, max_rounds=4, pgm_iters=500
    )
    calibrated = config.configure(zcdp_rho=10000)
    result = calibrated(np.random.default_rng(0), data)

    self.assertIsInstance(result, common.DiscreteMechanismResult)
    self.assertNotEmpty(result.measurements)
    for col in data.domain:
      expected = data.project([col]).datavector()
      actual = result.model.project([col]).datavector()
      np.testing.assert_allclose(actual, expected, atol=1)

  def test_correlated_workload_regression_with_aim(self):
    workload = [("a",), ("b",), ("c",), ("a", "b"), ("a", "c"), ("b", "c")]
    config = aim.AIMConfig(workload=workload, max_rounds=4, pgm_iters=500)
    baseline_config = independent.IndependentConfig()
    mechanism_error, baseline_error = (
        _correlated_workload_mechanism_baseline_errors(
            config.configure(zcdp_rho=5.0), baseline_config, workload
        )
    )
    self.assertLess(mechanism_error, 0.05 * baseline_error)

  def test_correlated_workload_regression_with_aim_gdp(self):
    workload = [("a",), ("b",), ("c",), ("a", "b"), ("a", "c"), ("b", "c")]
    config = aim_gdp.AIMGDPConfig(
        workload=workload, max_rounds=4, pgm_iters=500
    )
    baseline_config = independent.IndependentConfig()
    mechanism_error, baseline_error = (
        _correlated_workload_mechanism_baseline_errors(
            config.configure(zcdp_rho=5.0), baseline_config, workload
        )
    )
    self.assertLess(mechanism_error, 0.05 * baseline_error)

  def test_fits_one_way_marginals_with_aim_gdp_acc(self):
    data = mbi.Dataset.synthetic(mbi.Domain(["a", "b", "c"], [3, 4, 5]), N=1000)
    workload = [("a",), ("b",), ("c",)]
    config = aim_gdp_acc.AimGdpAccConfig(
        workload=workload, max_rounds=4, pgm_iters=500
    )

    # Same effectively-noiseless budget as the sibling tests, in GDP mu units.
    calibrated = config.configure(gdp_mu=accounting.zcdp_to_gdp(10000) ** 0.5)
    result = calibrated(np.random.default_rng(0), data)

    self.assertIsInstance(result, common.DiscreteMechanismResult)
    self.assertNotEmpty(result.measurements)
    for col in data.domain:
      expected = data.project([col]).datavector()
      actual = result.model.project([col]).datavector()
      np.testing.assert_allclose(actual, expected, atol=1)

  def test_correlated_workload_regression_with_aim_gdp_acc(self):
    workload = [("a",), ("b",), ("c",), ("a", "b"), ("a", "c"), ("b", "c")]
    config = aim_gdp_acc.AimGdpAccConfig(
        workload=workload, max_rounds=4, pgm_iters=500
    )
    baseline_config = independent.IndependentConfig()
    # mu^2 = 2 * rho matches the budget the baseline is given.
    mechanism_error, baseline_error = (
        _correlated_workload_mechanism_baseline_errors(
            config.configure(gdp_mu=accounting.zcdp_to_gdp(5.0) ** 0.5),
            baseline_config,
            workload,
        )
    )
    self.assertLess(mechanism_error, 0.05 * baseline_error)

  def test_aim_gdp_acc_budget_can_be_set_in_mu(self):
    config = aim_gdp_acc.AimGdpAccConfig()

    calibrated = config.configure(gdp_mu=3.0)

    self.assertEqual(calibrated.gdp_mu, 3.0)
    self.assertEqual(calibrated.dp_event.noise_multiplier, 1 / 3.0)

  def test_aim_gdp_acc_requires_gdp_mu_and_rejects_zcdp_rho(self):
    with self.assertRaises(ValueError):
      aim_gdp_acc.AimGdpAccConfig().configure(zcdp_rho=1.0)
    with self.assertRaises(ValueError):
      aim_gdp_acc.AimGdpAccConfig().configure()

  def test_default_configuration_values(self):
    config = aim.AIMConfig()
    self.assertEqual(config.pgm_iters, 1000)

    gdp_config = aim_gdp.AIMGDPConfig()
    self.assertEqual(gdp_config.pgm_iters, 1000)

    gdp_acc_config = aim_gdp_acc.AimGdpAccConfig()
    self.assertEqual(gdp_acc_config.pgm_iters, 1000)
    self.assertIsNone(gdp_acc_config.gdp_mu)


class BoundedRangeGdpAccountingTest(absltest.TestCase):

  def test_bounded_range_musq_matches_closed_form(self):
    for nu in [0.01, 0.5, 1.0, 4.0]:
      expected_mu = -2 * scipy.stats.norm.ppf(1 / (np.exp(nu / 2) + 1))
      self.assertAlmostEqual(
          accounting.gdp_bounded_range_musq(nu), expected_mu**2
      )

  def test_bounded_range_nu_inverts_musq(self):
    for nu in [0.01, 0.5, 1.0, 4.0, 20.0]:
      musq = accounting.gdp_bounded_range_musq(nu)
      self.assertAlmostEqual(accounting.gdp_bounded_range_nu(musq), nu)

  def test_bounded_range_musq_approaches_pi_over_eight_nu_squared(self):
    nu = 1e-3
    self.assertAlmostEqual(
        accounting.gdp_bounded_range_musq(nu) / nu**2, np.pi / 8
    )

  def test_zero_budget(self):
    self.assertEqual(accounting.gdp_bounded_range_musq(0.0), 0.0)
    self.assertEqual(accounting.gdp_bounded_range_nu(0.0), 0.0)

  def test_calibrate_meets_target_epsilon(self):
    eps, delta = 1.0, 1e-5
    calibrated = aim_gdp_acc.AimGdpAccConfig().calibrate(
        epsilon=eps, delta=delta
    )
    self.assertLessEqual(
        accounting.gdp_delta(calibrated.gdp_mu, eps), delta + 1e-12
    )


if __name__ == "__main__":
  absltest.main()
