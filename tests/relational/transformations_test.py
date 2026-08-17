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
from dpsynth.relational import transformations


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


if __name__ == '__main__':
  absltest.main()
