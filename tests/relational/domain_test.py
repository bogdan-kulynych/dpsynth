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

"""Unit tests for dpsynth.relational.domain."""

from absl.testing import absltest
from dpsynth.relational import domain


class DomainTest(absltest.TestCase):

  def test_foreign_key_relation_initialization(self):
    fk = domain.ForeignKeyRelation(
        parent_table='households',
        parent_primary_key='household_id',
        child_table='persons',
        child_foreign_key='household_id',
        max_children_per_parent=5,
    )
    self.assertEqual(fk.parent_table, 'households')
    self.assertEqual(fk.parent_primary_key, 'household_id')
    self.assertEqual(fk.child_table, 'persons')
    self.assertEqual(fk.child_foreign_key, 'household_id')
    self.assertEqual(fk.max_children_per_parent, 5)

  def test_foreign_key_relation_invalid_capacity(self):
    with self.assertRaisesRegex(
        ValueError, 'max_children_per_parent must be >= 1'
    ):
      domain.ForeignKeyRelation(
          parent_table='households',
          parent_primary_key='household_id',
          child_table='persons',
          child_foreign_key='household_id',
          max_children_per_parent=0,
      )


if __name__ == '__main__':
  absltest.main()
