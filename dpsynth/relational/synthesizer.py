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

"""Multi-table relational differential privacy synthesizer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any, Literal

from absl import logging
import dp_accounting
from dpsynth import api
from dpsynth import discrete_mechanisms
from dpsynth import domain
from dpsynth.relational import domain as rel_domain
import numpy as np
import pandas as pd

# pylint: disable=unused-import
_LOGGING_UNUSED = logging
# pylint: enable=unused-import


@dataclasses.dataclass(frozen=True)
class MultiDataGenerationResult:
  """Results of multi-table relational DP synthetic data generation.

  Attributes:
    synthetic_tables: Mapping from table names to synthetic DataFrames.
    discrete_mechanism_results: Mapping from link/table names to mechanism
      diagnostics.
  """

  synthetic_tables: Mapping[str, pd.DataFrame]
  discrete_mechanism_results: Mapping[str, Any] = dataclasses.field(
      default_factory=dict
  )


@dataclasses.dataclass
class MultiTableMechanism(api.CalibratedMechanism):
  """Calibrated, runnable multi-table relational differential privacy mechanism.

  Attributes:
    domains: Mapping from table name to per-column attribute specifications.
    foreign_keys: Sequence of foreign key relationships defining the hierarchy.
    calibrated_discrete_mechanisms: Mapping from link names to calibrated
      discrete mechanisms.
    calibrated_initializers: Mapping from table and column to calibrated
      initializers.
    total_count_sigma: Sigma for the root table total-count mechanism.
    num_permutation_slots: Permutation exploration slot count (o), default 2.
    exploration_strategy: Exploration strategy ('empty_token' or 'size_sliced').
    max_records_per_user: Assumed upper bound on records a single root user
      contributes.

  Note: For simplicity, user-defined contraints are not supported yet.
  """

  domains: Mapping[str, domain.Schema]
  foreign_keys: Sequence[rel_domain.ForeignKeyRelation]
  calibrated_discrete_mechanisms: Mapping[str, api.CalibratedMechanism]
  calibrated_initializers: Mapping[str, Mapping[str, api.CalibratedMechanism]]
  total_count_sigma: float = dataclasses.field(repr=False)
  num_permutation_slots: int = 2
  exploration_strategy: Literal['empty_token', 'size_sliced'] = 'empty_token'
  max_records_per_user: int = 1

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the composed DpEvent for all sub-mechanisms."""
    raise NotImplementedError('dp_event is not yet implemented.')

  def __call__(
      self,
      rng: np.random.Generator,
      data: Mapping[str, pd.DataFrame],
  ) -> MultiDataGenerationResult:
    """Generates synthetic multi-table relational data."""
    del rng, data
    raise NotImplementedError(
        'MultiTableMechanism.__call__ is not yet implemented.'
    )


@dataclasses.dataclass
class MultiTableConfig(api.MechanismConfig):
  """Configuration recipe for multi-table relational differential privacy synthesis.

  Attributes:
    domains: Mapping from table name to per-column attribute domain
      specifications.
    foreign_keys: Sequence of foreign key relationships defining the hierarchy.
    discrete_mechanism: Discrete mechanism config (e.g. AIM, MST) for relational
      links.
    numerical_bins: Number of bins for numerical attribute discretization.
    init_budget_fraction: Fraction of total zCDP budget allocated to Phase 1.
    num_permutation_slots: Permutation exploration slot count (o), default 2.
    exploration_strategy: Exploration strategy ('empty_token' or 'size_sliced').
  """

  domains: Mapping[str, domain.Schema]
  foreign_keys: Sequence[rel_domain.ForeignKeyRelation] = ()
  discrete_mechanism: discrete_mechanisms.DiscreteMechanismConfig = (
      dataclasses.field(default_factory=discrete_mechanisms.AIMConfig)
  )
  numerical_bins: int = 32
  init_budget_fraction: float = 0.1
  num_permutation_slots: int = 2
  exploration_strategy: Literal['empty_token', 'size_sliced'] = 'empty_token'

  def configure(
      self,
      *,
      zcdp_rho: float,
      delta: float = 0.0,
      max_records_per_user: int = 1,
  ) -> MultiTableMechanism:
    """Configures privacy budgets across Phase 1 and Phase 2 relational links."""
    del zcdp_rho, delta, max_records_per_user
    raise NotImplementedError('configure is not yet implemented.')
