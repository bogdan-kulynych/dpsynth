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
import math
from typing import Any, Literal

from absl import logging
import dp_accounting
from dpsynth import api
from dpsynth import data_generation_v3
from dpsynth import discrete_mechanisms
from dpsynth import domain
from dpsynth.relational import domain as rel_domain
import numpy as np
import pandas as pd

# pylint: disable=unused-import
_LOGGING_UNUSED = logging
# pylint: enable=unused-import


def _create_table_initializers(
    domains: Mapping[str, domain.Schema],
    numerical_bins: int,
) -> dict[str, dict[str, api.MechanismConfig]]:
  """Creates per-table and per-column initializers from relational schemas."""
  return {
      table: data_generation_v3.create_initializers(schema, numerical_bins)
      for table, schema in domains.items()
  }


def _compute_table_col_deltas(
    domains: Mapping[str, domain.Schema],
    delta: float,
    init_budget_fraction: float,
) -> dict[str, dict[str, float]]:
  """Splits thresholding delta additively across open-set columns in all tables.

  DP Note: Only open-set categorical attributes consume delta (for Gaussian
  partition selection thresholding). Categorical and numerical attributes
  operate under pure zCDP (delta = 0.0).

  Args:
    domains: Mapping from table names to per-column AttributeType schemas.
    delta: Total DP delta for partition selection thresholding.
    init_budget_fraction: Fraction of delta allocated to Phase 1 initialization.

  Returns:
    A nested mapping from table name and column name to its allocated delta.

  Raises:
    ValueError: If open-set columns are present but delta <= 0.

  Formal Guarantees:
    - Only open-set categorical attributes consume delta
    - Categorical and numerical attributes operate under pure zCDP (delta = 0.0)
    - Sum of per-column deltas across all tables equals init_budget_fraction *
    delta.
    - Invariance: If no open-set columns exist, all per-column deltas are 0.0.
  """
  num_open_set = 0
  for schema in domains.values():
    for attr in schema.values():
      if isinstance(attr, domain.OpenSetCategoricalAttribute):
        num_open_set += 1
  if num_open_set > 0 and delta <= 0:
    raise ValueError(
        'delta must be positive when open-set categorical attributes are'
        ' present. It is used for Gaussian partition selection.'
    )
  thresholding_delta = init_budget_fraction * delta
  per_col_delta = thresholding_delta / num_open_set if num_open_set > 0 else 0.0
  return {
      table: {
          col: (
              per_col_delta
              if isinstance(attr, domain.OpenSetCategoricalAttribute)
              else 0.0
          )
          for col, attr in schema.items()
      }
      for table, schema in domains.items()
  }


def _compute_link_sensitivities(
    hierarchy: Sequence[tuple[int, str, rel_domain.ForeignKeyRelation | None]],
    max_records_per_user: int = 1,
) -> dict[str, int]:
  """Computes cascading sensitivity (Delta_k = prod s_ancestors) per relational link.

  Overview:
    Relational multi-table synthesis generates data level-by-level down the
    foreign key hierarchy (e.g. Household -> Person -> Activity) using pairwise
    exploration datasets and wide graphical models stitched sequentially via
    quantile copula matching.

  Formal Guarantees:
    - Root Privacy Unit: Differential privacy is guaranteed with respect to the
      root parent entity (e.g. Household).
    - Descendant Bounded Impact: Modifying a single root parent record cascades
      to at most prod_{j=1}^{k-1} s_j immediate parent records in Link k, where
      s_j is the group capacity bound (max_children_per_parent) of ancestor j.
    - Sensitivity Scaling Soundness: Scaling discrete mechanism noise by
      Delta_k = max_records_per_user * prod s_ancestors strictly preserves
      root-level differential privacy across all child and subchild tables
      without materializing Cartesian joins.

  Args:
    hierarchy: Ordered topological synthesis levels from
      `topological_sort_hierarchy()`.
    max_records_per_user: Upper bound on root entity contributions (>= 1).

  Returns:
    A mapping from link name (f'{parent}->{child}') to its cascading integer
    sensitivity bound Delta_k.

  Example:
    Household (max_records_per_user = 1) -> Person (s_1 = 3)
    -> Activity (s_2 = 2):
      - 'Household->Person': Delta_1 = max_records_per_user * 1 = 1
      - 'Person->Activity': Delta_2 = max_records_per_user * s_1 = 3
    Result:
      {'Household->Person': 1, 'Person->Activity': 3}
  """
  cumulative_capacity: dict[str, int] = {}
  link_sensitivities: dict[str, int] = {}

  for _, table_name, fk in hierarchy:
    if fk is None:
      cumulative_capacity[table_name] = 1
    else:
      parent_capacity = cumulative_capacity[fk.parent_table]
      link_name = f'{fk.parent_table}->{fk.child_table}'
      link_sensitivities[link_name] = max_records_per_user * parent_capacity
      cumulative_capacity[table_name] = (
          parent_capacity * fk.max_children_per_parent
      )

  return link_sensitivities


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
    max_records_per_user: Assumed upper bound on records a single user
      contributes to the root table. Essentially the sensitivitiy at the root.

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
    """Returns the composed DpEvent combining all relational sub-mechanisms.

    Formally composes:
      - 1 GaussianDpEvent for the root parent table total count measurement.
      - Column initializer DpEvents across all tables.
      - Discrete mechanism DpEvents across all relational hierarchy links.
        E.g. Exploration of Household-Person and Person-Activity tables with AIM
    """
    events: list[dp_accounting.DpEvent] = []
    for table_inits in self.calibrated_initializers.values():
      for init in table_inits.values():
        events.append(init.dp_event)
    events.append(
        dp_accounting.GaussianDpEvent(noise_multiplier=self.total_count_sigma)
    )
    events.extend(
        mech.dp_event for mech in self.calibrated_discrete_mechanisms.values()
    )
    return dp_accounting.ComposedDpEvent(events)

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
  initializers: Mapping[str, Mapping[str, api.MechanismConfig]] | None = None
  num_permutation_slots: int = 2
  exploration_strategy: Literal['empty_token', 'size_sliced'] = 'empty_token'

  def __post_init__(self):
    if len(self.domains) < 2:
      raise ValueError(
          'MultiTableConfig requires at least two tables in domains, got'
          f' {len(self.domains)}. For single-table synthesis, use'
          ' TabularConfig.'
      )
    if not self.foreign_keys:
      raise ValueError(
          'MultiTableConfig requires at least one foreign key relationship in'
          ' foreign_keys. For single-table synthesis, use TabularConfig.'
      )
    if not 0.0 <= self.init_budget_fraction <= 1.0:
      raise ValueError(
          'init_budget_fraction must be in [0.0, 1.0], got'
          f' {self.init_budget_fraction}.'
      )
    if self.numerical_bins < 1:
      raise ValueError(
          f'numerical_bins must be >= 1, got {self.numerical_bins}.'
      )
    if self.num_permutation_slots < 1:
      raise ValueError(
          'num_permutation_slots must be >= 1, got'
          f' {self.num_permutation_slots}.'
      )
    if self.exploration_strategy not in ('empty_token', 'size_sliced'):
      raise ValueError(
          f'Unsupported exploration_strategy {self.exploration_strategy!r}.'
      )

  def configure(
      self,
      *,
      zcdp_rho: float,
      delta: float = 0.0,
      max_records_per_user: int = 1,
  ) -> MultiTableMechanism:
    """Configures privacy budgets across Phase 1 initializers and Phase 2 links.

    Formal Guarantees:
      - Additive zCDP Partitioning: The total zCDP budget zcdp_rho is
        additively split into init_rho (allocated to 1 root total-count
        measurement and N_total_columns per-column initializers) and
        total_discrete_rho (split evenly across relational hierarchy links).
      - Root-Anchored Population Count: Only the root parent table total count
        is measured with Gaussian noise (total_count_sigma).
      - Descendant table row counts are generated via post-processing from the
        wide discrete mechanism (unstacking non-empty slots under 'empty_token'
        strategy or group size K under 'size_sliced' strategy).
      - Pure zCDP for Gaussian Primitives: Numerical, closed categorical, and
        root count initializers operate under pure zCDP (delta = 0.0).
      - Thresholding Delta Partitioning: Open-set partition selection delta is
        split additively across all open-set columns in all tables.
      - Cascading Sensitivity Scaling: Downstream discrete mechanisms are
        configured with cascading sensitivities Delta_k = prod s_ancestors,
        guaranteeing root parent differential privacy without Cartesian joins.

    Args:
      zcdp_rho: The total zCDP privacy budget (rho > 0).
      delta: Approximate DP delta for open-set Gaussian partition selection.
      max_records_per_user: Upper bound on root entity contributions (>= 1).

    Returns:
      A calibrated, runnable MultiTableMechanism.

    Raises:
      ValueError: If configuration hyperparameters or budgets are invalid.
    """
    api.validate_max_records_per_user(max_records_per_user)
    if zcdp_rho <= 0:
      raise ValueError(f'zcdp_rho must be positive, got {zcdp_rho}.')

    hierarchy = rel_domain.topological_sort_hierarchy(
        list(self.domains.keys()), self.foreign_keys
    )
    link_sensitivities = _compute_link_sensitivities(
        hierarchy, max_records_per_user=max_records_per_user
    )

    per_col_deltas = _compute_table_col_deltas(
        self.domains,
        delta=delta,
        init_budget_fraction=self.init_budget_fraction,
    )
    inits = (
        self.initializers
        if self.initializers is not None
        else _create_table_initializers(self.domains, self.numerical_bins)
    )

    total_cols = sum(len(schema) for schema in self.domains.values())
    init_rho = self.init_budget_fraction * zcdp_rho
    per_col_rho = init_rho / (total_cols + 1)  # +1 for root table total count.
    total_count_rho = per_col_rho
    total_discrete_rho = zcdp_rho - init_rho
    per_link_rho = total_discrete_rho / len(link_sensitivities)

    calibrated_inits = {
        table: {
            col: init.configure(
                zcdp_rho=per_col_rho,
                delta=per_col_deltas[table][col],
                max_records_per_user=max_records_per_user,
            )
            for col, init in table_inits.items()
        }
        for table, table_inits in inits.items()
    }
    total_count_sigma = (
        math.sqrt(0.5 / total_count_rho) if total_count_rho > 0 else 0.0
    )

    calibrated_discrete = {
        link_name: self.discrete_mechanism.configure(
            zcdp_rho=per_link_rho,
            max_records_per_user=sensitivity,
        )
        for link_name, sensitivity in link_sensitivities.items()
    }

    return MultiTableMechanism(
        domains=self.domains,
        foreign_keys=self.foreign_keys,
        calibrated_discrete_mechanisms=calibrated_discrete,
        calibrated_initializers=calibrated_inits,
        total_count_sigma=total_count_sigma,
        num_permutation_slots=self.num_permutation_slots,
        exploration_strategy=self.exploration_strategy,
        max_records_per_user=max_records_per_user,
    )
