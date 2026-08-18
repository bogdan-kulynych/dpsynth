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

"""Pure, deterministic relational data transformers and mathematical helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from dpsynth.relational import domain as rel_domain
import mbi
import numpy as np
import pandas as pd


def _compute_row_root_mappings(
    tables: Mapping[str, pd.DataFrame],
    hierarchy: Sequence[tuple[int, str, rel_domain.ForeignKeyRelation | None]],
    rng: np.random.Generator | None = None,
) -> dict[str, pd.Series]:
  """Maps each row in every table to its root parent row index (0..N-1) or None.

  Traverses the relational hierarchy top-down. For each child table, maps
  foreign key references to parent row positions and checks capacity limits.
  When a parent record exceeds its `max_children_per_parent` bound (s), exactly
  `s` children are selected uniformly at random without replacement.
  Truncated, orphaned, or descendant records under dropped parents map to None.

  Formal Guarantees:
    - Bounded Lineage: For each root entity H_i and relation with bound s_k, at
      most s_k children per parent (and at most prod_{j=1}^k s_j descendants at
      depth k) retain active root mappings to H_i.
    - Cascading Truncation Invariant: If an ancestor evaluates to None, all of
      its transitive descendant records in downstream tables evaluate to None.
    - Order-Agnostic Truncation: Child subsampling is uniform without
      replacement, preventing privacy leakage from input DataFrame row order.
    - Data-Dependent Error Immunity: Orphan foreign keys, duplicate primary
      keys, and NaNs evaluate to None without runtime errors.
    - Positional Alignment via None Preservation: Evaluates truncated, orphaned,
      or invalid records to None rather than deleting them, preserving strict
      1-to-1 positional alignment with input DataFrames. This avoids mutating
      table shapes, prevents index-shifting artifacts during downstream
      parent-to-child lookups, and cleanly maps to weight w_r = 0.0 in
      sensitivity weighting.
    - Downstream Ingestion: Zero-weighted records are stripped in the
      synthesizer (via `weights > 0.0`) before entering column initializers
      for bin discovery, encoding, and domain compression.

  Example:
    households: [H0, H1]
    persons (s1=2): [P0(H0), P1(H0), P2(H0), P3(H1)] -> P2 truncated (None)
    activities (s2=2): [A0(P0), A1(P1), A2(P2), A3(P3), A4(orphan)]
      -> A0 maps to 0
      -> A1 maps to 0
      -> A2 maps to None (parent P2 was truncated)
      -> A3 maps to 1
      -> A4 maps to None (orphan foreign key)

    Result:
      {
        'households': pd.Series([0, 1]),
        'persons':    pd.Series([0, 0, None, 1]),
        'activities': pd.Series([0, 0, None, 1, None]),
      }

  Args:
    tables: Mapping from table name to input DataFrame.
    hierarchy: Ordered topological synthesis levels from
      `topological_sort_hierarchy()`.
    rng: Random number generator for uniform child record truncation.

  Returns:
    A dictionary mapping table name to a pd.Series of root row indices (int) or
    None, aligned 1-to-1 with DataFrame rows.

  Raises:
    ValueError: If required primary or foreign key columns are missing from
      schemas.
  """
  if rng is None:
    rng = np.random.default_rng()

  row_to_root: dict[str, pd.Series] = {}
  for depth, table_name, fk in hierarchy:
    child_df = tables[table_name]

    # Depth 0: Root privacy unit table (no incoming foreign key).
    # Each root record maps to its own 0-based integer row index.
    if depth == 0 or fk is None:
      row_to_root[table_name] = pd.Series(
          range(len(child_df)),
          index=child_df.index,
          dtype=object,
      )
      continue

    # Schema integrity validation (public schema check; safe to raise errors).
    if fk.parent_primary_key not in tables[fk.parent_table].columns:
      raise ValueError(
          f'Parent primary key column {fk.parent_primary_key!r} not in table'
          f' {fk.parent_table!r}.'
      )
    if fk.child_foreign_key not in child_df.columns:
      raise ValueError(
          f'Child foreign key column {fk.child_foreign_key!r} not in table'
          f' {table_name!r}.'
      )

    parent_df = tables[fk.parent_table]
    parent_roots = row_to_root[fk.parent_table]

    # Fast path for empty tables: returns all None without failing.
    if child_df.empty or parent_df.empty:
      row_to_root[table_name] = pd.Series(
          [None] * len(child_df), index=child_df.index, dtype=object
      )
      continue

    # 1. Parent lookup maps parent primary keys to row numbers in parent_df.
    # Ignores NaN primary keys and deduplicates repeated keys (keeping first).
    # parent_lookup: (Index = parent_pk, Value = parent_df row index).
    parent_pos = pd.Series(range(len(parent_df)), index=parent_df.index)
    parent_valid_mask = (
        parent_df[fk.parent_primary_key].notna()  # No NaN keys.
        & ~parent_df[fk.parent_primary_key].duplicated()  # Keep only first.
    )
    parent_keys = parent_df.loc[parent_valid_mask, fk.parent_primary_key]
    parent_lookup = pd.Series(
        parent_pos.loc[parent_valid_mask].values, index=parent_keys
    )

    # 2. Vectorized translation of child foreign keys to parent_df row indices.
    # Non-matching keys (orphans) and NaNs evaluate to NaN.
    # child_p_idx : (Index = child row, Value = parent row | NaN).
    child_p_idx = child_df[fk.child_foreign_key].map(parent_lookup)

    # 3. Discard unlinked children
    # valid_children: (Index = child row, Value = parent row).
    valid_children = child_p_idx.dropna().astype(int)
    if valid_children.empty:
      row_to_root[table_name] = pd.Series(
          [None] * len(child_df), index=child_df.index, dtype=object
      )
      continue

    # 4. Enforce cascading truncation: drop children whose parent root is None.
    # Map valid parent indices back to parent_roots positions
    parent_active_mask = parent_roots.notna().iloc[valid_children.values].values
    valid_children = valid_children[parent_active_mask]
    if valid_children.empty:
      row_to_root[table_name] = pd.Series(
          [None] * len(child_df), index=child_df.index, dtype=object
      )
      continue

    # 5. Intra-group uniform random ranking via Pandas, for uniform truncation.
    # Assigns random float to each child; ranks within each parent group.
    # random_scores: (Index = child row, Value = random float).
    # group_ranks: (Index = child row, Value = rank 1-to-n within parent group).
    random_scores = pd.Series(
        rng.random(len(valid_children)), index=valid_children.index
    )
    group_ranks = random_scores.groupby(valid_children.values).rank(
        method='first'
    )
    selected_children = valid_children[
        group_ranks <= fk.max_children_per_parent
    ]

    # 6. Assign root lineages to selected children in Pandas.
    # Initialize full child table to None; update only selected active rows.
    child_roots_series = pd.Series(
        [None] * len(child_df), index=child_df.index, dtype=object
    )
    child_roots_series.loc[selected_children.index] = parent_roots.iloc[
        selected_children.values
    ].values

    row_to_root[table_name] = child_roots_series
  return row_to_root


def compute_hierarchical_weights(
    tables: Mapping[str, pd.DataFrame],
    hierarchy: Sequence[tuple[int, str, rel_domain.ForeignKeyRelation | None]],
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
  """Computes standalone sensitivity weights (w = 1/k_eff) for Phase 1 initializers.

  Calculates a 1D weight array for each table such that the sum of weights
  associated with any single root entity (e.g. household) equals 1.0, ensuring
  global unit sensitivity (Delta = 1.0) without Cartesian joins or noise
  scaling.

  For each table, active records belonging to a root with k_eff active rows
  receive weight 1.0 / k_eff. Inactive rows (truncated or orphaned) receive 0.0.

  Formal Guarantees:
    - Unit Sensitivity (Delta = 1.0): For every table T and every root entity
      H_i, sum_{r in H_i} w_r <= 1.0, guaranteeing global ell_1-sensitivity
      Delta = 1.0 for weighted linear queries on T without Cartesian joins.
    - Equal Intra-Group Weighting: Each active record under root H_i with k_eff
      active rows receives identical weight w_r = 1.0 / k_eff.
    - Zero Inactive Weight: All truncated, orphaned, or unlinked records are
      assigned weight exactly w_r = 0.0.
    - Downstream Ingestion: Zero-weighted records (w_r = 0.0) are stripped at
      the synthesizer boundary (via `weights > 0.0`) before entering column
      initializers, bin discovery, discrete encoding, and domain compression.

  Args:
    tables: Mapping from table name to input DataFrame.
    hierarchy: Ordered topological synthesis levels from
      `topological_sort_hierarchy()`.
    rng: Random number generator for child record truncation.

  Returns:
    A dictionary mapping table name to a 1D float64 array of row weights.
  """
  row_to_root = _compute_row_root_mappings(tables, hierarchy, rng=rng)

  weights: dict[str, np.ndarray] = {}
  for depth, table_name, _ in hierarchy:
    if depth == 0:
      weights[table_name] = np.ones(len(tables[table_name]), dtype=np.float64)
    else:
      roots = row_to_root[table_name]
      root_counts = roots.value_counts()
      table_weights = (
          roots.map(1.0 / root_counts).fillna(0.0).to_numpy(dtype=np.float64)
      )
      weights[table_name] = table_weights

  return weights


def build_permuted_exploration_dataset(
    parent_dataset: mbi.Dataset,
    child_dataset: mbi.Dataset,
    parent_primary_keys: Sequence[str | int] | np.ndarray,
    child_foreign_keys: Sequence[str | int] | np.ndarray,
    num_permutation_slots: int = 2,
    strategy: Literal['empty_token', 'size_sliced'] = 'empty_token',
) -> mbi.Dataset:
  """Constructs the permuted multi-slot exploration dataset for candidate selection.

  Args:
    parent_dataset: Encoded discrete mbi.Dataset for the parent table.
    child_dataset: Encoded discrete mbi.Dataset for the child table.
    parent_primary_keys: Sequence or array of parent primary key identifiers.
    child_foreign_keys: Sequence or array of child foreign key references.
    num_permutation_slots: Number of permutation slots (o) in exploration table,
      default 2.
    strategy: Exploration strategy ('empty_token' with <EMPTY> or
      'size_sliced').

  Returns:
    An mbi.Dataset instance representing the permuted exploration table.

  Raises:
    ValueError: If strategy is unsupported or num_permutation_slots < 1.
  """
  del parent_dataset, child_dataset, parent_primary_keys
  del child_foreign_keys, num_permutation_slots, strategy
  raise NotImplementedError(
      'build_permuted_exploration_dataset is not yet implemented.'
  )


def create_slot_linear_chain_constraints(
    child_domain: mbi.Domain,
    num_permutation_slots: int = 2,
) -> list[mbi.Constraint]:
  """Creates adjacent pairwise mbi.Constraint objects for monolithic slot validity.

  For each slot, generates D-1 pairwise adjacent constraints ((S_i.A_1,
  S_i.A_2),
  (S_i.A_2, S_i.A_3), ...) setting log-potential to -inf on mixed states,
  ensuring
  sampled slots are 100% Real or 100% <EMPTY> with bounded treewidth <= 2.

  Args:
    child_domain: Sub-domain representing attributes of a single child record.
    num_permutation_slots: Number of permutation slots (o), default 2.

  Returns:
    A list of mbi.Constraint instances enforcing monolithic slot locking.
  """
  del child_domain, num_permutation_slots
  raise NotImplementedError(
      'create_slot_linear_chain_constraints is not yet implemented.'
  )


def symmetrize_to_wide_domain(
    measurements: Sequence[mbi.LinearMeasurement],
    max_children_per_parent: int,
    num_permutation_slots: int = 2,
) -> list[mbi.LinearMeasurement]:
  """Replicates selected exploration measurements across all generation slots.

  Equivariantly replicates candidate measurements from (S_1) and (S_1, S_2)
  to all s slots and all comb(s, 2) sibling pairs in the wide generation MRF.

  Args:
    measurements: Noisy marginal measurements from exploration candidate
      selection.
    max_children_per_parent: Maximum group capacity bound (s).
    num_permutation_slots: Number of permutation exploration slots (o), default
      2.

  Returns:
    A list of expanded LinearMeasurement objects for the wide generation MRF.
  """
  del measurements, max_children_per_parent, num_permutation_slots
  raise NotImplementedError('symmetrize_to_wide_domain is not yet implemented.')


def quantile_copula_coupling(
    synth_parents: mbi.Dataset,
    synth_wide_children: mbi.Dataset,
    parent_columns: Sequence[str],
    rng: np.random.Generator | None = None,
) -> mbi.Dataset:
  """Couples synthetic parents and wide child records via Quantile Copula Matching.

  Applies randomized within-bin tie-breaking and lexicographical sorting along
  parent feature coordinates to align parent records with wide child records.

  Args:
    synth_parents: Discrete mbi.Dataset of synthesized parent records.
    synth_wide_children: Discrete mbi.Dataset of synthesized wide child records.
    parent_columns: Parent feature columns used as the coupling anchor.
    rng: Random number generator for within-bin tie-breaking permutation.

  Returns:
    The coupled wide child discrete mbi.Dataset aligned with synthetic parents.
  """
  del synth_parents, synth_wide_children, parent_columns, rng
  raise NotImplementedError('quantile_copula_coupling is not yet implemented.')


def unstack_wide_family_records(
    synth_wide_dataset: mbi.Dataset,
    child_domain: mbi.Domain,
    max_children_per_parent: int,
) -> tuple[mbi.Dataset, np.ndarray]:
  """Unstacks wide family records into a standard normalized child mbi.Dataset.

  Reads group_size = k on each wide row, emits the active child records, and
  returns the unstacked child dataset along with a 1D mapping array of parent
  row indices.

  Args:
    synth_wide_dataset: Discrete mbi.Dataset of wide family records.
    child_domain: Single-child mbi.Domain defining attribute sizes.
    max_children_per_parent: Maximum group capacity bound (s).

  Returns:
    A tuple of (unstacked_child_dataset, parent_row_indices) where
    parent_row_indices maps each unstacked child record to its parent row.
  """
  del synth_wide_dataset, child_domain, max_children_per_parent
  raise NotImplementedError(
      'unstack_wide_family_records is not yet implemented.'
  )
