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

"""Post-processing, symmetrization, and domain metadata transformers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import itertools

import mbi
import numpy as np


def create_slot_linear_chain_constraints(
    child_domain: mbi.Domain,
    num_permutation_slots: int = 2,
) -> list[mbi.Constraint]:
  """Creates adjacent pairwise mbi.Constraint objects for monolithic slot validity.

  For each slot, generates D-1 pairwise adjacent constraints ((S_i.A_1,
  S_i.A_2), (S_i.A_2, S_i.A_3), ...) setting log-potential to -inf on mixed
  states, ensuring sampled slots are 100% Real or 100% <EMPTY> with bounded
  treewidth <= 2.

    - Transitive Monolithic Slot Locking: By chaining pairwise constraints
      (A_1 = E <=> A_2 = E <=> ... <=> A_D = E), any mixed state containing both
      real values and <EMPTY> has joint log-potential = -inf (P = 0.0).
    - Treewidth <= 2 Bounded Complexity: Pairwise linear chains avoid star-graph
      hubs and high-dimensional cliques, keeping maximum constraint clique size
      to 2 (memory <= (K+1)^2 entries per factor) to prevent junction tree OOMs.
    - Zero Private Information: Constraints are constructed purely from public
      domain metadata and slot count, inducing zero DP privacy loss (eps = 0).

  Args:
    child_domain: Sub-domain representing attributes of a single child record.
    num_permutation_slots: Number of permutation slots (o), default 2.

  Returns:
    A list of mbi.Constraint instances enforcing monolithic slot locking.

  Raises:
    ValueError: If num_permutation_slots < 1.
  """
  if num_permutation_slots < 1:
    raise ValueError(
        f'num_permutation_slots must be >= 1, got {num_permutation_slots}'
    )

  child_attrs = child_domain.attributes
  child_shape = child_domain.shape
  num_attrs = len(child_attrs)

  # Single- or 0-attribute child domain requires no cross-attribute constraints.
  if num_attrs < 2:
    return []

  constraints: list[mbi.Constraint] = []
  for slot_idx in range(1, num_permutation_slots + 1):  # slot_1 to slot_o
    for i in range(num_attrs - 1):
      attr1, attr2 = child_attrs[i], child_attrs[i + 1]
      k1, k2 = child_shape[i], child_shape[i + 1]

      slot_attr1 = f'slot_{slot_idx}.{attr1}'
      slot_attr2 = f'slot_{slot_idx}.{attr2}'
      pair_domain = mbi.Domain((slot_attr1, slot_attr2), (k1 + 1, k2 + 1))

      # Mixed states: (k1, [0..k2-1]) and ([0..k1-1], k2).
      invalid_combos = [(k1, a2) for a2 in range(k2)] + [
          (a1, k2) for a1 in range(k1)
      ]
      invalid_arr = np.array(invalid_combos, dtype=np.int64)

      constraints.append(
          mbi.Constraint(domain=pair_domain, invalid=invalid_arr)
      )

  return constraints


def _extract_slot_indices(clique: Sequence[str | int]) -> list[int]:
  """Extracts unique sorted 1-based slot indices present in a measurement clique.

    - Returns sorted unique integer slot indices
      ('income', 'slot_2.age', 'slot_1.gender') -> [1, 2]).
    - Attributes not adhering to the 'slot_<idx>.<attr>' pattern
      (such as parent features or 'group_size') are cleanly ignored.

  Example:
    Household -> Person -> Activity running schema:
      >>> _extract_slot_indices(('income', 'region'))
      []
      >>> _extract_slot_indices(('income', 'slot_1.age', 'slot_1.gender'))
      [1]
      >>> _extract_slot_indices(('group_size', 'slot_2.gender', 'slot_1.age'))
      [1, 2]
      >>> _extract_slot_indices(('age', 'slot_1.amount', 'slot_2.type'))
      [1, 2]

  Args:
    clique: Sequence of attribute names defining a marginal measurement.

  Returns:
    A sorted list of unique integer slot indices present in the clique.
  """
  slot_indices: set[int] = set()
  for attr in clique:
    if isinstance(attr, str) and attr.startswith('slot_') and '.' in attr:
      prefix = attr.split('.', 1)[0]
      slot_str = prefix[len('slot_') :]
      if slot_str.isdigit():
        slot_indices.add(int(slot_str))
  return sorted(slot_indices)


def _remap_clique_slots(
    clique: Sequence[str | int],
    slot_mapping: Mapping[int, int],
) -> tuple[str | int, ...]:
  """Remaps slot index prefixes in a clique according to a given slot mapping.

  - Order & Non-Slot Invariance: Attributes not matching 'slot_<idx>.<attr>'
    and their relative positions in the clique tuple are strictly preserved.
  - Deterministic Substitution: Replaces 'slot_{orig_idx}.{col}' with
    'slot_{target_idx}.{col}' for any orig_idx in slot_mapping.

  Example:
    Household -> Person -> Activity running schema:
      >>> _remap_clique_slots(('income', 'slot_1.age'), {1: 3})
      ('income', 'slot_3.age')
      >>> _remap_clique_slots(('slot_1.age', 'slot_2.gender'), {1: 2, 2: 5})
      ('slot_2.age', 'slot_5.gender')
      >>> _remap_clique_slots(('age', 'slot_1.amount'), {1: 2})
      ('age', 'slot_2.amount')

  Args:
    clique: Sequence of attribute names defining a marginal measurement.
    slot_mapping: Dictionary mapping original 1-based slot indices to target
      slot indices.

  Returns:
    A tuple of attribute names with remapped slot indices.
  """
  new_attrs: list[str | int] = []
  for attr in clique:
    if isinstance(attr, str) and attr.startswith('slot_') and '.' in attr:
      prefix, col_name = attr.split('.', 1)
      slot_str = prefix[len('slot_') :]
      if slot_str.isdigit():
        orig_slot = int(slot_str)
        target_slot = slot_mapping.get(orig_slot, orig_slot)
        new_attrs.append(f'slot_{target_slot}.{col_name}')
        continue
    new_attrs.append(attr)
  return tuple(new_attrs)


def symmetrize_to_wide_domain(
    measurements: Sequence[mbi.LinearMeasurement],
    max_children_per_parent: int,
    num_permutation_slots: int = 2,
) -> list[mbi.LinearMeasurement]:
  """Replicates selected exploration measurements across all generation slots.

  Candidate selection (e.g. AIM, MST) explores dependencies in a compact
  o-slot exploration table (typically o = 2). In generation, families can have
  up to s = max_children_per_parent slots. Because child records within a parent
  are exchangeable, this function equivariantly replicates measurements across
  all single slots and all comb(s, r) multi-slot combinations:
    - Single-slot measurements (P, S_1) replicate symmetrically across all s
      slots: (P, S_1), ..., (P, S_s).
    - Multi-slot measurements (S_1, S_2) replicate symmetrically across all
      comb(s, 2) sibling pairs: (S_i, S_j) for 1 <= i < j <= s.
    - Parent-only and metadata measurements (e.g. ('income', 'region') or
      ('group_size',)) are passed through directly without duplication.
    - Preserves noisy_measurement datavector and stddev across all copies.

  Example:
    3-Tier Hierarchy: Household -> Person (s=3) -> Activity (s=2):

    Exploration Measurements (Household -> Person, o=2):
      - M1: ('income', 'group_size')
          -> ('income', 'group_size') [1 copy]
      - M2: ('income', 'slot_1.age')
          -> ('income', 'slot_1.age'), ('income', 'slot_2.age'),
             ('income', 'slot_3.age') [3 copies]
      - M3: ('slot_1.age', 'slot_1.gender')
          -> ('slot_1.age', 'slot_1.gender'), ('slot_2.age', 'slot_2.gender'),
             ('slot_3.age', 'slot_3.gender') [3 copies]
      - M4: ('slot_1.age', 'slot_2.age')
          -> ('slot_1.age', 'slot_2.age'), ('slot_1.age', 'slot_3.age'),
             ('slot_2.age', 'slot_3.age') [3 copies]

    Exploration Measurements (Person -> Activity, o=2, s=2):
      - M5: ('age', 'slot_1.amount')
          -> ('age', 'slot_1.amount'), ('age', 'slot_2.amount') [2 copies]

  Args:
    measurements: Noisy marginal measurements from exploration candidate
      selection.
    max_children_per_parent: Maximum group capacity bound (s >= 1).
    num_permutation_slots: Number of permutation exploration slots (o >= 1),
      default 2.

  Returns:
    A list of expanded LinearMeasurement objects for the wide generation MRF.

  Raises:
    ValueError: If max_children_per_parent < 1 or num_permutation_slots < 1.
  """
  if max_children_per_parent < 1:
    raise ValueError(
        f'max_children_per_parent must be >= 1, got {max_children_per_parent}'
    )
  if num_permutation_slots < 1:
    raise ValueError(
        f'num_permutation_slots must be >= 1, got {num_permutation_slots}'
    )

  s = max_children_per_parent
  expanded: list[mbi.LinearMeasurement] = []

  for m in measurements:
    slots = _extract_slot_indices(m.clique)
    r = len(slots)

    if r == 0:
      expanded.append(m)
    elif r <= s:
      for target_combo in itertools.combinations(range(1, s + 1), r):
        mapping = dict(zip(slots, target_combo))
        new_clique = _remap_clique_slots(m.clique, mapping)
        expanded.append(
            mbi.LinearMeasurement(
                noisy_measurement=m.noisy_measurement,
                clique=new_clique,
                stddev=m.stddev,
                query=m.query,
            )
        )

  return expanded


def quantile_copula_coupling(
    synth_parents: mbi.Dataset,
    synth_wide_children: mbi.Dataset,
    parent_columns: Sequence[str],
    rng: np.random.Generator | None = None,
) -> mbi.Dataset:
  """Couples synthetic parent records with wide child records via Quantile Copula Matching.

  Overview & Problem Formulation:
    In hierarchical synthesis (e.g. Household -> Person -> Activity), Step 1
    synthesizes a parent table (e.g. Person with features 'age', 'gender'), and
    Step 2 independently samples wide child records (e.g. Activity slots
    conditioned on 'age', 'gender'). Because both steps are perturbed by
    independent DP noise, we need a statistically sound way to attach child
    records to parent records without materializing full cross-table graphical
    models.

  How Quantile Copula Matching Works:
    1. Count Synchronization: Downstream linkage table generation samples
       exactly N wide child records, matching upstream parent table
       `len(synth_parents) = N`.
    2. Empirical CDF Alignment (Quantile Matching): Both `synth_parents` and
       `synth_wide_children` approximate the same underlying marginal
       distribution over `parent_columns`. Lexicographically sorting both tables
       aligns their empirical quantile ranks (matching the p-th percentile
       parent to the p-th percentile child record).
    3. Seamless Boundary Overflow: Small count discrepancies due to independent
       DP noise naturally overflow into the nearest adjacent quantile rank,
       guaranteeing 1-to-1 foreign key pairing without orphaned records.
    4. Randomized Tie-Breaking: Eliminates spurious correlations stemming from
       sampling order. If multiple parents share identical `parent_columns`,
       child records are assigned uniformly at random among them,
       guaranteeing conditional exchangeability from the model's perspective.

    - Zero Privacy Loss: Coupling is a deterministic post-processing function on
      previously synthesized private datasets (epsilon = 0).

  Example:
    3-Tier Hierarchy: Household -> Person (age, gender) -> Activity:
      synth_parents (Person):
        Row 0: age=10, gender=0
        Row 1: age=5,  gender=1
      synth_wide_children (Activity slots with person features):
        Row 0: age=5,  gender=1, slot_1.amount=100
        Row 1: age=10, gender=0, slot_1.amount=200

      lexsort order (ranking along parent features):
        parents:  [1 (age=5, gender=1), 0 (age=10, gender=0)]
        children: [0 (age=5, gender=1), 1 (age=10, gender=0)]

      Coupled Output (aligned 1-to-1 with synth_parents row order):
        Row 0 (matches Person 0, age=10): child row 1 (age=10, amt=200)
        Row 1 (matches Person 1, age=5):  child row 0 (age=5,  amt=100)

  Args:
    synth_parents: Discrete mbi.Dataset of synthesized parent records.
    synth_wide_children: Discrete mbi.Dataset of synthesized wide child records.
    parent_columns: Parent feature columns used as the coupling anchor.
    rng: Random number generator for within-bin tie-breaking permutation.

  Returns:
    The coupled wide child discrete mbi.Dataset aligned with synthetic parents.

  Raises:
    ValueError: If record counts do not match or parent_columns are missing.
  """
  num_records = synth_parents.records
  if num_records != synth_wide_children.records:
    raise ValueError(
        f'synth_parents records ({num_records}) does not match'
        f' synth_wide_children records ({synth_wide_children.records}).'
    )
  for col in parent_columns:
    if col not in synth_parents.domain.attributes:
      raise ValueError(
          f'Anchor column {col!r} not in synth_parents domain attributes'
          f' {synth_parents.domain.attributes}.'
      )
    if col not in synth_wide_children.domain.attributes:
      raise ValueError(
          f'Anchor column {col!r} not in synth_wide_children domain attributes'
          f' {synth_wide_children.domain.attributes}.'
      )

  if num_records == 0 or not parent_columns:
    return synth_wide_children

  if rng is None:
    rng = np.random.default_rng()

  # Generate random floats for uniform within-bin tie-breaking.
  parent_rand = rng.random(num_records)
  child_rand = rng.random(num_records)

  # Lexicographical sorting by parent_col[0] then parent_col[1], etc.
  p_keys = (parent_rand,) + tuple(
      synth_parents.data[col] for col in reversed(parent_columns)
  )
  c_keys = (child_rand,) + tuple(
      synth_wide_children.data[col] for col in reversed(parent_columns)
  )

  sort_order_parents = np.lexsort(p_keys)
  sort_order_children = np.lexsort(c_keys)

  # Reorder children so rank r in synth_parents receives rank r in children.
  aligned_child_indices = np.empty(num_records, dtype=np.int64)
  aligned_child_indices[sort_order_parents] = sort_order_children

  new_data = {
      attr: np.asarray(arr)[aligned_child_indices]
      for attr, arr in synth_wide_children.data.items()
  }
  new_weights = (
      synth_wide_children.weights[aligned_child_indices]
      if synth_wide_children.weights is not None
      else None
  )
  return mbi.Dataset(new_data, synth_wide_children.domain, weights=new_weights)


def unstack_wide_family_records(
    synth_wide_dataset: mbi.Dataset,
    child_domain: mbi.Domain,
    max_children_per_parent: int,
) -> tuple[mbi.Dataset, np.ndarray]:
  """Unstacks wide family records into a standard normalized child mbi.Dataset.

  Reads `group_size = k` on each wide parent row, emits active child records
  from slots 1..k (filtering out `<EMPTY>` slots if present), and returns the
  unstacked child dataset along with a 1D mapping array of parent row indices.

    - Bounded Group Size: Emits at most min(k, max_children_per_parent) active
      children per parent row.
    - Clean Category Space: Filtered child records contain values strictly in
      [0, K_attr - 1], removing all <EMPTY> padding tokens (value = K_attr).
    - Stable Sibling Contiguity: Emitted children are grouped contiguously by
      parent row index, preserving deterministic intra-household sibling order.
    - Zero Privacy Loss: Deterministic post-processing (epsilon = 0).

  Example:
    max_children_per_parent = 2, child_domain = {'age': 10} (<EMPTY> = 10)
      Parent 0: group_size=2, slot_1.age=5, slot_2.age=8
      Parent 1: group_size=0, slot_1.age=10 (<EMPTY>), slot_2.age=10 (<EMPTY>)
      Parent 2: group_size=1, slot_1.age=3, slot_2.age=10 (<EMPTY>)

    Result:
      unstacked child dataset:
        Row 0: age=5 (from Parent 0, slot 1)
        Row 1: age=8 (from Parent 0, slot 2)
        Row 2: age=3 (from Parent 2, slot 1)
      parent_row_indices:
        np.array([0, 0, 2], dtype=np.int64)

  Args:
    synth_wide_dataset: Discrete mbi.Dataset of wide family records.
    child_domain: Single-child mbi.Domain defining attribute names and sizes.
    max_children_per_parent: Maximum group capacity bound (s >= 1).

  Returns:
    A tuple of (unstacked_child_dataset, parent_row_indices) where
    parent_row_indices maps each unstacked child record to its parent row.

  Raises:
    ValueError: If max_children_per_parent < 1 or child slot columns are missing
      from synth_wide_dataset domain.
  """
  if max_children_per_parent < 1:
    raise ValueError(
        f'max_children_per_parent must be >= 1, got {max_children_per_parent}'
    )

  num_parents = synth_wide_dataset.records
  child_attrs = list(child_domain.attributes)
  child_shapes = dict(zip(child_attrs, child_domain.shape))

  # Fast path for empty parent datasets.
  if num_parents == 0:
    empty_data = {attr: np.empty(0, dtype=np.int64) for attr in child_attrs}
    return mbi.Dataset(empty_data, child_domain), np.empty(0, dtype=np.int64)

  # Validate required slot columns exist in wide dataset domain.
  for slot_idx in range(1, max_children_per_parent + 1):
    for attr in child_attrs:
      col_name = f'slot_{slot_idx}.{attr}'
      if col_name not in synth_wide_dataset.domain.attributes:
        raise ValueError(
            f'Required slot column {col_name!r} not found in wide dataset'
            f' domain {synth_wide_dataset.domain.attributes}.'
        )

  # Read group_sizes (default to max_children_per_parent if column is absent).
  if 'group_size' in synth_wide_dataset.domain.attributes:
    group_sizes = np.asarray(
        synth_wide_dataset.data['group_size'], dtype=np.int64
    )
    group_sizes = np.clip(group_sizes, 0, max_children_per_parent)
  else:
    group_sizes = np.full(num_parents, max_children_per_parent, dtype=np.int64)

  parent_indices_list: list[np.ndarray] = []
  child_cols_list: dict[str | int, list[np.ndarray]] = {
      attr: [] for attr in child_attrs
  }

  for slot_idx in range(1, max_children_per_parent + 1):
    active_mask = slot_idx <= group_sizes

    # Under 'empty_token' mode, filter out slots with <EMPTY> (val == K_attr).
    # <EMPTY> encoded as k_attr in _build_exploration_domain
    for attr in child_attrs:
      col_name = f'slot_{slot_idx}.{attr}'
      vals = synth_wide_dataset.data[col_name]
      k_attr = child_shapes[attr]
      active_mask = active_mask & (vals < k_attr)

    active_parent_indices = np.where(active_mask)[0]
    if len(active_parent_indices) == 0:
      continue

    parent_indices_list.append(active_parent_indices)
    for attr in child_attrs:
      col_name = f'slot_{slot_idx}.{attr}'
      child_cols_list[attr].append(
          synth_wide_dataset.data[col_name][active_parent_indices]
      )

  if not parent_indices_list:
    empty_data: dict[str | int, np.ndarray] = {
        attr: np.empty(0, dtype=np.int64) for attr in child_attrs
    }
    return mbi.Dataset(empty_data, child_domain), np.empty(0, dtype=np.int64)

  parent_row_indices = np.concatenate(parent_indices_list)
  unstacked_data: dict[str | int, np.ndarray] = {
      attr: np.concatenate(child_cols_list[attr]) for attr in child_attrs
  }

  # Stable sort by parent row idx so all siblings under a parent are contiguous.
  sort_idx = np.argsort(parent_row_indices, kind='stable')
  parent_row_indices = parent_row_indices[sort_idx]
  unstacked_data = {attr: arr[sort_idx] for attr, arr in unstacked_data.items()}

  return mbi.Dataset(unstacked_data, child_domain), parent_row_indices
