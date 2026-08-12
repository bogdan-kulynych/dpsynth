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

"""Utilities for measuring and integer-encoding single columns."""

from __future__ import annotations

import dataclasses
import math
from typing import TypeVar

import dp_accounting
from dpsynth import api
from dpsynth import domain
from dpsynth.local_mode import _quantiles
from dpsynth.local_mode import primitives
from dpsynth.local_mode import vectorized_transformations as vtx
import mbi
import numpy as np
import scipy.stats

_M = TypeVar('_M')


def encode_to_grid(values, lower, upper, delta, **_):
  """Maps finite value(s) to quantile-grid index/indices in [0, grid_size - 1].

  Clipping to ``[lower, upper]`` folds out-of-grid values into the boundary bins
  and guarantees the returned index stays in range. Polymorphic over scalars and
  NumPy arrays; callers must handle NaN / out-of-domain filtering beforehand.

  Args:
    values: A finite scalar or NumPy array of standardized numerical values.
    lower: The inclusive lower bound of the candidate grid.
    upper: The inclusive upper bound of the candidate grid.
    delta: The spacing between adjacent grid points.

  Returns:
    The nearest grid index (or array of indices) as ``np.int64``.
  """
  clamped = np.clip(values, lower, upper)
  return np.round((clamped - lower) / delta).astype(np.int64)


@dataclasses.dataclass
class ColumnMeasurement:
  """Result of running a column initializer on raw data.

  Attributes:
    categorical_attribute: The discovered or constructed CategoricalAttribute
      defining the discrete domain for this column.
    bin_edges: Inner bin edges for numerical columns (used for
      discretize/undiscretize). None for categorical columns.
    measurement: A noisy one-way marginal measurement, or None if the
      initializer does not produce one (e.g. NumericalInitializer).
  """

  categorical_attribute: domain.CategoricalAttribute
  bin_edges: np.ndarray | None = None
  measurement: mbi.LinearMeasurement | None = None


def _validate_mechanism(mechanism: _M | None) -> _M:
  """Validates that the mechanism has been calibrated and returns it."""
  if mechanism is None:
    raise ValueError('Must call calibrate() before using the mechanism.')
  return mechanism


@dataclasses.dataclass
class NumericalInitializer(api.DPMechanism):
  """Mechanism that creates the data encoding transform for numerical data.

  Under the hood, this mechanism uses the Exponential Mechanism to privately
  identify quantile-based bin edges. These edges partition the continuous
  numerical domain into a discrete set of bins that maximize the uniform
  distribution of density.

  Internally calibrates and computes quantiles for privacy accounting.

  Attributes:
    name: Attribute name used as the clique key in the measurement.
    num_partitions: Number of quantile partitions (must be a power of 2).
    attribute: The NumericalAttribute defining the data domain.
    max_grid_size: Maximum size of the grid.
    max_records_per_user: Assumed upper bound on the number of records a single
      user contributes. Added noise (and mechanism sensitivity) is scaled by
      this factor to provide user-level rather than record-level DP; the privacy
      accounting is unchanged. Soundness relies on the caller enforcing this
      bound.
  """

  name: str
  num_partitions: int
  attribute: domain.NumericalAttribute
  max_grid_size: int = 10_000_000
  max_records_per_user: int = 1
  _epsilon_levels: tuple[float, ...] | None = dataclasses.field(
      default=None, repr=False
  )

  def __post_init__(self):
    api.validate_max_records_per_user(self.max_records_per_user)
    if self.max_grid_size < 2:
      raise ValueError(f'max_grid_size must be >= 2, got {self.max_grid_size}.')

  @property
  def _num_levels(self) -> int:
    result = int(np.log2(self.num_partitions))
    if 2**result != self.num_partitions:
      raise ValueError(f'{self.num_partitions=} must be a power of 2.')
    return result

  @property
  def _grid_spec(self) -> tuple[float, float, int]:
    """Returns (lower, upper, grid_size) for the quantile candidate grid."""
    attr = self.attribute
    if attr.dtype == 'int':
      # Reserve budget for the m-fold refinement so the refined grid fits.
      m = _quantiles.jitter_factor(self.num_partitions)
      budget = max(2, self.max_grid_size // m)
      int_range = int(attr.max_value - attr.min_value + 1)
      step = max(1, math.ceil(int_range / budget))
      gs = math.ceil(int_range / step)
      return (attr.min_value, attr.min_value + (gs - 1) * step, gs)
    return (attr.min_value, attr.exclusive_max_value, self.max_grid_size)

  @property
  def grid_size(self) -> int:
    """Grid size used for histogram construction."""
    return self._grid_spec[2]

  @property
  def zcdp_rho(self) -> float:
    if self._epsilon_levels is None:
      raise ValueError('Mechanism is uncalibrated.')
    return sum(e**2 / 8.0 for e in self._epsilon_levels)

  def configure(  # pyrefly: ignore[bad-override]
      self, *, zcdp_rho: float, delta: float = 0.0, epsilon_ratio: float = 2.0
  ) -> NumericalInitializer:
    """Returns a copy calibrated to the given zCDP budget."""
    levels = self._num_levels
    if levels == 0:
      return dataclasses.replace(self, _epsilon_levels=())
    rho_ratio = epsilon_ratio**2
    budget_weights = rho_ratio ** np.arange(levels)[::-1]
    rho_levels = zcdp_rho * budget_weights / budget_weights.sum()
    eps = np.sqrt(8.0 * rho_levels)
    return dataclasses.replace(self, _epsilon_levels=tuple(eps.tolist()))

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the composed privacy event for the quantile computation."""
    if self._epsilon_levels is None:
      raise ValueError('Mechanism is uncalibrated.')
    return dp_accounting.ComposedDpEvent([
        dp_accounting.ExponentialMechanismDpEvent(epsilon=float(eps))
        for eps in self._epsilon_levels
    ])

  def __call__(
      self,
      rng: np.random.Generator,
      data: np.ndarray,
      *,
      estimated_total: float | None = None,
  ) -> ColumnMeasurement:
    """Returns a ColumnMeasurement with the discretization transform."""
    counts = self._grid_histogram(data)
    return self.from_summary(rng, counts, estimated_total=estimated_total)

  def _grid_histogram(self, data):
    """Returns the quantile candidate-grid histogram (length grid_size)."""
    # Applies NumericalAttribute.standardize semantics in a vectorized manner.
    lower, upper, gs = self._grid_spec
    delta = (upper - lower) / (gs - 1)
    attr = self.attribute
    values = np.asarray(data, dtype=float)
    if attr.clip_to_range:
      values = np.where(np.isnan(values), attr.min_value, values)
    else:
      in_domain = (values >= attr.min_value) & (values <= attr.max_value)
      values = values[in_domain]
    if attr.dtype == 'int':
      values = np.round(values)
    indices = encode_to_grid(values, lower, upper, delta)
    return np.bincount(indices, minlength=gs)

  def from_summary(
      self,
      rng: np.random.Generator,
      counts: np.ndarray,
      *,
      estimated_total: float | None = None,
  ) -> ColumnMeasurement:
    """Returns a ColumnMeasurement from pre-aggregated histogram counts."""
    if self._epsilon_levels is None:
      raise ValueError('Mechanism is uncalibrated.')
    jitter_strategy = 'refine' if self.attribute.dtype == 'int' else 'symmetric'
    scaled_epsilon_levels = (
        np.asarray(self._epsilon_levels) / self.max_records_per_user
    )
    indices = _quantiles.quantiles_from_histogram(
        rng,
        counts,
        epsilon_levels=scaled_epsilon_levels,
        jitter_strategy=jitter_strategy,
    )
    lower, upper, _ = self._grid_spec
    delta = (upper - lower) / max(1, np.asarray(counts).size - 1)
    raw_edges = [lower + i * delta for i in indices]

    return edges_to_column_measurement(
        raw_edges=raw_edges,
        attribute=self.attribute,
        name=self.name,
        zcdp_rho=self.zcdp_rho,
        estimated_total=estimated_total,
        max_records_per_user=self.max_records_per_user,
    )


def edges_to_column_measurement(
    raw_edges,
    attribute,
    name,
    zcdp_rho,
    estimated_total=None,
    max_records_per_user=1,
):
  """Converts raw quantile edges into a ColumnMeasurement.

  Handles edge deduplication, degenerate-bin removal, and categorical
  attribute construction.  Shared between the data-based
  ``NumericalInitializer`` and the histogram-based
  ``HistogramNumericalInitializer``.

  Args:
    raw_edges: Quantile edge values (unsorted duplicates are fine).
    attribute: The ``NumericalAttribute`` defining the data domain.
    name: Attribute name used as the clique key in any measurement.
    zcdp_rho: Total zCDP rho consumed by the quantile mechanism.
    estimated_total: If provided, a heuristic one-way measurement is included.
    max_records_per_user: Assumed upper bound on the number of records a single
      user contributes. Added noise (and mechanism sensitivity) is scaled by
      this factor to provide user-level rather than record-level DP; the privacy
      accounting is unchanged. Soundness relies on the caller enforcing this
      bound.

  Returns:
    A ``ColumnMeasurement`` with bin edges and optionally a measurement.
  """
  raw_edges = np.asarray(raw_edges, dtype=float)
  bin_edges, edge_counts = np.unique(raw_edges, return_counts=True)
  # Edges at or above max_value produce a degenerate empty tail bin;
  # absorb their weight into the last real bin.
  max_val = attribute.max_value
  if len(bin_edges) > 0 and bin_edges[-1] >= max_val:
    tail_count = edge_counts[-1]
    bin_edges = bin_edges[:-1]
    edge_counts = edge_counts[:-1]
    bin_weights = np.append(edge_counts, tail_count + 1)
  else:
    bin_weights = np.append(edge_counts, 1)
  cat_attr = vtx.categorical_attribute_from_edges(bin_edges, attribute)

  measurement = None
  if estimated_total is not None:
    if not attribute.clip_to_range:
      # Prepend zero weight for the OUT_OF_DOMAIN slot at index 0.
      bin_weights = np.r_[0, bin_weights]
    counts = estimated_total * bin_weights / bin_weights.sum()
    stddev = max_records_per_user / np.sqrt(zcdp_rho)
    measurement = mbi.LinearMeasurement(
        counts,
        (name,),
        stddev=stddev,
        query=mbi.DatavectorQuery(use_for_total_estimation=False),
    )

  return ColumnMeasurement(cat_attr, bin_edges, measurement=measurement)


@dataclasses.dataclass
class CategoricalInitializer(api.DPMechanism):
  """Mechanism that measures a noisy histogram for categorical data.

  Under the hood, this mechanism uses the standard Gaussian Mechanism. It adds
  normally distributed noise to the exact counts of each closed-set category to
  produce a noisy marginal measurement.

  Attributes:
    name: Attribute name used as the clique key in the measurement.
    attribute: The CategoricalAttribute defining the closed domain.
    max_records_per_user: Assumed upper bound on the number of records a single
      user contributes. Added noise (and mechanism sensitivity) is scaled by
      this factor to provide user-level rather than record-level DP; the privacy
      accounting is unchanged. Soundness relies on the caller enforcing this
      bound.
  """

  name: str
  attribute: domain.CategoricalAttribute
  max_records_per_user: int = 1
  sigma: float | None = dataclasses.field(default=None, repr=False)

  def __post_init__(self):
    api.validate_max_records_per_user(self.max_records_per_user)

  def configure(  # pyrefly: ignore[bad-override]
      self, *, zcdp_rho: float, delta: float = 0.0
  ) -> CategoricalInitializer:
    """Returns a copy calibrated to the given zCDP budget."""
    return dataclasses.replace(self, sigma=math.sqrt(0.5 / zcdp_rho))

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the Gaussian privacy event for this mechanism."""
    if self.sigma is None:
      raise ValueError('Mechanism is uncalibrated.')
    return dp_accounting.GaussianDpEvent(noise_multiplier=self.sigma)

  def __call__(
      self, rng: np.random.Generator, data: np.ndarray
  ) -> ColumnMeasurement:
    """Returns a ColumnMeasurement with the noisy histogram."""
    encoded = vtx.discrete_encode(data, self.attribute)
    counts = np.bincount(encoded, minlength=self.attribute.size)
    return self.from_summary(rng, counts)

  def from_summary(
      self, rng: np.random.Generator, counts: np.ndarray
  ) -> ColumnMeasurement:
    """Returns a ColumnMeasurement from pre-aggregated counts."""
    if self.sigma is None:
      raise ValueError('Mechanism is uncalibrated.')
    noisy = primitives.add_gaussian_noise(
        rng, counts, self.sigma, self.max_records_per_user
    )
    noisy_counts = np.asarray(noisy)
    measurement = mbi.LinearMeasurement(
        noisy_counts,
        (self.name,),
        stddev=self.max_records_per_user * self.sigma,
    )
    return ColumnMeasurement(self.attribute, measurement=measurement)


@dataclasses.dataclass
class OpenSetCategoricalInitializer(api.DPMechanism):
  """Mechanism that discovers and measures an open-set categorical domain.

  Uses Gaussian Thresholding (Algorithm 2 from the DP-SIPS paper) to privately
  select significant partitions from the data and simultaneously obtain noisy
  counts for each discovered partition. The discovered partitions, together
  with the attribute's default_value (used as a catch-all for undiscovered
  values), form a CategoricalAttribute used for downstream synthesis.

  Under the hood, this mechanism relies on Gaussian Thresholding discovery to
  privately filter low-count tokens and outputs standard Gaussian noisy counts
  for only the tokens that cross the discovery threshold.

  Attributes:
    name: Attribute name used as the clique key in the measurement.
    attribute: The OpenSetCategoricalAttribute specifying the default value.
    delta: Failure probability for the partition selection threshold.
    min_count: Minimum true count for a partition to be discovered.
    max_records_per_user: Assumed upper bound on the number of records a single
      user contributes. Added noise (and mechanism sensitivity) is scaled by
      this factor to provide user-level rather than record-level DP; the privacy
      accounting is unchanged. Soundness relies on the caller enforcing this
      bound.
  """

  name: str
  attribute: domain.OpenSetCategoricalAttribute
  delta: float
  min_count: int = 1
  max_records_per_user: int = 1
  sigma: float | None = dataclasses.field(default=None, repr=False)

  def __post_init__(self):
    api.validate_max_records_per_user(self.max_records_per_user)

  def configure(  # pyrefly: ignore[bad-override]
      self, *, zcdp_rho: float, delta: float = 0.0
  ) -> OpenSetCategoricalInitializer:
    """Returns a copy calibrated to the given zCDP budget."""
    return dataclasses.replace(self, sigma=math.sqrt(0.5 / zcdp_rho))

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the privacy event including thresholding delta."""
    if self.sigma is None:
      raise ValueError('Mechanism is uncalibrated.')
    main_event = dp_accounting.GaussianDpEvent(noise_multiplier=self.sigma)
    failure_event = dp_accounting.dp_event.EpsilonDeltaDpEvent(0, self.delta)
    return dp_accounting.ComposedDpEvent([main_event, failure_event])

  def __call__(
      self, rng: np.random.Generator, data: np.ndarray
  ) -> ColumnMeasurement:
    """Returns a differentially private measurement of the given data."""
    unique_values, inverse = np.unique(data, return_inverse=True)
    counts = np.bincount(inverse)
    return self.from_summary(rng, unique_values, counts)

  def from_summary(
      self,
      rng: np.random.Generator,
      unique_values: np.ndarray,
      counts: np.ndarray,
  ) -> ColumnMeasurement:
    """Returns a ColumnMeasurement from pre-aggregated value counts."""
    if self.sigma is None:
      raise ValueError('Mechanism is uncalibrated.')

    above_min = counts >= self.min_count
    eligible_idx = np.where(above_min)[0]
    eligible_counts = counts[above_min].astype(float)

    noisy = primitives.add_gaussian_noise(
        rng, eligible_counts, self.sigma, self.max_records_per_user
    )
    noisy_counts = np.asarray(noisy)

    stddev = self.max_records_per_user * self.sigma
    base = float(self.max_records_per_user + self.min_count - 1)
    threshold = base + stddev * scipy.stats.norm.ppf(1.0 - self.delta)
    passed = noisy_counts >= threshold

    selected_partitions = eligible_idx[passed]
    estimated_counts = noisy_counts[passed]

    selected_values = np.array(
        [str(v) for v in unique_values[selected_partitions]]
    )

    if self.attribute.public_possible_values:
      pub = np.array(self.attribute.public_possible_values)
      selected_values, estimated_counts = primitives.ensure_public_partitions(
          rng,
          selected_values,
          estimated_counts,
          stddev,
          pub,
      )

    # Build the discovered domain: default first, then selected values.
    possible_values = [self.attribute.default_value] + selected_values.tolist()
    cat_attr = domain.CategoricalAttribute(
        possible_values=possible_values,  # pyrefly: ignore[unexpected-keyword]
        out_of_domain_index=0,  # pyrefly: ignore[unexpected-keyword]
    )

    # The measurement covers only the discovered partitions (indices 1:),
    # not the unmeasured default at index 0.
    measurement = mbi.LinearMeasurement(
        estimated_counts,  # pyrefly: ignore[bad-argument-type]
        (self.name,),
        stddev=stddev,
        query=mbi.SlicedQuery(start=1),
    )
    return ColumnMeasurement(cat_attr, measurement=measurement)
