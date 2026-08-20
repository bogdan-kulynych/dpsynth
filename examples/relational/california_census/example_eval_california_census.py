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

"""Evaluation and validation utilities for California Census relational synthesis.

Provides modular evaluation routines for:
1. Relational Integrity: Validating PK uniqueness, referential integrity (0
orphans), and group size capacity bounds.
2. Statistical Utility: 1-way marginal Total Variation Distance (TVD).
3. Machine Learning Utility: Train-on-Synthetic, Test-on-Real (TSTR) classifier
for EMPSTAT.
4. Multi-Table Quality & Diagnostics: SDMetrics Multi-Table Diagnostic and
Quality reports.
"""

from __future__ import annotations

from collections.abc import Mapping

from absl import logging
import pandas as pd
import sdmetrics
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection


# ==============================================================================
# 1. Validate Relational Integrity
# ==============================================================================
def validate_relational_integrity(
    synthetic_tables: Mapping[str, pd.DataFrame],
    max_children_per_parent: int = 8,
) -> None:
  """Validates relational consistency and foreign key integrity.

  Args:
    synthetic_tables: Dictionary of synthesized DataFrames.
    max_children_per_parent: Upper bound on child records per parent.

  Raises:
    ValueError: If primary key uniqueness is violated, orphaned records exist,
      or max child capacity bound is exceeded.
  """
  synth_h = synthetic_tables['household']
  synth_i = synthetic_tables['individual']

  # 1. Primary key uniqueness in parent table
  if synth_h['HOUSEHOLD'].nunique() != len(synth_h):
    raise ValueError('Primary key uniqueness violation in household table!')
  logging.info('Household primary key uniqueness: PASSED')

  # 2. Referential integrity (no orphaned children)
  orphan_mask = ~synth_i['HOUSEHOLD'].isin(synth_h['HOUSEHOLD'])
  num_orphans = int(orphan_mask.sum())
  if num_orphans > 0:
    raise ValueError(f'Found {num_orphans} orphaned individual records!')
  logging.info('Referential integrity (0 orphans): PASSED')

  # 3. Capacity constraints
  counts_per_h = synth_i['HOUSEHOLD'].value_counts()
  max_found = int(counts_per_h.max()) if not counts_per_h.empty else 0
  if max_found > max_children_per_parent:
    raise ValueError(
        f'Max children capacity exceeded: {max_found} >'
        f' {max_children_per_parent}'
    )
  mean_found = float(counts_per_h.mean()) if not counts_per_h.empty else 0.0
  logging.info(
      'Max children capacity check (max found = %d <= %d, mean = %.2f): PASSED',
      max_found,
      max_children_per_parent,
      mean_found,
  )


# ==============================================================================
# 2. Statistical Fidelity (Total Variation Distance)
# ==============================================================================
def compute_tvd(real_series: pd.Series, synth_series: pd.Series) -> float:
  """Computes Total Variation Distance (TVD in [0, 1]) between two series."""
  real_dist = real_series.value_counts(normalize=True)
  synth_dist = synth_series.value_counts(normalize=True)
  all_cats = real_dist.index.union(synth_dist.index)
  real_dist = real_dist.reindex(all_cats, fill_value=0.0)
  synth_dist = synth_dist.reindex(all_cats, fill_value=0.0)
  return float(0.5 * (real_dist - synth_dist).abs().sum())


def evaluate_statistical_fidelity(
    real_tables: Mapping[str, pd.DataFrame],
    synthetic_tables: Mapping[str, pd.DataFrame],
) -> dict[str, float]:
  """Evaluates 1-way marginal TVD across key household and individual columns."""
  scores = {
      'household.OWNERSHP': compute_tvd(
          real_tables['household']['OWNERSHP'],
          synthetic_tables['household']['OWNERSHP'],
      ),
      'household.ROOMS': compute_tvd(
          real_tables['household']['ROOMS'],
          synthetic_tables['household']['ROOMS'],
      ),
      'individual.EDUC': compute_tvd(
          real_tables['individual']['EDUC'],
          synthetic_tables['individual']['EDUC'],
      ),
      'individual.MARST': compute_tvd(
          real_tables['individual']['MARST'],
          synthetic_tables['individual']['MARST'],
      ),
  }
  for col_name, score in scores.items():
    logging.info('TVD [%s]: %.4f', col_name, score)
  return scores


# ==============================================================================
# 3. Machine Learning Utility (TSTR)
# ==============================================================================
def evaluate_downstream_ml_utility(
    real_tables: Mapping[str, pd.DataFrame],
    synthetic_tables: Mapping[str, pd.DataFrame],
    random_state: int = 42,
) -> tuple[float, float, float, float]:
  """Evaluates multi-table cross-feature correlations using TSTR benchmark.

  Trains a classifier to predict individual employment status (EMPSTAT)
  using features from both individual and household tables.

  Args:
    real_tables: Real input tables dictionary.
    synthetic_tables: Synthesized tables dictionary.
    random_state: Random seed for model reproducibility.

  Returns:
    Tuple of (real_acc, real_fscore, synth_acc, synth_fscore).
  """
  real_joined = real_tables['individual'].merge(
      real_tables['household'], on='HOUSEHOLD', how='inner'
  )
  synth_joined = synthetic_tables['individual'].merge(
      synthetic_tables['household'], on='HOUSEHOLD', how='inner'
  )

  features = ['AGE', 'SEX', 'EDUC', 'MARST', 'OWNERSHP', 'ROOMS', 'PUMA']
  target = 'EMPSTAT'

  train_real, test_real = model_selection.train_test_split(
      real_joined, test_size=0.3, random_state=random_state
  )

  # 1. Baseline: Train on Real, Test on Real
  clf_real = ensemble.RandomForestClassifier(
      n_estimators=50, random_state=random_state
  )
  clf_real.fit(train_real[features], train_real[target])
  pred_real = clf_real.predict(test_real[features])
  real_acc = float(metrics.accuracy_score(test_real[target], pred_real))
  real_fscore = float(
      metrics.f1_score(test_real[target], pred_real, average='macro')
  )

  # 2. Synthetic: Train on Synthetic, Test on Real (TSTR)
  clf_synth = ensemble.RandomForestClassifier(
      n_estimators=50, random_state=random_state
  )
  clf_synth.fit(synth_joined[features], synth_joined[target])
  pred_synth = clf_synth.predict(test_real[features])
  synth_acc = float(metrics.accuracy_score(test_real[target], pred_synth))
  synth_fscore = float(
      metrics.f1_score(test_real[target], pred_synth, average='macro')
  )

  logging.info(
      'Downstream ML (EMPSTAT) - Real Baseline (TRTR) -> Acc: %.4f,'
      ' Macro-FScore: %.4f',
      real_acc,
      real_fscore,
  )
  logging.info(
      'Downstream ML (EMPSTAT) - Synthetic (TSTR)    -> Acc: %.4f,'
      ' Macro-FScore: %.4f',
      synth_acc,
      synth_fscore,
  )
  return real_acc, real_fscore, synth_acc, synth_fscore


# ==============================================================================
# 4. SDMetrics Multi-Table Reports
# ==============================================================================
def generate_sdmetrics_reports(
    real_tables: Mapping[str, pd.DataFrame],
    synthetic_tables: Mapping[str, pd.DataFrame],
) -> tuple[
    sdmetrics.reports.multi_table.DiagnosticReport,
    sdmetrics.reports.multi_table.QualityReport,
]:
  """Generates SDMetrics Multi-Table Diagnostic and Quality reports.

  Args:
    real_tables: Real input tables dictionary.
    synthetic_tables: Synthesized tables dictionary.

  Returns:
    A tuple of (diagnostic_report, quality_report).
  """
  synth_h = synthetic_tables['household']
  synth_i = synthetic_tables['individual']

  # Construct multi-table metadata for SDMetrics
  metadata = {
      'tables': {
          'household': {
              'primary_key': 'HOUSEHOLD',
              'columns': {
                  col: {'sdtype': 'categorical'}
                  for col in synth_h.columns
                  if col != 'HOUSEHOLD'
              },
          },
          'individual': {
              'columns': {
                  col: {'sdtype': 'categorical'}
                  for col in synth_i.columns
                  if col != 'HOUSEHOLD'
              },
          },
      },
      'relationships': [{
          'parent_table_name': 'household',
          'parent_primary_key': 'HOUSEHOLD',
          'child_table_name': 'individual',
          'child_foreign_key': 'HOUSEHOLD',
      }],
  }

  logging.info('Generating SDMetrics DiagnosticReport...')
  diag_report = sdmetrics.reports.multi_table.DiagnosticReport()
  diag_report.generate(real_tables, synthetic_tables, metadata)
  logging.info('SDMetrics Diagnostic Score: %.4f', diag_report.get_score())

  logging.info('Generating SDMetrics QualityReport...')
  qual_report = sdmetrics.reports.multi_table.QualityReport()
  qual_report.generate(real_tables, synthetic_tables, metadata)
  logging.info('SDMetrics Quality Score: %.4f', qual_report.get_score())

  return diag_report, qual_report
