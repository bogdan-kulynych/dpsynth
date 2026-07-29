# Discrete Mechanisms for Synthetic Tabular Data

<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

This package contains the in-memory differentially private mechanisms that
learn a distribution over **discrete, integer-encoded tabular data**. They
select and measure low-dimensional marginals, fit a Private-PGM model to the
noisy measurements, and sample synthetic records from that model.

Most users should supply one of these mechanisms to
`dpsynth.TabularSynthesizer`. Direct use of this package is intended for
advanced workflows that already have an `mbi.Dataset`, an `mbi.CliqueVector`,
or precomputed noisy measurements.

## The 30-Second Mental Model

A mechanism does not privatize and copy individual rows. Instead, it learns
selected aggregate views of the data.

*   A **domain** gives each column a finite set of integer values.
*   A **clique** is a tuple of columns, such as `("age",)` or
    `("age", "income")`.
*   A **marginal** is the contingency table for a clique.
*   A **measurement** is a marginal with calibrated DP noise.
*   An **estimate** is a Markov random field whose marginals approximately
    agree with those measurements.

The common execution flow is:

```text
sensitive discrete data
        |
        v
measure or accept initial one-way marginals
        |
        v
optionally compress rare domain values
        |
        v
select additional cliques -> measure them with DP noise
        |
        v
estimate an mbi.MarkovRandomField with mirror descent
        |
        v
sample synthetic rows -> decompress to the original domain
```

Mechanisms mainly differ in **which additional cliques they select** and
whether selection happens once or adaptively over multiple rounds.

## Where This Package Fits

### Normal use: `TabularSynthesizer`

`dpsynth.TabularSynthesizer` is the public DataFrame-level interface. It:

1.  Privately initializes categorical and numerical columns.
2.  Encodes the input DataFrame into an integer `mbi.Dataset`.
3.  Supplies the noisy total and one-way column measurements to a discrete
    mechanism.
4.  Decodes the mechanism's synthetic integer records back to the original
    column types.

```python
import dpsynth
from dpsynth import discrete_mechanisms
import numpy as np

synthesizer = dpsynth.TabularSynthesizer(
    domains=domains,
    discrete_mechanism=discrete_mechanisms.MSTMechanism(),
)

calibrated = synthesizer.calibrate(epsilon=1.0, delta=1e-6)
result = calibrated(np.random.default_rng(42), sensitive_df)
synthetic_df = result.synthetic_data
```

See the [in-memory API guide](../../docs/in_memory_api.md) for domain and
DataFrame examples.

### Advanced use: run a mechanism directly

The low-level API expects data that is already discrete and integer encoded.

```python
from dpsynth import discrete_mechanisms
import mbi
import numpy as np

domain = mbi.Domain(["a", "b", "c"], [3, 4, 5])
data = mbi.Dataset(
    {"a": a_values, "b": b_values, "c": c_values},
    domain,
)

mechanism = discrete_mechanisms.MSTMechanism(
    compress_columns=True,
).configure(zcdp_rho=0.5)

result = mechanism(np.random.default_rng(42), data)
synthetic_data = result.synthetic_data
model = result.model
```

Use `configure(zcdp_rho=...)` when working directly in zCDP units. Use
`calibrate(epsilon=..., delta=...)` when an `(epsilon, delta)` guarantee is the
public privacy contract. Passing `zcdp_rho` to `calibrate()` is deprecated.

Both methods return a new configured dataclass; they do not mutate the original
mechanism.

## Shared Select-Measure-Estimate Architecture

All mechanisms inherit from [`base.DiscreteMechanism`](base.py). The base class
owns the common orchestration that used to be duplicated in every mechanism.

Its public call path is:

```text
__call__
  -> _check_calibration
  -> _measure_one_way
  -> _compress
  -> _run
       -> _select
       -> measure selected cliques
       -> estimate with mbi.estimation.MirrorDescent
       -> generate synthetic data
  -> decompress
  -> build diagnostics and DiscreteMechanismResult
```

The principal extension points are:

| Method | Responsibility | Usually overridden by |
|---|---|---|
| `supporting_cliques(domain)` | Declare the marginals the mechanism may need or support. | Every mechanism |
| `_allocate_budget(remaining_rho)` | Split the budget left after one-way measurement. | Every mechanism that measures or selects more cliques |
| `_one_way_cliques(data)` | Choose the initial one-way marginals. | AIM and AIM-GDP |
| `_select(...)` | Select additional cliques in one pass. | Direct, Independent, and MST |
| `_run(...)` | Replace the standard one-pass select-measure-estimate flow. | AIM, AIM-GDP, and SWIFT |
| `dp_event` | Describe the mechanism to `dp_accounting`. | Every mechanism |

The standard `_run()` also starts Private-PGM precompilation before measuring
the selected marginals. Compilation failures are logged and treated as
non-fatal; estimation still runs normally.

## Mechanisms and When to Use Them

| Mechanism | What it learns | Selection style | Good starting point when... |
|---|---|---|---|
| `IndependentMechanism` | One-way marginals only | No additional selection | You need a fast baseline or want to check whether correlations matter. |
| `DirectMechanism` | A caller-provided workload | Fixed, non-private choice supplied by the caller | You already know exactly which marginals matter. |
| `MSTMechanism` | One-way marginals plus a private spanning tree of pairwise marginals | One private selection pass | You want a practical default for pairwise relationships. |
| `AIMMechanism` | A workload refined over repeated rounds | Adaptive selection and measurement | Utility on a chosen workload matters more than runtime. |
| `AIMGDPMechanism` | The AIM workload using GDP-based internal allocation | Adaptive selection and measurement | You want the AIM algorithm with its GDP-oriented budgeting variant. |
| `SWIFTMechanism` | A workload supported by a bounded clique tree | Workload-informed selection with a custom estimator path | You need a denser workload on higher-dimensional data and can tune model-size limits. |

`MSTMechanism` is the default used by `TabularSynthesizer`.

### Independent

`IndependentMechanism` spends its standalone budget on one-way marginals and
selects nothing else. The resulting model preserves per-column distributions
but intentionally does not learn correlations between columns.

### Direct

`DirectMechanism` measures `prespecified_marginal_queries`. The selection is
not itself private because the workload must be chosen without inspecting the
sensitive data.

```python
mechanism = discrete_mechanisms.DirectMechanism(
    prespecified_marginal_queries=[("a", "b"), ("a", "c")],
)
```

It does not automatically spend budget on one-way marginals, although initial
measurements supplied by an orchestration layer are included in estimation.

### MST

`MSTMechanism` first builds an independent model from one-way measurements. It
privately scores pairwise errors, selects a maximum spanning tree with the
exponential mechanism, measures the selected two-way marginals, and fits the
final model.

The tree structure limits model complexity while representing one relationship
for every edge connecting the attributes.

### AIM and AIM-GDP

AIM is adaptive: it repeatedly compares the current model with candidate
marginals, privately selects a poorly approximated candidate, measures it, and
updates the model. It can anneal its per-round budget when additional noisy
measurements stop improving the estimate.

The `workload` may be an iterable of cliques or a mapping from cliques to
weights. `max_model_size`, `max_marginal_size`, and `max_rounds` control the
quality/runtime trade-off.

`AIMGDPMechanism` retains the same high-level loop but converts the loop's zCDP
allocation into GDP units internally. Its externally reported event is a
composition of the shared one-way Gaussian event and the adaptive loop's zCDP
event.

### SWIFT

SWIFT compiles a workload, estimates initial errors, chooses a subset of
marginals that fits within `max_clique_size`, builds a clique tree, and uses a
junction-tree-aware final estimation and sampling path. It therefore overrides
the shared `_run()` rather than only `_select()`.

Important tuning fields include `max_clique_size`, `max_marginal_size`,
`select_budget_frac`, and `pgm_iters`.

## Privacy Budget Model

The shared intermediate currency is zCDP rho because rho composes additively.
For a standalone mechanism with no externally supplied measurements:

```text
total rho = one_way_rho + remaining_rho
remaining_rho = selection rho + additional-measurement rho
```

Adaptive mechanisms allocate `remaining_rho` across their loop instead of a
single selection/measurement pass. Gaussian noise uses
`rho = 0.5 / sigma^2`.

The current default standalone allocations are:

| Mechanism | One-way share of total rho | Use of the remaining rho |
|---|---:|---|
| Independent | `1` | Nothing remains. |
| Direct | `0` | All of it measures the prespecified workload. |
| MST | `1/3` | `1/3` of the remainder selects; `2/3` measures. |
| AIM | `1/3` | All of the remainder funds the adaptive loop. |
| AIM-GDP | `1/3` | All of the remainder funds the GDP-based adaptive loop. |
| SWIFT | `0.1` | `0.1` of the remainder selects; `0.9` measures. |

For example, default MST receives `1/3` of total rho for one-way marginals,
`2/9` for selection, and `4/9` for measuring the selected pairwise marginals.

`dp_event` is the authoritative accounting description used by the high-level
calibration search. Depending on the mechanism, it is a `GaussianDpEvent`, a
`ZCDpEvent`, or a `ComposedDpEvent`.

## Initial Measurements

`initial_measurements` allows an upstream DP process to provide measurements
that can be reused as post-processing. This is how `TabularSynthesizer` passes
its privately initialized total and per-column measurements into the discrete
mechanism.

At the low level, two details matter:

1.  Passing any non-`None` `initial_measurements` to `__call__()` makes the base
    class use that list verbatim instead of making its own one-way
    measurements. It does not fill in missing one-way cliques.
2.  Pass the same non-`None` value to `configure()` when the mechanism should
    reallocate the normally reserved one-way rho to its later phases.

```python
configured = config.configure(
    zcdp_rho=0.5,
    initial_measurements=initial_measurements,
)
result = configured(
    rng,
    data,
    initial_measurements=initial_measurements,
)
```

The measurements must use cliques and domains compatible with the input data.
Callers should normally provide the complete one-way set required by their
workflow, not a partial set.

## Domain Compression

Set `compress_columns=True` to consider every measured one-way column, or pass
a sequence of column names to compress selectively. Compression merges rare
domain values using noisy one-way counts, reducing graphical-model size and
runtime.

Compression is post-processing of DP measurements:

*   The sensitive encoded dataset and measurements are mapped into the smaller
    domain.
*   Estimation runs in the compressed domain.
*   `result.model` remains defined over the compressed domain.
*   `result.synthetic_data` is decompressed back to the original domain.
*   `result.mappings` records the mappings that were applied.

Columns participating in structural constraints are skipped because their
constraints are expressed in the original domain.

## Inputs, Workloads, and `supporting_cliques`

`DiscreteMechanism.__call__()` accepts either:

*   `mbi.Dataset`: row-oriented encoded data, required when the mechanism needs
    operations such as dataset compression.
*   `mbi.CliqueVector`: precomputed exact marginals, useful when an upstream
    system has already aggregated the data.

`supporting_cliques(domain)` declares the cliques sufficient to run a
mechanism from a `CliqueVector`. For workload-based mechanisms it also explains
which marginal queries the resulting model is intended to answer.

A workload must be chosen independently of the sensitive data unless its
selection is itself differentially private. Workload weights describe relative
importance; they do not consume privacy by themselves.

## Result Object

Every mechanism returns a [`common.DiscreteMechanismResult`](common.py) with:

| Field | Meaning |
|---|---|
| `model` | The estimated `mbi.MarkovRandomField`; compressed when compression was used. |
| `synthetic_data` | Sampled `mbi.Dataset`, always restored to the original domain. |
| `measurements` | Initial and newly created noisy linear measurements used for estimation. |
| `diagnostics` | Phase timings and graphical-model/clique-tree size statistics. |
| `mappings` | Domain-compression mappings, or an empty dictionary. |

The DataFrame-level `TabularSynthesizer` wraps this inside
`DataGenerationResult.discrete_mechanism_result` and exposes the decoded
DataFrame as `DataGenerationResult.synthetic_data`.

## File Map

| Module | Role |
|---|---|
| [`base.py`](base.py) | Shared select-measure-estimate orchestration and budget hooks. |
| [`common.py`](common.py) | Measurement, compression, workload, timing, and diagnostic utilities. |
| [`accounting.py`](accounting.py) | zCDP, approximate-DP, GDP, Gaussian, and exponential-mechanism conversions. |
| [`independent.py`](independent.py) | One-way baseline. |
| [`direct.py`](direct.py) | Prespecified-workload mechanism. |
| [`mst.py`](mst.py) | Private maximum-spanning-tree selection. |
| [`aim.py`](aim.py) | Adaptive iterative mechanism with zCDP allocation. |
| [`aim_gdp.py`](aim_gdp.py) | AIM variant using GDP units internally. |
| [`swift.py`](swift.py) | Workload-informed clique-tree mechanism. |
| [`clique_tree.py`](clique_tree.py) | Clique-tree construction and local updates used by SWIFT. |
| [`swift_utils.py`](swift_utils.py) | SWIFT subset selection and budget-allocation utilities. |

## Migration Notes: Select-Measure-Estimate Refactor

Code written before the shared-base-class refactor should account for these
changes:

*   `dpsynth.discrete_mechanisms.DiscreteMechanism` now names the concrete
    shared base class, not an alias for the general `dpsynth.api.DPMechanism`.
*   Mechanism-specific `configure()` and `__call__()` implementations moved
    into `base.py`; simple mechanisms now expose hooks rather than complete
    pipelines.
*   AIM and AIM-GDP now inherit `pgm_iters=5000` instead of defining `1000`,
    and their default one-way share is now `1/3` instead of `0.1`.
*   Selection fractions for MST and SWIFT apply to the budget **remaining
    after** one-way measurement, not directly to total rho.
*   SWIFT uses `one_way_budget_fraction`; the old
    `one_way_budget_frac` constructor keyword is not accepted.
*   SWIFT reports a `ZCDpEvent` rather than a `GaussianDpEvent`.
*   `MechanismDiagnostics.num_rounds` was removed. Diagnostics now contain
    phase timings and structural model statistics.
*   `DirectMechanism.prespecified_marginal_queries` now defaults to an empty
    list instead of being a required constructor argument.
*   Initial measurements are treated as the complete initial set. In
    particular, `IndependentMechanism` no longer measures missing one-way
    cliques when given a partial initial set.

The result type and the public mechanism names remain available. The older
`AIMConfig`, `AIMGDPConfig`, `DirectConfig`, `IndependentConfig`, `MSTConfig`,
and `SWIFTConfig` names are compatibility aliases for the corresponding
`*Mechanism` classes.

## Relationship to Other Packages

*   [`data_generation_v3.py`](../data_generation_v3.py) provides
    `TabularSynthesizer`, the normal DataFrame-level orchestration layer.
*   [`domain.py`](../domain.py) defines public categorical and numerical domain
    specifications.
*   [`local_mode/`](../local_mode/) privately initializes columns before they
    are encoded and passed here.
*   [`transformations.py`](../transformations.py) provides shared encoding and
    domain transformation utilities.
*   [`pipeline_transformations/`](../pipeline_transformations/) contains the
    separate Apache Beam execution path. It shares mathematical ideas with this
    package but does not run through `base.DiscreteMechanism`.

## Tests

Shared base behavior is tested in
[`tests/discrete_mechanisms/base_test.py`](../../tests/discrete_mechanisms/base_test.py).
Cross-mechanism properties such as compression, calibration, result shape, and
`CliqueVector` support are tested in
[`tests/discrete_mechanisms/discrete_mechanisms_test.py`](../../tests/discrete_mechanisms/discrete_mechanisms_test.py).
Each mechanism also has its own focused test module.
