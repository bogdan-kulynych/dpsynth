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

"""Standard AIM with Gaussian DP accounting of its component mechanisms."""

from collections.abc import Iterable, Mapping
from collections.abc import Sequence
import dataclasses

from absl import logging
import dp_accounting
from dpsynth import api
from dpsynth.discrete_mechanisms import accounting
from dpsynth.discrete_mechanisms import aim
from dpsynth.discrete_mechanisms import common
import mbi
import mbi.junction_tree
import numpy as np


@dataclasses.dataclass(frozen=True)
class AimGdpAccConfig(api.MechanismConfig):
    """Configuration for AIM with Gaussian DP accounting.

    This is the standard AIM mechanism of
    [AIM: An Adaptive and Iterative Mechanism for Differentially Private Synthetic
    Data](https://arxiv.org/abs/2201.12677) -- the selection step is the
    exponential mechanism and the measurement step is the Gaussian mechanism --
    but the privacy budget is tracked in mu^2 GDP units rather than in zCDP rho.

    Attributes:
      workload: A collection of marginal queries (and weights) the synthetic data
        should be tailored to.
      max_rounds: The maximum number of rounds to run the mechanism.
      max_model_size: The maximum size of the graphical model in megabytes.
        Controls the utility/runtime trade-off.
      max_marginal_size: The maximum size of a marginal query to consider.
      anneal_factor: The factor by which to anneal the privacy.
      select_budget_fraction: The fraction of the total budget to use for
        selecting two-way marginal queries.
      gdp_mu: If set, the total privacy budget as a GDP mu, used directly instead
        of converting a zCDP budget. `configure` then rejects `zcdp_rho`.
    """

    workload: Mapping[mbi.Clique, float] | Iterable[mbi.Clique] | None = None
    max_rounds: int | None = None
    max_model_size: int = 80
    max_marginal_size: float = 1e6
    anneal_factor: float = 4.0
    select_budget_fraction: float = 0.1
    pgm_iters: int = 1000
    marginal_oracle: mbi.MarginalOracle | None = None
    gdp_mu: float | None = None

    def supporting_cliques(self, domain: mbi.Domain) -> list[mbi.Clique]:
        """Returns the workload cliques filtered by max_marginal_size."""
        return common.supporting_cliques(domain, self.workload, self.max_marginal_size)

    def configure(
        self,
        _=None,
        *,
        gdp_mu=None,
        zcdp_rho=None,
        delta=0,
        max_records_per_user=1,
    ):
        """Set GDP, for compatibility with the interface."""
        api.validate_max_records_per_user(max_records_per_user)
        if zcdp_rho is not None:
            raise ValueError(
                f"This mechanism takes no zCDP budget, got zcdp_rho={zcdp_rho}."
                " Pass gdp_mu instead."
            )
        if delta:
            raise ValueError(f"This mechanism consumes no delta, got delta={delta}.")
        if gdp_mu is None:
            raise ValueError("gdp_mu must be given.")
        return AimGdpAcc(
            config=self,
            gdp_mu=gdp_mu,
            max_records_per_user=max_records_per_user,
        )

    def calibrate(
        self,
        domain=None,
        /,
        *,
        epsilon: float,
        delta: float,
        poisson_sampling_prob: float = 1.0,
        max_records_per_user: int = 1,
    ) -> "AimGdpAcc":
        """Calibrate the mechanism to a target (epsilon, delta)-DP guarantee.

        The mechanism is exactly mu-GDP, so the largest admissible mu follows in
        closed form from the Gaussian tradeoff function and needs no search over
        zCDP budgets.

        Args:
          domain: Unused.
          epsilon: Target epsilon for (epsilon, delta)-DP.
          delta: Target delta for (epsilon, delta)-DP.
          poisson_sampling_prob: Unsupported, a subsampled mechanism is not GDP.
          max_records_per_user: Assumed upper bound on the number of records a
            single user contributes.

        Returns:
          A calibrated, runnable mechanism.
        """
        del domain
        if poisson_sampling_prob != 1.0:
            raise NotImplementedError("Poisson subsampling is not supported.")
        return self.configure(
            gdp_mu=accounting.gdp_mu(epsilon, delta),
            max_records_per_user=max_records_per_user,
        )


@dataclasses.dataclass(frozen=True, kw_only=True)
class AimGdpAcc(api.CalibratedMechanism):
    """Calibrated AimGdpAcc instance."""

    config: AimGdpAccConfig
    gdp_mu: float
    max_records_per_user: int = 1

    @property
    def dp_event(self) -> dp_accounting.DpEvent:
        """Returns the DP event for the mechanism."""
        return dp_accounting.GaussianDpEvent(1.0 / self.gdp_mu)

    def __call__(
        self,
        rng: np.random.Generator,
        data: mbi.Dataset | mbi.CliqueVector,
        *,
        initial_measurements: Sequence[mbi.LinearMeasurement] | None = None,
        constraints: Sequence[mbi.Constraint] = (),
    ) -> common.DiscreteMechanismResult:
        common.validate_initial_measurements(initial_measurements)
        measurements = list(initial_measurements) if initial_measurements else []
        phase_times = {}
        logging.info("[AIM-GDP-ACC]: Starting Mechanism.")
        gdp_musq = self.gdp_mu**2
        terminate = False
        musq_remaining = gdp_musq
        max_rounds = self.config.max_rounds or 16 * len(data.domain)
        musq_per_round = gdp_musq / max_rounds

        #########################################################################
        # Compile workload into candidate measurements.                         #
        #########################################################################
        candidates = common.compiled_workload(
            data.domain, self.config.workload, self.config.max_marginal_size
        )

        estimator = mbi.estimation.MirrorDescent(self.config.marginal_oracle)
        model = estimator.estimate(
            data.domain,
            measurements,
            iters=self.config.pgm_iters,
            constraints=constraints,
        )
        assert isinstance(model, mbi.MarkovRandomField)

        t = 0
        while not terminate:
            t += 1
            if musq_remaining < 2 * musq_per_round:
                logging.info(
                    "[AIM-GDP-ACC] Final round, Using all remaining privacy budget."
                )
                musq_per_round = musq_remaining
                terminate = True

            ########################################################################
            # Select a marginal query worst approximated by the current model.     #
            ########################################################################
            with common.timed(phase_times, "selection"):
                musq_remaining -= musq_per_round
                fraction = self.config.select_budget_fraction
                sigma = accounting.gdp_gaussian_sigma((1 - fraction) * musq_per_round)
                epsilon = accounting.gdp_bounded_range_nu(fraction * musq_per_round)
                size_limit = (
                    self.config.max_model_size * (gdp_musq - musq_remaining) / gdp_musq
                )
                small_candidates = aim._filter_candidates(
                    candidates, model, size_limit
                )  # pylint: disable=protected-access

                estimates = mbi.marginal_oracles.bulk_variable_elimination(
                    model.potentials,
                    list(small_candidates),
                    total=model.total,  # pyrefly: ignore[bad-argument-type]
                )
                marginal_query = (
                    aim._worst_approximated(  # pylint: disable=protected-access
                        rng,
                        small_candidates,
                        data,
                        estimates,
                        epsilon,
                        sigma,
                        data.domain,
                        max_records_per_user=self.max_records_per_user,
                    )
                )

            summary = mbi.summarize(
                data.domain, [m.clique for m in measurements] + [marginal_query]
            )
            logging.info(
                "[AIM-GDP-ACC] Round %d, Budget used: %.4f, Measuring: %s,"
                " Candidates: %d, cliques: %d, treewidth: %d, memory: %d bytes",
                t,
                (gdp_musq - musq_remaining) / gdp_musq,
                marginal_query,
                len(small_candidates),
                summary.num_cliques,
                summary.treewidth,
                summary.memory_bytes,
            )

            ######################################################################
            # Measure the marginal query privately using the Gaussian mechanism. #
            ######################################################################
            with common.timed(phase_times, "measurement"):
                measurement = common.measure_marginals_with_noise(
                    rng,
                    data,  # pyrefly: ignore[bad-argument-type]
                    [marginal_query],  # pyrefly: ignore[bad-argument-type]
                    sigma,
                    max_records_per_user=self.max_records_per_user,
                )[0]
                measurements.append(measurement)
                old_estimate = model.project(marginal_query).datavector()

            #####################################################
            # Estimate the data distribution using Private-PGM. #
            #####################################################
            with common.timed(phase_times, "estimation"):
                callback_fn = mbi.callbacks.default(measurements, data.domain)
                model = estimator.estimate(
                    data.domain,
                    measurements,
                    warm_start=model,
                    iters=self.config.pgm_iters,
                    callback_fn=callback_fn,
                    constraints=constraints,
                )
                assert isinstance(model, mbi.MarkovRandomField)

            new_estimate = model.project(marginal_query).datavector()

            ##########################################
            # Anneal epsilon and sigma if necessary. #
            ##########################################
            threshold = (
                self.max_records_per_user
                * sigma
                * np.sqrt(2 / np.pi)
                * data.domain.size(marginal_query)
            )
            if np.linalg.norm(new_estimate - old_estimate, ord=1) <= threshold:
                # No useful information at this noise level, increase budget per round.
                musq_per_round *= self.config.anneal_factor
                fraction = self.config.select_budget_fraction
                sigma = accounting.gdp_gaussian_sigma((1 - fraction) * musq_per_round)
                logging.info("[AIM-GDP-ACC] Reducing sigma: %.1f", sigma)

        return common.DiscreteMechanismResult(
            measurements=measurements,
            model=model,
            diagnostics=common.clique_stats(model),
        )
