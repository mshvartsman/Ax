#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.core import (
    Experiment,
    Objective,
    OptimizationConfig,
    ParameterType,
    RangeParameter,
    SearchSpace,
)
from ax.modelbridge.dispatch_utils import choose_generation_strategy
from ax.runners.submitit import SubmitItMetricFetcher, SubmitItRunner
from ax.service.scheduler import FailureRateExceededError, Scheduler, SchedulerOptions
from ax.utils.common.testutils import TestCase

from ax.utils.measurement.synthetic_functions import branin
from submitit import DebugExecutor


class BraninCallable:
    def __call__(self, trial):
        params = trial.arm.parameters.values()
        return branin(*params)


class SubmitItRunnerTest(TestCase):
    def test_submitit_smoketest(self):
        parameters = [
            RangeParameter(
                name="x1",
                parameter_type=ParameterType.FLOAT,
                lower=-5,
                upper=10,
            ),
            RangeParameter(
                name="x2",
                parameter_type=ParameterType.FLOAT,
                lower=0,
                upper=15,
            ),
        ]

        objective = Objective(
            metric=SubmitItMetricFetcher(name="branin"), minimize=True
        )

        experiment = Experiment(
            name="branin_test_experiment",
            search_space=SearchSpace(parameters=parameters),
            optimization_config=OptimizationConfig(objective=objective),
            runner=SubmitItRunner(
                train_evaluate_fn=BraninCallable(),
                executor=DebugExecutor(folder="/tmp/submitit_test"),
            ),
            is_test=True,  # Marking this experiment as a test experiment.
        )

        generation_strategy = choose_generation_strategy(
            search_space=experiment.search_space,
            max_parallelism_cap=3,
        )

        scheduler = Scheduler(
            experiment=experiment,
            generation_strategy=generation_strategy,
            options=SchedulerOptions(),
        )
        try:
            scheduler.run_n_trials(max_trials=3)

        except FailureRateExceededError:
            self.fail()
