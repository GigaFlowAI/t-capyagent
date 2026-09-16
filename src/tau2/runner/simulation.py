"""
Layer 1: Simulation execution.

Runs a pre-built orchestrator and evaluates the result.
No registry dependency, no config parsing, no side effects.
"""

import atexit
import os
from typing import Optional, Union

from gigaflow import agent, configure, shutdown
from loguru import logger
from opentelemetry import trace

from tau2.data_model.simulation import SimulationRun
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.full_duplex_orchestrator import FullDuplexOrchestrator
from tau2.orchestrator.modes import CommunicationMode
from tau2.orchestrator.orchestrator import Orchestrator

_gigaflow_configured = False


def _configure_gigaflow() -> None:
    """Configure GigaFlow once and flush its spans at process exit."""
    global _gigaflow_configured
    if _gigaflow_configured:
        return
    configure(
        service_name="tau2",
        endpoint=os.environ["GIGAFLOW_OTLP_ENDPOINT"],
        api_key=os.environ["GIGAFLOW_OTLP_TOKEN"],
    )
    atexit.register(shutdown)
    _gigaflow_configured = True


def run_simulation(
    orchestrator: Union[Orchestrator, FullDuplexOrchestrator],
    *,
    evaluation_type: EvaluationType = EvaluationType.ALL,
    env_kwargs: Optional[dict] = None,
) -> SimulationRun:
    """Run a simulation and evaluate the result.

    Takes a fully constructed orchestrator (with agent, user, environment, and task
    already wired in), runs the simulation, evaluates it, and returns the result
    with reward_info attached.

    This is the lowest-level entry point. It has no dependency on the registry
    or RunConfig -- everything is already encapsulated in the orchestrator.

    Args:
        orchestrator: A fully constructed Orchestrator (half-duplex) or
            FullDuplexOrchestrator (full-duplex/voice). Must have agent, user,
            environment, and task set.
        evaluation_type: The type of evaluation to perform. Defaults to ALL.
        env_kwargs: Additional kwargs passed to the evaluator's environment
            constructor (e.g., retrieval_variant for banking_knowledge).

    Returns:
        SimulationRun with reward_info attached.

    Example:
        # Build your own instances (no registry needed):
        env = MyEnvironment()
        agent = MyAgent(tools=env.get_tools(), domain_policy=env.get_policy())
        user = UserSimulator(llm="gpt-4.1", instructions=task.user_scenario,
                             tools=env.get_user_tools())
        orchestrator = Orchestrator(
            domain="airline", agent=agent, user=user,
            environment=env, task=task, max_steps=100,
        )
        result = run_simulation(orchestrator)
        print(result.reward_info.reward)
    """
    _configure_gigaflow()
    with agent(
        name="tau2.simulation", conversation_id=orchestrator.simulation_id
    ):
        # Run the orchestrator
        simulation = orchestrator.run()

        # Save the actual policy used for this simulation
        simulation.policy = orchestrator.environment.get_policy()

        # Extract context from the orchestrator -- no external params needed
        domain = orchestrator.environment.get_domain_name()
        task = orchestrator.task
        is_full_duplex = isinstance(orchestrator, FullDuplexOrchestrator)
        mode = (
            CommunicationMode.FULL_DUPLEX
            if is_full_duplex
            else CommunicationMode.HALF_DUPLEX
        )
        solo_mode = getattr(orchestrator, "solo_mode", False)

        # Evaluate
        reward_info = evaluate_simulation(
            simulation=simulation,
            task=task,
            evaluation_type=evaluation_type,
            solo_mode=solo_mode,
            domain=domain,
            mode=mode,
            env_kwargs=env_kwargs,
        )
        simulation.reward_info = reward_info

        span = trace.get_current_span()
        span.set_attribute("tau2.simulation.id", simulation.id)
        span.set_attribute("tau2.task_id", simulation.task_id)
        span.set_attribute("tau2.reward", reward_info.reward)
        span.set_attribute("tau2.duration_seconds", simulation.duration)
        span.set_attribute(
            "tau2.termination_reason", simulation.termination_reason.value
        )
        span.set_attribute("tau2.tool_call_count", orchestrator.num_tool_calls)
        span.set_attribute("tau2.tool_error_count", orchestrator.num_errors)

        logger.info(
            f"Simulation complete: domain={domain}, task={task.id}, "
            f"reward={reward_info.reward}"
        )

    return simulation
