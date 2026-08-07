"""LangGraph checkpointer factory for SAP Agent Memory.

Usage::

    from sap_cloud_sdk.agent_memory.factory.langgraph_checkpoint import create_checkpointer

    # Auto-detects Agent Memory Service credentials.
    # Returns HanaAgentMemorySaver when credentials are present (persistent),
    # or InMemorySaver when they are not (local dev / no binding).
    checkpointer = create_checkpointer()

    # With TTL — only applies when falling back to in-memory (no credentials found).
    # Ignored when using HanaAgentMemorySaver (retention is managed server-side).
    checkpointer = create_checkpointer(ttl_seconds=3600)

    app = workflow.compile(checkpointer=checkpointer)

    # Or with LangChain create_agent:
    from langchain.agents import create_agent
    agent = create_agent(model="...", tools=[...], checkpointer=checkpointer)
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def create_checkpointer(*, ttl_seconds: Optional[int] = None):
    """Create a LangGraph checkpointer for the current environment.

    Automatically detects whether Agent Memory Service credentials are available:

    - **Credentials present** — returns ``HanaAgentMemorySaver`` backed by the
      Agent Memory Service (persistent across restarts).  Requires the
      ``langgraph-checkpoint-sap-agent-memory`` optional extra.
    - **No credentials** — falls back to ``InMemorySaver`` (or
      ``TimedInMemorySaver`` when *ttl_seconds* is set). State is in-process
      only and will not survive restarts.

    Args:
        ttl_seconds: Evict threads inactive for this many seconds.
                     Only applies to the in-memory fallback path.
                     Ignored (with a warning) when ``HanaAgentMemorySaver``
                     is returned — use the Agent Memory Service retention
                     config to control thread lifetime there.

    Returns:
        BaseCheckpointSaver instance.

    Raises:
        ImportError: If ``langgraph`` is not installed, or if credentials are
            found but ``langgraph-checkpoint-sap-agent-memory`` is not installed.

    Example — auto-detect (recommended)::

        checkpointer = create_checkpointer()
        app = workflow.compile(checkpointer=checkpointer)

    Example — in-memory with TTL (local dev)::

        checkpointer = create_checkpointer(ttl_seconds=3600)
        app = workflow.compile(checkpointer=checkpointer)
    """
    from sap_cloud_sdk.agent_memory.config import _load_config_from_env
    from sap_cloud_sdk.agent_memory.exceptions import AgentMemoryConfigError

    try:
        config = _load_config_from_env()
    except AgentMemoryConfigError:
        config = None

    if config is not None:
        try:
            from langgraph.checkpoint.sap.agent_memory import HanaAgentMemorySaver
        except ImportError as exc:
            raise ImportError(
                "langgraph-checkpoint-sap-agent-memory is required for persistent checkpointing. "
                "Install it with: "
                "pip install 'sap-cloud-sdk[langgraph-checkpoint-sap-agent-memory]'"
            ) from exc
        if ttl_seconds is not None:
            logger.warning(
                "create_checkpointer(): ttl_seconds=%d is ignored when using "
                "HanaAgentMemorySaver — thread retention is managed server-side "
                "via the Agent Memory Service retention config.",
                ttl_seconds,
            )
        logger.info("create_checkpointer(): using HanaAgentMemorySaver (persistent).")
        return HanaAgentMemorySaver(
            base_url=config.base_url,
            token_url=config.token_url,
            client_id=config.client_id,
            client_secret=config.client_secret,
            timeout=config.timeout,
        )

    try:
        from langgraph.checkpoint.memory import InMemorySaver
    except ImportError as exc:
        raise ImportError(
            "langgraph is required for create_checkpointer(). "
            "Install it with: pip install sap-cloud-sdk[langgraph] or pip install langgraph"
        ) from exc

    if ttl_seconds is not None:
        from sap_cloud_sdk.agent_memory.factory._timed_memory import TimedInMemorySaver

        logger.warning(
            "create_checkpointer(): using TimedInMemorySaver(ttl_seconds=%d) — "
            "session state is in-process only and will be lost on process exit.",
            ttl_seconds,
        )
        return TimedInMemorySaver(ttl_seconds=ttl_seconds)

    logger.warning(
        "create_checkpointer(): no Agent Memory Service credentials found, "
        "using InMemorySaver — session state is in-process only and will be "
        "lost on process exit."
    )
    return InMemorySaver()
