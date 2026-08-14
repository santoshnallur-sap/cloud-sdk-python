"""Unit tests for the create_checkpointer() LangGraph factory."""

import builtins
import logging
from typing import Any, TypedDict
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from sap_cloud_sdk.agent_memory.config import AgentMemoryConfig
from sap_cloud_sdk.agent_memory.exceptions import AgentMemoryConfigError
from sap_cloud_sdk.agent_memory.factory.langgraph_checkpoint import create_checkpointer

_NO_CREDENTIALS = "sap_cloud_sdk.agent_memory.config._load_config_from_env"
_VALID_CONFIG = AgentMemoryConfig(
    base_url="https://agent-memory.example.com",
    token_url="https://tenant.authentication.region.hana.ondemand.com/oauth/token",
    client_id="client-id",
    client_secret="client-secret",
)



class TestCreateCheckpointer:
    """Tests for create_checkpointer() factory."""

    # ── Return type ───────────────────────────────────────────────────────────

    def test_returns_in_memory_saver(self):
        """Factory returns LangGraph's InMemorySaver."""
        result = create_checkpointer()
        assert isinstance(result, InMemorySaver)

    def test_ttl_seconds_still_returns_in_memory_saver(self):
        """ttl_seconds returns TimedInMemorySaver."""
        from sap_cloud_sdk.agent_memory.factory._timed_memory import TimedInMemorySaver
        result = create_checkpointer(ttl_seconds=3600)
        assert isinstance(result, TimedInMemorySaver)

    def test_ttl_seconds_logs_warning(self, caplog):
        """ttl_seconds logs a warning about in-process state."""
        import logging
        with caplog.at_level(
            logging.WARNING,
            logger="sap_cloud_sdk.agent_memory.factory.langgraph_checkpoint",
        ):
            create_checkpointer(ttl_seconds=3600)
        assert "TimedInMemorySaver" in caplog.text

    # ── Missing langgraph ─────────────────────────────────────────────────────

    def test_missing_langgraph_raises_import_error(self):
        """Clear ImportError when langgraph is not installed."""
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "langgraph.checkpoint.memory":
                raise ImportError("No module named 'langgraph'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="langgraph is required"):
                create_checkpointer()

    # ── LangGraph integration ─────────────────────────────────────────────────

    def test_checkpointer_compiles_with_langgraph_graph(self):
        """Returned checkpointer can compile a LangGraph StateGraph."""
        class SimpleState(TypedDict):
            value: str

        def noop(state: SimpleState) -> SimpleState:
            return state

        builder = StateGraph(SimpleState)  # type: ignore
        builder.add_node("noop", noop)
        builder.add_edge(START, "noop")
        builder.add_edge("noop", END)

        app: Any = builder.compile(checkpointer=create_checkpointer())
        assert app is not None

    def test_checkpointer_persists_state_across_invocations(self):
        """State is preserved across invocations on the same thread_id."""
        class TickState(TypedDict):
            values: list

        def append_node(state: TickState) -> TickState:
            return {"values": state["values"] + ["tick"]}

        builder = StateGraph(TickState)  # type: ignore
        builder.add_node("append", append_node)
        builder.add_edge(START, "append")
        builder.add_edge("append", END)

        app: Any = builder.compile(checkpointer=create_checkpointer())
        config: RunnableConfig = {"configurable": {"thread_id": "test-thread-persist"}}

        result1 = app.invoke({"values": []}, config)
        assert result1["values"] == ["tick"]

        result2 = app.invoke({"values": result1["values"]}, config)
        assert result2["values"] == ["tick", "tick"]

    def test_different_thread_ids_are_isolated(self):
        """Two thread IDs maintain independent state."""
        class NameState(TypedDict):
            name: str

        def noop(state: NameState) -> NameState:
            return state

        builder = StateGraph(NameState)  # type: ignore
        builder.add_node("noop", noop)
        builder.add_edge(START, "noop")
        builder.add_edge("noop", END)

        app: Any = builder.compile(checkpointer=create_checkpointer())
        config_a: RunnableConfig = {"configurable": {"thread_id": "thread-isolation-a"}}
        config_b: RunnableConfig = {"configurable": {"thread_id": "thread-isolation-b"}}

        app.invoke({"name": "alice"}, config_a)
        app.invoke({"name": "bob"}, config_b)

        assert app.get_state(config_a).values["name"] == "alice"
        assert app.get_state(config_b).values["name"] == "bob"


@pytest.mark.no_autouse_credentials
class TestCreateCheckpointerPersistentBackend:
    """Tests for the HanaAgentMemorySaver path in create_checkpointer()."""

    # ── Happy path ────────────────────────────────────────────────────────────

    def test_returns_hana_saver_when_credentials_available(self):
        """Returns HanaAgentMemorySaver when credentials are present and extra is installed."""
        mock_saver_instance = MagicMock()
        mock_saver_class = MagicMock(return_value=mock_saver_instance)

        with patch(_NO_CREDENTIALS, return_value=_VALID_CONFIG):
            with patch.dict(
                "sys.modules",
                {"langgraph.checkpoint.sap.agent_memory": MagicMock(HanaAgentMemorySaver=mock_saver_class)},
            ):
                result = create_checkpointer()

        mock_saver_class.assert_called_once_with(
            base_url=_VALID_CONFIG.base_url,
            token_url=_VALID_CONFIG.token_url,
            client_id=_VALID_CONFIG.client_id,
            client_secret=_VALID_CONFIG.client_secret,
            timeout=_VALID_CONFIG.timeout,
            ttl_seconds=None,
        )
        assert result is mock_saver_instance

    def test_credentials_passed_correctly_to_hana_saver(self):
        """Each AgentMemoryConfig field is forwarded to HanaAgentMemorySaver."""
        config = AgentMemoryConfig(
            base_url="https://custom.example.com",
            token_url="https://custom.auth.example.com/oauth/token",
            client_id="my-client",
            client_secret="my-secret",
            timeout=60.0,
        )
        mock_saver_class = MagicMock()

        with patch(_NO_CREDENTIALS, return_value=config):
            with patch.dict(
                "sys.modules",
                {"langgraph.checkpoint.sap.agent_memory": MagicMock(HanaAgentMemorySaver=mock_saver_class)},
            ):
                create_checkpointer()

        mock_saver_class.assert_called_once_with(
            base_url="https://custom.example.com",
            token_url="https://custom.auth.example.com/oauth/token",
            client_id="my-client",
            client_secret="my-secret",
            timeout=60.0,
            ttl_seconds=None,
        )

    # ── Fallback paths ────────────────────────────────────────────────────────

    def test_falls_back_to_in_memory_when_no_credentials(self):
        """Falls back to InMemorySaver when no credentials are found."""
        with patch(_NO_CREDENTIALS, side_effect=AgentMemoryConfigError("no credentials")):
            result = create_checkpointer()
        assert isinstance(result, InMemorySaver)

    def test_falls_back_to_timed_saver_when_no_credentials_and_ttl_set(self):
        """Falls back to TimedInMemorySaver when no credentials but ttl_seconds is given."""
        from sap_cloud_sdk.agent_memory.factory._timed_memory import TimedInMemorySaver

        with patch(_NO_CREDENTIALS, side_effect=AgentMemoryConfigError("no credentials")):
            result = create_checkpointer(ttl_seconds=3600)
        assert isinstance(result, TimedInMemorySaver)

    # ── Error handling ────────────────────────────────────────────────────────

    def test_raises_import_error_when_extra_not_installed(self):
        """Raises ImportError with install hint when extra is missing despite credentials."""
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "langgraph.checkpoint.sap.agent_memory":
                raise ImportError("No module named 'langgraph.checkpoint.sap'")
            return original_import(name, *args, **kwargs)

        with patch(_NO_CREDENTIALS, return_value=_VALID_CONFIG):
            with patch("builtins.__import__", side_effect=mock_import):
                with pytest.raises(ImportError, match="langgraph-checkpoint-sap-agent-memory"):
                    create_checkpointer()

    # ── TTL forwarding ────────────────────────────────────────────────────────

    def test_ttl_seconds_forwarded_to_hana_saver(self):
        """ttl_seconds is passed through to HanaAgentMemorySaver constructor."""
        mock_saver_class = MagicMock()

        with patch(_NO_CREDENTIALS, return_value=_VALID_CONFIG):
            with patch.dict(
                "sys.modules",
                {"langgraph.checkpoint.sap.agent_memory": MagicMock(HanaAgentMemorySaver=mock_saver_class)},
            ):
                create_checkpointer(ttl_seconds=3600)

        _, kwargs = mock_saver_class.call_args
        assert kwargs["ttl_seconds"] == 3600

    def test_ttl_seconds_none_forwarded_to_hana_saver(self):
        """ttl_seconds=None is forwarded when no TTL is requested."""
        mock_saver_class = MagicMock()

        with patch(_NO_CREDENTIALS, return_value=_VALID_CONFIG):
            with patch.dict(
                "sys.modules",
                {"langgraph.checkpoint.sap.agent_memory": MagicMock(HanaAgentMemorySaver=mock_saver_class)},
            ):
                create_checkpointer()

        _, kwargs = mock_saver_class.call_args
        assert kwargs["ttl_seconds"] is None
