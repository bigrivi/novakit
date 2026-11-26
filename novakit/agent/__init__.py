"""Agent runtime primitives shared by hosts and standalone novakit users."""

from novakit.agent.compaction import CompactionError
from novakit.agent.config import AgentConfig, CoreConfig
from novakit.agent.core import (
    Agent,
    AgentOptions,
    AgentRuntimeDependencies,
    ChatOptions,
    PreparedRequest,
)
from novakit.agent.llm_stream import EventData, TurnOutcome, TurnStreamReader
from novakit.agent.messages import (
    build_user_message,
    convert_to_llm_messages,
    count_user_turns,
)
from novakit.agent.overflow import (
    NON_OVERFLOW_PATTERNS,
    OVERFLOW_PATTERNS,
    is_context_overflow,
)
from novakit.agent.ports import AgentPorts
from novakit.agent.protocol import ChatAgent, InterruptibleAgent
from novakit.agent.session import AgentSession, AgentSessionContext
from novakit.agent.state import TurnState
from novakit.agent.title import (
    MAX_TITLE_LENGTH,
    TITLE_CALL_TIMEOUT_SECONDS,
    TITLE_SYSTEM_PROMPT,
    TITLE_USER_TEMPLATE,
    build_title_messages,
    clean_title,
    generate_session_title,
)
from novakit.agent.tool_guardrails import (
    GuardrailAction,
    GuardrailObservation,
    ToolGuardrails,
)
from novakit.agent.tool_invoker import (
    ToolInvoker,
    ToolOutcome,
    has_parsable_arguments,
    parse_tool_arguments,
)

__all__ = [
    "MAX_TITLE_LENGTH",
    "NON_OVERFLOW_PATTERNS",
    "OVERFLOW_PATTERNS",
    "TITLE_CALL_TIMEOUT_SECONDS",
    "TITLE_SYSTEM_PROMPT",
    "TITLE_USER_TEMPLATE",
    "Agent",
    "AgentConfig",
    "AgentOptions",
    "AgentPorts",
    "AgentRuntimeDependencies",
    "AgentSession",
    "AgentSessionContext",
    "ChatAgent",
    "ChatOptions",
    "CompactionError",
    "CoreConfig",
    "EventData",
    "GuardrailAction",
    "GuardrailObservation",
    "InterruptibleAgent",
    "PreparedRequest",
    "ToolGuardrails",
    "ToolInvoker",
    "ToolOutcome",
    "TurnOutcome",
    "TurnState",
    "TurnStreamReader",
    "build_title_messages",
    "build_user_message",
    "clean_title",
    "convert_to_llm_messages",
    "count_user_turns",
    "generate_session_title",
    "has_parsable_arguments",
    "is_context_overflow",
    "parse_tool_arguments",
]
