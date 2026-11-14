from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager

from ..exceptions import MaxDelegationDepthExceeded

_CURRENT_DELEGATION_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar(
    "current_agent_delegation_depth", default=0
)


def get_current_delegation_depth() -> int:
    """Return the current nesting depth of Agent.as_tool() invocations (0 for top-level)."""
    return _CURRENT_DELEGATION_DEPTH.get()


@contextmanager
def enter_agent_tool_delegation(max_depth: int | None) -> Iterator[int]:
    """Check delegation depth budget, increment depth for the nested run, and safely reset."""
    current = _CURRENT_DELEGATION_DEPTH.get()
    if max_depth is not None and current >= max_depth:
        raise MaxDelegationDepthExceeded(
            f"Maximum agent delegation depth ({max_depth}) exceeded. "
            f"Attempted to start nested agent at depth {current + 1}.",
            current_depth=current,
            max_depth=max_depth,
        )

    token = _CURRENT_DELEGATION_DEPTH.set(current + 1)
    try:
        yield current + 1
    finally:
        _CURRENT_DELEGATION_DEPTH.reset(token)
