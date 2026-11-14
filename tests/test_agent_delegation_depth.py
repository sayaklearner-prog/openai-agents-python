"""Unit tests for inherited recursion-depth budget across nested Agent.as_tool() calls."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agents import Agent, RunConfig, Runner
from agents.exceptions import MaxDelegationDepthExceeded
from agents.run_internal.delegation_context import (
    enter_agent_tool_delegation,
    get_current_delegation_depth,
)


def test_delegation_context_manager_basic_lifecycle():
    assert get_current_delegation_depth() == 0

    with enter_agent_tool_delegation(2) as depth1:
        assert depth1 == 1
        assert get_current_delegation_depth() == 1

        with enter_agent_tool_delegation(2) as depth2:
            assert depth2 == 2
            assert get_current_delegation_depth() == 2

            # Third level should exceed limit of 2
            with pytest.raises(MaxDelegationDepthExceeded) as exc_info:
                with enter_agent_tool_delegation(2):
                    pass
            assert exc_info.value.max_depth == 2
            assert exc_info.value.current_depth == 2

        assert get_current_delegation_depth() == 1

    assert get_current_delegation_depth() == 0


@pytest.mark.asyncio
async def test_delegation_context_parallel_task_isolation():
    """Verify that sibling tasks running in parallel inherit but do not pollute each other's depth."""
    assert get_current_delegation_depth() == 0

    with enter_agent_tool_delegation(2):
        assert get_current_delegation_depth() == 1

        async def subtask(task_id: int) -> int:
            # Sibling starts at depth 1 inherited from parent
            assert get_current_delegation_depth() == 1
            with enter_agent_tool_delegation(2) as child_depth:
                assert child_depth == 2
                assert get_current_delegation_depth() == 2
                await asyncio.sleep(0.01)
                return task_id

        results = await asyncio.gather(subtask(1), subtask(2), subtask(3))
        assert results == [1, 2, 3]
        assert get_current_delegation_depth() == 1

    assert get_current_delegation_depth() == 0


def test_delegation_context_zero_limit_blocks_immediate_nesting():
    """A max_depth of 0 should immediately block any nested agent invocation."""
    assert get_current_delegation_depth() == 0

    with pytest.raises(MaxDelegationDepthExceeded) as exc_info:
        with enter_agent_tool_delegation(0):
            pass

    assert exc_info.value.max_depth == 0
    assert exc_info.value.current_depth == 0
    assert get_current_delegation_depth() == 0


def test_delegation_context_none_allows_unbounded_nesting():
    """max_depth=None disables the limit and allows unbounded recursion."""
    assert get_current_delegation_depth() == 0

    with enter_agent_tool_delegation(None):
        assert get_current_delegation_depth() == 1
        with enter_agent_tool_delegation(None):
            assert get_current_delegation_depth() == 2
            with enter_agent_tool_delegation(None):
                assert get_current_delegation_depth() == 3

    assert get_current_delegation_depth() == 0


def test_delegation_context_resets_on_unhandled_exception():
    """Verify that any unhandled exception cleanly restores ancestral depth."""
    assert get_current_delegation_depth() == 0

    try:
        with enter_agent_tool_delegation(5):
            assert get_current_delegation_depth() == 1
            raise RuntimeError("Unexpected failure during tool execution")
    except RuntimeError:
        pass

    assert get_current_delegation_depth() == 0
