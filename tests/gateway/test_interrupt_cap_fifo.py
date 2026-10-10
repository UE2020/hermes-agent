"""Accepted completions must survive the recursive follow-up handoff."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource
from gateway.run import GatewayRunner


@pytest.mark.asyncio
@pytest.mark.parametrize("internal", [True, False])
@pytest.mark.parametrize("tail_count", [0, 1, 3])
async def test_interrupt_cap_preserves_promoted_completion_and_fifo(internal, tail_count):
    runner = object.__new__(GatewayRunner)
    runner._MAX_INTERRUPT_DEPTH = 3
    runner._draining = False
    runner._session_states = {}
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="test", chat_type="dm")
    key = "agent:main:telegram:dm:test"
    events = [MessageEvent(text=f"completion-{i}", source=source, internal=internal)
              for i in range(tail_count + 2)]
    adapter = SimpleNamespace(_pending_messages={key: events[0]}, _active_sessions={})
    adapter.get_pending_message = lambda k: adapter._pending_messages.pop(k, None)
    runner._delivery_adapter_for = MagicMock(return_value=adapter)
    runner._session_state(key).conversation.queued_events.extend(events[1:])
    result = {"final_response": "still working", "messages": []}
    ctx = SimpleNamespace(source=source, session_id="test-session", session_key=key,
                          run_generation=1, _interrupt_depth=3, history=[],
                          _status_thread_metadata=None, result_holder=[result])
    pending_event, pending = await runner._run_agent_drain_pending(result, adapter, source, key)
    assert pending_event is events[0]
    assert adapter._pending_messages[key] is events[1]
    await runner._run_agent_queued_followup(ctx, adapter, pending, pending_event,
                                          result["final_response"], result, None)
    observed = [adapter.get_pending_message(key)]
    while runner._overflow_queue(key):
        observed.append(runner._promote_queued_event(key, adapter, None))
    assert observed == events, "recursion cap silently dropped an admitted completion"


@pytest.mark.asyncio
@pytest.mark.parametrize("occupied", [True, False])
@pytest.mark.parametrize("eventless", [True, False])
async def test_interrupt_cap_restores_head_without_merge_or_duplicate(occupied, eventless):
    runner = object.__new__(GatewayRunner)
    runner._MAX_INTERRUPT_DEPTH = 3
    runner._session_states = {}
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="test", chat_type="dm")
    key = "agent:main:telegram:dm:test"
    head = MessageEvent(text="deferred head", source=source, internal=True)
    staged = MessageEvent(text="accepted result", source=source, internal=True)
    adapter = SimpleNamespace(_pending_messages={key: staged} if occupied else {}, _active_sessions={})
    runner._delivery_adapter_for = MagicMock(return_value=adapter)
    ctx = SimpleNamespace(source=source, session_id="test-session", session_key=key,
                          run_generation=1, _interrupt_depth=3, history=[],
                          _status_thread_metadata=None, result_holder=[{}])
    await runner._run_agent_queued_followup(ctx, adapter, head.text, None if eventless else head,
                                          "response", {}, None)
    restored = adapter._pending_messages[key]
    assert restored.text == head.text
    if not eventless:
        assert restored is head
    assert (runner._overflow_queue(key) or []) == ([staged] if occupied else [])


@pytest.mark.asyncio
async def test_acknowledged_result_reaches_parent_after_real_adapter_cap_handoff():
    import asyncio
    import time
    from gateway.config import GatewayConfig, PlatformConfig
    from gateway.session import build_session_key
    from plugins.platforms.discord.adapter import DiscordAdapter
    from tools import async_delegation as delegation

    runner = GatewayRunner(GatewayConfig())
    runner._is_user_authorized = lambda _source: True
    adapter = DiscordAdapter(PlatformConfig(enabled=True, typing_indicator=False))
    from unittest.mock import AsyncMock
    from gateway.platforms.base import SendResult
    adapter.send = AsyncMock(return_value=SendResult(success=True))
    runner.adapters = {Platform.DISCORD: adapter}
    source = SessionSource(platform=Platform.DISCORD, chat_type="dm", chat_id="42", user_id="42")
    key = build_session_key(source)
    evt = {"type": "async_delegation", "session_key": key, "delegation_id": "cap-admitted",
           "summary": "auth-child-finished", "status": "completed", "dispatched_at": time.time()}
    delegation._persist_dispatch(evt)
    delegation._persist_completion(evt, {"status": "completed", "summary": evt["summary"]})
    entered, release = asyncio.Event(), asyncio.Event()
    received = []

    async def handler(event):
        received.append(event.text)
        if event.text == "active-human":
            entered.set()
            await release.wait()
            result = {"final_response": "still working", "messages": []}
            pending_event, pending = await runner._run_agent_drain_pending(result, adapter, source, key)
            ctx = SimpleNamespace(source=source, session_id="test-session", session_key=key,
                                  run_generation=1, _interrupt_depth=runner._MAX_INTERRUPT_DEPTH,
                                  history=[], _status_thread_metadata=None, result_holder=[result])
            await runner._run_agent_queued_followup(ctx, adapter, pending, pending_event,
                                                  result["final_response"], result, None)
        elif key not in adapter._pending_messages:
            staged = runner._promote_queued_event(key, adapter, None)
            if staged is not None:
                adapter._pending_messages[key] = staged

    async def drain():
        while adapter._background_tasks:
            await asyncio.gather(*list(adapter._background_tasks))
            await asyncio.sleep(0)

    adapter.set_message_handler(handler)
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)
    try:
        await adapter.handle_message(MessageEvent(text="active-human", source=source))
        await asyncio.wait_for(entered.wait(), 3)
        await adapter.handle_message(MessageEvent(text="pending-human", source=source))
        assert await runner._deliver_async_delegation_group([evt]) is True
        assert delegation.get_durable_delegation(evt["delegation_id"])["delivery_state"] == "delivered"
        release.set()
        await asyncio.wait_for(drain(), 5)
        assert received[:2] == ["active-human", "pending-human"]
        assert len(received) == 3 and evt["summary"] in received[2]
        assert await runner._deliver_async_delegation_group([evt]) is None
        assert not adapter._pending_messages
        assert not runner._overflow_queue(key)
    finally:
        release.set()
        await asyncio.wait_for(drain(), 5)
        await runner._cancel_process_completion_batch_tasks()
