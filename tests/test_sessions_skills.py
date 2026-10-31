"""Restoration of approved skill grants from durable session events."""

from __future__ import annotations

import json
from pathlib import Path

from lingcore.agent import Agent
from lingcore.message import Message, ToolCall
from lingcore.sessions import (
    SessionStore,
    new_session_id,
    open_store,
)
from tests.fakes import FakeLLMClient, ScriptedTurn
from tests.support.sessions import _profile
from tests.support.sessions import store as store


async def test_agent_dynamic_skill_state_survives_rebuild(tmp_path: Path):
    profile = _profile(
        tmp_path,
        """
name: skill-session
workspace: .
llm:
  model: gpt-4o
tools:
  - activate_skill
  - read_file
  - search
initial_tools:
  - activate_skill
""",
    )
    store, _ = open_store(profile)
    assert store is not None
    with store:
        activate = ToolCall(
            id="activate-review",
            name="activate_skill",
            arguments={"name": "code-review"},
        )
        first_llm = FakeLLMClient(
            [
                ScriptedTurn(tool_calls=[activate], finish_reason="tool_calls"),
                ScriptedTurn(text="activated"),
            ]
        )
        first = Agent.from_profile(
            profile, llm=first_llm, base_dir=tmp_path, session_store=store
        )
        async for _ in first.run("review this"):
            pass
        sid = first.memory.session_id
        assert store.active_skills(sid) == ("code-review",)

        second_llm = FakeLLMClient([ScriptedTurn(text="still active")])
        await first.aclose()
        resumed = Agent.from_profile(
            profile,
            llm=second_llm,
            base_dir=tmp_path,
            session_store=store,
            session_id=sid,
        )
        assert resumed.skill_state is not None
        assert resumed.skill_state.active == ["code-review"]
        async for _ in resumed.run("continue"):
            pass

        system = second_llm.calls[0][0].content
        assert "Group findings by severity" in system
        names = {
            schema["function"]["name"] for schema in (second_llm.tool_schemas[0] or [])
        }
        assert {"activate_skill", "read_file", "search"} <= names


async def test_skill_restore_refuses_unapproved_high_risk_grant_drift(tmp_path: Path):
    profile = _profile(
        tmp_path,
        """
name: expanded-skill-session
workspace: .
llm:
  model: gpt-4o
tools:
  - activate_skill
  - read_file
  - search
  - run_shell
initial_tools:
  - activate_skill
""",
    )
    store, _ = open_store(profile)
    assert store is not None
    with store:
        sid = new_session_id()
        store.append(sid, Message.user("activated under an older safe ceiling"))
        # Legacy/safe activation: code-review previously received read/search,
        # but no high-risk grant was approved.
        store.save_skill_state(
            sid,
            active=["code-review"],
            activated=["code-review"],
            deactivated=[],
        )

        resumed = Agent.from_profile(
            profile,
            llm=FakeLLMClient([ScriptedTurn(text="unused")]),
            base_dir=tmp_path,
            session_store=store,
            session_id=sid,
        )
        assert resumed.skill_state is not None
        assert resumed.skill_state.active == []
        assert "run_shell" not in resumed._authorized_tool_names()

        # Once that exact risky grant has recorded consent, durable restoration
        # is allowed without widening it further.
        store.save_skill_state(
            sid,
            active=["code-review"],
            activated=["code-review"],
            deactivated=[],
            approved_high_risk={"code-review": ["run_shell"]},
        )
        await resumed.aclose()
        approved = Agent.from_profile(
            profile,
            llm=FakeLLMClient([ScriptedTurn(text="unused")]),
            base_dir=tmp_path,
            session_store=store,
            session_id=sid,
        )
        assert approved.skill_state is not None
        assert approved.skill_state.active == ["code-review"]
        assert "run_shell" in approved._authorized_tool_names()


def test_corrupt_latest_skill_state_fails_closed(store: SessionStore):
    sid = new_session_id()
    store.append(sid, Message.user("question"))
    store.save_skill_state(
        sid,
        active=["review"],
        activated=["review"],
        deactivated=[],
    )
    with store._lock, store._conn:
        corrupt = store._conn.execute(
            "INSERT INTO session_events "
            "(session_id, message_seq, kind, created_at, payload) "
            "VALUES (?, 0, 'skill_state', '2026-01-01T00:00:00+00:00', '{bad')",
            (sid,),
        )
    assert store.active_skills(sid) == ()

    # A syntactically valid activation without its claimed message anchor is
    # equally untrusted and must not restore authorization.
    orphaned = {
        "active": ["review"],
        "activated": ["review"],
        "deactivated": [],
        "approved_high_risk": {},
    }
    with store._lock, store._conn:
        store._conn.execute(
            "UPDATE session_events SET message_seq = 99, payload = ? "
            "WHERE event_seq = ?",
            (json.dumps(orphaned), int(corrupt.lastrowid)),
        )
    assert store.active_skills(sid) == ()
