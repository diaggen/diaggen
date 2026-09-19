from __future__ import annotations

import secrets
import threading
from collections.abc import Callable, Collection, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from hag4r.agentic.runtime_tools.diagnostics import LiveToolHandler
from hag4r.tools.genesis.live_client import GenesisLiveApiSession


class DiagnosticSessionState(str, Enum):
    CREATING = "creating"
    CREATED = "created"
    BOUND = "bound"
    RESET = "reset"
    ACTIVE = "active"
    RUNTIME_FAILED = "runtime_failed"
    FAILED_CLOSING = "failed_closing"
    CLOSED = "closed"
    CLOSED_FAILED = "closed_failed"


class DiagnosticSessionEvent(str, Enum):
    CONNECT_SUCCEEDED = "connect_succeeded"
    CONNECT_FAILED = "connect_failed"
    BIND_SUCCEEDED = "bind_succeeded"
    RESET_SUCCEEDED = "reset_succeeded"
    ACTIVITY_SUCCEEDED = "activity_succeeded"
    RUNTIME_FAILURE_RECORDED = "runtime_failure_recorded"
    EXPLICIT_CLOSE_SUCCEEDED = "explicit_close_succeeded"
    FAILURE_CLEANUP_STARTED = "failure_cleanup_started"
    FAILURE_CLEANUP_FINISHED = "failure_cleanup_finished"


class DiagnosticSessionRegistryError(RuntimeError):
    """Invalid registry identity, ownership, or lifecycle operation."""


DIAGNOSTIC_SESSION_TRANSITIONS: dict[
    tuple[DiagnosticSessionState, DiagnosticSessionEvent], DiagnosticSessionState
] = {
    (DiagnosticSessionState.CREATING, DiagnosticSessionEvent.CONNECT_SUCCEEDED): DiagnosticSessionState.CREATED,
    (DiagnosticSessionState.CREATING, DiagnosticSessionEvent.CONNECT_FAILED): DiagnosticSessionState.FAILED_CLOSING,
    (DiagnosticSessionState.CREATED, DiagnosticSessionEvent.BIND_SUCCEEDED): DiagnosticSessionState.BOUND,
    (DiagnosticSessionState.BOUND, DiagnosticSessionEvent.RESET_SUCCEEDED): DiagnosticSessionState.RESET,
    (DiagnosticSessionState.RESET, DiagnosticSessionEvent.ACTIVITY_SUCCEEDED): DiagnosticSessionState.ACTIVE,
    (DiagnosticSessionState.ACTIVE, DiagnosticSessionEvent.ACTIVITY_SUCCEEDED): DiagnosticSessionState.ACTIVE,
    **{
        (state, DiagnosticSessionEvent.RUNTIME_FAILURE_RECORDED): DiagnosticSessionState.RUNTIME_FAILED
        for state in (
            DiagnosticSessionState.BOUND,
            DiagnosticSessionState.RESET,
            DiagnosticSessionState.ACTIVE,
        )
    },
    **{
        (state, DiagnosticSessionEvent.EXPLICIT_CLOSE_SUCCEEDED): DiagnosticSessionState.CLOSED
        for state in (
            DiagnosticSessionState.CREATED,
            DiagnosticSessionState.BOUND,
            DiagnosticSessionState.RESET,
            DiagnosticSessionState.ACTIVE,
            DiagnosticSessionState.RUNTIME_FAILED,
        )
    },
    **{
        (state, DiagnosticSessionEvent.FAILURE_CLEANUP_STARTED): DiagnosticSessionState.FAILED_CLOSING
        for state in (
            DiagnosticSessionState.CREATING,
            DiagnosticSessionState.CREATED,
            DiagnosticSessionState.BOUND,
            DiagnosticSessionState.RESET,
            DiagnosticSessionState.ACTIVE,
            DiagnosticSessionState.RUNTIME_FAILED,
        )
    },
    (
        DiagnosticSessionState.FAILED_CLOSING,
        DiagnosticSessionEvent.FAILURE_CLEANUP_FINISHED,
    ): DiagnosticSessionState.CLOSED_FAILED,
}

TERMINAL_DIAGNOSTIC_SESSION_STATES = frozenset(
    {DiagnosticSessionState.CLOSED, DiagnosticSessionState.CLOSED_FAILED}
)
OPEN_DIAGNOSTIC_SESSION_STATES = frozenset(
    state for state in DiagnosticSessionState if state not in TERMINAL_DIAGNOSTIC_SESSION_STATES
)
NORMAL_CLOSE_STATES = frozenset(
    {
        DiagnosticSessionState.CREATED,
        DiagnosticSessionState.BOUND,
        DiagnosticSessionState.RESET,
        DiagnosticSessionState.ACTIVE,
        DiagnosticSessionState.RUNTIME_FAILED,
    }
)
FAILURE_CLOSE_STATES = frozenset({DiagnosticSessionState.CREATING, *NORMAL_CLOSE_STATES})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_run_root(run_root: str | Path) -> Path:
    if isinstance(run_root, str) and not run_root.strip():
        raise DiagnosticSessionRegistryError("run_root must be non-empty")
    return Path(run_root).expanduser().resolve()


def _require_identity(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DiagnosticSessionRegistryError(f"{field_name} must be a non-empty string")
    return value


def _format_error(error: BaseException | str) -> str:
    if isinstance(error, BaseException):
        return f"{type(error).__name__}: {error}"
    return str(error)


@dataclass
class DiagnosticSessionContext:
    handle: str
    run_root: Path
    episode_id: str
    agent_invocation_id: str
    live_session: GenesisLiveApiSession
    state: DiagnosticSessionState = DiagnosticSessionState.CREATING
    handlers: dict[str, LiveToolHandler] = field(default_factory=dict)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    registered_at: str = ""
    created_at: str | None = None
    bound_at: str | None = None
    reset_at: str | None = None
    active_at: str | None = None
    runtime_failed_at: str | None = None
    close_started_at: str | None = None
    closed_at: str | None = None
    tool_sequence: list[str] = field(default_factory=list)
    tool_events: list[dict[str, Any]] = field(default_factory=list)
    causal_error: str | None = None
    cleanup_error: str | None = None


class DiagnosticSessionRegistry:
    def __init__(
        self,
        *,
        handle_factory: Callable[[], str] | None = None,
        timestamp_factory: Callable[[], str] | None = None,
    ) -> None:
        self._handle_factory = handle_factory or (lambda: secrets.token_urlsafe(32))
        self._timestamp_factory = timestamp_factory or _utc_now
        self._lock = threading.RLock()
        self._contexts_by_handle: dict[str, DiagnosticSessionContext] = {}
        self._open_handle_by_run: dict[Path, str] = {}
        self._open_handle_by_run_episode: dict[tuple[Path, str], str] = {}
        self._shutting_down = False

    def create(
        self,
        *,
        run_root: str | Path,
        episode_id: str,
        agent_invocation_id: str,
        live_session: GenesisLiveApiSession,
    ) -> DiagnosticSessionContext:
        canonical_run_root = _canonical_run_root(run_root)
        episode_id = _require_identity(episode_id, field_name="episode_id")
        agent_invocation_id = _require_identity(agent_invocation_id, field_name="agent_invocation_id")
        with self._lock:
            if self._shutting_down:
                raise DiagnosticSessionRegistryError("diagnostic session registry is shutting down")
            run_episode = (canonical_run_root, episode_id)
            if run_episode in self._open_handle_by_run_episode:
                raise DiagnosticSessionRegistryError(
                    f"an open diagnostic session already exists for run/episode: {canonical_run_root} / {episode_id}"
                )
            if canonical_run_root in self._open_handle_by_run:
                raise DiagnosticSessionRegistryError(
                    f"an open diagnostic session already exists for run: {canonical_run_root}"
                )
            handle = _require_identity(self._handle_factory(), field_name="generated handle")
            if handle in self._contexts_by_handle:
                raise DiagnosticSessionRegistryError("generated diagnostic session handle is not unique")
            context = DiagnosticSessionContext(
                handle=handle,
                run_root=canonical_run_root,
                episode_id=episode_id,
                agent_invocation_id=agent_invocation_id,
                live_session=live_session,
                registered_at=self._timestamp_factory(),
            )
            self._contexts_by_handle[handle] = context
            self._open_handle_by_run[canonical_run_root] = handle
            self._open_handle_by_run_episode[run_episode] = handle
            return context

    def _resolve_context_locked(
        self,
        *,
        handle: str,
        run_root: Path,
        allow_terminal: bool,
        allowed_states: Collection[DiagnosticSessionState] | None,
    ) -> DiagnosticSessionContext:
        handle = _require_identity(handle, field_name="handle")
        context = self._contexts_by_handle.get(handle)
        if context is None:
            raise DiagnosticSessionRegistryError(f"unknown diagnostic session handle: {handle}")
        if context.run_root != run_root:
            raise DiagnosticSessionRegistryError("diagnostic session handle belongs to a different run root")
        if not allow_terminal and context.state in TERMINAL_DIAGNOSTIC_SESSION_STATES:
            raise DiagnosticSessionRegistryError(f"diagnostic session is terminal: {context.state.value}")
        if allowed_states is not None and context.state not in allowed_states:
            expected = ", ".join(sorted(state.value for state in allowed_states))
            raise DiagnosticSessionRegistryError(
                f"diagnostic session state {context.state.value} is not one of: {expected}"
            )
        return context

    def lookup(
        self,
        *,
        handle: str,
        run_root: str | Path,
        allowed_states: Collection[DiagnosticSessionState] | None = None,
    ) -> DiagnosticSessionContext:
        canonical_run_root = _canonical_run_root(run_root)
        with self._lock:
            return self._resolve_context_locked(
                handle=handle,
                run_root=canonical_run_root,
                allow_terminal=False,
                allowed_states=allowed_states,
            )

    def inspect(self, *, handle: str, run_root: str | Path) -> DiagnosticSessionContext:
        canonical_run_root = _canonical_run_root(run_root)
        with self._lock:
            return self._resolve_context_locked(
                handle=handle,
                run_root=canonical_run_root,
                allow_terminal=True,
                allowed_states=None,
            )

    def open_contexts(
        self,
        *,
        run_root: str | Path | None = None,
        agent_invocation_id: str | None = None,
    ) -> tuple[DiagnosticSessionContext, ...]:
        canonical_run_root = _canonical_run_root(run_root) if run_root is not None else None
        if agent_invocation_id is not None:
            agent_invocation_id = _require_identity(
                agent_invocation_id,
                field_name="agent_invocation_id",
            )
        with self._lock:
            return tuple(
                context
                for context in self._contexts_by_handle.values()
                if context.state in OPEN_DIAGNOSTIC_SESSION_STATES
                and (canonical_run_root is None or context.run_root == canonical_run_root)
                and (
                    agent_invocation_id is None
                    or context.agent_invocation_id == agent_invocation_id
                )
            )

    def has_open_session(self, *, run_root: str | Path) -> bool:
        canonical_run_root = _canonical_run_root(run_root)
        with self._lock:
            return canonical_run_root in self._open_handle_by_run

    @contextmanager
    def locked_context(
        self,
        *,
        handle: str,
        run_root: str | Path,
        allowed_states: Collection[DiagnosticSessionState] | None = None,
    ) -> Iterator[DiagnosticSessionContext]:
        canonical_run_root = _canonical_run_root(run_root)
        with self._lock:
            context = self._resolve_context_locked(
                handle=handle,
                run_root=canonical_run_root,
                allow_terminal=False,
                allowed_states=allowed_states,
            )
        with context.lock:
            with self._lock:
                self._resolve_context_locked(
                    handle=handle,
                    run_root=canonical_run_root,
                    allow_terminal=False,
                    allowed_states=allowed_states,
                )
            yield context

    def transition(
        self,
        *,
        context: DiagnosticSessionContext,
        event: DiagnosticSessionEvent,
        causal_error: BaseException | str | None = None,
        cleanup_error: BaseException | str | None = None,
    ) -> DiagnosticSessionContext:
        with context.lock:
            with self._lock:
                registered = self._contexts_by_handle.get(context.handle)
                if registered is not context:
                    raise DiagnosticSessionRegistryError("diagnostic session context is not registered")
                transition = (context.state, event)
                next_state = DIAGNOSTIC_SESSION_TRANSITIONS.get(transition)
                if next_state is None:
                    raise DiagnosticSessionRegistryError(
                        f"invalid diagnostic session transition: {context.state.value} + {event.value}"
                    )
                if event is DiagnosticSessionEvent.BIND_SUCCEEDED and not context.handlers:
                    raise DiagnosticSessionRegistryError(
                        "diagnostic session handlers must be stored before bind_succeeded"
                    )
                if event in {
                    DiagnosticSessionEvent.CONNECT_FAILED,
                    DiagnosticSessionEvent.FAILURE_CLEANUP_STARTED,
                } and causal_error is None:
                    raise DiagnosticSessionRegistryError(f"{event.value} requires a causal error")
                timestamp = self._timestamp_factory()
                if event is DiagnosticSessionEvent.CONNECT_SUCCEEDED:
                    context.created_at = timestamp
                elif event in {
                    DiagnosticSessionEvent.CONNECT_FAILED,
                    DiagnosticSessionEvent.FAILURE_CLEANUP_STARTED,
                }:
                    context.close_started_at = timestamp
                    if context.causal_error is None:
                        assert causal_error is not None
                        context.causal_error = _format_error(causal_error)
                elif event is DiagnosticSessionEvent.BIND_SUCCEEDED:
                    context.bound_at = timestamp
                elif event is DiagnosticSessionEvent.RESET_SUCCEEDED:
                    context.reset_at = timestamp
                elif event is DiagnosticSessionEvent.ACTIVITY_SUCCEEDED:
                    context.active_at = timestamp
                elif event is DiagnosticSessionEvent.RUNTIME_FAILURE_RECORDED:
                    context.runtime_failed_at = timestamp
                elif event is DiagnosticSessionEvent.EXPLICIT_CLOSE_SUCCEEDED:
                    context.closed_at = timestamp
                elif event is DiagnosticSessionEvent.FAILURE_CLEANUP_FINISHED:
                    context.closed_at = timestamp
                    if cleanup_error is not None:
                        context.cleanup_error = _format_error(cleanup_error)
                context.state = next_state
                if next_state in TERMINAL_DIAGNOSTIC_SESSION_STATES:
                    run_episode = (context.run_root, context.episode_id)
                    if self._open_handle_by_run.get(context.run_root) == context.handle:
                        del self._open_handle_by_run[context.run_root]
                    if self._open_handle_by_run_episode.get(run_episode) == context.handle:
                        del self._open_handle_by_run_episode[run_episode]
                return context

    def close(self, *, handle: str, run_root: str | Path) -> DiagnosticSessionContext:
        with self.locked_context(handle=handle, run_root=run_root, allowed_states=NORMAL_CLOSE_STATES) as context:
            try:
                context.live_session.close()
            except BaseException as error:
                self.transition(
                    context=context,
                    event=DiagnosticSessionEvent.FAILURE_CLEANUP_STARTED,
                    causal_error=error,
                )
                self.transition(
                    context=context,
                    event=DiagnosticSessionEvent.FAILURE_CLEANUP_FINISHED,
                )
                raise
            return self.transition(context=context, event=DiagnosticSessionEvent.EXPLICIT_CLOSE_SUCCEEDED)

    def fail_and_close(
        self,
        *,
        handle: str,
        run_root: str | Path,
        causal_error: BaseException | str,
    ) -> DiagnosticSessionContext:
        with self.locked_context(handle=handle, run_root=run_root, allowed_states=FAILURE_CLOSE_STATES) as context:
            self.transition(
                context=context,
                event=DiagnosticSessionEvent.FAILURE_CLEANUP_STARTED,
                causal_error=causal_error,
            )
            return self._finish_failure_cleanup(context)

    def _finish_failure_cleanup(self, context: DiagnosticSessionContext) -> DiagnosticSessionContext:
        cleanup_error: BaseException | None = None
        try:
            context.live_session.close()
        except BaseException as error:
            cleanup_error = error
        return self.transition(
            context=context,
            event=DiagnosticSessionEvent.FAILURE_CLEANUP_FINISHED,
            cleanup_error=cleanup_error,
        )

    def _shutdown_context(
        self,
        *,
        context: DiagnosticSessionContext,
        causal_error: BaseException | str,
    ) -> DiagnosticSessionContext | None:
        with context.lock:
            with self._lock:
                registered = self._contexts_by_handle.get(context.handle)
                if registered is not context:
                    raise DiagnosticSessionRegistryError("diagnostic session context is not registered")
                state = context.state
            if state in TERMINAL_DIAGNOSTIC_SESSION_STATES:
                return None
            if state is not DiagnosticSessionState.FAILED_CLOSING:
                if state not in FAILURE_CLOSE_STATES:
                    raise DiagnosticSessionRegistryError(
                        f"cannot shut down diagnostic session in state: {state.value}"
                    )
                self.transition(
                    context=context,
                    event=DiagnosticSessionEvent.FAILURE_CLEANUP_STARTED,
                    causal_error=causal_error,
                )
            return self._finish_failure_cleanup(context)

    @staticmethod
    def _record_shutdown_failure(
        context: DiagnosticSessionContext,
        *,
        causal_error: BaseException | str,
        shutdown_error: BaseException,
    ) -> None:
        with context.lock:
            if context.causal_error is None:
                context.causal_error = _format_error(causal_error)
            formatted_error = _format_error(shutdown_error)
            if context.cleanup_error is None:
                context.cleanup_error = formatted_error
            else:
                context.cleanup_error = f"{context.cleanup_error}; shutdown failure: {formatted_error}"

    def shutdown(
        self,
        *,
        causal_error: BaseException | str = "diagnostic MCP runtime shutdown",
    ) -> tuple[DiagnosticSessionContext, ...]:
        with self._lock:
            if self._shutting_down:
                return ()
            self._shutting_down = True
            open_contexts = tuple(
                context
                for context in self._contexts_by_handle.values()
                if context.state in OPEN_DIAGNOSTIC_SESSION_STATES
            )
        closed_contexts: list[DiagnosticSessionContext] = []
        for context in open_contexts:
            try:
                closed_context = self._shutdown_context(
                    context=context,
                    causal_error=causal_error,
                )
            except BaseException as error:
                self._record_shutdown_failure(
                    context,
                    causal_error=causal_error,
                    shutdown_error=error,
                )
                continue
            if closed_context is not None:
                closed_contexts.append(closed_context)
        return tuple(closed_contexts)
