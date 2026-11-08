"""The event envelope and the twenty event kinds (spec section 2)."""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated, Literal, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    PlainSerializer,
    TypeAdapter,
    field_serializer,
    field_validator,
    model_validator,
)

JsonObject = dict[str, JsonValue]

# Money is Decimal in memory and a JSON number on disk, like the Messages API's own
# numbers; tokens stay the source of truth, so a float's rounding cannot matter.
Usd = Annotated[Decimal, PlainSerializer(float, return_type=float, when_used="json")]


class _Model(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, populate_by_name=True, serialize_by_alias=True
    )


class _Envelope(_Model):
    schema_: Literal["loomcore.event.v1"] = Field("loomcore.event.v1", alias="schema")
    seq: int = Field(ge=0)
    timestamp: AwareDatetime
    session_id: str = Field(alias="sessionId")
    run_id: str = Field(alias="runId")

    @field_validator("timestamp")
    @classmethod
    def _to_millis(cls, value: datetime) -> datetime:
        # The disk keeps milliseconds, so keep no more in memory: an event equals
        # its own read-back.
        return value.replace(microsecond=value.microsecond // 1000 * 1000)

    @field_serializer("timestamp")
    def _millis(self, value: datetime) -> str:
        return (
            value.astimezone(UTC)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )


class ForkOrigin(_Model):
    run_id: str = Field(alias="runId")
    seq: int = Field(ge=0)


class RunStarted(_Envelope):
    kind: Literal["run_started"] = "run_started"
    task: JsonValue
    config: JsonObject
    limits: JsonObject
    config_digest: str


class RunForked(_Envelope):
    kind: Literal["run_forked"] = "run_forked"
    from_: ForkOrigin = Field(alias="from")
    hint: str | None = None
    limits: JsonObject | None = None


class TurnStarted(_Envelope):
    kind: Literal["turn_started"] = "turn_started"
    turn: int


class ModelCall(_Envelope):
    kind: Literal["model_call"] = "model_call"
    type: Literal["assistant"] = "assistant"
    message: JsonObject
    duration_ms: int
    cost_usd: Usd
    pricing_version: str


class ToolRequested(_Envelope):
    kind: Literal["tool_requested"] = "tool_requested"
    tool_use_id: str
    name: str
    input: JsonObject
    call_hash: str


class ToolCompleted(_Envelope):
    kind: Literal["tool_completed"] = "tool_completed"
    tool_use_id: str
    ok: bool
    content: JsonValue
    duration_ms: int
    cached: bool


class GuardDecision(_Envelope):
    kind: Literal["guard_decision"] = "guard_decision"
    stage: Literal["pre", "post"]
    tool_use_id: str
    decision: Literal["allow", "deny", "ask", "redact"]
    reason: str

    @model_validator(mode="after")
    def _decision_fits_stage(self) -> Self:
        if self.decision == "ask" and self.stage != "pre":
            raise ValueError("ask is only a pre-stage decision")
        if self.decision == "redact" and self.stage != "post":
            raise ValueError("redact is only a post-stage decision")
        return self


class ApprovalRequested(_Envelope):
    kind: Literal["approval_requested"] = "approval_requested"
    tool_use_id: str
    question: str


class _Decided(_Envelope):
    tool_use_id: str
    by: str
    note: str


class ApprovalGranted(_Decided):
    kind: Literal["approval_granted"] = "approval_granted"


class ApprovalDenied(_Decided):
    kind: Literal["approval_denied"] = "approval_denied"


class SandboxDenied(_Envelope):
    kind: Literal["sandbox_denied"] = "sandbox_denied"
    tool_use_id: str | None = None
    action: str
    target: str
    reason: str


class ContextPacked(_Envelope):
    kind: Literal["context_packed"] = "context_packed"
    selected: list[JsonValue]
    dropped: list[JsonValue]
    tokens: int
    budget: int


class ContextCompacted(_Envelope):
    kind: Literal["context_compacted"] = "context_compacted"
    strategy: str
    tokens_before: int
    tokens_after: int


class _Verification(_Envelope):
    verifier: str
    message: str


class VerificationPassed(_Verification):
    kind: Literal["verification_passed"] = "verification_passed"


class VerificationFailed(_Verification):
    kind: Literal["verification_failed"] = "verification_failed"


class Escalation(_Envelope):
    kind: Literal["escalation"] = "escalation"
    rung: int
    reason: str
    tool_use_id: str | None = None
    call_hash: str | None = None
    config: JsonObject | None = None
    config_digest: str | None = None


class NoteWritten(_Envelope):
    kind: Literal["note_written"] = "note_written"
    path: str


class RunPaused(_Envelope):
    kind: Literal["run_paused"] = "run_paused"
    reason: Literal["awaiting_approval", "limit"]
    limit: str | None = None
    resume_from: int

    @model_validator(mode="after")
    def _limit_names_which(self) -> Self:
        if (self.reason == "limit") != (self.limit is not None):
            raise ValueError("limit is given when, and only when, the reason is limit")
        return self


class RunResumed(_Envelope):
    kind: Literal["run_resumed"] = "run_resumed"
    reason: Literal["approval", "limit", "crash"]
    resume_from: int
    limits: JsonObject | None = None

    @model_validator(mode="after")
    def _limit_resume_carries_limits(self) -> Self:
        if self.reason == "limit" and self.limits is None:
            raise ValueError("a resume after a limit pause must carry limits")
        return self


class RunFinished(_Envelope):
    kind: Literal["run_finished"] = "run_finished"
    reason: Literal["finished", "end_turn", "refused", "stuck", "killed", "error"]
    result: JsonValue = None


Event = Annotated[
    RunStarted
    | RunForked
    | TurnStarted
    | ModelCall
    | ToolRequested
    | ToolCompleted
    | GuardDecision
    | ApprovalRequested
    | ApprovalGranted
    | ApprovalDenied
    | SandboxDenied
    | ContextPacked
    | ContextCompacted
    | VerificationPassed
    | VerificationFailed
    | Escalation
    | NoteWritten
    | RunPaused
    | RunResumed
    | RunFinished,
    Field(discriminator="kind"),
]

_EVENT: TypeAdapter[Event] = TypeAdapter(Event)


def parse_event(line: str | bytes) -> Event:
    """Parse one log line; raises pydantic.ValidationError if it is not an event."""
    return _EVENT.validate_json(line)


def dump_event(event: Event) -> str:
    """One log line, without the newline."""
    return _EVENT.dump_json(event).decode()
