"""Tool call retry state validator (Python 3.8.2 compatible).

Pure state-machine validation over a conversation event log. This module
never invokes real tools; it only decides what the client *should* do next
for a given tool call after a connection interruption, retry, or cancel.

Guarantees enforced:
- The same call (tool_call_id + idempotency_key + params_digest) executes
  at most once; retries reuse or recover the existing outcome.
- A retry with different parameters must NOT reuse the old result; it is
  reported as a conflict with evidence.
- If the tool executed but the response was lost, the call is recoverable:
  the retained receipt is returned and re-execution is forbidden.
- A cancel cannot interrupt an irreversible tool that already started.
"""

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class CallStatus(str, Enum):
    PENDING = "pending"
    IN_FLIGHT = "in_flight"
    RESPONSE_LOST = "response_lost"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    CONFLICT = "conflict"


class Action(str, Enum):
    EXECUTE = "execute"
    REUSE_RESULT = "reuse_result"
    RECOVER_RESULT = "recover_result"
    WAIT = "wait"
    REJECT_PARAM_CONFLICT = "reject_param_conflict"
    ACK_CANCEL = "ack_cancel"
    IGNORE_CANCEL = "ignore_cancel"
    NOOP = "noop"


def params_digest(params: Any) -> str:
    """Canonical digest of tool call parameters (order-independent)."""
    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class Verdict:
    status: CallStatus
    action: Action
    should_execute: bool
    reusable_result: Optional[Any] = None
    conflict_evidence: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "action": self.action.value,
            "should_execute": self.should_execute,
            "reusable_result": self.reusable_result,
            "conflict_evidence": list(self.conflict_evidence),
        }


def _events_for(conversation: List[Dict[str, Any]], tool_call_id: str):
    for event in conversation:
        if event.get("tool_call_id") == tool_call_id:
            yield event


def evaluate(
    conversation: List[Dict[str, Any]],
    tool_call_id: str,
    digest: str,
    idempotency_key: str,
    receipt: Optional[Dict[str, Any]] = None,
) -> Verdict:
    """Compute the next action for one tool call.

    Parameters:
        conversation: ordered event log. Recognised event types:
            request, started, receipt, response_delivered, response_lost,
            retry, cancel.
        tool_call_id: the call under evaluation.
        digest: params_digest() of the parameters the client wants to run now.
        idempotency_key: idempotency key the client presents now.
        receipt: execution receipt known to the client/server, if any. Must
            contain idempotency_key, params_digest and result.
    """
    evidence = []  # type: List[str]

    started = False
    delivered = False
    lost = False
    irreversible = False
    cancel_count = 0
    recorded_digests = set()
    recorded_keys = set()
    logged_receipt = None

    for event in _events_for(conversation, tool_call_id):
        etype = event.get("type")
        if etype == "request":
            irreversible = bool(event.get("irreversible", False))
            if event.get("params_digest"):
                recorded_digests.add(event["params_digest"])
            if event.get("idempotency_key"):
                recorded_keys.add(event["idempotency_key"])
        elif etype == "started":
            started = True
        elif etype == "receipt":
            logged_receipt = event.get("receipt")
        elif etype == "response_delivered":
            delivered = True
        elif etype == "response_lost":
            lost = True
        elif etype == "retry":
            if event.get("params_digest"):
                recorded_digests.add(event["params_digest"])
            if event.get("idempotency_key"):
                recorded_keys.add(event["idempotency_key"])
        elif etype == "cancel":
            cancel_count += 1

    effective_receipt = receipt or logged_receipt
    if effective_receipt:
        recorded_digests.add(effective_receipt.get("params_digest"))
        recorded_keys.add(effective_receipt.get("idempotency_key"))

    # --- Parameter / key drift: never reuse a result across different input.
    drifted_digests = sorted(d for d in recorded_digests if d and d != digest)
    drifted_keys = sorted(k for k in recorded_keys if k and k != idempotency_key)
    if drifted_digests or drifted_keys:
        for d in drifted_digests:
            evidence.append(
                "params_digest drift: recorded=%s requested=%s" % (d, digest)
            )
        for k in drifted_keys:
            evidence.append(
                "idempotency_key drift: recorded=%s requested=%s"
                % (k, idempotency_key)
            )
        evidence.append("refusing to reuse result computed for different input")
        return Verdict(
            status=CallStatus.CONFLICT,
            action=Action.REJECT_PARAM_CONFLICT,
            should_execute=False,
            reusable_result=None,
            conflict_evidence=evidence,
        )

    # --- Cancel handling.
    if cancel_count:
        if cancel_count > 1:
            evidence.append(
                "duplicate cancel: %d cancel events, extra cancels are no-ops"
                % cancel_count
            )
        if started and irreversible:
            evidence.append(
                "cancel arrived after irreversible tool started; "
                "cancel cannot interrupt execution"
            )
        elif not started:
            evidence.append("cancel accepted before execution started")
            return Verdict(
                status=CallStatus.CANCELLED,
                action=Action.ACK_CANCEL,
                should_execute=False,
                conflict_evidence=evidence,
            )
        # started and reversible: fall through; execution state wins.

    # --- Execution outcome.
    if effective_receipt:
        result = effective_receipt.get("result")
        if lost and not delivered:
            evidence.append(
                "tool executed but response was lost; receipt retained, "
                "recover result via idempotency_key=%s instead of re-running"
                % idempotency_key
            )
            return Verdict(
                status=CallStatus.RESPONSE_LOST,
                action=Action.RECOVER_RESULT,
                should_execute=False,
                reusable_result=result,
                conflict_evidence=evidence,
            )
        evidence.append(
            "call already executed once; retry reuses the stored receipt"
        )
        return Verdict(
            status=CallStatus.COMPLETED,
            action=Action.REUSE_RESULT,
            should_execute=False,
            reusable_result=result,
            conflict_evidence=evidence,
        )

    if started:
        evidence.append(
            "execution already in flight; retry must wait, not duplicate"
        )
        action = Action.IGNORE_CANCEL if (cancel_count and irreversible) else Action.WAIT
        return Verdict(
            status=CallStatus.IN_FLIGHT,
            action=action,
            should_execute=False,
            conflict_evidence=evidence,
        )

    return Verdict(
        status=CallStatus.PENDING,
        action=Action.EXECUTE,
        should_execute=True,
        conflict_evidence=evidence,
    )
