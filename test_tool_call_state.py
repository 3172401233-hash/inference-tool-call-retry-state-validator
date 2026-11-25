import unittest

from tool_call_state import (
    Action,
    CallStatus,
    evaluate,
    params_digest,
)


PARAMS = {"path": "/tmp/a.txt", "content": "hello"}
DIGEST = params_digest(PARAMS)
KEY = "idem-001"
CALL = "call-1"


def receipt(result="ok", digest=DIGEST, key=KEY):
    return {
        "tool_call_id": CALL,
        "idempotency_key": key,
        "params_digest": digest,
        "result": result,
    }


def base_conversation():
    return [
        {
            "type": "request",
            "tool_call_id": CALL,
            "idempotency_key": KEY,
            "params_digest": DIGEST,
            "irreversible": True,
        },
        {"type": "started", "tool_call_id": CALL},
    ]


class ResponseLossTest(unittest.TestCase):
    def test_response_lost_is_recoverable(self):
        conv = base_conversation() + [
            {"type": "receipt", "tool_call_id": CALL, "receipt": receipt()},
            {"type": "response_lost", "tool_call_id": CALL},
            {"type": "retry", "tool_call_id": CALL,
             "idempotency_key": KEY, "params_digest": DIGEST},
        ]
        verdict = evaluate(conv, CALL, DIGEST, KEY, receipt=receipt())
        self.assertEqual(verdict.status, CallStatus.RESPONSE_LOST)
        self.assertEqual(verdict.action, Action.RECOVER_RESULT)
        self.assertFalse(verdict.should_execute)
        self.assertEqual(verdict.reusable_result, "ok")
        self.assertTrue(any("response was lost" in e
                            for e in verdict.conflict_evidence))

    def test_delivered_response_is_reused_not_recovered(self):
        conv = base_conversation() + [
            {"type": "receipt", "tool_call_id": CALL, "receipt": receipt()},
            {"type": "response_delivered", "tool_call_id": CALL},
        ]
        verdict = evaluate(conv, CALL, DIGEST, KEY)
        self.assertEqual(verdict.status, CallStatus.COMPLETED)
        self.assertEqual(verdict.action, Action.REUSE_RESULT)
        self.assertFalse(verdict.should_execute)
        self.assertEqual(verdict.reusable_result, "ok")


class ParamDriftTest(unittest.TestCase):
    def test_param_drift_rejects_reuse(self):
        new_digest = params_digest({"path": "/tmp/a.txt", "content": "CHANGED"})
        conv = base_conversation() + [
            {"type": "receipt", "tool_call_id": CALL, "receipt": receipt()},
            {"type": "retry", "tool_call_id": CALL,
             "idempotency_key": KEY, "params_digest": new_digest},
        ]
        verdict = evaluate(conv, CALL, new_digest, KEY, receipt=receipt())
        self.assertEqual(verdict.status, CallStatus.CONFLICT)
        self.assertEqual(verdict.action, Action.REJECT_PARAM_CONFLICT)
        self.assertFalse(verdict.should_execute)
        self.assertIsNone(verdict.reusable_result)
        self.assertTrue(any("params_digest drift" in e
                            for e in verdict.conflict_evidence))

    def test_idempotency_key_drift_rejects_reuse(self):
        conv = base_conversation() + [
            {"type": "receipt", "tool_call_id": CALL, "receipt": receipt()},
        ]
        verdict = evaluate(conv, CALL, DIGEST, "idem-OTHER", receipt=receipt())
        self.assertEqual(verdict.status, CallStatus.CONFLICT)
        self.assertIsNone(verdict.reusable_result)
        self.assertTrue(any("idempotency_key drift" in e
                            for e in verdict.conflict_evidence))


class CancelTest(unittest.TestCase):
    def test_cancel_before_start_is_accepted(self):
        conv = [
            {"type": "request", "tool_call_id": CALL,
             "idempotency_key": KEY, "params_digest": DIGEST},
            {"type": "cancel", "tool_call_id": CALL},
        ]
        verdict = evaluate(conv, CALL, DIGEST, KEY)
        self.assertEqual(verdict.status, CallStatus.CANCELLED)
        self.assertEqual(verdict.action, Action.ACK_CANCEL)
        self.assertFalse(verdict.should_execute)

    def test_cancel_cannot_interrupt_irreversible_started_tool(self):
        conv = base_conversation() + [
            {"type": "cancel", "tool_call_id": CALL},
        ]
        verdict = evaluate(conv, CALL, DIGEST, KEY)
        self.assertEqual(verdict.status, CallStatus.IN_FLIGHT)
        self.assertEqual(verdict.action, Action.IGNORE_CANCEL)
        self.assertFalse(verdict.should_execute)
        self.assertTrue(any("cannot interrupt" in e
                            for e in verdict.conflict_evidence))

    def test_duplicate_cancel_is_idempotent(self):
        conv = [
            {"type": "request", "tool_call_id": CALL,
             "idempotency_key": KEY, "params_digest": DIGEST},
            {"type": "cancel", "tool_call_id": CALL},
            {"type": "cancel", "tool_call_id": CALL},
            {"type": "cancel", "tool_call_id": CALL},
        ]
        verdict = evaluate(conv, CALL, DIGEST, KEY)
        self.assertEqual(verdict.status, CallStatus.CANCELLED)
        self.assertEqual(verdict.action, Action.ACK_CANCEL)
        self.assertTrue(any("duplicate cancel: 3" in e
                            for e in verdict.conflict_evidence))

    def test_duplicate_cancel_after_irreversible_start_keeps_state(self):
        conv = base_conversation() + [
            {"type": "cancel", "tool_call_id": CALL},
            {"type": "cancel", "tool_call_id": CALL},
        ]
        verdict = evaluate(conv, CALL, DIGEST, KEY)
        self.assertEqual(verdict.status, CallStatus.IN_FLIGHT)
        self.assertEqual(verdict.action, Action.IGNORE_CANCEL)
        self.assertTrue(any("duplicate cancel: 2" in e
                            for e in verdict.conflict_evidence))


class ExecuteOnceTest(unittest.TestCase):
    def test_pending_call_executes_once(self):
        conv = [
            {"type": "request", "tool_call_id": CALL,
             "idempotency_key": KEY, "params_digest": DIGEST},
        ]
        verdict = evaluate(conv, CALL, DIGEST, KEY)
        self.assertEqual(verdict.status, CallStatus.PENDING)
        self.assertEqual(verdict.action, Action.EXECUTE)
        self.assertTrue(verdict.should_execute)

    def test_retry_with_same_key_and_digest_reuses_result(self):
        conv = base_conversation() + [
            {"type": "receipt", "tool_call_id": CALL, "receipt": receipt()},
            {"type": "retry", "tool_call_id": CALL,
             "idempotency_key": KEY, "params_digest": DIGEST},
            {"type": "retry", "tool_call_id": CALL,
             "idempotency_key": KEY, "params_digest": DIGEST},
        ]
        verdict = evaluate(conv, CALL, DIGEST, KEY, receipt=receipt())
        self.assertEqual(verdict.action, Action.REUSE_RESULT)
        self.assertFalse(verdict.should_execute)
        self.assertEqual(verdict.reusable_result, "ok")

    def test_retry_while_in_flight_waits(self):
        conv = base_conversation() + [
            {"type": "retry", "tool_call_id": CALL,
             "idempotency_key": KEY, "params_digest": DIGEST},
        ]
        verdict = evaluate(conv, CALL, DIGEST, KEY)
        self.assertEqual(verdict.status, CallStatus.IN_FLIGHT)
        self.assertEqual(verdict.action, Action.WAIT)
        self.assertFalse(verdict.should_execute)


if __name__ == "__main__":
    unittest.main()
