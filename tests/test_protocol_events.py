from cortex.protocol.events import EventEmitter


def test_event_emitter_increments_seq_per_session() -> None:
    emitted = []
    emitter = EventEmitter(send=emitted.append)

    emitter.emit(session_id="s1", event_type="system.notice", payload={"message": "a"})
    emitter.emit(session_id="s1", event_type="system.notice", payload={"message": "b"})
    emitter.emit(session_id="s2", event_type="system.notice", payload={"message": "c"})

    assert emitted[0]["params"]["seq"] == 1
    assert emitted[1]["params"]["seq"] == 2
    assert emitted[2]["params"]["seq"] == 1
    assert emitted[0]["params"]["session_id"] == "s1"
    assert emitted[2]["params"]["session_id"] == "s2"
