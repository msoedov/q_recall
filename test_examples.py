from inline_snapshot import snapshot


def test_basic():
    from examples.basic import mem0

    state = mem0("what is spec-driven development?")
    assert len(state.trace) == snapshot(30)


def test_sec_lease():
    from examples.sec_lease import lease_agent

    state = lease_agent("what is spec-driven development?")
    assert len(state.trace) == snapshot(26)


def test_self_healing():
    from examples.self_healing import mem0

    state = mem0("what is spec-driven development?")
    assert len(state.trace) == snapshot(26)
