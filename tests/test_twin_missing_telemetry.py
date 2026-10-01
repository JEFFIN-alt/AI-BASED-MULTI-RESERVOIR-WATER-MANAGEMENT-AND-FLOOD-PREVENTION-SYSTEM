"""Missing telemetry must not look like an empty, safe reservoir."""
from src.dashboard.twin_component.state_adapter import adapt_state_for_twin

def test_absent_reservoirs_have_no_fabricated_quantities():
    state = adapt_state_for_twin({})
    assert state['downstream_flow'] is None
    for row in state['reservoirs'].values():
        assert row['risk'] == 'unknown'
        for key in ('storage', 'water_level', 'gate', 'release', 'inflow'):
            assert row[key] is None

def test_partial_reservoir_preserves_unknown_fields():
    row = adapt_state_for_twin({'reservoirs': {'Virtual Reservoir D': {
        'storage_pct': 55, 'gate_position_pct': 0,
    }}})['reservoirs']['reservoir_4']
    assert row['water_level'] == .55
    assert row['gate'] == 0
    assert row['inflow'] is None
    assert row['release'] is None
    assert row['trend'] is None
    assert row['risk'] == 'unknown'
