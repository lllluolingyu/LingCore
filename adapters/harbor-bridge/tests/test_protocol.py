import json
from pathlib import Path

import pytest

from lingcore_harbor_bridge.protocol import BridgeProtocolError, decode, encode

FIXTURES = Path(__file__).resolve().parents[1] / "protocol/v1"


@pytest.mark.parametrize(
    "path", sorted(FIXTURES.glob("*.json")), ids=lambda path: path.stem
)
def test_shared_golden_frames(path):
    data = path.read_bytes()
    assert encode(decode(data)) == data
    invalid = json.loads(data)
    invalid["extra"] = "secret-value"
    with pytest.raises(BridgeProtocolError) as error:
        decode(json.dumps(invalid).encode() + b"\n")
    assert "secret-value" not in str(error.value)


def test_shared_atif_fixture_validates_with_pinned_harbor():
    from pathlib import Path

    from harbor.models.trajectories.trajectory import Trajectory

    path = Path(__file__).parents[1] / "protocol/atif/trajectory.json"
    trajectory = Trajectory.model_validate_json(path.read_bytes())
    assert trajectory.schema_version == "ATIF-v1.7"
    assert trajectory.steps[2].observation.results[0].source_call_id == "b"
