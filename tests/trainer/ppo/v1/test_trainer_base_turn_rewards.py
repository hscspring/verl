import numpy as np
import pytest

from verl.trainer.ppo.v1.trainer_base import PPOTrainer


def test_extracts_configured_turn_rewards_from_extra_fields():
    config = {
        "turn_reward_keys": ["intent_preservation", "relationship_continuity"],
        "turn_language_gate_key": "language",
        "turn_boundary_violation_key": "boundary_violation",
        "turn_platform_refusal_key": "platform_refusal",
    }
    keys = PPOTrainer._turn_reward_component_keys(config)
    extra_fields = np.array(
        [
            {
                "reward_extra_info": {
                    "intent_preservation": [1.0, 0.0],
                    "relationship_continuity": [0.5, 1.0],
                    "language": [0.0, -1.0],
                    "boundary_violation": [0.0, None],
                    "platform_refusal": [0.0, None],
                }
            },
            {
                "reward_extra_info": {
                    "intent_preservation": [0.0, 1.0],
                    "relationship_continuity": [1.0, 0.5],
                    "language": [0.0, 0.0],
                    "boundary_violation": [1.0, 0.0],
                    "platform_refusal": [0.0, 1.0],
                }
            },
        ],
        dtype=object,
    )

    result = PPOTrainer._extract_reward_extra_info(extra_fields, keys)

    assert keys == [
        "intent_preservation",
        "relationship_continuity",
        "language",
        "boundary_violation",
        "platform_refusal",
    ]
    assert result["intent_preservation"].tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert result["language"].tolist() == [[0.0, -1.0], [0.0, 0.0]]
    assert result["boundary_violation"].tolist() == [[0.0, None], [1.0, 0.0]]


def test_turn_reward_extraction_fails_on_missing_component():
    extra_fields = np.array(
        [{"reward_extra_info": {"intent_preservation": [1.0]}}],
        dtype=object,
    )

    with pytest.raises(KeyError, match="relationship_continuity"):
        PPOTrainer._extract_reward_extra_info(
            extra_fields,
            ["intent_preservation", "relationship_continuity"],
        )
