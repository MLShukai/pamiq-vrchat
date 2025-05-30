import pytest
import torch

from pamiq_vrchat import ActionType
from pamiq_vrchat.actuators import MouseButton, OscAxes, OscButtons
from sample.transforms.action import (
    ACTION_CHOICES,
    MOUSE_ACTION_CHOICES,
    OSC_ACTION_CHOICES,
    ActionTransform,
    MouseTransform,
    OscTransform,
)


class TestMouseTransform:
    """Tests for the MouseTransform class."""

    def test_init(self):
        """Test initialization with different velocity values."""
        MouseTransform(10.0, 20.0)

    @pytest.mark.parametrize("h_idx,h_val", [(0, 0.0), (1, 100.0), (2, -100.0)])
    @pytest.mark.parametrize("v_idx,v_val", [(0, 0.0), (1, 50.0), (2, -50.0)])
    @pytest.mark.parametrize("left", [0, 1])
    @pytest.mark.parametrize("right", [0, 1])
    @pytest.mark.parametrize("middle", [0, 1])
    def test_call_valid_input(self, h_idx, h_val, v_idx, v_val, left, right, middle):
        """Test transformation with valid input tensor."""
        transform = MouseTransform(100.0, 50.0)
        action = torch.tensor([h_idx, v_idx, left, right, middle], dtype=torch.long)

        result = transform(action)

        assert result.get("move_velocity") == (h_val, v_val)
        assert "button_press" in result
        assert result["button_press"][MouseButton.LEFT] == bool(left)
        assert result["button_press"][MouseButton.RIGHT] == bool(right)
        assert result["button_press"][MouseButton.MIDDLE] == bool(middle)

    @pytest.mark.parametrize(
        "action,error_msg",
        [
            (torch.zeros((2, 5)), "Action tensor must be 1-dimensional"),
            (
                torch.zeros(3),
                f"Action tensor must have {len(MOUSE_ACTION_CHOICES)} elements",
            ),
            (torch.tensor([3, 0, 0, 0, 0]), "Invalid horizontal velocity action"),
            (torch.tensor([0, 3, 0, 0, 0]), "Invalid vertical velocity action"),
        ],
    )
    def test_call_invalid_input(self, action, error_msg):
        """Test transformation with invalid input tensors."""
        transform = MouseTransform(100.0, 50.0)

        with pytest.raises(ValueError, match=error_msg):
            transform(action)


class TestOscTransform:
    """Tests for the OscTransform class."""

    def test_init(self):
        """Test initialization."""
        OscTransform()

    @pytest.mark.parametrize("v_idx,v_val", [(0, 0.0), (1, 1.0), (2, -1.0)])
    @pytest.mark.parametrize("h_idx,h_val", [(0, 0.0), (1, 1.0), (2, -1.0)])
    @pytest.mark.parametrize("jump", [0, 1])
    @pytest.mark.parametrize("run", [0, 1])
    def test_call_valid_input(self, v_idx, v_val, h_idx, h_val, jump, run):
        """Test transformation with valid input tensor."""
        transform = OscTransform()
        action = torch.tensor([v_idx, h_idx, jump, run], dtype=torch.long)

        result = transform(action)

        assert "axes" in result
        assert "buttons" in result
        assert result["axes"][OscAxes.Vertical] == v_val
        assert result["axes"][OscAxes.Horizontal] == h_val
        assert result["buttons"][OscButtons.Jump] == bool(jump)
        assert result["buttons"][OscButtons.Run] == bool(run)

    @pytest.mark.parametrize(
        "action,error_msg",
        [
            (torch.zeros((2, 4)), "Action tensor must be 1-dimensional"),
            (
                torch.zeros(3),
                f"Action tensor must have {len(OSC_ACTION_CHOICES)} elements",
            ),
            (torch.tensor([3, 0, 0, 0]), "Invalid vertical movement action"),
            (torch.tensor([0, 3, 0, 0]), "Invalid horizontal movement action"),
        ],
    )
    def test_call_invalid_input(self, action, error_msg):
        """Test transformation with invalid input tensors."""
        transform = OscTransform()

        with pytest.raises(ValueError, match=error_msg):
            transform(action)


class TestActionTransform:
    """Tests for the ActionTransform class."""

    def test_init(self):
        """Test initialization with different mouse velocity values."""
        ActionTransform()

    @pytest.mark.parametrize(
        "mouse_part,osc_part,expected_mouse,expected_osc",
        [
            (
                torch.tensor(
                    [1, 2, 1, 0, 1]
                ),  # Right, Up, Left press, Right release, Middle press
                torch.tensor([1, 0, 1, 0]),  # Forward, Stop, Jump press, Run release
                {
                    "move_velocity": (100.0, -50.0),
                    "button_press": {
                        MouseButton.LEFT: True,
                        MouseButton.RIGHT: False,
                        MouseButton.MIDDLE: True,
                    },
                },
                {
                    "axes": {OscAxes.Vertical: 1.0, OscAxes.Horizontal: 0.0},
                    "buttons": {OscButtons.Jump: True, OscButtons.Run: False},
                },
            ),
            (
                torch.tensor([0, 0, 0, 0, 0]),  # Stop, Stop, all buttons released
                torch.tensor([2, 2, 0, 1]),  # Backward, Left, Jump release, Run press
                {
                    "move_velocity": (0.0, 0.0),
                    "button_press": {
                        MouseButton.LEFT: False,
                        MouseButton.RIGHT: False,
                        MouseButton.MIDDLE: False,
                    },
                },
                {
                    "axes": {OscAxes.Vertical: -1.0, OscAxes.Horizontal: -1.0},
                    "buttons": {OscButtons.Jump: False, OscButtons.Run: True},
                },
            ),
        ],
    )
    def test_call_valid_input(self, mouse_part, osc_part, expected_mouse, expected_osc):
        """Test transformation with valid input tensor."""
        transform = ActionTransform(mouse_vx=100.0, mouse_vy=50.0)
        action = torch.cat([mouse_part, osc_part])

        result = transform(action)

        # Check structure
        assert ActionType.MOUSE in result
        assert ActionType.OSC in result

        # Check mouse action
        mouse_action = result[ActionType.MOUSE]
        assert mouse_action["move_velocity"] == expected_mouse["move_velocity"]
        for btn, state in expected_mouse["button_press"].items():
            assert mouse_action["button_press"][btn] is state

        # Check OSC action
        osc_action = result[ActionType.OSC]
        for axis, value in expected_osc["axes"].items():
            assert osc_action["axes"][axis] == value
        for btn, state in expected_osc["buttons"].items():
            assert osc_action["buttons"][btn] is state

    @pytest.mark.parametrize(
        "action,error_msg",
        [
            (torch.zeros(3), f"Action tensor must have {len(ACTION_CHOICES)} elements"),
        ],
    )
    def test_call_invalid_input(self, action, error_msg):
        """Test transformation with invalid input tensors."""
        transform = ActionTransform()

        with pytest.raises(ValueError, match=error_msg):
            transform(action)
