from typing import Final

import torch

from pamiq_vrchat import ActionType, VRChatAction
from pamiq_vrchat.actuators import (
    MouseAction,
    MouseButton,
    OscAction,
    OscAxes,
    OscButtons,
)

# Action space dimensions for each controller type
MOUSE_ACTION_CHOICES: Final[tuple[int, ...]] = (3, 3, 2, 2, 2)
OSC_ACTION_CHOICES: Final[tuple[int, ...]] = (3, 3, 2, 2)
ACTION_CHOICES: Final[tuple[int, ...]] = MOUSE_ACTION_CHOICES + OSC_ACTION_CHOICES


class MouseTransform:
    """Transforms discrete action indices into MouseAction format.

    Converts a tensor of discrete action indices into a structured
    MouseAction dictionary that can be used by the MouseActuator. Maps
    velocity directions and button states based on action indices.
    """

    def __init__(self, vx: float, vy: float) -> None:
        """Initialize the mouse transform with velocities.

        Args:
            vx: Horizontal mouse velocity.
            vy: Vertical mouse velocity.
        """
        self._vx_scale = vx
        self._vy_scale = vy
        # Mapping from discrete actions (0,1,2) to directional values (0, +1, -1)
        self._velocity_map: dict[int, float] = {0: 0.0, 1: 1.0, 2: -1.0}

    def __call__(self, action: torch.Tensor) -> MouseAction:
        """Transform discrete action tensor into MouseAction format.

        Args:
            action: Tensor of discrete action indices with shape (5,), where:
                - index 0: horizontal velocity (0=none, 1=right, 2=left)
                - index 1: vertical velocity (0=none, 1=down, 2=up)
                - index 2: left mouse button (0=release, 1=press)
                - index 3: right mouse button (0=release, 1=press)
                - index 4: middle mouse button (0=release, 1=press)

        Returns:
            MouseAction dictionary with move_velocity and button_press fields.

        Raises:
            ValueError: If action tensor shape is invalid or contains unsupported values.
        """
        if action.ndim != 1:
            raise ValueError("Action tensor must be 1-dimensional")
        if action.numel() != len(MOUSE_ACTION_CHOICES):
            raise ValueError(
                f"Action tensor must have {len(MOUSE_ACTION_CHOICES)} elements"
            )

        action_list: list[int] = action.detach().cpu().long().tolist()
        if (vx := self._velocity_map.get(action_list[0])) is None:
            raise ValueError(f"Invalid horizontal velocity action: {action_list[0]}")
        if (vy := self._velocity_map.get(action_list[1])) is None:
            raise ValueError(f"Invalid vertical velocity action: {action_list[1]}")

        return MouseAction(
            move_velocity=(vx * self._vx_scale, vy * self._vy_scale),
            button_press={
                MouseButton.LEFT: bool(action_list[2]),
                MouseButton.RIGHT: bool(action_list[3]),
                MouseButton.MIDDLE: bool(action_list[4]),
            },
        )


class OscTransform:
    """Transforms discrete action indices into OscAction format.

    Converts a tensor of discrete action indices into a structured
    OscAction dictionary that can be used by the OscActuator. Maps
    avatar movement directions and button states based on action
    indices.
    """

    def __init__(self) -> None:
        """Initialize the OSC transform."""
        # Mapping from discrete actions (0,1,2) to directional values (0, +1, -1)
        self._velocity_map: dict[int, float] = {0: 0.0, 1: 1.0, 2: -1.0}

    def __call__(self, action: torch.Tensor) -> OscAction:
        """Transform discrete action tensor into OscAction format.

        Args:
            action: Tensor of discrete action indices with shape (4,), where:
                - index 0: vertical movement (0=stop, 1=forward, 2=backward)
                - index 1: horizontal movement (0=stop, 1=right, 2=left)
                - index 2: jump button (0=release, 1=press)
                - index 3: run button (0=release, 1=press)

        Returns:
            OscAction dictionary with axes and buttons fields.

        Raises:
            ValueError: If action tensor shape is invalid or contains unsupported values.
        """
        if action.ndim != 1:
            raise ValueError("Action tensor must be 1-dimensional")
        if action.numel() != len(OSC_ACTION_CHOICES):
            raise ValueError(
                f"Action tensor must have {len(OSC_ACTION_CHOICES)} elements"
            )

        action_list: list[int] = action.detach().cpu().long().tolist()

        if (vertical := self._velocity_map.get(action_list[0])) is None:
            raise ValueError(f"Invalid vertical movement action: {action_list[0]}")
        if (horizontal := self._velocity_map.get(action_list[1])) is None:
            raise ValueError(f"Invalid horizontal movement action: {action_list[1]}")

        return OscAction(
            axes={
                OscAxes.Vertical: vertical,
                OscAxes.Horizontal: horizontal,
            },
            buttons={
                OscButtons.Jump: bool(action_list[2]),
                OscButtons.Run: bool(action_list[3]),
            },
        )


class ActionTransform:
    """Transforms discrete action tensor into VRChat action format.

    Combines MouseTransform and OscTransform to convert a single action
    tensor into a complete VRChatAction dictionary that can control both
    mouse movements and avatar actions simultaneously.
    """

    def __init__(self, mouse_vx: float = 100.0, mouse_vy: float = 100.0) -> None:
        """Initialize the action transform with velocity scaling factors.

        Args:
            mouse_vx: Horizontal mouse velocity (pixels/sec).
            mouse_vy: Vertical mouse velocity (pixels/sec).
        """
        self._mouse = MouseTransform(mouse_vx, mouse_vy)
        self._osc = OscTransform()

    def __call__(self, action: torch.Tensor) -> VRChatAction:
        """Transform discrete action tensor into VRChatAction format.

        Args:
            action: Tensor of discrete action indices with shape (N,),
                   where N = len(ACTION_CHOICES). The tensor combines both
                   mouse and OSC action indices.

        Returns:
            VRChatAction dictionary with MOUSE and OSC components.

        Raises:
            ValueError: If action tensor shape is invalid.
        """
        action = action.flatten()
        if action.numel() != len(ACTION_CHOICES):
            raise ValueError(f"Action tensor must have {len(ACTION_CHOICES)} elements")

        mouse, osc = action.split([len(MOUSE_ACTION_CHOICES), len(OSC_ACTION_CHOICES)])
        return {ActionType.MOUSE: self._mouse(mouse), ActionType.OSC: self._osc(osc)}
