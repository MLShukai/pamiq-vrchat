"""Tests for the VRChat environment implementation."""

import pytest
from pamiq_core.interaction.modular_env import Actuator, Sensor
from pytest_mock import MockerFixture

from pamiq_vrchat.env import (
    ActionType,
    ObservationType,
    VRChatAction,
    VRChatEnvironment,
)


class TestVRChatEnvironment:
    """Tests for the VRChatEnvironment class."""

    @pytest.fixture
    def mock_image_sensor(self, mocker: MockerFixture):
        """Create a mock image sensor."""
        sensor = mocker.Mock(Sensor)
        sensor.read.return_value = "image_data"
        return sensor

    @pytest.fixture
    def mock_osc_actuator(self, mocker: MockerFixture):
        """Create a mock OSC actuator."""
        actuator = mocker.Mock(Actuator)
        return actuator

    @pytest.fixture
    def mock_mouse_actuator(self, mocker: MockerFixture):
        """Create a mock mouse actuator."""
        actuator = mocker.Mock(Actuator)
        return actuator

    @pytest.fixture
    def sensors(self, mock_image_sensor):
        """Create a mapping of sensors."""
        return {ObservationType.IMAGE: mock_image_sensor}

    @pytest.fixture
    def actuators(self, mock_osc_actuator, mock_mouse_actuator):
        """Create a mapping of actuators."""
        return {
            ActionType.OSC: mock_osc_actuator,
            ActionType.MOUSE: mock_mouse_actuator,
        }

    @pytest.fixture
    def vrchat_env(self, sensors, actuators):
        """Create a VRChatEnvironment instance for testing."""
        return VRChatEnvironment(sensors, actuators)

    def test_init(self, vrchat_env, sensors, actuators):
        """Test initialization of VRChatEnvironment."""
        assert vrchat_env.sensors == sensors
        assert vrchat_env.actuators == actuators

    def test_observe(self, vrchat_env, mock_image_sensor):
        """Test the observe method."""
        observations = vrchat_env.observe()

        mock_image_sensor.read.assert_called_once()
        assert isinstance(observations, dict)
        assert ObservationType.IMAGE in observations
        assert observations[ObservationType.IMAGE] == "image_data"

    def test_affect(self, vrchat_env, mock_osc_actuator, mock_mouse_actuator):
        """Test the affect method."""
        osc_action = {"axes": {"vertical": 0.5}, "buttons": {"jump": True}}
        mouse_action = {"move_velocity": (100, 50)}
        action: VRChatAction = {
            ActionType.OSC: osc_action,
            ActionType.MOUSE: mouse_action,
        }

        vrchat_env.affect(action)

        mock_osc_actuator.operate.assert_called_once_with(osc_action)
        mock_mouse_actuator.operate.assert_called_once_with(mouse_action)

    def test_affect_missing_action_key(self, vrchat_env):
        """Test the affect method with missing action key."""
        action: VRChatAction = {
            ActionType.OSC: {"axes": {"vertical": 0.5}},
        }

        with pytest.raises(
            ValueError,
            match=f"Action key {ActionType.MOUSE} does not exists in action.",
        ):
            vrchat_env.affect(action)

    def test_empty_sensors(self, mock_osc_actuator):
        """Test initialization with empty sensors."""
        actuators = {ActionType.OSC: mock_osc_actuator}
        env = VRChatEnvironment({}, actuators)

        observations = env.observe()
        assert isinstance(observations, dict)
        assert len(observations) == 0

    def test_empty_actuators(self, mock_image_sensor):
        """Test initialization with empty actuators."""
        sensors = {ObservationType.IMAGE: mock_image_sensor}
        env = VRChatEnvironment(sensors, {})

        env.affect({})
