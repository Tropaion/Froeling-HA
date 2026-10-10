"""Sensor platform for the Fröling Heater integration.

Creates all HA sensor entities from the data supplied by the coordinator:

  - One :class:`FroelingValueSensor` per discovered ValueSpec whose menu_type
    is MESSWERT, MESSWERT1 (measured values) or ANL_OUT (analogue outputs).
  - :class:`FroelingStateSensor`            – heater state text ("Heizen" …)
  - :class:`FroelingModeSensor`             – operating mode text
  - :class:`FroelingActiveErrorCountSensor` – count of ARRIVED (active) errors
  - :class:`FroelingLastErrorSensor`        – text of the most recent error

HA sensor platform docs:
  https://developers.home-assistant.io/docs/core/entity/sensor
"""

from __future__ import annotations

import logging

from homeassistant.components.sensor import (
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import UNIT_DEVICE_CLASS_MAP
from .coordinator import FroelingCoordinator
from .entity import FroelingEntity
from .pyfroeling import ErrorState

_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Platform setup
# ---------------------------------------------------------------------------

async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Create and register sensor entities for this config entry.

    Called once by HA when the integration is loaded.  The coordinator's
    ``data.specs`` list (populated during ``_async_setup``) determines which
    FroelingValueSensor instances are created.  The fixed "meta" sensors
    (state, mode, error count, last error) are always added.

    Parameters
    ----------
    hass:
        Home Assistant instance.
    entry:
        The config entry to set up sensors for.
    async_add_entities:
        Callback that registers the entity objects with HA.
    """
    # Retrieve the coordinator stored on the entry at integration setup time.
    coordinator: FroelingCoordinator = entry.runtime_data

    entities: list[FroelingEntity] = []

    # --- Dynamic value sensors from selected specs ---
    # Only create entities for sensors the user selected during setup.
    # The coordinator's _get_selected_specs() handles the filtering for
    # polling; here we create matching entities.
    from .const import CONF_SELECTED_SENSORS

    selected = entry.data.get(CONF_SELECTED_SENSORS, [])
    selected_addrs = set()
    for addr_str in selected:
        try:
            selected_addrs.add(int(addr_str, 16))
        except ValueError:
            pass

    for spec in coordinator.data.specs:
        # If a selection exists, only create entities for selected addresses.
        # If no selection (backwards compat), create entities for all.
        if selected and spec.address not in selected_addrs:
            continue
        entities.append(
            FroelingValueSensor(
                coordinator, spec.address, spec.title, spec.unit, sensor_type="VA"
            )
        )

    # --- Fixed "meta" sensors ---
    entities.append(FroelingStateSensor(coordinator))
    entities.append(FroelingModeSensor(coordinator))
    entities.append(FroelingActiveErrorCountSensor(coordinator))
    entities.append(FroelingErrorCountTotalSensor(coordinator))
    entities.append(FroelingLastErrorSensor(coordinator))

    _LOGGER.debug(
        "sensor.async_setup_entry: adding %d entities (%d value sensors + 4 meta)",
        len(entities), len(entities) - 4,
    )

    async_add_entities(entities)


# ---------------------------------------------------------------------------
# Value sensor (measured values and analogue outputs)
# ---------------------------------------------------------------------------

class FroelingValueSensor(FroelingEntity, SensorEntity):
    """Represents a single numeric sensor value from the heater.

    Covers both MESSWERT/MESSWERT1 (measured temperatures, pressures, …) and
    ANL_OUT (analogue output channels like pump speed).

    The unit and device class are set once at construction time from the
    ValueSpec metadata.  The native_value is read from coordinator data on
    each HA state write, rounded to 2 decimal places to avoid floating-point
    noise in the history graph.
    """

    def __init__(
        self,
        coordinator: FroelingCoordinator,
        address: int,
        name: str,
        unit: str,
        sensor_type: str = "VA",
    ) -> None:
        """Initialise the value sensor.

        Parameters
        ----------
        coordinator:
            Shared data coordinator for this config entry.
        address:
            16-bit register address of this sensor.
        name:
            German sensor title from the controller menu (preserved as-is).
        unit:
            Physical unit string (e.g. "°C", "%", "kW").
        sensor_type:
            Short category tag: "VA" for measured values, "AO" for analogue
            outputs.  Included in the unique_id to avoid collisions.
        """
        super().__init__(coordinator, sensor_type, address, name)

        # Physical unit as returned by the controller (e.g. "°C", "%").
        self._attr_native_unit_of_measurement = unit

        # Map the unit to an HA SensorDeviceClass if known; None otherwise.
        # Sensors with an unknown unit (e.g. "kW", "l/h") will still display
        # correctly – they just won't get the coloured history icon.
        self._attr_device_class = UNIT_DEVICE_CLASS_MAP.get(unit)

        # All numeric heater sensors are instantaneous measurements.
        self._attr_state_class = SensorStateClass.MEASUREMENT

        # Store address for data lookup in native_value.
        self._address = address

    @property
    def native_value(self) -> float | None:
        """Return the current sensor reading.

        Returns None if the sensor address is absent from coordinator data
        (e.g., the controller did not respond for this address during the
        last poll cycle).  HA will then show "Unavailable" for this entity.
        """
        if self.coordinator.data is None:
            return None

        sv = self.coordinator.data.values.get(self._address)
        if sv is None:
            return None

        # Return integer for whole numbers (e.g., "Anzahl Brennerstarts = 16497")
        # and rounded float for fractional values (e.g., "Kesseltemperatur = 65.3")
        rounded = round(sv.value, 2)
        if rounded == int(rounded):
            return int(rounded)
        return rounded


# ---------------------------------------------------------------------------
# Fixed "meta" sensors
# ---------------------------------------------------------------------------

class FroelingStateSensor(FroelingEntity, SensorEntity):
    """Text sensor reporting the heater's current operating state.

    Reads from ``coordinator.data.status.state_text``, which is the German
    state name looked up from STATE_TABLE (e.g. "Heizen", "Abstellen Warten").
    This is the primary status sensor most users will display on their dashboard.

    Bug 3 fix: uses ``translation_key="heater_state"`` so HA displays
    "Heizungszustand" for German users and "Heater State" for English users
    (resolved from strings.json / translations/*.json).
    """

    def __init__(self, coordinator: FroelingCoordinator) -> None:
        """Initialise the state sensor with a fixed virtual address 0x0001."""
        # Virtual address 0x0001 is chosen to avoid colliding with any real
        # controller address while still fitting the unique_id pattern.
        # translation_key maps to entity.sensor.froeling.heater_state.name
        # in the integration's translation files.
        super().__init__(coordinator, "UD", 0x0001, translation_key="heater_state")

    @property
    def native_value(self) -> str | None:
        """Return the current state text (e.g. 'Heizen')."""
        if self.coordinator.data is None:
            return None
        return self.coordinator.data.status.state_text


class FroelingModeSensor(FroelingEntity, SensorEntity):
    """Text sensor reporting the heater's current operating mode.

    Reads from ``coordinator.data.status.mode_text``, which carries the
    mode name decoded from the GET_STATE response (e.g. "Automatik").

    Bug 3 fix: uses ``translation_key="operating_mode"`` for i18n support.
    """

    def __init__(self, coordinator: FroelingCoordinator) -> None:
        """Initialise the mode sensor with a fixed virtual address 0x0002."""
        # translation_key maps to entity.sensor.froeling.operating_mode.name
        super().__init__(coordinator, "UD", 0x0002, translation_key="operating_mode")

    @property
    def native_value(self) -> str | None:
        """Return the current operating mode text."""
        if self.coordinator.data is None:
            return None
        return self.coordinator.data.status.mode_text


class FroelingActiveErrorCountSensor(FroelingEntity, SensorEntity):
    """Numeric sensor counting errors that are still active (not acknowledged, not gone).

    Error states from the heater (discrete values, NOT bitmasks):
      - ARRIVED (1): error is active and has NOT been acknowledged
      - ACKNOWLEDGED (2): operator has acknowledged the error
      - GONE (4): error condition has cleared

    Only errors with state == ARRIVED (exactly 1) are counted as "active".
    Acknowledged and gone errors are not counted.

    Bug 3 fix: uses ``translation_key="active_errors"`` for i18n support.
    """

    def __init__(self, coordinator: FroelingCoordinator) -> None:
        # translation_key maps to entity.sensor.froeling.active_errors.name
        super().__init__(coordinator, "ERR", 0x0001, translation_key="active_errors")
        self._attr_icon = "mdi:alert-circle"
        self._attr_state_class = SensorStateClass.MEASUREMENT

    @property
    def native_value(self) -> int | None:
        """Count errors with state == ARRIVED (exactly 1)."""
        if self.coordinator.data is None:
            return None
        # Discrete state check: only count errors that have NOT been
        # acknowledged or cleared by the operator.
        return sum(
            1 for err in self.coordinator.data.errors
            if err.state == ErrorState.ARRIVED
        )


class FroelingErrorCountTotalSensor(FroelingEntity, SensorEntity):
    """Total number of entries in the heater's error log (all states).

    Bug 3 fix: uses ``translation_key="error_log_entries"`` for i18n support.
    """

    def __init__(self, coordinator: FroelingCoordinator) -> None:
        # translation_key maps to entity.sensor.froeling.error_log_entries.name
        super().__init__(coordinator, "ERR", 0x0003, translation_key="error_log_entries")
        self._attr_icon = "mdi:format-list-bulleted"
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_entity_registry_enabled_default = False  # disabled by default

    @property
    def native_value(self) -> int | None:
        if self.coordinator.data is None:
            return None
        return len(self.coordinator.data.errors)


class FroelingLastErrorSensor(FroelingEntity, SensorEntity):
    """Text sensor showing the most recent error's description and state.

    Format: "Störung STB (quittiert)" or "Fehler Saugzug (aktiv)"

    Bug 3 fix: uses ``translation_key="last_error"`` for i18n support.
    """

    # Map error states to German display text (values from the heater are German
    # regardless of HA language; this mapping is intentionally German).
    _STATE_LABELS = {
        ErrorState.ARRIVED: "aktiv",
        ErrorState.ACKNOWLEDGED: "quittiert",
        ErrorState.GONE: "gegangen",
    }

    def __init__(self, coordinator: FroelingCoordinator) -> None:
        # translation_key maps to entity.sensor.froeling.last_error.name
        super().__init__(coordinator, "ERR", 0x0002, translation_key="last_error")
        self._attr_icon = "mdi:alert"

    @property
    def native_value(self) -> str | None:
        """Return the most recent error text with its state label."""
        if self.coordinator.data is None or not self.coordinator.data.errors:
            return None

        err = self.coordinator.data.errors[0]
        state_label = self._STATE_LABELS.get(err.state, f"state={err.state}")
        return f"{err.text} ({state_label})"
