"""Binary sensor platform for the Fröling Heater integration.

Creates binary (on/off) entities for:

  - One :class:`FroelingDigitalSensor` per discovered ValueSpec whose
    menu_type is DIG_OUT (digital outputs, device_class=RUNNING) or
    DIG_IN (digital inputs, device_class=SAFETY).
  - :class:`FroelingErrorBinarySensor` – overall fault indicator derived from
    the heater status (is_on when state is a "Störung" or "Fehler" code).

HA binary sensor platform docs:
  https://developers.home-assistant.io/docs/core/entity/binary-sensor
"""

from __future__ import annotations

import logging

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .coordinator import FroelingCoordinator
from .entity import FroelingEntity

_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Platform setup
# ---------------------------------------------------------------------------

async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Create and register binary sensor entities for this config entry.

    Iterates over the sensor specs in coordinator data to create one
    :class:`FroelingDigitalSensor` per DIG_OUT or DIG_IN channel.
    The fixed :class:`FroelingErrorBinarySensor` is always added.

    Parameters
    ----------
    hass:
        Home Assistant instance.
    entry:
        The config entry to set up binary sensors for.
    async_add_entities:
        Callback that registers the entity objects with HA.
    """
    coordinator: FroelingCoordinator = entry.runtime_data

    entities: list[FroelingEntity] = []

    # NOTE: Digital I/O sensors (DIG_OUT, DIG_IN) are not created as binary
    # sensors because ALL discovered ValueSpecs are read with cmdGetValue (0x30)
    # and exposed as numeric sensors on the sensor platform.  The binary_sensor
    # platform only provides the overall heater fault indicator.

    # --- Fixed fault-indicator binary sensor ---
    entities.append(FroelingErrorBinarySensor(coordinator))

    _LOGGER.debug(
        "binary_sensor.async_setup_entry: adding %d entities (%d I/O + 1 fault indicator)",
        len(entities), len(entities) - 1,
    )

    async_add_entities(entities)


# ---------------------------------------------------------------------------
# Digital I/O binary sensor
# ---------------------------------------------------------------------------

class FroelingDigitalSensor(FroelingEntity, BinarySensorEntity):
    """Binary sensor for a single digital output or input channel.

    The heater controller exposes digital channels via GET_DIG_OUT and
    GET_DIG_IN.  The raw ``state`` field of the :class:`~pyfroeling.IoValue`
    (stored as ``SensorValue.raw_value`` by the client) is 0 or 1.

    Parameters
    ----------
    coordinator:
        Shared coordinator for this config entry.
    address:
        16-bit channel address in the controller's I/O table.
    name:
        German channel title from the controller menu (preserved as-is).
    sensor_type:
        "DO" for digital outputs, "DI" for digital inputs.
    device_class:
        The HA BinarySensorDeviceClass: RUNNING for outputs, SAFETY for inputs.
    """

    def __init__(
        self,
        coordinator: FroelingCoordinator,
        address: int,
        name: str,
        sensor_type: str,
        device_class: BinarySensorDeviceClass,
    ) -> None:
        """Initialise the digital sensor."""
        super().__init__(coordinator, sensor_type, address, name)

        # Assign the device class so HA picks the right icon and state strings.
        self._attr_device_class = device_class

    @property
    def is_on(self) -> bool | None:
        """Return True when the channel is active (raw_value == 1).

        Returns None if coordinator data is not yet available, which causes
        HA to mark the entity as "Unknown" rather than on or off.
        """
        if self.coordinator.data is None:
            return None

        sv = self.coordinator.data.values.get(self._address)
        if sv is None:
            # The channel was not read during the last poll (likely a transient
            # error); return None so HA keeps the previous state.
            return None

        # raw_value is the integer state field returned by the controller:
        # 0 = off / inactive, 1 = on / active.
        return sv.raw_value == 1


# ---------------------------------------------------------------------------
# Overall fault indicator
# ---------------------------------------------------------------------------

class FroelingErrorBinarySensor(FroelingEntity, BinarySensorEntity):
    """Binary sensor indicating whether the heater is in a fault/error state.

    Reads ``coordinator.data.status.is_error``, which is pre-computed by the
    protocol client from the state code: True when the state is in
    ERROR_STATE_CODES (any state name starting with "Störung" or "Fehler").

    device_class=PROBLEM causes HA to show the entity as "Problem" (red) when
    on and "OK" (green) when off.

    Bug 3 fix: uses ``translation_key="heater_error"`` so HA displays
    "Heizungsfehler" for German users and "Heater Error" for English users.

    A typical automation monitors this entity: if it turns on, send a push
    notification with the value of FroelingLastErrorSensor.
    """

    def __init__(self, coordinator: FroelingCoordinator) -> None:
        """Initialise the error binary sensor with virtual address 0x0000."""
        # Virtual address 0x0000 is safe here because the controller's real
        # DIG_OUT/DIG_IN addresses start at 0x0001.
        # translation_key maps to entity.binary_sensor.froeling.heater_error.name
        super().__init__(coordinator, "ERR", 0x0000, translation_key="heater_error")

        # PROBLEM device class: on = problem detected, off = everything OK.
        self._attr_device_class = BinarySensorDeviceClass.PROBLEM

        # Alert-octagon icon visually differentiates this from the other
        # "alert" variants used by the error-count and last-error sensors.
        self._attr_icon = "mdi:alert-octagon"

    @property
    def is_on(self) -> bool | None:
        """Return True when the heater is in a fault or error state."""
        if self.coordinator.data is None:
            return None
        return self.coordinator.data.status.is_error
