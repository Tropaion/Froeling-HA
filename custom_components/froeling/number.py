"""Number platform for writable numeric parameters (mstPar, 0x07).

Creates NumberEntity instances for heater parameters like temperature
setpoints.  Values are bounded by min/max from the heater's parameter
definition.  Uses input box mode (not slider) for safety – a slider
makes it too easy to accidentally set a dangerous value with a single
swipe, while a text box forces the user to type the intended value.

Only parameters with a numeric range (digits > 0 OR min_value != max_value)
are registered here.  Choice-style parameters (small integer range) are
handled by the select platform instead.
"""

from __future__ import annotations

import logging

from homeassistant.components.number import NumberEntity, NumberMode
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import CONF_SELECTED_PARAMETERS, CONF_WRITE_ENABLED
from .coordinator import FroelingCoordinator
from .entity import FroelingEntity
from .pyfroeling import WritableParameter

_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Platform setup
# ---------------------------------------------------------------------------

async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Create and register NumberEntity instances for writable numeric parameters.

    Only runs when write mode was enabled during the config flow.  Reads the
    list of selected parameter addresses from the config entry and creates one
    entity per parameter that qualifies as numeric (has a meaningful value range
    rather than a small integer choice set).

    Parameters
    ----------
    hass:
        Home Assistant instance.
    entry:
        The config entry to set up number entities for.
    async_add_entities:
        Callback that registers the entity objects with HA.
    """
    # Skip this platform entirely if write mode was not enabled
    if not entry.data.get(CONF_WRITE_ENABLED, False):
        _LOGGER.debug("number.async_setup_entry: write mode disabled, skipping")
        return

    coordinator: FroelingCoordinator = entry.runtime_data

    # Parse the selected parameter addresses (stored as hex strings)
    selected_addrs: set[int] = set()
    for addr_str in entry.data.get(CONF_SELECTED_PARAMETERS, []):
        try:
            selected_addrs.add(int(addr_str, 16))
        except (ValueError, TypeError):
            pass

    if not selected_addrs:
        _LOGGER.debug("number.async_setup_entry: no selected parameters")
        return

    entities: list[FroelingNumberEntity] = []

    # Iterate over parameters that were actually read on the first poll
    for addr, param in coordinator.data.parameters.items():
        if addr not in selected_addrs:
            continue  # User didn't select this parameter

        # Determine if this is a numeric (range) parameter or a choice parameter.
        # A parameter is treated as numeric when it has a meaningful non-trivial range
        # that cannot reasonably be expressed as a small choice list.
        # Heuristic: if max - min > 10 it is almost certainly a numeric range param.
        # This keeps the split clean: select handles 0-5 style booleans/enums,
        # number handles temperature setpoints, time durations, percentages, etc.
        value_range = param.max_value - param.min_value
        is_numeric = value_range > 10 or param.digits > 0

        if not is_numeric:
            # Will be handled by the select platform
            _LOGGER.debug(
                "number: skipping 0x%04X '%s' (range=%s, digits=%d) – defer to select",
                addr, param.title, value_range, param.digits,
            )
            continue

        # Register this parameter's title with the coordinator so polling uses
        # the real name rather than the generic "Parameter 0xXXXX" fallback.
        coordinator.set_parameter_title(addr, param.title)

        entities.append(FroelingNumberEntity(coordinator, param))

    _LOGGER.debug(
        "number.async_setup_entry: adding %d number entities", len(entities)
    )
    async_add_entities(entities)


# ---------------------------------------------------------------------------
# NumberEntity
# ---------------------------------------------------------------------------

class FroelingNumberEntity(FroelingEntity, NumberEntity):
    """A numeric control entity for a single writable heater parameter.

    The user can type a value into an input box in the HA UI or send it
    through automations / scripts.  The value is validated against the
    heater's own min/max limits before being written.

    Uses BOX mode (text input) rather than SLIDER because:
    - Sliders make it dangerously easy to swipe to an extreme value.
    - Heater parameters like setpoint temperatures need precise input.
    - BOX mode is the safer and more explicit choice.
    """

    # Use text input (box) rather than a slider for safety
    _attr_mode = NumberMode.BOX

    def __init__(
        self,
        coordinator: FroelingCoordinator,
        param: WritableParameter,
    ) -> None:
        """Initialise the number entity.

        Parameters
        ----------
        coordinator:
            Shared data coordinator for this config entry.
        param:
            The :class:`~pyfroeling.WritableParameter` this entity represents.
            Used to set bounds, units, step size, and the entity name.
        """
        # Use "NR" (Number Register) as the sensor type tag so the unique_id
        # does not collide with sensor entities (which use "VA").
        super().__init__(coordinator, "NR", param.address, param.title)

        # Store the parameter address for coordinator data lookups
        self._address = param.address

        # Physical unit for the HA UI (e.g. "°C", "min", "%")
        self._attr_native_unit_of_measurement = param.unit if param.unit else None

        # Value bounds from the heater's EEPROM parameter definition
        self._attr_native_min_value = param.min_value
        self._attr_native_max_value = param.max_value

        # Step size derived from the scale factor:
        #   factor=10 → step=0.1 (one decimal place)
        #   factor=1  → step=1.0 (whole numbers only)
        self._attr_native_step = 1.0 / param.factor if param.factor > 1 else 1.0

        # Remember the factor for encode/decode during set operations
        self._factor = param.factor

        # Value shown while a write is in flight (None: show coordinator data)
        self._optimistic_value: float | None = None

    @property
    def native_value(self) -> float | None:
        """Return the current parameter value.

        Returns optimistic value if a write is pending, otherwise from coordinator.
        """
        if self._optimistic_value is not None:
            return self._optimistic_value

        if self.coordinator.data is None:
            return None
        wp = self.coordinator.data.parameters.get(self._address)
        return wp.value if wp is not None else None

    async def async_set_native_value(self, value: float) -> None:
        """Write a new value to the heater parameter.

        Optimistically updates the UI immediately, then writes to the heater.
        """
        _LOGGER.debug(
            "FroelingNumberEntity: setting 0x%04X '%s' to %s",
            self._address, self.name, value,
        )
        # Optimistic: show the new value immediately
        self._optimistic_value = value
        self.async_write_ha_state()

        try:
            await self.coordinator.async_write_parameter(
                self._address, value, self._factor
            )
        except Exception as exc:
            _LOGGER.error("Failed to write 0x%04X: %s", self._address, exc)
            self._optimistic_value = None
            self.async_write_ha_state()

    def _handle_coordinator_update(self) -> None:
        """Clear optimistic state when coordinator confirms the real value."""
        self._optimistic_value = None
        super()._handle_coordinator_update()
