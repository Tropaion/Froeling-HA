"""Select platform for writable choice parameters (mstParDig, 0x08).

Creates SelectEntity instances for parameters that use a small integer range
to represent a set of discrete options (e.g. operating mode: 0=Off, 1=Auto,
2=Manual).

The heater does not supply text labels for individual option values in the
binary protocol – the labels exist only on the controller's own LCD display.
For now, options are displayed as their numeric string representations
("0", "1", "2", …).  Users can create template entities or automations to
map these numbers to meaningful labels if desired.

Selection criteria (complementary to the number platform):
- max_value – min_value <= 10  (small integer choice range)
- digits == 0                  (no decimal places – whole-number selection)
"""

from __future__ import annotations

import logging

from homeassistant.components.select import SelectEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import CONF_SELECTED_PARAMETERS, CONF_WRITE_ENABLED
from .coordinator import FroelingCoordinator
from .entity import FroelingEntity
from .pyfroeling import WritableParameter

_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Option label lookup
# ---------------------------------------------------------------------------

def _get_option_labels(param: WritableParameter) -> dict[int, str] | None:
    """Look up known option labels for a parameter by address.

    Uses the address-based KNOWN_BASIC_PARAMS table from known_params.py.
    Falls back to generic Aus/Ein for boolean-like parameters (min=0, max=1).
    """
    from .known_params import get_option_labels

    # Address-based lookup (most reliable)
    labels = get_option_labels(param.address)
    if labels:
        return labels

    # Generic fallback: boolean-like parameters (min=0, max=1)
    if int(param.min_value) == 0 and int(param.max_value) == 1:
        return {0: "Aus", 1: "Ein"}

    return None


# ---------------------------------------------------------------------------
# Platform setup
# ---------------------------------------------------------------------------

async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Create and register SelectEntity instances for writable choice parameters.

    Only runs when write mode was enabled during the config flow.  Reads the
    list of selected parameter addresses from the config entry and creates one
    entity per parameter that qualifies as a choice (small integer range,
    integer step – the complementary set to what the number platform handles).

    Parameters
    ----------
    hass:
        Home Assistant instance.
    entry:
        The config entry to set up select entities for.
    async_add_entities:
        Callback that registers the entity objects with HA.
    """
    # Skip this platform entirely if write mode was not enabled
    if not entry.data.get(CONF_WRITE_ENABLED, False):
        _LOGGER.debug("select.async_setup_entry: write mode disabled, skipping")
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
        _LOGGER.debug("select.async_setup_entry: no selected parameters")
        return

    entities: list[FroelingSelectEntity] = []

    # Iterate over parameters that were actually read on the first poll
    for addr, param in coordinator.data.parameters.items():
        if addr not in selected_addrs:
            continue  # User didn't select this parameter

        # Only create a select entity for parameters with a small integer range.
        # This is the complement of the number platform's numeric range check:
        #   - max - min <= 10  → select (discrete choices)
        #   - max - min > 10   → number (continuous range)
        #   - digits > 0       → number (fractional values)
        value_range = param.max_value - param.min_value
        is_choice = value_range <= 10 and param.digits == 0

        if not is_choice:
            # Will be handled by the number platform
            _LOGGER.debug(
                "select: skipping 0x%04X '%s' (range=%s, digits=%d) – defer to number",
                addr, param.title, value_range, param.digits,
            )
            continue

        # Register this parameter's title with the coordinator
        coordinator.set_parameter_title(addr, param.title)

        entities.append(FroelingSelectEntity(coordinator, param))

    _LOGGER.debug(
        "select.async_setup_entry: adding %d select entities", len(entities)
    )
    async_add_entities(entities)


# ---------------------------------------------------------------------------
# SelectEntity
# ---------------------------------------------------------------------------

class FroelingSelectEntity(FroelingEntity, SelectEntity):
    """A drop-down select entity for a single writable heater choice parameter.

    Options are the integer values from min_value to max_value (inclusive),
    displayed as strings ("0", "1", "2", …).  Writing a value back converts
    the option string to a float and delegates to the coordinator.

    The heater's LCD may show German text labels for each option value (e.g.
    "Automatik" for mode 0).  Those labels are not available in the binary
    protocol response, so we show numbers.  The user can create a HA template
    entity on top of this one to apply friendly labels.
    """

    def __init__(
        self,
        coordinator: FroelingCoordinator,
        param: WritableParameter,
    ) -> None:
        """Initialise the select entity.

        Parameters
        ----------
        coordinator:
            Shared data coordinator for this config entry.
        param:
            The :class:`~pyfroeling.WritableParameter` this entity represents.
        """
        # Use "SR" (Select Register) as the sensor type tag to avoid unique_id
        # collisions with "VA" (sensor) and "NR" (number) entities.
        super().__init__(coordinator, "SR", param.address, param.title)

        # Store the parameter address for coordinator data lookups
        self._address = param.address

        # Build the option list from the allowed integer range.
        low = int(param.min_value)
        high = int(param.max_value)

        # Look up known labels for this parameter (e.g., Betriebsart -> Sommerbetrieb)
        self._labels = _get_option_labels(param)

        if self._labels:
            # Use readable labels as options
            self._attr_options = [
                self._labels.get(i, str(i)) for i in range(low, high + 1)
            ]
            # Reverse map: label -> int value
            self._label_to_value = {
                self._labels.get(i, str(i)): i for i in range(low, high + 1)
            }
        else:
            # No known labels: show numeric strings
            self._attr_options = [str(i) for i in range(low, high + 1)]
            self._label_to_value = {str(i): i for i in range(low, high + 1)}

        # Remember the factor for encode/decode during set operations.
        self._factor = param.factor
        self._min_value = low

        # Option shown while a write is in flight (None: show coordinator data)
        self._optimistic_option: str | None = None

    @property
    def current_option(self) -> str | None:
        """Return the currently active option label.

        Returns the optimistic value if a write is pending, otherwise
        reads from the coordinator snapshot.
        """
        # Return optimistic value immediately after a write
        if self._optimistic_option is not None:
            return self._optimistic_option

        if self.coordinator.data is None:
            return None
        wp = self.coordinator.data.parameters.get(self._address)
        if wp is None:
            return None
        int_val = int(wp.value)
        if self._labels:
            return self._labels.get(int_val, str(int_val))
        return str(int_val)

    def _handle_coordinator_update(self) -> None:
        """Clear optimistic state when the coordinator confirms the real value."""
        self._optimistic_option = None
        super()._handle_coordinator_update()

    async def async_select_option(self, option: str) -> None:
        """Write the selected option to the heater.

        Optimistically updates the UI immediately, then writes to the heater
        in the background. The next poll cycle will confirm the actual value.
        """
        _LOGGER.debug(
            "FroelingSelectEntity: setting 0x%04X '%s' to option '%s'",
            self._address, self.name, option,
        )
        # Optimistically update the UI immediately so the user sees feedback
        self._optimistic_option = option
        self.async_write_ha_state()

        # Convert the label back to the numeric value
        int_value = self._label_to_value.get(option)
        if int_value is None:
            int_value = int(option)
        value = float(int_value)

        try:
            await self.coordinator.async_write_parameter(
                self._address, value, self._factor
            )
        except Exception as exc:
            # Write failed -- revert optimistic state
            _LOGGER.error("Failed to write 0x%04X: %s", self._address, exc)
            self._optimistic_option = None
            self.async_write_ha_state()
