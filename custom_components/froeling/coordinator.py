"""DataUpdateCoordinator for the Fröling Heater integration.

The coordinator is the single point of truth for all heater data within a
config entry.  It owns the FroelingClient, drives the polling loop, and
exposes a typed FroelingData snapshot to all entity subclasses.

HA coordinator docs:
  https://developers.home-assistant.io/docs/integration_fetching_data
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .const import (
    CONF_CONNECTION_TYPE,
    CONF_HOST,
    CONF_PARAMETER_TITLES,
    CONF_PORT,
    CONF_SCAN_INTERVAL,
    CONF_SELECTED_PARAMETERS,
    CONF_SELECTED_SENSORS,
    CONF_SENSOR_SPECS,
    CONF_SERIAL_DEVICE,
    CONF_WRITE_ENABLED,
    CONN_TYPE_NETWORK,
    CONN_TYPE_SERIAL,
    DEFAULT_SCAN_INTERVAL,
    DOMAIN,
)
from .pyfroeling import (
    ErrorEntry,
    FroelingClient,
    FroelingConnectionError,
    FroelingError,
    HeaterStatus,
    SensorValue,
    ValueSpec,
    WritableParameter,
)
from .pyfroeling.const import MenuStructType

_LOGGER = logging.getLogger(__name__)


def cached_specs(data: Mapping[str, Any]) -> list[ValueSpec]:
    """Return the sensor specs the config flow cached in entry ``data``.

    Empty for entries created before the cache existed; those have to walk
    the controller's value list (GET_VALUE_LIST_FIRST / NEXT) instead.
    """
    return [
        ValueSpec(
            address   = s["address"],
            factor    = s["factor"],
            unit      = s["unit"],
            title     = s["title"],
            menu_type = s["menu_type"],
        )
        for s in data.get(CONF_SENSOR_SPECS, [])
    ]


def cached_parameter_titles(data: Mapping[str, Any]) -> dict[int, str]:
    """Return the writable-parameter titles the config flow cached in entry ``data``.

    Keys are stored as hex strings like "0x00A3"; malformed ones are skipped.
    The config flow stores every writable parameter it found, not only the
    selected ones.
    """
    titles: dict[int, str] = {}
    for addr_str, title in data.get(CONF_PARAMETER_TITLES, {}).items():
        try:
            titles[int(addr_str, 16)] = title
        except (ValueError, TypeError):
            _LOGGER.debug("Ignoring invalid parameter title key: %r", addr_str)
    return titles


def create_client(data: Mapping[str, Any]) -> FroelingClient:
    """Create an (unconnected) client for the connection stored in entry ``data``.

    Shared by entry setup and the config/options flows so the TCP vs. serial
    decision lives in one place.
    """
    if data.get(CONF_CONNECTION_TYPE, CONN_TYPE_NETWORK) == CONN_TYPE_SERIAL:
        return FroelingClient(serial_device=data.get(CONF_SERIAL_DEVICE, ""))
    return FroelingClient(host=data.get(CONF_HOST, ""), port=data.get(CONF_PORT, 0))


# ---------------------------------------------------------------------------
# Typed snapshot of all heater data
# ---------------------------------------------------------------------------

@dataclass
class FroelingData:
    """Immutable snapshot of all heater data fetched during one poll cycle.

    All entity ``native_value`` / ``is_on`` implementations read from this
    dataclass rather than querying the heater directly.  This ensures that all
    entities within a single refresh cycle see a consistent view of the data.

    Fields
    ------
    status:
        Combined heater state + firmware version, as returned by
        :meth:`FroelingClient.get_status`.
    values:
        Dict mapping 16-bit sensor address → :class:`~pyfroeling.SensorValue`.
        Includes measured values, digital outputs, analogue outputs and digital
        inputs, depending on what was discovered during ``_async_setup``.
    errors:
        List of error log entries from the controller's ring buffer, ordered
        from newest (index 0) to oldest.
    specs:
        The full list of :class:`~pyfroeling.ValueSpec` objects discovered at
        startup.  Entities use this to know which sensors exist.
    parameters:
        Dict mapping 16-bit parameter address → :class:`~pyfroeling.WritableParameter`.
        Only populated when write mode is enabled and parameter addresses are
        configured.  Empty dict when read-only mode is in use.
    """

    status: HeaterStatus
    values: dict[int, SensorValue] = field(default_factory=dict)
    errors: list[ErrorEntry] = field(default_factory=list)
    specs: list[ValueSpec] = field(default_factory=list)
    # Writable parameter snapshots – keyed by 16-bit register address.
    # Empty when write mode is disabled (read-only entries).
    parameters: dict[int, WritableParameter] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------

class FroelingCoordinator(DataUpdateCoordinator[FroelingData]):
    """Manages periodic polling of the Fröling heater and distributes data.

    Lifecycle
    ---------
    1. ``_async_setup()`` is called once at entry load time to discover all
       sensor specs from the controller.  The specs are cached in ``_specs``
       and re-used on every subsequent poll so the slow sensor-discovery phase
       is not repeated each minute.

    2. ``_async_update_data()`` is called by the HA scheduler every
       SCAN_INTERVAL seconds.  It fetches status, all sensor values, the error
       log, and (when write mode is enabled) the current writable parameter
       values, bundling them all into a :class:`FroelingData` snapshot.

    On reconnect: if the TCP connection has dropped between polls (closed or
    reset by the bridge, or reset by the client after it could not
    re-synchronise the byte stream), ``client.ensure_connected()`` reconnects
    before any data is read; it is also called before every write.
    Recovering from stray bytes or late replies on a still-open connection is
    handled inside :class:`FroelingClient`.
    """

    # The base class types this as optional (a coordinator may live without a
    # config entry); this one always gets its entry in __init__.
    config_entry: ConfigEntry

    def __init__(
        self,
        hass: HomeAssistant,
        config_entry: ConfigEntry,
        client: FroelingClient,
    ) -> None:
        """Initialise the coordinator.

        Parameters
        ----------
        hass:
            The Home Assistant instance.
        config_entry:
            The config entry this coordinator belongs to (used for logging and
            to give the coordinator a meaningful name).
        client:
            An already-instantiated (but possibly not yet connected)
            :class:`FroelingClient`.  The coordinator takes ownership of the
            connection lifecycle after this point.
        """
        # Read the polling interval from options (user-configurable), falling
        # back to the default if not set.
        scan_interval = config_entry.options.get(
            CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL
        )

        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=timedelta(seconds=scan_interval),
            always_update=False,
        )

        # The underlying protocol client – shared with __init__.py for
        # teardown (disconnect on entry unload).
        self.client = client

        # Sensor specs discovered during _async_setup().  Stored as an
        # instance attribute so _async_update_data can use them without
        # re-querying the controller.
        self._specs: list[ValueSpec] = []

        # Keep a reference to the config entry for unique_id / logging.
        self.config_entry = config_entry

        # ---------------------------------------------------------------------------
        # Write mode configuration (populated from config_entry.data)
        # ---------------------------------------------------------------------------

        # Whether the user enabled write mode during the config flow.
        self._write_enabled: bool = config_entry.data.get(CONF_WRITE_ENABLED, False)

        # The 16-bit integer addresses of parameters the user selected for polling.
        # Stored as a set for O(1) membership tests.
        param_strs = config_entry.data.get(CONF_SELECTED_PARAMETERS, [])
        self._parameter_addresses: set[int] = set()
        for addr_str in param_strs:
            try:
                # Config entry stores addresses as hex strings like "0x00A3"
                self._parameter_addresses.add(int(addr_str, 16))
            except (ValueError, TypeError):
                # Skip malformed entries rather than crashing at startup
                _LOGGER.debug(
                    "FroelingCoordinator: ignoring invalid parameter address: %r",
                    addr_str,
                )

        # -----------------------------------------------------------------------
        # Bug 1 fix: load parameter title mapping from the config entry.
        # During the config flow the user selects writable parameters whose
        # titles are stored as CONF_PARAMETER_TITLES (dict hex-addr → title).
        # Loading them here means every poll can use the real German title
        # (e.g. "Betriebsart") instead of the "Parameter 0x02F5" placeholder.
        # -----------------------------------------------------------------------
        self._parameter_titles: dict[int, str] = cached_parameter_titles(config_entry.data)

        _LOGGER.debug(
            "FroelingCoordinator: write_enabled=%s, parameter_addresses=%s",
            self._write_enabled,
            {f"0x{a:04X}" for a in self._parameter_addresses},
        )

    # ------------------------------------------------------------------
    # Setup (called once at integration load time)
    # ------------------------------------------------------------------

    async def _async_setup(self) -> None:
        """Load sensor specs, preferring the cached copy in the config entry.

        Called automatically by HA before the first
        :meth:`_async_update_data` invocation (via
        :meth:`async_config_entry_first_refresh`).

        Bug 4 / Bug 5 fix:
        The slow GET_VALUE_LIST_FIRST/NEXT discovery exchange (~90 s on the
        P1) is now skipped on startup if the config entry already contains
        cached ``sensor_specs`` data.  The config flow stores these specs
        when it creates the entry.  Old entries without the cache field fall
        back to live discovery so existing installations keep working.

        Raises
        ------
        UpdateFailed
            If live discovery is needed and it fails for any reason.
        """
        # Fast path: rebuild ValueSpec objects from the cached dicts.
        # This avoids the ~90-second discovery exchange on every HA restart.
        self._specs = cached_specs(self.config_entry.data)

        if self._specs:
            _LOGGER.info(
                "FroelingCoordinator: loaded %d cached sensor specs from config entry",
                len(self._specs),
            )
        else:
            # Slow path / fallback: ask the controller for the full spec list.
            # This runs once for old config entries that predate the spec cache,
            # and whenever the user sets up the integration from scratch without
            # a cached entry (e.g. first install before this fix was shipped).
            _LOGGER.debug(
                "FroelingCoordinator: no cached specs – discovering sensors from heater…"
            )
            try:
                self._specs = await self.client.discover_sensors()
                _LOGGER.info(
                    "FroelingCoordinator: discovered %d sensor specs from heater",
                    len(self._specs),
                )
            except FroelingError as exc:
                # Wrap library errors in UpdateFailed so HA marks the entry as
                # "unavailable" rather than crashing the event loop.
                raise UpdateFailed(
                    f"Failed to discover sensors from Fröling heater: {exc}"
                ) from exc

    # ------------------------------------------------------------------
    # Polling (called every SCAN_INTERVAL seconds)
    # ------------------------------------------------------------------

    async def _async_update_data(self) -> FroelingData:
        """Fetch a fresh data snapshot from the heater.

        Steps
        -----
        1. Reconnect if the TCP connection has been lost since the last poll.
        2. Send GET_STATE + GET_VERSION to build a :class:`~pyfroeling.HeaterStatus`.
        3. Send GET_VALUE / GET_DIG_OUT / etc. for every selected sensor spec.
        4. Read the error log (GET_ERROR_FIRST / _NEXT).
        5. If write mode is enabled, read the current value of every selected
           writable parameter via GET_PARAMETER (0x55).
        6. Bundle everything into a :class:`FroelingData` and return it.

        Raises
        ------
        UpdateFailed
            Wraps any :class:`~pyfroeling.FroelingError` so HA can mark
            entities as unavailable instead of propagating the exception.
        """
        try:
            # --- Reconnect if needed ---
            # The TCP connection might have been dropped by the serial bridge
            # (many converters have a configurable idle timeout), or reset by
            # the client because the heater stopped answering (e.g. while it
            # restarts).  A fresh connection starts with an empty stream.
            # Always called: it also notices a session the bridge closed
            # behind stray bytes, which is_connected cannot see yet.
            await self.client.ensure_connected()

            # --- Fetch heater status (state + firmware version) ---
            status = await self.client.get_status()
            _LOGGER.debug(
                "FroelingCoordinator: state=%s, mode=%s, is_error=%s",
                status.state_text, status.mode_text, status.is_error,
            )

            # --- Fetch only SELECTED sensor values ---
            # The user chose which sensors to enable during setup.
            # Only poll those addresses to minimize serial traffic.
            active_specs = self._get_selected_specs()
            _LOGGER.debug(
                "FroelingCoordinator: polling %d of %d sensors",
                len(active_specs), len(self._specs),
            )

            values = await self.client.get_all_values(active_specs)
            _LOGGER.debug(
                "FroelingCoordinator: fetched %d sensor values", len(values)
            )

            # --- Fetch the error log ---
            errors = await self.client.get_errors()
            _LOGGER.debug(
                "FroelingCoordinator: fetched %d error entries", len(errors)
            )

        except FroelingConnectionError as exc:
            raise UpdateFailed(
                f"Connection to Fröling heater lost: {exc}"
            ) from exc
        except FroelingError as exc:
            raise UpdateFailed(
                f"Error reading from Fröling heater: {exc}"
            ) from exc

        # --- Fetch writable parameter values (write mode only) ---
        # This is done AFTER the sensor/error reads so that a failure here
        # does not prevent the sensor data from being returned.
        parameters: dict[int, WritableParameter] = {}
        if self._write_enabled and self._parameter_addresses:
            for addr in self._parameter_addresses:
                try:
                    # GET_PARAMETER returns the current value, limits, unit etc.
                    param = await self.client.get_parameter(addr)

                    # Cache the title once we have it (first successful read).
                    # Subsequent reads use the cached title to avoid losing it.
                    # Note: get_parameter returns an empty title string because
                    # the GET_PARAMETER wire response does not carry the name.
                    # We rely on _parameter_titles populated from config flow data.
                    title = self._parameter_titles.get(addr, f"Parameter 0x{addr:04X}")

                    parameters[addr] = WritableParameter(
                        address=addr,
                        title=title,
                        # Assume numeric type; actual type is irrelevant for reading
                        menu_type=MenuStructType.PAR,
                        value=param.value,
                        unit=param.unit,
                        digits=param.digits,
                        factor=param.factor,
                        min_value=param.min_value,
                        max_value=param.max_value,
                        default_value=param.default_value,
                    )
                    _LOGGER.debug(
                        "FroelingCoordinator: parameter 0x%04X '%s' = %s %s",
                        addr, title, param.value, param.unit,
                    )

                except Exception as exc:
                    # A failed parameter read is non-fatal: log and continue.
                    # The entity will show "Unavailable" until the next poll.
                    _LOGGER.debug(
                        "FroelingCoordinator: failed to read parameter 0x%04X: %s",
                        addr, exc,
                    )

        return FroelingData(
            status=status,
            values=values,
            errors=errors,
            specs=self._specs,
            parameters=parameters,
        )

    # ------------------------------------------------------------------
    # Write support
    # ------------------------------------------------------------------

    async def async_write_parameter(
        self, address: int, value: float, factor: int
    ) -> float:
        """Write a parameter value to the heater and refresh data.

        Uses SET_PARAMETER (0x39) via the FroelingClient which handles the
        full multi-step write sequence (send, consume two echo frames, confirm
        via GET_PARAMETER).

        The single-connection constraint is upheld inside FroelingClient:
        no disconnect/reconnect happens during or between the write steps.

        Parameters
        ----------
        address:
            16-bit parameter register address to write.
        value:
            Physical float value to set (e.g. 65.0 for 65 °C).
        factor:
            Scale factor for this parameter (raw = int(value * factor)).

        Returns
        -------
        float
            The confirmed physical value read back from the controller after the
            write.  May differ from ``value`` due to controller rounding/clamping.
        """
        _LOGGER.debug(
            "async_write_parameter: writing 0x%04X = %s (factor=%d)",
            address, value, factor,
        )
        # The client may have reset the connection since the last poll (e.g.
        # the heater stopped answering during the write-mode parameter reads,
        # which do not fail the poll), so reconnect here instead of failing
        # every write until the next poll.
        await self.client.ensure_connected()

        # Delegate to the client; it handles the protocol handshake.
        # set_parameter() sends the write, reads two echo frames, and
        # re-reads the parameter with GET_PARAMETER to confirm (~1-2s total).
        confirmed = await self.client.set_parameter(address, value, factor)

        # Update the coordinator's data directly with the confirmed value
        # instead of triggering a full poll (which reads ALL sensors and
        # takes 30-60s). The optimistic entity update already shows the
        # new value; this just updates the coordinator's backing data so
        # the next regular poll doesn't revert it.
        if self.data and address in self.data.parameters:
            self.data.parameters[address].value = confirmed

        return confirmed

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_selected_specs(self) -> list[ValueSpec]:
        """Return only the specs that the user selected during setup.

        The selected sensor addresses are stored as a list of hex strings
        (e.g. ["0x0000", "0x0001", ...]) in config_entry.data.

        If no selection is stored (e.g. migrating from an older version),
        all discovered specs are returned as a fallback.
        """
        selected = self.config_entry.data.get(CONF_SELECTED_SENSORS)

        if not selected:
            # No selection stored -- poll everything (backwards compat)
            return self._specs

        # Convert hex strings to int addresses for fast lookup
        selected_addrs = set()
        for addr_str in selected:
            try:
                selected_addrs.add(int(addr_str, 16))
            except ValueError:
                pass

        return [s for s in self._specs if s.address in selected_addrs]

    def set_parameter_title(self, address: int, title: str) -> None:
        """Store the human-readable title for a parameter address.

        Called by the number/select platforms when they are set up, allowing
        the coordinator to label parameters with their actual names from the
        config flow discovery rather than the generic "Parameter 0xXXXX" fallback.

        Parameters
        ----------
        address:
            16-bit parameter register address.
        title:
            Human-readable parameter name (from the config flow discovery).
        """
        self._parameter_titles[address] = title
