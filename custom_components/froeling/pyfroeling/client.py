"""High-level async client for the Fröling Lambdatronic P/S 3200 controller.

This module provides :class:`FroelingClient`, which is the main entry point for
application code.  It sits on top of:

  - :mod:`pyfroeling.connection` – raw TCP byte stream (``FroelingConnection``)
  - :mod:`pyfroeling.protocol`   – frame building / CRC (``build_frame``)
  - :mod:`pyfroeling.commands`   – request builders and response parsers

Typical usage::

    from pyfroeling.client import FroelingClient

    client = FroelingClient("192.168.1.100", 8899, timeout=5.0)
    await client.connect()

    status = await client.get_status()
    print(status.state_text)   # e.g. "Heizen"

    sensors = await client.discover_sensors()
    values  = await client.get_all_values(sensors)

    await client.disconnect()

All public methods acquire the connection lock internally so that concurrent
callers from the same event-loop do not interleave frames.  Paginated reads
(FIRST/NEXT list walks) hold it across all their pages, so no other request
(a poll's, or a write) can land between two pages of the same list.

Stream synchronisation
----------------------
The protocol is strictly one request → one response with no request IDs, so
a single extra or missing byte would otherwise shift every following
request/response pair (the classic symptom: everything "unavailable" after the
heater restarts until the integration is reloaded).  The client therefore:

  - discards stale input before every request,
  - lets the connection skip garbage in front of a response's sync word,
  - marks the stream "dirty" after ANY failed exchange (timeout, framing or
    CRC error, unparseable payload, cancellation), and
  - before the next request on a dirty stream, drains the line until it is
    quiet, sends ONE CHECK ping and reads frames until its echo arrives.
    The controller answers strictly in order, so that echo proves every older
    reply has been consumed (linux-p4d drains after every request and
    re-opens the line after a failed check).  If no echo arrives the
    connection is reset, so the coordinator's next poll reconnects with a
    fresh, empty stream.

All CHECK echoes look identical.  A CHECK whose echo was never read (it timed
out, was answered by a stale frame, or its caller was cancelled) could
therefore be mistaken for the echo of the next resync's CHECK, leaving that
CHECK's echo to be read as the reply to the following request.  Two guards
prevent that from ever producing wrong data:

  - the client remembers when an unanswered CHECK was sent and, before the
    next resync, drains until two timeouts have passed since then (this
    survives reconnects: the bridge's serial line outlives the TCP session);
  - a reply carrying the CHECK text is never accepted as the answer to any
    other request; it fails that request and triggers a fresh resync instead.
"""

from __future__ import annotations

import asyncio
import logging
import struct
from collections.abc import Iterable, Mapping
from typing import Any, Callable, Optional

from .commands import (
    _normalize_unit,
    build_check_request,
    build_get_error_request,
    build_get_menu_list_request,
    build_get_parameter_request,
    build_get_state_request,
    build_get_value_list_request,
    build_get_value_request,
    build_get_version_request,
    build_set_parameter_request,
    parse_error_response,
    parse_menu_entry_response,
    parse_parameter_response,
    parse_set_parameter_response,
    parse_state_response,
    parse_value_response,
    parse_value_spec_response,
    parse_version_response,
)
from .connection import FroelingConnection
from .connection import ConnectionError as _ConnErr
from .connection import TimeoutError as _TimeoutErr
from .const import COMM_ID, ERROR_STATE_CODES, Command, MenuStructType
from .models import (
    ConfigParameter,
    ErrorEntry,
    HeaterStatus,
    MenuItem,
    SensorValue,
    ValueSpec,
    WritableParameter,
)
from .protocol import build_frame, calculate_crc

_log = logging.getLogger(__name__)

# Maximum number of pages to read from paginated commands.
# Prevents an infinite loop if the controller misbehaves.
# Bug 2 fix: raised from 500 to 1000 because the P1 sends many duplicate
# address entries in its value list (e.g. "Außentemperatur" at 0x0004 can
# appear 5+ times).  The deduplication logic skips these, so we need more
# page budget than there are unique sensors.  1000 pages is sufficient for
# the ~120 unique sensors on a P1 even with heavy duplication, while still
# providing a safety cap against infinite loops from misbehaving firmware.
_MAX_PAGES: int = 1000

# How long the line must stay silent before a re-synchronisation drain stops.
# TCP-to-serial bridges forward UART data in packets, so this is well above
# p4d's 10 ms for a local tty.  Only spent after an error.
_RESYNC_QUIET: float = 0.25

# Upper bound on a single re-synchronisation drain.
_RESYNC_MAX_DRAIN: float = 3.0

# Frames read while waiting for the resync CHECK's echo before the connection
# is reset.  Stale replies still in flight (at most one or two per failed
# request) are skipped; the budget only guards against an endless stream.
_RESYNC_MAX_FRAMES: int = 8

# The CHECK payload; the controller echoes it verbatim.
_CHECK_CMD, _CHECK_PAYLOAD = build_check_request()


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class FroelingError(Exception):
    """Base exception for all Fröling client errors.

    Derive from this class when you want to catch any error originating from
    this library without distinguishing between connection and protocol issues.
    """


class FroelingConnectionError(FroelingError):
    """Raised when the TCP connection to the heater bridge cannot be
    established or is unexpectedly lost.

    Wraps :exc:`~pyfroeling.connection.ConnectionError` and
    :exc:`~pyfroeling.connection.TimeoutError` from the lower layer.
    """


class FroelingProtocolError(FroelingError):
    """Raised when the controller returns a response that cannot be parsed
    or that does not match the expected command.

    This typically indicates a firmware version mismatch or a corrupted frame.
    """


# ---------------------------------------------------------------------------
# FroelingClient
# ---------------------------------------------------------------------------

class FroelingClient:
    """High-level async API for reading data from a Fröling heater.

    All I/O is performed over a single persistent TCP connection to a
    TCP-to-serial converter connected to the heater's COM1 port.

    Parameters
    ----------
    host:
        Hostname or IP address of the TCP-to-serial bridge.
    port:
        TCP port the bridge listens on (commonly 8899).
    timeout:
        Per-operation timeout in seconds (default: 5.0).
    resync_quiet:
        Seconds of line silence that end a re-synchronisation drain
        (default: 0.25).  Only spent after a failed exchange or a (re)connect.
    """

    def __init__(
        self,
        host: str = "",
        port: int = 0,
        serial_device: str = "",
        timeout: float = 5.0,
        resync_quiet: float = _RESYNC_QUIET,
    ) -> None:
        # Store connection parameters so we can reconnect if needed.
        self._host = host
        self._port = port
        self._serial_device = serial_device
        self._timeout = timeout
        self._resync_quiet = resync_quiet

        # The low-level connection object (works for both TCP and serial).
        self._conn: FroelingConnection = FroelingConnection(timeout=timeout)

        # True when the byte stream may no longer be aligned on a frame
        # boundary (see "Stream synchronisation" in the module docstring).
        # Checked and cleared by _resync() before the next request.
        self._needs_resync: bool = False

        # Event-loop time at which a CHECK was sent whose echo has not been
        # read yet (None if there is none).  That echo could still arrive and
        # would be indistinguishable from the echo of the next resync's CHECK.
        self._unanswered_check_at: Optional[float] = None

        # Set by disconnect(): ensure_connected() must not reopen a client
        # that was closed on purpose (e.g. on unload while a write was
        # waiting).  An explicit connect() clears it.
        self._closed: bool = False

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        """Delegate to the underlying connection's ``is_connected`` property."""
        return self._conn.is_connected

    async def connect(self) -> None:
        """Open the connection to the heater (TCP or serial).

        Automatically chooses TCP or serial based on which parameters
        were provided at construction time. If serial_device is set,
        opens a direct USB serial connection. Otherwise uses TCP.

        Any previous connection is released first.  The new stream starts
        out "dirty": some bridges replay serial data they buffered while no
        client was connected, so the first request drains the line and
        verifies alignment with a CHECK ping.

        Raises
        ------
        FroelingConnectionError
            If the connection cannot be established.
        """
        # Hold the lock so a concurrent request never sees the stream swapped
        # out underneath it.
        async with self._conn.lock:
            self._closed = False
            await self._connect_locked()

    async def ensure_connected(self) -> None:
        """Connect unless the connection is already open.

        Used by the coordinator before every poll and every write.  Checking
        under the lock means two callers that both find the connection gone
        (e.g. a poll and a write) reconnect only once.

        An open-looking connection is probed by discarding whatever arrived
        while idle: a bridge that closed the session (FIN) right after stray
        bytes is only noticed once those bytes have been read.

        Raises
        ------
        FroelingConnectionError
            If the connection cannot be established, or the client was closed
            with :meth:`disconnect`.
        """
        async with self._conn.lock:
            if self._closed:
                raise FroelingConnectionError("Client was disconnected.")
            if self._conn.is_connected:
                try:
                    stale = await self._conn.discard_input()
                except _ConnErr:
                    self._conn.abort()          # the bridge closed the session
                else:
                    _warn_discarded(stale, "received while idle")
                    return
            _log.info("Connection to the heater lost, reconnecting")
            await self._connect_locked()

    async def _connect_locked(self) -> None:
        """Open the connection (TCP or serial).  Lock must be held."""
        try:
            if self._serial_device:
                # Direct USB serial connection to COM1
                await self._conn.connect_serial(self._serial_device)
            else:
                # Network connection via TCP-to-serial converter
                await self._conn.connect(self._host, self._port)
        except (_ConnErr, _TimeoutErr) as exc:
            raise FroelingConnectionError(str(exc)) from exc
        if self._closed:
            # disconnect() ran while the connection was being opened (it does
            # not take the lock); do not leave the new session behind.
            self._conn.abort()
            raise FroelingConnectionError("Client was disconnected.")
        # The bridge may replay serial data buffered while no client was
        # connected, so verify alignment first.  _unanswered_check_at is kept:
        # the serial line outlives the TCP session, so the echo of a CHECK
        # sent on the old session can still arrive on this one.
        self._needs_resync = True

    async def disconnect(self) -> None:
        """Close the connection gracefully. Safe to call when already disconnected.

        Afterwards :meth:`ensure_connected` refuses to reconnect until
        :meth:`connect` is called explicitly.
        """
        self._closed = True
        await self._conn.disconnect()

    # ------------------------------------------------------------------
    # Public API – read methods
    # ------------------------------------------------------------------

    async def check_connection(self) -> bool:
        """Verify that the heater answers a CHECK (0x22) ping with an exact echo.

        The CHECK command sends the fixed string "Tescht ;-)" and the
        controller echoes the same bytes.  That is exactly what a
        re-synchronisation does, so this is a forced :meth:`_resync`: it also
        leaves the stream aligned, and right after a (re)connect, which needs
        a resync anyway, it costs one CHECK round trip instead of two.

        Returns
        -------
        bool
            True if the controller echoed the CHECK, False otherwise (the
            connection is then reset, as after any failed resync).
        """
        async with self._conn.lock:
            if not self._conn.is_connected:
                return False
            # Marked before the resync so a cancellation still leaves the
            # stream dirty for the next request.
            self._needs_resync = True
            try:
                await self._resync()
            except FroelingConnectionError:
                return False
            return True

    async def get_status(self) -> HeaterStatus:
        """Fetch the overall heater status by combining GET_STATE + GET_VERSION.

        Sends GET_STATE to read the current operating mode and state, then
        GET_VERSION to read the firmware version string and the controller's
        real-time clock.  The two responses are merged into a single
        :class:`~pyfroeling.models.HeaterStatus` object.

        Returns
        -------
        HeaterStatus
            Combined state + version information.

        Raises
        ------
        FroelingConnectionError
            On TCP failure.
        FroelingProtocolError
            If either response cannot be parsed.
        """
        state_cmd, state_payload = build_get_state_request()
        ver_cmd, ver_payload = build_get_version_request()

        # Both requests under one lock: a write queued in between would
        # consume a misaligned GET_STATE reply's successor and hide the
        # misalignment that the GET_VERSION parse would otherwise reveal.
        async with self._conn.lock:
            # --- GET_STATE ---
            state_data = await self._exchange(
                state_cmd, state_payload, parse_state_response
            )
            # --- GET_VERSION ---
            ver_data = await self._exchange(
                ver_cmd, ver_payload, parse_version_response
            )

        state_code: int = state_data["state"]

        return HeaterStatus(
            state      = state_code,
            mode       = state_data["mode"],
            state_text = state_data["state_text"],
            mode_text  = state_data["mode_text"],
            version    = ver_data["version"],
            datetime   = ver_data["datetime"],
            # An error state is any code listed in ERROR_STATE_CODES.
            is_error   = state_code in ERROR_STATE_CODES,
        )

    async def get_value(self, address: int, spec: ValueSpec) -> SensorValue:
        """Read a single sensor value by address.

        The raw signed integer returned by the controller is divided by
        ``spec.factor`` to produce the physical measurement.

        Parameters
        ----------
        address:
            16-bit register address to read.
        spec:
            :class:`~pyfroeling.models.ValueSpec` that provides metadata
            (factor, unit, title, sensor_type) for this address.

        Returns
        -------
        SensorValue
            Scaled physical reading with metadata.

        Raises
        ------
        FroelingConnectionError / FroelingProtocolError
            On communication failure.
        """
        cmd, payload = build_get_value_request(address)
        raw_value = await self._send_and_receive(cmd, payload, parse_value_response)

        # Apply the scale factor from the spec to get the physical value.
        factor = spec.factor if spec.factor else 1
        value  = raw_value / factor

        return SensorValue(
            address     = address,
            value       = value,
            raw_value   = raw_value,
            factor      = factor,
            unit        = spec.unit,
            title       = spec.title,
            sensor_type = MenuStructType(spec.menu_type).name
                          if spec.menu_type in [m.value for m in MenuStructType]
                          else str(spec.menu_type),
        )

    async def get_all_values(
        self, specs: list[ValueSpec]
    ) -> dict[int, SensorValue]:
        """Read sensor values for all specs in the list.

        Dispatches each spec to the appropriate GET_* command based on its
        ``menu_type`` field:

        ============  ===========  ======================================
        menu_type     Command      Mapping
        ============  ===========  ======================================
        MESSWERT      GET_VALUE    Standard sensor measurement
        MESSWERT1     GET_VALUE    Extended sensor measurement
        DIG_OUT       GET_DIG_OUT  Digital output channel state
        ANL_OUT       GET_ANL_OUT  Analogue output channel state
        DIG_IN        GET_DIG_IN   Digital input channel state
        ============  ===========  ======================================

        Errors for individual sensors are logged at WARNING level and the
        sensor is omitted from the result dict rather than aborting the entire
        batch.  The next request re-synchronises the stream first, so a failed
        sensor can never hand its reply to the following one.  If the
        connection had to be reset (heater not answering at all), the batch
        is aborted with :class:`FroelingConnectionError`: every remaining
        request would only fail with "Not connected".

        Parameters
        ----------
        specs:
            List of :class:`~pyfroeling.models.ValueSpec` objects as returned
            by :meth:`discover_sensors`.

        Returns
        -------
        dict[int, SensorValue]
            Mapping from register address to :class:`~pyfroeling.models.SensorValue`.
            Addresses that could not be read are absent from the dict.
        """
        results: dict[int, SensorValue] = {}

        for spec in specs:
            try:
                # ALL sensors discovered via cmdGetValueListFirst/Next are
                # readable with cmdGetValue (0x30), regardless of their
                # menu_type.  This matches linux-p4d's behaviour in
                # specific.c:initValueFacts() where every ValueSpec entry
                # is stored as type "VA" and read with request->getValue().
                #
                # The menu_type field is metadata about the sensor's role
                # in the heater's menu structure, NOT the read method.
                sv = await self.get_value(spec.address, spec)
                results[spec.address] = sv

            except (FroelingConnectionError, FroelingProtocolError) as exc:
                if not self.is_connected:
                    # Re-synchronisation failed and the connection was reset:
                    # every remaining request would fail too.
                    raise
                # Log and continue so one bad sensor doesn't abort the batch.
                _log.warning(
                    "Failed to read sensor at address 0x%04X (%s): %s",
                    spec.address, spec.title, exc,
                )

        return results

    async def discover_sensors(self) -> list[ValueSpec]:
        """Read the full sensor/value specification list from the controller.

        Uses the paginated GET_VALUE_LIST_FIRST / GET_VALUE_LIST_NEXT protocol
        to enumerate all entries in the controller's menu-structure table.
        Each entry becomes one :class:`~pyfroeling.models.ValueSpec`.

        The loop stops when the controller sets ``more=0`` or after
        ``_MAX_PAGES`` pages (safety guard against misbehaving firmware).

        Returns
        -------
        list[ValueSpec]
            All sensor specifications found on the controller.

        Raises
        ------
        FroelingConnectionError / FroelingProtocolError
            On communication failure during any page request.
        """
        specs: list[ValueSpec] = []
        seen_addresses: set[int] = set()  # Track addresses to skip duplicates

        for data in await self._read_pages(
            build_get_value_list_request, parse_value_spec_response, "discover_sensors"
        ):
            address = data["address"]

            # The heater often reports the same address multiple times in the
            # value list (e.g. "Außentemperatur" at 0x0004 appears 5+ times).
            # Reading the same address twice gives the same value, so skip dupes.
            if address in seen_addresses:
                _log.debug(
                    "discover_sensors: skipping duplicate address 0x%04X '%s'",
                    address, data["title"],
                )
                continue
            seen_addresses.add(address)

            specs.append(ValueSpec(
                address   = address,
                factor    = data["factor"],
                unit      = data["unit"],
                title     = data["title"],
                menu_type = data["menu_type"],
            ))
            _log.debug("discover_sensors: 0x%04X '%s'", address, data["title"])

        _log.debug("discover_sensors: %d specs", len(specs))
        return specs

    async def get_parameter(self, address: int) -> ConfigParameter:
        """Read a single configurable parameter by address.

        Parameters
        ----------
        address:
            16-bit parameter address in the controller's EEPROM.

        Returns
        -------
        ConfigParameter
            Current value, unit, limits, and scale factor.

        Raises
        ------
        FroelingConnectionError / FroelingProtocolError
        """
        cmd, payload = build_get_parameter_request(address)
        data = await self._send_and_receive(cmd, payload, parse_parameter_response)

        return ConfigParameter(
            address       = data["address"],
            value         = data["value"],
            unit          = data["unit"],
            digits        = data["digits"],
            factor        = data["factor"],
            min_value     = data["min_value"],
            max_value     = data["max_value"],
            default_value = data["default_value"],
            # The parameter title is not in the GET_PARAMETER response; it must
            # come from the menu-structure list.  Leave blank here.
            title         = "",
        )

    async def get_errors(self) -> list[ErrorEntry]:
        """Read the controller's error log.

        Iterates over paginated GET_ERROR_FIRST / GET_ERROR_NEXT responses
        until the end-of-list sentinel is received or ``_MAX_PAGES`` is hit.

        Returns
        -------
        list[ErrorEntry]
            All error log entries currently stored in the controller.

        Raises
        ------
        FroelingConnectionError / FroelingProtocolError
            On communication failure.
        """
        return [
            ErrorEntry(
                number    = data["number"],
                text      = data["text"],
                state     = data["state"],
                timestamp = data["timestamp"],
                info      = data["info"],
            )
            for data in await self._read_pages(
                build_get_error_request, parse_error_response, "get_errors"
            )
        ]

    async def discover_menu(self) -> list[MenuItem]:
        """Read the full menu-tree structure from the controller.

        Uses the paginated GET_MENU_LIST_FIRST (0x37) / GET_MENU_LIST_NEXT (0x38)
        protocol to enumerate every node in the controller's internal menu
        hierarchy.  Nodes include both read-only sensors and writable parameters.

        The enumeration pattern mirrors :meth:`discover_sensors`:
          - Send GET_MENU_LIST_FIRST on the first iteration.
          - Send GET_MENU_LIST_NEXT on all subsequent iterations.
          - Stop when the controller returns ``more=0`` (end-of-list) or after
            ``_MAX_PAGES`` pages (safety guard against misbehaving firmware).
          - Skip "empty" entries (more=1 but payload too short – linux-p4d wrnEmpty).
          - Deduplicate by address so each register appears only once.

        Returns
        -------
        list[MenuItem]
            All menu entries found on the controller (sensors and parameters).

        Raises
        ------
        FroelingConnectionError / FroelingProtocolError
            On communication failure during any page request.
        """
        items: list[MenuItem] = []
        # Track seen addresses to avoid duplicate entries (same address can
        # appear more than once in the menu tree under different parent nodes).
        seen_addresses: set[int] = set()

        for data in await self._read_pages(
            build_get_menu_list_request, parse_menu_entry_response, "discover_menu"
        ):
            address = data["address"]

            # The menu tree frequently lists the same address under multiple
            # parent nodes.  Deduplicate by address so callers see each register
            # only once.
            #
            # Bug 2 fix: do NOT deduplicate address 0x0000.  Menu GROUP nodes
            # (container/folder entries that are not associated with any
            # register) all share address 0x0000.  If we deduplicated on
            # 0x0000 we would keep only the first group node and silently drop
            # every subsequent one, causing get_writable_parameters to miss
            # entire parameter sub-trees.  All other addresses are still
            # deduplicated as before.
            if address != 0 and address in seen_addresses:
                _log.debug(
                    "discover_menu: skipping duplicate address 0x%04X '%s'",
                    address, data["title"],
                )
                continue
            # Only add non-zero addresses to the seen set; 0x0000 group nodes
            # are allowed to appear multiple times.
            if address != 0:
                seen_addresses.add(address)

            items.append(MenuItem(
                menu_type = data["menu_type"],
                parent    = data["parent"],
                child     = data["child"],
                address   = address,
                title     = data["title"],
            ))
            _log.debug(
                "discover_menu: 0x%04X type=0x%02X '%s'",
                address, data["menu_type"], data["title"],
            )

        _log.debug("discover_menu: %d items", len(items))
        return items

    async def set_parameter(self, address: int, value: float, factor: int) -> float:
        """Write a parameter value to the controller and confirm the new value.

        The Lambdatronic protocol requires a specific multi-step write sequence:
          1. Send SET_PARAMETER (0x39) with the address and raw integer value.
          2. Read the FIRST echo frame – the controller echoes address + new value.
          3. Read the SECOND echo frame – an additional confirmation frame that
             the controller always sends after a write (discarded, no useful data).
          4. Re-read the parameter with GET_PARAMETER (0x55) to confirm the value
             was stored correctly in EEPROM.

        All steps are executed while holding the connection lock for the whole
        sequence.  Otherwise a poll waiting for the lock could slip in between
        the two echo frames, read the second echo as its own response and
        leave its real response behind for the write to consume.

        IMPORTANT: The connection is NEVER disconnected or reconnected during or
        between steps.  All communication happens on the single persistent TCP or
        serial connection.

        Parameters
        ----------
        address:
            16-bit parameter register address to write.
        value:
            Physical float value to set (e.g. 75.0 for 75 °C).
        factor:
            Scale factor for this parameter.  The raw integer sent on the wire
            is ``int(value * factor)``.  Must be >= 1.

        Returns
        -------
        float
            The confirmed physical value read back from the controller after the
            write.  This may differ slightly from ``value`` due to the controller's
            internal rounding or range clamping.

        Raises
        ------
        FroelingConnectionError
            On TCP / serial communication failure.
        FroelingProtocolError
            If the first echo frame cannot be parsed.
        """
        # Convert the physical float to the raw integer value for the wire.
        # For example: value=75.5, factor=10 → raw_value=755.  Round rather
        # than truncate: 0.29 * 100 is 28.999999999999996 in floating point.
        raw_value: int = round(value * factor)

        async with self._conn.lock:
            # --- Step 1: Send SET_PARAMETER and receive the first echo frame ---
            # Parse it to confirm the controller acknowledged the write.
            cmd, req_payload = build_set_parameter_request(address, raw_value)
            await self._exchange(cmd, req_payload, parse_set_parameter_response)

            # --- Steps 2-3: Read the second echo frame ---
            # The Lambdatronic sends a second confirmation frame after a write.
            # We must consume it to keep the protocol synchronised, but its
            # content is not useful.  Problems here are non-fatal: the write
            # already succeeded, and _consume_extra_frame marks the stream for
            # re-synchronisation so a late echo cannot shift later replies.
            await self._consume_extra_frame(f"second echo for 0x{address:04X}")

            # --- Step 4: Re-read to confirm the value was stored in EEPROM ---
            gp_cmd, gp_payload = build_get_parameter_request(address)
            confirmed = await self._exchange(gp_cmd, gp_payload, parse_parameter_response)

        confirmed_value = confirmed["value"]

        _log.debug(
            "set_parameter: address=0x%04X requested=%s confirmed=%s",
            address, value, confirmed_value,
        )

        return confirmed_value

    async def get_writable_parameters(
        self, menu_items: list[MenuItem]
    ) -> list[WritableParameter]:
        """Build writable parameter objects from a filtered menu-tree list.

        Filters ``menu_items`` to only the writable parameter types:
          - 0x07 (PAR)      – numeric configuration parameters
          - 0x08 (PAR_DIG)  – digital / boolean parameters
          - 0x0A (PAR_ZEIT) – time-programme parameters

        For each writable menu item, calls :meth:`get_parameter` to fetch the
        current value, limits, unit, scale factor, and display digits.  The two
        results are merged into a :class:`~pyfroeling.models.WritableParameter`.

        Failures for individual parameters are logged at DEBUG level and the item
        is silently skipped so that one inaccessible parameter does not abort the
        entire enumeration.

        Parameters
        ----------
        menu_items:
            List of :class:`~pyfroeling.models.MenuItem` objects as returned by
            :meth:`discover_menu`.  Only those with writable menu_types are used.

        Returns
        -------
        list[WritableParameter]
            All successfully read writable parameters.
        """
        # Set of MenuStructType codes that correspond to writable parameters.
        # From const.py / service.h:
        #   PAR      = 0x07  Numeric parameter (temperature, %, etc.)
        #   PAR_DIG  = 0x08  Boolean/choice parameter (ja/nein, 0/1)
        #   PAR_ZEIT = 0x0A  Time parameter (HH:MM)
        #   PAR_SET  = 0x32  Parameter set
        #   PAR_SET1 = 0x39  Parameter set variant 1
        #   PAR_SET2 = 0x40  Parameter set variant 2
        #   WORKMODE = 0x2F  Operating mode (Sommer/Übergang/Winter)
        WRITABLE_TYPES: frozenset[int] = frozenset({
            0x07, 0x08, 0x0A, 0x0B,  # PAR, PAR_DIG, PAR_ZEIT, PAR_WEEKDAY
            0x2F,                      # WORKMODE (operating mode)
            0x32, 0x39, 0x40,          # PAR_SET, PAR_SET1, PAR_SET2
        })

        # Log all unique menu types found for debugging
        type_counts: dict[int, int] = {}
        for item in menu_items:
            type_counts[item.menu_type] = type_counts.get(item.menu_type, 0) + 1
        _log.info(
            "Menu tree types found: %s",
            {f"0x{t:02X}": c for t, c in sorted(type_counts.items())},
        )

        # Skip non-writable menu types (sensors, I/O channels, etc.).
        return await self._read_writable(
            (item.address, item.title, item.menu_type)
            for item in menu_items
            if item.menu_type in WRITABLE_TYPES
        )

    async def read_parameters(self, titles: Mapping[int, str]) -> list[WritableParameter]:
        """Read known writable parameters by address, without walking the menu.

        For parameters found earlier (e.g. the titles the config flow cached at
        setup), this skips the slow GET_MENU_LIST walk.  Their menu type is not
        known, so it is reported as PAR.

        Parameters
        ----------
        titles:
            Mapping of parameter address to its human-readable title.

        Returns
        -------
        list[WritableParameter]
            All successfully read parameters (see :meth:`get_writable_parameters`).
        """
        return await self._read_writable(
            (address, title, MenuStructType.PAR) for address, title in titles.items()
        )

    async def _read_writable(
        self, items: Iterable[tuple[int, str, int]]
    ) -> list[WritableParameter]:
        """Read each ``(address, title, menu_type)`` via GET_PARAMETER.

        For each item, :meth:`get_parameter` fetches the current value, limits,
        unit, scale factor and display digits; they are merged with the title
        and menu type into a :class:`~pyfroeling.models.WritableParameter`.

        Failures for individual parameters are logged at DEBUG level and the
        item is skipped, so one inaccessible parameter does not abort the
        whole read.  If the connection was reset, the read is aborted with
        :class:`FroelingConnectionError` instead of returning a truncated list.
        """
        results: list[WritableParameter] = []

        for address, title, menu_type in items:
            try:
                # Fetch the current value and metadata via GET_PARAMETER (0x55).
                param: ConfigParameter = await self.get_parameter(address)
            except (FroelingConnectionError, FroelingProtocolError) as exc:
                if not self.is_connected:
                    # The connection was reset; every remaining read would
                    # fail too, so report it instead of a truncated list.
                    raise
                # Skip this parameter; it may be inaccessible on this model.
                _log.debug(
                    "Skipping parameter 0x%04X ('%s'): %s", address, title, exc,
                )
                continue

            # Apply unit normalisation using the title for context
            # (e.g. "m" → "min" for runtime parameters).
            unit: str = _normalize_unit(param.unit, title)

            results.append(WritableParameter(
                address       = address,
                title         = title,                # From the menu tree (has the name)
                menu_type     = menu_type,
                value         = param.value,
                unit          = unit,
                digits        = param.digits,
                factor        = param.factor,
                min_value     = param.min_value,
                max_value     = param.max_value,
                default_value = param.default_value,
            ))

            _log.debug(
                "Parameter 0x%04X '%s' = %s %s (min=%s max=%s)",
                address, title, param.value, unit, param.min_value, param.max_value,
            )

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _send_and_receive(
        self,
        command: Command,
        payload: bytes,
        parser: Callable[[bytes], Any],
    ) -> Any:
        """Acquire the lock and perform one request/response exchange.

        This is the single choke-point for all request/response exchanges
        except :meth:`set_parameter`, which holds the lock across several of
        them.  See :meth:`_exchange` for the details.

        Parameters
        ----------
        command:
            The :class:`~pyfroeling.const.Command` opcode to send.
        payload:
            Request payload bytes (may be empty).
        parser:
            Function applied to the response payload (CRC removed) while the
            lock is still held.  Its exceptions are treated like a malformed
            frame.

        Returns
        -------
        Any
            ``parser(payload)``.
        """
        async with self._conn.lock:
            return await self._exchange(command, payload, parser)

    async def _exchange(
        self,
        command: Command,
        payload: bytes,
        parser: Callable[[bytes], Any],
    ) -> Any:
        """Send one request and return its (parsed) response.  Lock must be held.

        Steps:
          1. Fail fast when not connected.
          2. Re-synchronise first if the previous exchange failed.
          3. Discard stale input, so the read below can only see the reply to
             THIS request.
          4. Build, escape and write the request frame.
          5. Read the response frame (skipping garbage before its sync word).
          6. Verify the CRC and strip it.
          7. Apply ``parser``.

        If anything fails – including cancellation of the calling task – the
        stream is marked dirty so the next exchange re-synchronises first.

        Raises
        ------
        FroelingConnectionError
            On timeout, connection loss, or a failed re-synchronisation.  On
            connection loss the connection is reset (``is_connected`` becomes
            False) so the coordinator reconnects on its next poll.
        FroelingProtocolError
            If the response frame is malformed, fails the CRC check, or its
            payload cannot be parsed.
        """
        if not self._conn.is_connected:
            raise FroelingConnectionError(
                f"Not connected – cannot send {command.name}."
            )

        if self._needs_resync:
            await self._resync()

        # Dirty until this exchange completes: any exception, and also a
        # cancellation of the calling task, leaves it set.
        self._needs_resync = True
        try:
            # --- Discard stale input ---
            _warn_discarded(await self._conn.discard_input(), f"before {command.name}")

            # --- Build and transmit the request ---
            await self._conn.write_frame(build_frame(command, payload))

            # --- Read the response, verify and strip the CRC ---
            resp_cmd, size, payload_with_crc = await self._conn.read_response()
            payload_only = self._check_frame(command, resp_cmd, size, payload_with_crc)
            if _is_check_echo(payload_only):
                # The echo of an earlier CHECK (only _resync sends CHECK):
                # replies are misaligned.
                raise FroelingProtocolError(
                    f"Received a CHECK echo in reply to {command.name}."
                )

            result = parser(payload_only)
            self._needs_resync = False
            return result

        except _TimeoutErr as exc:
            raise FroelingConnectionError(
                f"Failed to receive response for {command.name}: {exc}"
            ) from exc
        except _ConnErr as exc:
            # The session is gone (closed / reset by the bridge).  Drop it so
            # is_connected turns False and the next poll reconnects.
            self._conn.abort()
            raise FroelingConnectionError(
                f"Connection lost during {command.name}: {exc}"
            ) from exc
        except (ValueError, IndexError, KeyError, struct.error) as exc:
            # ValueError from read_response: no sync word, implausible size or
            # invalid escape.  The others (and ValueError, e.g. from datetime()
            # or an enum) come from a parser choking on a CRC-valid payload,
            # which usually means it is the reply to a different request.
            raise FroelingProtocolError(
                f"Malformed response for {command.name}: {exc}"
            ) from exc

    async def _read_pages(
        self,
        build_request: Callable[..., tuple[Command, bytes]],
        parser: Callable[[bytes], dict[str, Any]],
        what: str,
    ) -> list[dict[str, Any]]:
        """Walk a paginated FIRST/NEXT list and return its entries' parsed pages.

        Holds the client lock across all pages, so no other request (a poll's
        or a write) can land between two pages of the same list; the
        controller keeps a single list cursor per walk.

        Sends ``build_request(first=True)`` once, then ``first=False`` until a
        page reports ``more=0`` (end of list) or ``_MAX_PAGES`` pages were
        read (safety guard against misbehaving firmware).  Pages flagged
        ``empty`` (more=1 but too short, linux-p4d's wrnEmpty) are skipped.

        Parameters
        ----------
        build_request:
            Request builder taking ``first: bool``, e.g.
            :func:`~pyfroeling.commands.build_get_value_list_request`.
        parser:
            Page parser returning a dict with at least ``more`` (and
            ``empty`` for short pages).
        what:
            Name used in log messages.

        Raises
        ------
        FroelingConnectionError / FroelingProtocolError
            On communication failure during any page request.
        """
        pages: list[dict[str, Any]] = []
        async with self._conn.lock:
            for page_num in range(_MAX_PAGES):
                cmd, payload = build_request(first=page_num == 0)
                data = await self._exchange(cmd, payload, parser)
                if not data.get("more", False):
                    break                       # end-of-list sentinel
                if data.get("empty", False):
                    _log.debug("%s page %d: skipping short/empty entry", what, page_num)
                    continue
                pages.append(data)
            else:
                _log.warning(
                    "%s hit _MAX_PAGES limit (%d); list may be incomplete",
                    what, _MAX_PAGES,
                )
        return pages

    async def _resync(self) -> None:
        """Bring a dirty stream back onto a frame boundary.  Lock must be held.

        1. Drain the line until it has been quiet for ``resync_quiet``
           seconds.  If an earlier CHECK is still unanswered, keep draining
           until two timeouts have passed since it was sent, so its echo is
           drained instead of being mistaken for ours.
        2. Send exactly ONE CHECK ping.
        3. Read frames until its echo arrives, skipping stale replies and
           garbage that were still in flight.  The controller answers in
           order, so the echo proves every older reply has been consumed.

        Never send a second CHECK while the first may be unanswered: both
        echoes are identical, and the second one would be read as the reply
        to the next request.  If the heater does not answer, or no echo shows
        up within ``_RESYNC_MAX_FRAMES`` frames, the connection is reset so
        the coordinator reconnects with a fresh stream on its next poll.

        Raises
        ------
        FroelingConnectionError
            If re-synchronisation failed (the connection has been reset).
        """
        loop = asyncio.get_running_loop()
        quiet = self._resync_quiet
        if self._unanswered_check_at is not None:
            # Give an unanswered CHECK's echo up to two timeouts (from when it
            # was sent) to arrive, so it is drained instead of matched.
            quiet = max(quiet, self._unanswered_check_at + 2 * self._timeout - loop.time())
        reason = f"no CHECK echo within {_RESYNC_MAX_FRAMES} frames"

        try:
            _warn_discarded(
                await self._conn.drain_input(
                    quiet=quiet, max_duration=quiet + _RESYNC_MAX_DRAIN,
                ),
                "while re-synchronising",
            )

            # Bounded in total, not just per byte, because the client lock is
            # held throughout.
            async with asyncio.timeout(2 * self._timeout):
                sent_at = loop.time()
                await self._conn.write_frame(build_frame(_CHECK_CMD, _CHECK_PAYLOAD))
                # Only now is a CHECK in flight (a failed send queued none).
                self._unanswered_check_at = sent_at

                for _ in range(_RESYNC_MAX_FRAMES):
                    try:
                        resp_cmd, size, payload_with_crc = await self._conn.read_response()
                        reply = self._check_frame(
                            _CHECK_CMD, resp_cmd, size, payload_with_crc
                        )
                    except (ValueError, FroelingProtocolError) as exc:
                        # Garbage or a corrupted stale frame: keep looking.
                        _log.debug(
                            "Skipping malformed frame while re-synchronising: %s", exc
                        )
                        continue
                    if _is_check_echo(reply):
                        self._unanswered_check_at = None
                        self._needs_resync = False
                        _log.debug("Re-synchronised with the heater")
                        return
                    # A reply to an earlier request that arrived late: skip it.
                    _log.debug(
                        "Skipping stale reply while re-synchronising: %s", reply[:16].hex()
                    )

        except _TimeoutErr as exc:
            reason = f"no reply to CHECK: {exc}"
        except asyncio.TimeoutError:
            reason = f"no CHECK echo within {2 * self._timeout:.1f} s"
        except _ConnErr as exc:
            reason = f"connection lost: {exc}"

        self._conn.abort()
        raise FroelingConnectionError(
            f"Lost synchronisation with the heater ({reason}); connection reset."
        )

    async def _consume_extra_frame(self, what: str) -> None:
        """Read and discard one extra response frame.  Lock must be held.

        Used for the second echo the controller sends after SET_PARAMETER.
        Failures are logged but not raised; they mark the stream dirty so the
        next exchange re-synchronises (and a lost connection is reset).
        """
        self._needs_resync = True            # until the frame has been read
        try:
            await self._conn.read_response()
        except _ConnErr as exc:
            self._conn.abort()
            _log.debug("Connection lost while reading %s: %s", what, exc)
        except (_TimeoutErr, ValueError) as exc:
            _log.debug("Could not read %s: %s", what, exc)
        else:
            self._needs_resync = False
            _log.debug("Consumed %s", what)

    @staticmethod
    def _check_frame(
        command: Command, resp_cmd: int, size: int, payload_with_crc: bytes
    ) -> bytes:
        """Verify a response frame's CRC and return its payload without the CRC.

        Parameters
        ----------
        command:
            The request command (for logging / error messages only).
        resp_cmd, size, payload_with_crc:
            The values returned by :meth:`FroelingConnection.read_response`.

        Raises
        ------
        FroelingProtocolError
            If the frame is empty or the CRC does not match.
        """
        # Log the response command for debugging but do NOT enforce a
        # strict match.  The Lambdatronic controller does not always echo
        # the same command code -- e.g. it responds to GET_VALUE_LIST_NEXT
        # (0x32) with 0x31 (GET_VALUE_LIST_FIRST) in the response header.
        # linux-p4d (the reference implementation) never checks this field.
        if resp_cmd != command.value:
            _log.debug(
                "Response command 0x%02X differs from request 0x%02X (%s) "
                "– this is normal for some Lambdatronic commands",
                resp_cmd, command.value, command.name,
            )

        # Validate that there is at least one byte (the CRC).
        if len(payload_with_crc) == 0:
            raise FroelingProtocolError(
                f"Empty response for command {command.name} – missing CRC byte."
            )

        # --- Verify the response CRC ---
        # The CRC is the last byte of payload_with_crc.
        received_crc: int = payload_with_crc[-1]
        payload_only: bytes = payload_with_crc[:-1]

        # Reconstruct the full unescaped response frame (excluding CRC) so we
        # can recompute the expected CRC.  The frame header is:
        #   sync_id (2 bytes, 0x02FD) + size (2 bytes, BE) + command (1 byte)
        # followed by the payload bytes (without CRC).
        # Note: read_response() returns (command_byte, size, payload_with_crc)
        # where size == len(payload_with_crc) (includes the CRC byte itself).
        sync_bytes: bytes = COMM_ID.to_bytes(2, byteorder="big")
        size_bytes: bytes = size.to_bytes(2, byteorder="big")
        crc_input: bytes = (
            sync_bytes
            + size_bytes
            + bytes([resp_cmd])
            + payload_only
        )
        expected_crc: int = calculate_crc(crc_input)

        if received_crc != expected_crc:
            raise FroelingProtocolError(
                f"CRC mismatch for command {command.name}: "
                f"expected 0x{expected_crc:02X}, got 0x{received_crc:02X}."
            )

        return payload_only


def _is_check_echo(payload: bytes) -> bool:
    """Return True if a response payload is the controller's echo of a CHECK.

    Only the payload is compared, like linux-p4d's check(): the response
    command byte is not reliable on this controller (see _check_frame), and
    no other reply carries the CHECK text.
    """
    return payload == _CHECK_PAYLOAD


def _warn_discarded(data: bytes, when: str) -> None:
    """Log bytes thrown away to keep the stream aligned (helps diagnose line noise)."""
    if data:
        _log.warning("Discarded %d byte(s) %s: %s", len(data), when, data[:32].hex())
