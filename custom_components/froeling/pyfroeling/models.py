"""Data models for the Fröling Lambdatronic P/S 3200 protocol responses.

Each dataclass in this module maps directly to a response payload returned by one
or more of the protocol commands defined in :mod:`pyfroeling.const`.  All field
names and semantics follow the linux-p4d source code (Jörg Wendel) and the
protocol reverse-engineering notes.

Typical data flow:
  1. :func:`~pyfroeling.protocol.build_frame` constructs a request frame.
  2. The raw response bytes are received over TCP from the serial bridge.
  3. :func:`~pyfroeling.protocol.parse_frame_header` extracts the command and
     payload size.
  4. A parser function (to be implemented in a higher layer) populates one of
     the dataclasses below from the raw payload bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ErrorState(IntEnum):
    """Bitmask flags describing the lifecycle state of a single error log entry.

    These flags can be OR-combined.  In practice the controller sets individual
    bits as the error progresses from arrival to acknowledgement and eventually
    to being resolved (gone).

    Source: linux-p4d/lib/service.h (ErrorState enum / errorState field).
    """

    # The error has just appeared and is currently active.
    ARRIVED      = 1  # 0b001

    # The error has been acknowledged by the operator on the controller.
    ACKNOWLEDGED = 2  # 0b010

    # The error condition has been resolved and is no longer active.
    GONE         = 4  # 0b100


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class HeaterStatus:
    """Overall heater status returned by the GET_STATE command (0x51).

    The GET_STATE response delivers the numeric state code, an operating mode
    code, and optionally the controller's firmware version and current date/time.
    Convenience fields derived from these raw values are also included.

    Fields
    ------
    state : int
        Raw numeric state code from the controller (see STATE_TABLE in const.py).
    mode : int
        Operating mode code (e.g. manual, automatic, stand-by).
    state_text : str
        Human-readable German state name looked up from STATE_TABLE.
    mode_text : str
        Human-readable mode name (typically decoded by the caller).
    version : str
        Firmware version string read via GET_VERSION (0x41).
    datetime : datetime
        Controller's current date/time (may be read alongside state).
    is_error : bool
        True when ``state`` is in ERROR_STATE_CODES, i.e. the controller
        is in a fault or error condition that requires operator attention.
    """

    state: int
    mode: int
    state_text: str
    mode_text: str
    version: str
    datetime: datetime
    is_error: bool


@dataclass
class SensorValue:
    """A single measured sensor reading returned in GET_VALUE / GET_VALUE_LIST_*.

    Sensor values are transmitted as raw 16-bit integers.  The actual physical
    value is obtained by dividing the raw integer by the ``factor`` field
    (e.g. raw_value=230, factor=10 → value=23.0 °C).

    Fields
    ------
    address : int
        16-bit register address identifying this sensor in the controller.
    value : float
        Physical value after applying the scale factor (raw_value / factor).
    raw_value : int
        Raw 16-bit signed integer as received from the controller wire frame.
    factor : int
        Divisor used to convert raw_value to value.  Common values: 1, 10, 100.
    unit : str
        Physical unit string (e.g. "°C", "%", "kW", "bar").
    title : str
        Human-readable sensor name as defined in the controller menu structure.
    sensor_type : str
        Sensor or menu-structure type descriptor (maps to MenuStructType names).
    """

    address: int
    value: float
    raw_value: int
    factor: int
    unit: str
    title: str
    sensor_type: str


@dataclass
class ValueSpec:
    """Specification record for a single value/sensor entry from GET_MENU_LIST_*.

    During start-up the host queries the controller's entire menu tree to learn
    which sensor addresses exist, their scale factors, units and titles.  This
    dataclass stores that metadata and is used later to interpret raw GET_VALUE
    responses.

    Fields
    ------
    address : int
        16-bit register address in the controller.
    factor : int
        Scale divisor (see SensorValue.factor).
    unit : str
        Physical unit string.
    title : str
        Human-readable name from the controller menu.
    menu_type : int
        Raw MenuStructType code (use MenuStructType enum for comparison).
    """

    address: int
    factor: int
    unit: str
    title: str
    menu_type: int


@dataclass
class IoValue:
    """State of a single digital or analogue I/O channel.

    Returned by GET_DIG_OUT (0x44), GET_ANL_OUT (0x45), GET_DIG_IN (0x46) and
    their SET_* counterparts.

    Fields
    ------
    address : int
        16-bit channel address within the I/O table.
    mode : int
        Operating mode of the channel.
        For digital outputs: 0=automatic, 1=manual-off, 2=manual-on.
        For analogue outputs: 0=automatic, or a manual setpoint value.
    state : int
        Current physical state of the channel.
        For digital: 0=off, 1=on.
        For analogue: raw ADC/DAC value (scale factor determined by channel spec).
    """

    address: int
    mode: int
    state: int


@dataclass
class ErrorEntry:
    """A single entry from the controller's error log (GET_ERROR_FIRST / NEXT).

    The controller maintains a ring buffer of fault/warning events.  Each event
    carries a text message, a timestamp, and a bitmask indicating whether the
    error is still active, has been acknowledged, or has cleared.

    Fields
    ------
    number : int
        Sequential error number (index within the controller's error log).
    text : str
        Error description text as stored in the controller (German).
    state : ErrorState
        Bitmask of ErrorState flags (ARRIVED | ACKNOWLEDGED | GONE).
    timestamp : datetime
        Date and time when the error event was recorded by the controller.
    info : int
        Optional additional information code; meaning depends on error type.
    """

    number: int
    text: str
    state: ErrorState
    timestamp: datetime
    info: int


@dataclass
class ConfigParameter:
    """A configurable parameter read via GET_PARAMETER (0x55) or SET_PARAMETER (0x39).

    Parameters are writable values stored in the controller's EEPROM.  Each
    parameter has defined limits and a default value.  The ``digits`` field
    controls how many decimal places are shown on the controller's LCD.

    Fields
    ------
    address : int
        16-bit parameter address in the controller's parameter table.
    value : float
        Current parameter value after applying the scale factor.
    unit : str
        Physical unit string (e.g. "°C", "min", "%").
    digits : int
        Number of decimal digits to display (0 = integer, 1 = one decimal, …).
    factor : int
        Scale divisor used to convert the raw integer representation to ``value``.
    min_value : float
        Minimum allowed value (inclusive lower bound for SET_PARAMETER).
    max_value : float
        Maximum allowed value (inclusive upper bound for SET_PARAMETER).
    default_value : float
        Factory default value for this parameter.
    title : str
        Human-readable parameter name as shown on the controller LCD.
    """

    address: int
    value: float
    unit: str
    digits: int
    factor: int
    min_value: float
    max_value: float
    default_value: float
    title: str


@dataclass
class MenuItem:
    """An entry from the heater's menu tree (cmdGetMenuListFirst/Next 0x37/0x38).

    The menu tree contains all parameters, sensors, and settings in a hierarchy.
    It covers both readable sensors (MenuStructType MESSWERT, DIG_OUT, etc.) and
    writable parameters (PAR, PAR_DIG, PAR_ZEIT).

    Wire format parsed from the response payload (source: linux-p4d p4io.c:1157,
    getMenuItem function):
        [more:1 byte]      - 0 = end-of-list, 1 = entry follows
        [type:1 byte]      - MenuStructType code (e.g. 0x07 = mstPar)
        [unknown1:1 byte]  - reserved / unused byte, skipped
        [parent:2 bytes BE]- parent node ID in the menu hierarchy
        [child:2 bytes BE] - this entry's own node ID
        [18 spare bytes]   - reserved fields, skipped
        [address:2 bytes BE] - parameter/sensor register address for read/write
        [unknown2:2 bytes] - reserved bytes, skipped
        [description:N bytes] - null-terminated latin-1 string (the title)
        [terminator:1 byte]   - null byte ending the title string
        [crc:1 byte]          - frame CRC (stripped before parsing)

    Fields
    ------
    menu_type : int
        MenuStructType code (1 byte).  Key values:
          0x07 (PAR)      – numeric configuration parameter (writable)
          0x08 (PAR_DIG)  – digital/boolean parameter (writable)
          0x0A (PAR_ZEIT) – time-programme parameter (writable)
          0x03 (MESSWERT) – standard measured value (read-only sensor)
    parent : int
        Parent menu node ID, identifies where this entry sits in the hierarchy.
    child : int
        This entry's own node ID, used as a cursor for GET_MENU_LIST_NEXT.
    address : int
        16-bit register address for GET_PARAMETER / SET_PARAMETER commands.
    title : str
        Human-readable parameter or sensor name (German, latin-1 decoded).
    """

    menu_type: int      # MenuStructType code (e.g. 0x07=PAR, 0x03=MESSWERT)
    parent: int         # Parent menu node ID in the heater's menu hierarchy
    child: int          # This entry's own node ID
    address: int        # Register address for GET_PARAMETER / SET_PARAMETER
    title: str          # Human-readable name decoded from null-terminated latin-1


@dataclass
class WritableParameter:
    """A writable parameter discovered from the menu tree with its current value.

    This dataclass combines:
      - :class:`MenuItem` metadata (from cmdGetMenuListFirst/Next): address, title,
        menu type, and the node hierarchy used to navigate the menu.
      - :class:`ConfigParameter` runtime state (from cmdGetParameter 0x55): the
        current value, unit string, display digits, scale factor, and allowed range.

    Only entries with menu_type in {PAR=0x07, PAR_DIG=0x08, PAR_ZEIT=0x0A} are
    writable; the client filters to these before building WritableParameter objects.

    Fields
    ------
    address : int
        16-bit register address in the controller's parameter table.
    title : str
        Human-readable parameter name (German, from the menu tree).
    menu_type : int
        MenuStructType code – one of 0x07 (PAR), 0x08 (PAR_DIG), 0x0A (PAR_ZEIT).
    value : float
        Current parameter value scaled by factor (raw_value / factor).
    unit : str
        Physical unit string (e.g. "°C", "min", "%").
    digits : int
        Number of decimal places to display on the controller LCD.
    factor : int
        Scale divisor; raw integer values on the wire are divided by this.
    min_value : float
        Minimum allowed value (inclusive lower bound for SET_PARAMETER).
    max_value : float
        Maximum allowed value (inclusive upper bound for SET_PARAMETER).
    default_value : float
        Factory default value for this parameter.
    """

    address: int        # 16-bit register address
    title: str          # Human-readable German name from the menu tree
    menu_type: int      # mstPar (0x07), mstParDig (0x08), or mstParZeit (0x0A)
    value: float        # Current value (raw_value / factor)
    unit: str           # Physical unit string (e.g. "°C", "%", "min")
    digits: int         # Decimal places shown on the controller LCD
    factor: int         # Scale divisor for raw ↔ physical conversion
    min_value: float    # Minimum allowed value (inclusive)
    max_value: float    # Maximum allowed value (inclusive)
    default_value: float  # Factory default value
