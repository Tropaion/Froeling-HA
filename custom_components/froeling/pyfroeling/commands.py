"""Request builders and response parsers for the Fröling binary protocol.

Each protocol command has:
  1. A *builder* function that returns ``(Command, payload_bytes)`` – the
     caller passes these to :func:`~pyfroeling.protocol.build_frame`.
  2. A *parser* function that takes the raw (un-escaped, CRC-stripped) payload
     bytes returned by the controller and produces a plain Python dict.

All multi-byte integer fields in the wire format are big-endian.
Signed 16-bit values use Python's ``struct.unpack(">h", ...)`` semantics.

Unit normalisation
------------------
The controller firmware returns the degree symbol as the single byte 0xB0
(ISO-8859-1 / latin-1 "°") for temperature sensors.  We normalise every
standalone "°" to "°C" so that Home Assistant can display it correctly.

Error/value list pagination
----------------------------
The controller sends paginated responses for commands like GET_VALUE_LIST and
GET_ERROR.  The first byte of each response is a ``more`` flag:
  - ``0`` → this is the only (or last) page; no more data follows.
  - ``1`` → more pages are available; repeat the request with the _NEXT variant.
When ``more == 0`` and the response carries only that single byte, the parser
returns ``{"more": False}`` immediately without attempting to parse further.
"""

from __future__ import annotations

import struct
from datetime import datetime
from typing import Any

from .const import Command
from .models import ErrorState


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Unit correction
# ---------------------------------------------------------------------------

# The Lambdatronic protocol uses a 2-byte field for units.  Units longer than
# 2 characters get truncated by the controller firmware, producing misleading
# abbreviations.  This table maps known truncated units back to their correct
# full form.
#
# Additionally, some units are just shorthand that the controller uses
# internally and don't match standard abbreviations.
_UNIT_CORRECTION: dict[str, str] = {
    "°":  "°C",    # Lone degree sign -> degrees Celsius
    "U":  "U/min", # Revolutions -> RPM (Saugzugdrehzahl)
    "l":  "l/h",   # Litres -> litres per hour (flow rate sensors)
}

# Some sensors have units that are technically correct as a 2-byte truncation
# but ambiguous.  This table maps (sensor_title_substring, raw_unit) -> corrected_unit.
# Checked AFTER _UNIT_CORRECTION, so "°" is already "°C" at this point.
_TITLE_UNIT_CORRECTION: dict[tuple[str, str], str] = {
    # "m" is ambiguous: could be meters, minutes, or millivolts.
    # Only correct for specific sensor names where the meaning is unambiguous.
    ("Spannung",        "m"):  "mV",   # Truncated "mV" -> millivolts (if it were V, heater sends "V" directly)
    ("Betriebsstunden", "m"):  "h",    # Operating hours
    ("Laufzeit",        "m"):  "min",  # Runtime -> minutes
    ("Lastspiele",      ""):   "",     # Count, no unit
}


def _normalize_unit(unit: str, title: str = "") -> str:
    """Normalise a raw unit string from the controller.

    Applies two correction passes:
    1. Direct unit replacement from _UNIT_CORRECTION (e.g., "°" -> "°C")
    2. Context-aware correction from _TITLE_UNIT_CORRECTION using the sensor
       title to disambiguate (e.g., "m" -> "mV" for voltage sensors)

    Parameters
    ----------
    unit:
        Raw unit string decoded from the 2-byte payload field.
    title:
        Sensor title, used for context-aware disambiguation.

    Returns
    -------
    str
        Corrected unit string.
    """
    unit = unit.strip()

    # Pass 1: Direct unit correction
    if unit in _UNIT_CORRECTION:
        unit = _UNIT_CORRECTION[unit]

    # Pass 2: Context-aware correction based on sensor title
    for (title_substr, raw_unit), corrected in _TITLE_UNIT_CORRECTION.items():
        if title_substr in title and unit == raw_unit:
            unit = corrected
            break

    return unit


def _decode_string(data: bytes, encoding: str = "latin-1") -> str:
    """Decode bytes to a string, stripping null terminators and whitespace.

    Parameters
    ----------
    data:
        Raw bytes from the payload.
    encoding:
        Character encoding to use (default: latin-1 / ISO-8859-1 because the
        controller uses single-byte encodings for German characters like ä/ö/ü).

    Returns
    -------
    str
        Decoded, stripped string.
    """
    return data.decode(encoding, errors="replace").strip("\x00").strip()


# ---------------------------------------------------------------------------
# Request builders
# ---------------------------------------------------------------------------

def build_check_request() -> tuple[Command, bytes]:
    """Build a CHECK (0x22) keepalive / connectivity-test request.

    The payload is the fixed ASCII string "Tescht ;-)" which the controller
    echoes back verbatim in the response.

    Returns
    -------
    tuple[Command, bytes]
        ``(Command.CHECK, b"Tescht ;-)")``
    """
    return Command.CHECK, b"Tescht ;-)"


def build_get_state_request() -> tuple[Command, bytes]:
    """Build a GET_STATE (0x51) request.

    No payload is required – the controller responds with the current operating
    state and mode codes.

    Returns
    -------
    tuple[Command, bytes]
        ``(Command.GET_STATE, b"")``
    """
    return Command.GET_STATE, b""


def build_get_version_request() -> tuple[Command, bytes]:
    """Build a GET_VERSION (0x41) firmware-version request.

    No payload is required.  The response contains the 4-byte version number
    and the controller's current date/time.

    Returns
    -------
    tuple[Command, bytes]
        ``(Command.GET_VERSION, b"")``
    """
    return Command.GET_VERSION, b""


def build_get_value_request(address: int) -> tuple[Command, bytes]:
    """Build a GET_VALUE (0x30) request for a single sensor register.

    Parameters
    ----------
    address:
        16-bit sensor register address (big-endian on the wire).

    Returns
    -------
    tuple[Command, bytes]
        ``(Command.GET_VALUE, <2-byte address>)``
    """
    # Pack the address as a big-endian unsigned 16-bit integer.
    return Command.GET_VALUE, struct.pack(">H", address)


def build_get_value_list_request(first: bool) -> tuple[Command, bytes]:
    """Build a paginated value-list request.

    The controller's value list must be read page-by-page.  On the first call
    use ``first=True`` (sends GET_VALUE_LIST_FIRST = 0x31); on subsequent calls
    use ``first=False`` (sends GET_VALUE_LIST_NEXT = 0x32).

    Parameters
    ----------
    first:
        True for the first page request, False for continuation pages.

    Returns
    -------
    tuple[Command, bytes]
        ``(GET_VALUE_LIST_FIRST or GET_VALUE_LIST_NEXT, b"")``
    """
    cmd = Command.GET_VALUE_LIST_FIRST if first else Command.GET_VALUE_LIST_NEXT
    return cmd, b""


def build_get_parameter_request(address: int) -> tuple[Command, bytes]:
    """Build a GET_PARAMETER (0x55) request.

    Parameters
    ----------
    address:
        16-bit parameter address.

    Returns
    -------
    tuple[Command, bytes]
        ``(Command.GET_PARAMETER, <2-byte address>)``
    """
    return Command.GET_PARAMETER, struct.pack(">H", address)


def build_get_dig_out_request(address: int) -> tuple[Command, bytes]:
    """Build a GET_DIG_OUT (0x44) digital-output state request.

    Parameters
    ----------
    address:
        16-bit channel address.

    Returns
    -------
    tuple[Command, bytes]
    """
    return Command.GET_DIG_OUT, struct.pack(">H", address)


def build_get_anl_out_request(address: int) -> tuple[Command, bytes]:
    """Build a GET_ANL_OUT (0x45) analogue-output state request.

    Parameters
    ----------
    address:
        16-bit channel address.

    Returns
    -------
    tuple[Command, bytes]
    """
    return Command.GET_ANL_OUT, struct.pack(">H", address)


def build_get_dig_in_request(address: int) -> tuple[Command, bytes]:
    """Build a GET_DIG_IN (0x46) digital-input state request.

    Parameters
    ----------
    address:
        16-bit channel address.

    Returns
    -------
    tuple[Command, bytes]
    """
    return Command.GET_DIG_IN, struct.pack(">H", address)


def build_get_error_request(first: bool) -> tuple[Command, bytes]:
    """Build a paginated error-log request.

    Parameters
    ----------
    first:
        True for the first page (GET_ERROR_FIRST = 0x47), False for
        continuation pages (GET_ERROR_NEXT = 0x48).

    Returns
    -------
    tuple[Command, bytes]
        ``(GET_ERROR_FIRST or GET_ERROR_NEXT, b"")``
    """
    cmd = Command.GET_ERROR_FIRST if first else Command.GET_ERROR_NEXT
    return cmd, b""


# ---------------------------------------------------------------------------
# Response parsers
# ---------------------------------------------------------------------------

def parse_state_response(payload: bytes) -> dict[str, Any]:
    """Parse a GET_STATE (0x51) response payload.

    Wire format (unescaped, CRC already stripped):
        [mode: 1 byte] [state: 1 byte] [text: variable, semicolon-separated]

    The ``text`` field contains two German strings joined by a semicolon:
        ``"<mode_text>;<state_text>"``

    Example payload (hex + ASCII):
        ``01 03 Automatik;Heizen``

    Parameters
    ----------
    payload:
        Raw payload bytes with CRC already removed.

    Returns
    -------
    dict
        Keys: ``mode`` (int), ``state`` (int), ``mode_text`` (str),
        ``state_text`` (str).
    """
    # First byte: operating mode code (e.g. 0=manual, 1=automatic).
    mode: int = payload[0]
    # Second byte: state code (maps to STATE_TABLE).
    state: int = payload[1]

    # Remaining bytes are a latin-1 string in the form "ModeText;StateText".
    text_raw: str = _decode_string(payload[2:])
    if ";" in text_raw:
        mode_text, state_text = text_raw.split(";", 1)
    else:
        # Defensive fallback if the separator is missing.
        mode_text = text_raw
        state_text = ""

    return {
        "mode": mode,
        "state": state,
        "mode_text": mode_text.strip(),
        "state_text": state_text.strip(),
    }


def parse_version_response(payload: bytes) -> dict[str, Any]:
    """Parse a GET_VERSION (0x41) response payload.

    Wire format (unescaped, CRC stripped):
        [v1][v2][v3][v4]  – 4 firmware version bytes (displayed as hex pairs)
        [SS][MM][HH]      – seconds, minutes, hours (BCD/integer)
        [DD][MM][DOW][YY] – day, month, day-of-week, year (2-digit, + 2000)

    The version string is formatted as ``"xx.xx.xx.xx"`` where each ``xx`` is
    the hexadecimal representation of one version byte.

    Parameters
    ----------
    payload:
        Raw payload bytes with CRC stripped.

    Returns
    -------
    dict
        Keys: ``version`` (str, e.g. ``"03.12.04.01"``),
        ``datetime`` (:class:`datetime.datetime`).
    """
    # --- Version bytes (4 bytes, each displayed as two hex digits) ---
    v1, v2, v3, v4 = payload[0], payload[1], payload[2], payload[3]
    version: str = f"{v1:02x}.{v2:02x}.{v3:02x}.{v4:02x}"

    # --- Date/time bytes ---
    # The controller encodes time as SS MM HH DD MM DOW YY (7 bytes).
    seconds  = payload[4]
    minutes  = payload[5]
    hours    = payload[6]
    day      = payload[7]
    month    = payload[8]
    # payload[9] is day-of-week (0=Sunday etc.), not needed for datetime.
    year_raw = payload[10]
    year     = 2000 + year_raw  # 2-digit year stored as offset from 2000

    dt = datetime(year, month, day, hours, minutes, seconds)

    return {
        "version": version,
        "datetime": dt,
    }


def parse_value_response(payload: bytes) -> int:
    """Parse a GET_VALUE (0x30) response payload.

    Wire format:
        [value_hi][value_lo]  – signed 16-bit big-endian integer

    Parameters
    ----------
    payload:
        2 bytes: big-endian signed 16-bit raw sensor value.

    Returns
    -------
    int
        Signed raw integer value.  Divide by the sensor's ``factor`` to get
        the physical measurement.
    """
    # ">h" = big-endian signed 16-bit integer.
    (raw_value,) = struct.unpack(">h", payload[0:2])
    return raw_value


def parse_value_spec_response(payload: bytes) -> dict[str, Any]:
    """Parse one page of a GET_VALUE_LIST_* (0x31/0x32) response.

    Wire format when ``more == 1`` (there is an entry on this page):
        [more: 1]          – always 1 if there is data
        [factor: 2]        – unsigned 16-bit big-endian scale divisor
        [menu_type: 2]     – unsigned 16-bit MenuStructType code
        [unit: 2]          – 2-byte ASCII/latin-1 unit string
        [address: 2]       – unsigned 16-bit register address
        [title: variable]  – null-terminated latin-1 string

    When ``more == 0``, the payload is just the single byte 0x00 (end-of-list
    sentinel).  In this case the function returns ``{"more": False}`` without
    attempting to parse remaining bytes.

    Factor normalisation:
        A raw factor value of 0 is treated as 1 (no scaling).

    Unit normalisation:
        A standalone "°" is expanded to "°C".

    Parameters
    ----------
    payload:
        Raw payload bytes with CRC stripped.

    Returns
    -------
    dict
        When ``more == 1``:
            ``more`` (bool True), ``factor`` (int), ``menu_type`` (int),
            ``unit`` (str), ``address`` (int), ``title`` (str).
        When ``more == 0``:
            ``{"more": False}``
    """
    more: int = payload[0]
    if more == 0:
        # End-of-list sentinel.
        return {"more": False}

    # The heater occasionally sends a response with more=1 but a payload too
    # short to contain a valid entry.  linux-p4d handles this at p4io.c:1089:
    #     if (size < 11) { ... return wrnEmpty; }
    # Minimum valid entry: more(1) + factor(2) + type(2) + unit(2) + addr(2) = 9
    # (plus at least a title byte and terminator, but we're lenient here).
    if len(payload) < 10:
        return {"more": True, "empty": True}

    # --- factor (2 bytes, unsigned big-endian) ---
    (factor_raw,) = struct.unpack(">H", payload[1:3])
    # A factor of 0 means no scaling (treat as 1).
    factor: int = factor_raw if factor_raw != 0 else 1

    # --- menu_type (2 bytes, unsigned big-endian) ---
    (menu_type,) = struct.unpack(">H", payload[3:5])

    # --- unit (2 bytes, latin-1 text) -- decoded but not yet normalized ---
    unit_raw: str = _decode_string(payload[5:7])

    # --- address (2 bytes, unsigned big-endian) ---
    (address,) = struct.unpack(">H", payload[7:9])

    # --- title (remaining bytes, null-terminated latin-1 string) ---
    title_raw: bytes = payload[9:]
    null_pos = title_raw.find(b"\x00")
    if null_pos != -1:
        title_raw = title_raw[:null_pos]
    title: str = _decode_string(title_raw)

    # Normalize unit AFTER title is known, so context-aware corrections
    # (e.g., "m" -> "mV" for voltage sensors) can use the title.
    unit: str = _normalize_unit(unit_raw, title)

    return {
        "more": True,
        "factor": factor,
        "menu_type": menu_type,
        "unit": unit,
        "address": address,
        "title": title,
    }


def parse_error_response(payload: bytes) -> dict[str, Any]:
    """Parse one page of a GET_ERROR_* (0x47/0x48) response.

    Wire format when ``more == 1``:
        [more: 1]         – 1 if data follows
        [number: 2]       – unsigned 16-bit error log sequence number
        [info: 1]         – additional info byte
        [state: 1]        – ErrorState bitmask byte
        [SS][MM][HH]      – time: seconds, minutes, hours
        [DD][MM][YY]      – date: day, month, 2-digit year (+ 2000)
        [text: variable]  – null-terminated latin-1 error description

    When ``more == 0``, returns ``{"more": False}``.

    Parameters
    ----------
    payload:
        Raw payload bytes with CRC stripped.

    Returns
    -------
    dict
        When ``more == 1``:
            ``more`` (bool True), ``number`` (int), ``info`` (int),
            ``state`` (:class:`~pyfroeling.models.ErrorState`),
            ``timestamp`` (:class:`datetime.datetime`), ``text`` (str).
        When ``more == 0``:
            ``{"more": False}``
    """
    more: int = payload[0]
    if more == 0:
        return {"more": False}

    # Short payload guard (similar to value spec's size < 11 check).
    # Need at least: more(1) + number(2) + info(1) + state(1) + time(6) = 11 bytes
    if len(payload) < 11:
        return {"more": True, "empty": True}

    # --- error number (2 bytes, unsigned big-endian) ---
    (number,) = struct.unpack(">H", payload[1:3])

    # --- info byte ---
    info: int = payload[3]

    # --- error state bitmask ---
    state: ErrorState = ErrorState(payload[4])

    # --- timestamp: SS MM HH DD MM YY (6 bytes) ---
    seconds = payload[5]
    minutes = payload[6]
    hours   = payload[7]
    day     = payload[8]
    month   = payload[9]
    year    = 2000 + payload[10]

    timestamp = datetime(year, month, day, hours, minutes, seconds)

    # --- text (remaining bytes, null-terminated latin-1) ---
    text_raw: bytes = payload[11:]
    null_pos = text_raw.find(b"\x00")
    if null_pos != -1:
        text_raw = text_raw[:null_pos]
    text: str = _decode_string(text_raw)

    return {
        "more": True,
        "number": number,
        "info": info,
        "state": state,
        "timestamp": timestamp,
        "text": text,
    }


def parse_io_response(payload: bytes) -> dict[str, Any]:
    """Parse a GET_DIG_OUT / GET_ANL_OUT / GET_DIG_IN response payload.

    Wire format:
        [mode: 1]   – channel operating mode
        [state: 1]  – current physical channel state

    For digital channels:
        mode  0 = automatic, 1 = manual-off, 2 = manual-on
        state 0 = off, 1 = on

    For analogue channels:
        mode  0 = automatic, otherwise a manual setpoint
        state raw ADC/DAC value

    Parameters
    ----------
    payload:
        2 bytes from the response, CRC stripped.

    Returns
    -------
    dict
        Keys: ``mode`` (int), ``state`` (int).
    """
    return {
        "mode": payload[0],
        "state": payload[1],
    }


def build_get_menu_list_request(first: bool) -> tuple[Command, bytes]:
    """Build a paginated menu-tree list request.

    The controller's menu tree (which contains all parameters and sensors in a
    hierarchical structure) must be read page-by-page.  On the first call use
    ``first=True`` (sends GET_MENU_LIST_FIRST = 0x37); on subsequent calls
    use ``first=False`` (sends GET_MENU_LIST_NEXT = 0x38).

    No payload is needed for either variant – the controller tracks pagination
    state internally, advancing the cursor on each GET_MENU_LIST_NEXT request.

    Parameters
    ----------
    first:
        True for the initial request, False for continuation pages.

    Returns
    -------
    tuple[Command, bytes]
        ``(GET_MENU_LIST_FIRST or GET_MENU_LIST_NEXT, b"")``
    """
    cmd = Command.GET_MENU_LIST_FIRST if first else Command.GET_MENU_LIST_NEXT
    return cmd, b""


def parse_menu_entry_response(payload: bytes) -> dict[str, Any]:
    """Parse one page of a GET_MENU_LIST_* (0x37/0x38) response.

    The menu tree response delivers one entry per page.  Each entry describes a
    single node in the controller's internal menu hierarchy.  Nodes may be
    sensors (read-only) or parameters (writable, types PAR/PAR_DIG/PAR_ZEIT).

    Wire format when ``more == 1`` (source: linux-p4d p4io.c:1157, getMenuItem):
        [more: 1 byte]          – 1 if entry follows, 0 if end-of-list
        [type: 1 byte]          – MenuStructType code (e.g. 0x07=mstPar)
        [unknown1: 1 byte]      – reserved byte, skipped
        [parent: 2 bytes BE]    – parent node ID (unsigned 16-bit)
        [child: 2 bytes BE]     – this entry's node ID (unsigned 16-bit)
        [18 spare bytes]        – reserved/unknown, skipped
        [address: 2 bytes BE]   – register address for GET/SET_PARAMETER
        [unknown2: 2 bytes]     – reserved bytes, skipped
        [title: N bytes]        – null-terminated latin-1 description string

    Total fixed bytes before the title: 1+1+1+2+2+18+2+2 = 29 bytes minimum
    (plus at least one byte for the null terminator → 30 bytes minimum total).

    Parameters
    ----------
    payload:
        Raw payload bytes with CRC already stripped by the caller.

    Returns
    -------
    dict
        When ``more == 0`` (end-of-list sentinel):
            ``{"more": False}``
        When ``more == 1`` but payload is too short to contain a valid entry
        (linux-p4d p4io.c: wrnEmpty pattern, size < 30 after the more byte):
            ``{"more": True, "empty": True}``
        When ``more == 1`` and payload is long enough:
            ``{"more": True, "menu_type": int, "parent": int, "child": int,
               "address": int, "title": str}``
    """
    # First byte is always the 'more' flag.
    more: int = payload[0]
    if more == 0:
        # End-of-list sentinel – no further entries.
        return {"more": False}

    # Minimum meaningful payload: 1 (more) + 1 (type) + 1 (unk1) + 2 (parent)
    # + 2 (child) + 18 (spare) + 2 (address) + 2 (unk2) + 1 (null terminator)
    # = 30 bytes total.  Anything shorter is an "empty" entry (linux-p4d wrnEmpty).
    if len(payload) < 30:
        return {"more": True, "empty": True}

    # --- menu type (byte 1) ---
    menu_type: int = payload[1]

    # --- unknown1 (byte 2) – skip ---

    # --- parent node ID (bytes 3-4, unsigned 16-bit big-endian) ---
    (parent,) = struct.unpack(">H", payload[3:5])

    # --- child node ID (bytes 5-6, unsigned 16-bit big-endian) ---
    (child,) = struct.unpack(">H", payload[5:7])

    # --- 18 spare / reserved bytes (bytes 7-24) – skip ---

    # --- register address (bytes 25-26, unsigned 16-bit big-endian) ---
    (address,) = struct.unpack(">H", payload[25:27])

    # --- unknown2 (bytes 27-28) – skip ---

    # --- title string (byte 29 onwards, null-terminated latin-1) ---
    title_raw: bytes = payload[29:]
    null_pos = title_raw.find(b"\x00")
    if null_pos != -1:
        # Strip everything at and after the null terminator.
        title_raw = title_raw[:null_pos]
    title: str = _decode_string(title_raw)

    return {
        "more":      True,
        "menu_type": menu_type,
        "parent":    parent,
        "child":     child,
        "address":   address,
        "title":     title,
    }


def build_set_parameter_request(address: int, value: int) -> tuple[Command, bytes]:
    """Build a SET_PARAMETER (0x39) request frame payload.

    Sends the target parameter address and the new raw integer value to the
    controller.  The caller is responsible for converting the physical float
    value to the raw integer (multiply by factor) before calling this function.

    Wire format (payload only, sent after the frame header):
        [address: 2 bytes BE]  – 16-bit unsigned register address
        [value:   2 bytes BE]  – 16-bit unsigned raw value

    Note: the value on the wire is unsigned (">HH") even though the controller
    stores and returns signed values.  For negative physical values, multiply
    the signed float by factor first, then pass the resulting int here and it
    will be packed correctly as an unsigned 16-bit (two's complement).

    Parameters
    ----------
    address:
        16-bit parameter register address.
    value:
        Raw integer value to write (physical_value * factor, cast to int).

    Returns
    -------
    tuple[Command, bytes]
        ``(Command.SET_PARAMETER, <4-byte payload>)``
    """
    # Pack address and value as two consecutive unsigned 16-bit big-endian words.
    return Command.SET_PARAMETER, struct.pack(">HH", address, value)


def parse_set_parameter_response(payload: bytes) -> dict[str, Any]:
    """Parse the first echo frame returned after a SET_PARAMETER (0x39) command.

    After the controller accepts a SET_PARAMETER write it sends back two response
    frames.  This function handles the FIRST frame, which echoes the address and
    the new value back to the host for confirmation.

    Wire format (unescaped, CRC stripped):
        [address: 2 bytes BE]  – 16-bit register address (unsigned)
        [value:   2 bytes BE]  – new value as set (signed 16-bit, big-endian)

    The second echo frame is discarded by the high-level client after this parser
    returns; it carries no additional useful data.

    Parameters
    ----------
    payload:
        Raw payload bytes with CRC already removed.

    Returns
    -------
    dict
        Keys: ``address`` (int, unsigned 16-bit) and ``value`` (int, signed 16-bit).
    """
    # Address: unsigned 16-bit big-endian (">H")
    (address,) = struct.unpack(">H", payload[0:2])

    # Value: signed 16-bit big-endian (">h") – the controller echoes the signed int.
    (value,) = struct.unpack(">h", payload[2:4])

    return {
        "address": address,
        "value":   value,
    }


def parse_parameter_response(payload: bytes) -> dict[str, Any]:
    """Parse a GET_PARAMETER (0x55) response payload.

    Wire format (unescaped, CRC stripped):
        [ub1: 1]       – unknown byte 1 (ignored)
        [addr: 2]      – unsigned 16-bit parameter address (big-endian)
        [unit: 1]      – single latin-1 unit character (e.g. b'\\xB0' = "°")
        [digits: 1]    – number of decimal places to display
        [ub2: 1]       – unknown byte 2 (ignored)
        [factor: 1]    – unsigned 8-bit scale divisor (0 treated as 1)
        [value: 2]     – signed 16-bit current value (big-endian)
        [min: 2]       – signed 16-bit minimum allowed value
        [max: 2]       – signed 16-bit maximum allowed value
        [default: 2]   – signed 16-bit factory-default value
        [uw1: 2]       – unknown word (ignored)
        [ub3: 1]       – unknown byte 3 (ignored)

    All numeric values are divided by ``factor`` to get the physical float.

    Parameters
    ----------
    payload:
        Raw payload bytes with CRC stripped.

    Returns
    -------
    dict
        Keys: ``address`` (int), ``value`` (float), ``unit`` (str),
        ``digits`` (int), ``factor`` (int), ``min_value`` (float),
        ``max_value`` (float), ``default_value`` (float).

    Notes
    -----
    The ``title`` field is not present in the parameter response itself; it
    must be looked up from the menu-structure list.  The high-level client
    fills this in when constructing a :class:`~pyfroeling.models.ConfigParameter`.
    """
    # Byte 0: unknown, skip.
    # Bytes 1-2: parameter address.
    (address,) = struct.unpack(">H", payload[1:3])

    # Byte 3: single-character unit in latin-1.
    unit: str = _normalize_unit(payload[3:4].decode("latin-1", errors="replace"))

    # Byte 4: display decimal digits.
    digits: int = payload[4]

    # Byte 5: unknown, skip.

    # Byte 6: scale factor (unsigned 8-bit).
    factor_raw: int = payload[6]
    factor: int = factor_raw if factor_raw != 0 else 1

    # Bytes 7-8: current value (signed 16-bit).
    (raw_value,)   = struct.unpack(">h", payload[7:9])
    # Bytes 9-10: minimum value (signed 16-bit).
    (raw_min,)     = struct.unpack(">h", payload[9:11])
    # Bytes 11-12: maximum value (signed 16-bit).
    (raw_max,)     = struct.unpack(">h", payload[11:13])
    # Bytes 13-14: default value (signed 16-bit).
    (raw_default,) = struct.unpack(">h", payload[13:15])

    # Convert raw integers to physical floats by applying the scale factor.
    value         = raw_value   / factor
    min_value     = raw_min     / factor
    max_value     = raw_max     / factor
    default_value = raw_default / factor

    return {
        "address":       address,
        "value":         value,
        "unit":          unit,
        "digits":        digits,
        "factor":        factor,
        "min_value":     min_value,
        "max_value":     max_value,
        "default_value": default_value,
    }
