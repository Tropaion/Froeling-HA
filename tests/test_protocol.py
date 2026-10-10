"""Tests for pyfroeling protocol constants, data models, and frame encode/decode.

Test classes:
  - TestConstants     : validates const.py values and tables
  - TestModels        : validates dataclass construction and field semantics
  - TestCRC           : validates the CRC algorithm
  - TestByteEscaping  : validates escape_bytes / unescape_bytes
  - TestFrameBuilding : validates build_frame / parse_frame_header
"""

from __future__ import annotations

from datetime import datetime

import pytest

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------

from custom_components.froeling.pyfroeling.const import (
    COMM_ID,
    ERROR_STATE_CODES,
    ESCAPE_BYTES,
    ESCAPE_TABLE,
    HEADER_SIZE,
    MAX_PAYLOAD_SIZE,
    SIZE_CRC,
    STATE_TABLE,
    UNESCAPE_TABLE,
    Command,
    MenuStructType,
)
from custom_components.froeling.pyfroeling.models import (
    ConfigParameter,
    ErrorEntry,
    ErrorState,
    HeaterStatus,
    IoValue,
    SensorValue,
    ValueSpec,
)
from custom_components.froeling.pyfroeling.protocol import (
    build_frame,
    calculate_crc,
    escape_bytes,
    parse_frame_header,
    unescape_bytes,
)


# ===========================================================================
# TestConstants
# ===========================================================================

class TestConstants:
    """Verify that protocol constants have the exact values specified."""

    # --- Framing constants ---

    def test_comm_id(self) -> None:
        """Sync word must be 0x02FD."""
        assert COMM_ID == 0x02FD

    def test_header_size(self) -> None:
        """Header is 5 bytes: sync(2) + size(2) + command(1)."""
        assert HEADER_SIZE == 5

    def test_size_crc(self) -> None:
        """CRC field is one byte."""
        assert SIZE_CRC == 1

    def test_max_payload_size(self) -> None:
        """Maximum payload is 254 bytes."""
        assert MAX_PAYLOAD_SIZE == 254

    # --- Command enum ---

    def test_command_check(self) -> None:
        assert Command.CHECK == 0x22

    def test_command_get_value(self) -> None:
        assert Command.GET_VALUE == 0x30

    def test_command_get_state(self) -> None:
        assert Command.GET_STATE == 0x51

    def test_command_get_error_first(self) -> None:
        assert Command.GET_ERROR_FIRST == 0x47

    def test_command_get_value_list_first(self) -> None:
        assert Command.GET_VALUE_LIST_FIRST == 0x31

    def test_command_get_version(self) -> None:
        assert Command.GET_VERSION == 0x41

    def test_command_set_parameter(self) -> None:
        assert Command.SET_PARAMETER == 0x39

    def test_command_set_date_time(self) -> None:
        assert Command.SET_DATE_TIME == 0x54

    def test_command_get_force(self) -> None:
        assert Command.GET_FORCE == 0x5E

    def test_command_set_force(self) -> None:
        assert Command.SET_FORCE == 0x7E

    # --- State table ---

    def test_state_table_0_is_stoerung(self) -> None:
        """State 0 is 'Störung' (general fault)."""
        assert STATE_TABLE[0] == "Störung"

    def test_state_table_3_is_heizen(self) -> None:
        """State 3 is 'Heizen' (normal heating)."""
        assert STATE_TABLE[3] == "Heizen"

    def test_state_table_1_is_brenner_aus(self) -> None:
        assert STATE_TABLE[1] == "Brenner aus"

    def test_state_table_19_is_betriebsbereit(self) -> None:
        assert STATE_TABLE[19] == "Betriebsbereit"

    def test_state_table_72_is_sh_heizen(self) -> None:
        assert STATE_TABLE[72] == "SH Heizen"

    def test_state_table_41_stoerung_stb(self) -> None:
        assert STATE_TABLE[41] == "Störung STB"

    # --- ERROR_STATE_CODES ---

    def test_error_state_codes_contains_0(self) -> None:
        """State 0 ('Störung') must be an error state."""
        assert 0 in ERROR_STATE_CODES

    def test_error_state_codes_contains_41(self) -> None:
        """State 41 ('Störung STB') must be an error state."""
        assert 41 in ERROR_STATE_CODES

    def test_error_state_codes_does_not_contain_3(self) -> None:
        """State 3 ('Heizen') must NOT be an error state."""
        assert 3 not in ERROR_STATE_CODES

    def test_error_state_codes_does_not_contain_1(self) -> None:
        """State 1 ('Brenner aus') must NOT be an error state."""
        assert 1 not in ERROR_STATE_CODES

    def test_error_state_codes_contains_fehler_codes(self) -> None:
        """All states starting with 'Fehler' must be in ERROR_STATE_CODES."""
        for code, name in STATE_TABLE.items():
            if name.startswith("Fehler") or name.startswith("Störung"):
                assert code in ERROR_STATE_CODES, (
                    f"State {code} ('{name}') should be in ERROR_STATE_CODES"
                )

    # --- MenuStructType ---

    def test_menu_struct_type_messwert(self) -> None:
        assert MenuStructType.MESSWERT == 0x03

    def test_menu_struct_type_par(self) -> None:
        assert MenuStructType.PAR == 0x07

    def test_menu_struct_type_dig_out(self) -> None:
        assert MenuStructType.DIG_OUT == 0x11

    # --- Escape tables ---

    def test_escape_table_has_five_entries(self) -> None:
        """ESCAPE_TABLE must have exactly 5 entries."""
        assert len(ESCAPE_TABLE) == 5

    def test_unescape_table_is_inverse_of_escape_table(self) -> None:
        """UNESCAPE_TABLE must be the exact inverse of ESCAPE_TABLE."""
        for raw_byte, escaped_seq in ESCAPE_TABLE.items():
            assert UNESCAPE_TABLE[escaped_seq] == raw_byte

    def test_escape_bytes_frozenset(self) -> None:
        """ESCAPE_BYTES must contain all keys of ESCAPE_TABLE."""
        assert ESCAPE_BYTES == frozenset(ESCAPE_TABLE.keys())


# ===========================================================================
# TestModels
# ===========================================================================

class TestModels:
    """Verify dataclass construction and field semantics."""

    # --- ErrorState ---

    def test_error_state_arrived_value(self) -> None:
        assert ErrorState.ARRIVED == 1

    def test_error_state_acknowledged_value(self) -> None:
        assert ErrorState.ACKNOWLEDGED == 2

    def test_error_state_gone_value(self) -> None:
        assert ErrorState.GONE == 4

    def test_error_state_bitmask_combination(self) -> None:
        """Multiple ErrorState flags can be combined with bitwise OR."""
        combined = ErrorState.ARRIVED | ErrorState.ACKNOWLEDGED
        assert combined == 3

    # --- HeaterStatus ---

    def test_heater_status_is_error_true(self) -> None:
        """HeaterStatus.is_error should be True when state is an error code."""
        status = HeaterStatus(
            state=0,
            mode=0,
            state_text="Störung",
            mode_text="",
            version="1.0",
            datetime=datetime(2024, 1, 1, 12, 0),
            is_error=True,
        )
        assert status.is_error is True

    def test_heater_status_is_error_false(self) -> None:
        """HeaterStatus.is_error should be False for normal operating states."""
        status = HeaterStatus(
            state=3,
            mode=1,
            state_text="Heizen",
            mode_text="Automatik",
            version="2.3",
            datetime=datetime(2024, 6, 15, 8, 30),
            is_error=False,
        )
        assert status.is_error is False

    def test_heater_status_fields(self) -> None:
        """All HeaterStatus fields are stored correctly."""
        dt = datetime(2025, 3, 10, 14, 45)
        status = HeaterStatus(
            state=3,
            mode=1,
            state_text="Heizen",
            mode_text="Auto",
            version="3.12",
            datetime=dt,
            is_error=False,
        )
        assert status.state == 3
        assert status.mode == 1
        assert status.state_text == "Heizen"
        assert status.version == "3.12"
        assert status.datetime == dt

    # --- SensorValue ---

    def test_sensor_value_fields(self) -> None:
        """SensorValue stores raw_value and computed value independently."""
        sensor = SensorValue(
            address=0x0001,
            value=23.0,
            raw_value=230,
            factor=10,
            unit="°C",
            title="Kessel Ist",
            sensor_type="MESSWERT",
        )
        assert sensor.address == 0x0001
        assert sensor.raw_value == 230
        assert sensor.factor == 10
        assert sensor.unit == "°C"

    def test_sensor_value_computation(self) -> None:
        """value == raw_value / factor for standard temperature sensors."""
        raw = 856
        factor = 10
        expected_value = raw / factor  # 85.6

        sensor = SensorValue(
            address=0x0010,
            value=expected_value,
            raw_value=raw,
            factor=factor,
            unit="°C",
            title="Abgastemperatur",
            sensor_type="MESSWERT",
        )
        assert sensor.value == pytest.approx(85.6)

    def test_sensor_value_factor_100(self) -> None:
        """value == raw_value / 100 for high-precision sensors."""
        sensor = SensorValue(
            address=0x00FF,
            value=1.23,
            raw_value=123,
            factor=100,
            unit="bar",
            title="Druck",
            sensor_type="MESSWERT",
        )
        assert sensor.value == pytest.approx(1.23)

    # --- ErrorEntry ---

    def test_error_entry_creation(self) -> None:
        """ErrorEntry can be created with all required fields."""
        ts = datetime(2024, 12, 1, 9, 0)
        entry = ErrorEntry(
            number=1,
            text="Störung STB",
            state=ErrorState.ARRIVED,
            timestamp=ts,
            info=0,
        )
        assert entry.number == 1
        assert entry.text == "Störung STB"
        assert entry.state == ErrorState.ARRIVED
        assert entry.timestamp == ts
        assert entry.info == 0

    def test_error_entry_state_gone(self) -> None:
        """ErrorEntry.state can be set to GONE."""
        entry = ErrorEntry(
            number=5,
            text="Fehler Hydraulik",
            state=ErrorState.GONE,
            timestamp=datetime(2024, 11, 20, 18, 0),
            info=42,
        )
        assert entry.state == ErrorState.GONE

    # --- IoValue ---

    def test_io_value_creation(self) -> None:
        """IoValue stores address, mode, and state correctly."""
        io = IoValue(address=0x0005, mode=0, state=1)
        assert io.address == 0x0005
        assert io.mode == 0
        assert io.state == 1

    def test_io_value_manual_mode(self) -> None:
        """IoValue can represent a manually-forced digital output."""
        io = IoValue(address=0x000A, mode=2, state=1)  # manual-on
        assert io.mode == 2
        assert io.state == 1

    # --- ConfigParameter ---

    def test_config_parameter_fields(self) -> None:
        """ConfigParameter stores all limit and metadata fields."""
        param = ConfigParameter(
            address=0x0100,
            value=75.0,
            unit="°C",
            digits=1,
            factor=10,
            min_value=40.0,
            max_value=90.0,
            default_value=70.0,
            title="Kesseltemperatur Soll",
        )
        assert param.address == 0x0100
        assert param.value == pytest.approx(75.0)
        assert param.min_value == pytest.approx(40.0)
        assert param.max_value == pytest.approx(90.0)
        assert param.default_value == pytest.approx(70.0)
        assert param.digits == 1
        assert param.factor == 10
        assert param.title == "Kesseltemperatur Soll"

    # --- ValueSpec ---

    def test_value_spec_fields(self) -> None:
        """ValueSpec stores address, factor, unit, title, and menu_type."""
        spec = ValueSpec(
            address=0x0001,
            factor=10,
            unit="°C",
            title="Kessel Ist",
            menu_type=0x03,  # MenuStructType.MESSWERT
        )
        assert spec.address == 0x0001
        assert spec.factor == 10
        assert spec.menu_type == 0x03


# ===========================================================================
# TestCRC
# ===========================================================================

class TestCRC:
    """Verify the CRC calculation algorithm."""

    def test_crc_empty_data(self) -> None:
        """CRC of empty byte string must be 0."""
        assert calculate_crc(b"") == 0

    def test_crc_single_byte_0x41(self) -> None:
        """CRC of a single byte 0x41 ('A') must be 0xC3.

        Manual calculation:
            b = 0x41 = 65
            dummy = (65 * 2) & 0xFF = 130 = 0x82
            crc = 0 ^ (0x41 ^ 0x82) = 0 ^ 0xC3 = 0xC3
        """
        assert calculate_crc(b"\x41") == 0xC3

    def test_crc_single_byte_zero(self) -> None:
        """CRC of a single 0x00 byte must be 0."""
        # b=0, dummy=0, crc=0^(0^0)=0
        assert calculate_crc(b"\x00") == 0

    def test_crc_single_byte_0xFF(self) -> None:
        """CRC of 0xFF.

        b = 0xFF = 255
        dummy = (255 * 2) & 0xFF = 510 & 0xFF = 0xFE
        crc = 0 ^ (0xFF ^ 0xFE) = 0 ^ 0x01 = 0x01
        """
        assert calculate_crc(b"\xFF") == 0x01

    def test_crc_known_frame_check_command(self) -> None:
        """CRC is deterministic for a known byte sequence (CHECK command, no payload)."""
        # CHECK = 0x22
        # b=0x22: dummy=(0x22*2)&0xFF=0x44; crc=0^(0x22^0x44)=0x66
        data = bytes([Command.CHECK])
        crc = calculate_crc(data)
        assert crc == 0x66

    def test_crc_two_bytes(self) -> None:
        """CRC over two bytes is computed correctly."""
        # b1=0x30: dummy=0x60; crc=0^(0x30^0x60)=0x50
        # b2=0x00: dummy=0x00; crc=0x50^(0x00^0x00)=0x50
        data = bytes([Command.GET_VALUE, 0x00])
        crc = calculate_crc(data)
        assert crc == 0x50

    def test_crc_deterministic(self) -> None:
        """Calling calculate_crc twice on the same data returns the same result."""
        data = b"\x22\x30\x41\xAB\xCD"
        assert calculate_crc(data) == calculate_crc(data)

    def test_crc_is_order_independent(self) -> None:
        """The CRC is a pure XOR fold and is therefore commutative / order-independent.

        Because each byte contributes exactly (b XOR (b*2)&0xFF) via XOR to the
        accumulator, and XOR is commutative and associative, the result is the same
        regardless of byte order.  This is an intentional property of the algorithm.
        """
        data = b"\x22\x30\x41"
        # All six permutations of three bytes must yield the same CRC.
        from itertools import permutations
        crcs = {calculate_crc(bytes(p)) for p in permutations(data)}
        assert len(crcs) == 1, "All permutations of the input must give the same CRC"


# ===========================================================================
# TestByteEscaping
# ===========================================================================

class TestByteEscaping:
    """Verify escape_bytes and unescape_bytes."""

    # --- escape_bytes ---

    def test_escape_0x02(self) -> None:
        """0x02 (sync byte) must be escaped to 0x02 0x00."""
        assert escape_bytes(b"\x02") == b"\x02\x00"

    def test_escape_0x2B(self) -> None:
        """0x2B ('+') must be escaped to 0x2B 0x00."""
        assert escape_bytes(b"\x2B") == b"\x2B\x00"

    def test_escape_0xFE(self) -> None:
        """0xFE must be escaped to 0xFE 0x00."""
        assert escape_bytes(b"\xFE") == b"\xFE\x00"

    def test_escape_0x11_xon(self) -> None:
        """0x11 (XON) must be escaped to 0xFE 0x12."""
        assert escape_bytes(b"\x11") == b"\xFE\x12"

    def test_escape_0x13_xoff(self) -> None:
        """0x13 (XOFF) must be escaped to 0xFE 0x14."""
        assert escape_bytes(b"\x13") == b"\xFE\x14"

    def test_normal_bytes_unchanged(self) -> None:
        """Bytes that are not in ESCAPE_BYTES pass through unchanged."""
        normal = bytes(range(0x20, 0x7F))  # printable ASCII (no special bytes)
        # Remove any bytes that happen to need escaping
        normal = bytes(b for b in normal if b not in ESCAPE_BYTES)
        assert escape_bytes(normal) == normal

    def test_escape_single_safe_byte(self) -> None:
        """A single safe byte (e.g. 0x51) is not changed."""
        assert escape_bytes(b"\x51") == b"\x51"

    def test_escape_mixed_sequence(self) -> None:
        """A mixed sequence with both safe and special bytes is handled correctly."""
        # Input: 0x30, 0x02, 0x41
        # 0x30 → 0x30 (safe)
        # 0x02 → 0x02 0x00 (escaped)
        # 0x41 → 0x41 (safe)
        result = escape_bytes(b"\x30\x02\x41")
        assert result == b"\x30\x02\x00\x41"

    def test_escape_all_special_bytes_in_one_call(self) -> None:
        """All five special bytes are escaped correctly in a single call."""
        special = bytes([0x02, 0x2B, 0xFE, 0x11, 0x13])
        result = escape_bytes(special)
        expected = b"\x02\x00\x2B\x00\xFE\x00\xFE\x12\xFE\x14"
        assert result == expected

    # --- unescape_bytes ---

    def test_unescape_reverses_escape(self) -> None:
        """unescape_bytes must be the exact inverse of escape_bytes for all inputs."""
        for raw_byte in range(256):
            original = bytes([raw_byte])
            assert unescape_bytes(escape_bytes(original)) == original

    def test_unescape_0x02_0x00(self) -> None:
        """0x02 0x00 must unescape to 0x02."""
        assert unescape_bytes(b"\x02\x00") == b"\x02"

    def test_unescape_0xFE_0x12(self) -> None:
        """0xFE 0x12 must unescape to 0x11 (XON)."""
        assert unescape_bytes(b"\xFE\x12") == b"\x11"

    def test_unescape_0xFE_0x14(self) -> None:
        """0xFE 0x14 must unescape to 0x13 (XOFF)."""
        assert unescape_bytes(b"\xFE\x14") == b"\x13"

    def test_unescape_mixed_sequence(self) -> None:
        """Unescaping a mixed sequence with safe and escaped bytes is correct."""
        # Escaped: 0x30 (safe), 0x02 0x00 (→0x02), 0x41 (safe)
        escaped = b"\x30\x02\x00\x41"
        assert unescape_bytes(escaped) == b"\x30\x02\x41"

    def test_unescape_round_trip_all_special(self) -> None:
        """Round-trip for the sequence of all special bytes."""
        original = bytes([0x02, 0x2B, 0xFE, 0x11, 0x13])
        assert unescape_bytes(escape_bytes(original)) == original

    def test_unescape_empty(self) -> None:
        """Unescaping empty bytes returns empty bytes."""
        assert unescape_bytes(b"") == b""


# ===========================================================================
# TestFrameBuilding
# ===========================================================================

class TestFrameBuilding:
    """Verify build_frame and parse_frame_header."""

    def _unpack_frame(self, frame: bytes) -> tuple[int, int, int, bytes, int]:
        """Helper: split a built frame into (sync, size, command, payload, crc).

        Returns the raw (unescaped) values by stripping the sync and unescaping.
        """
        # First 2 bytes are the unescaped sync word.
        sync = int.from_bytes(frame[0:2], "big")
        # Unescape the rest.
        body = unescape_bytes(frame[2:])
        # body = [size hi][size lo][command][payload...][crc]
        size = int.from_bytes(body[0:2], "big")
        command = body[2]
        # payload is everything between command and the last byte (crc)
        payload = body[3:-1]
        crc = body[-1]
        return sync, size, command, payload, crc

    def test_build_check_command(self) -> None:
        """build_frame for CHECK command must produce a valid 6-byte unescaped frame."""
        frame = build_frame(Command.CHECK)
        sync, size, command, payload, crc = self._unpack_frame(frame)
        assert sync == COMM_ID
        assert command == Command.CHECK
        assert payload == b""
        # size = len(payload) + SIZE_CRC = 0 + 1 = 1
        assert size == 1
        # CRC is computed over the full unescaped frame:
        # sync_id(2) + size(2) + command(1) + payload
        full_frame = COMM_ID.to_bytes(2, "big") + size.to_bytes(2, "big") + bytes([Command.CHECK])
        assert crc == calculate_crc(full_frame)

    def test_build_get_value_with_address(self) -> None:
        """build_frame for GET_VALUE with a 2-byte address payload."""
        address = 0x00A4
        address_bytes = address.to_bytes(2, byteorder="big")
        frame = build_frame(Command.GET_VALUE, address_bytes)
        sync, size, command, payload, crc = self._unpack_frame(frame)
        assert sync == COMM_ID
        assert command == Command.GET_VALUE
        assert payload == address_bytes
        # size = 2 (payload) + 1 (CRC) = 3
        assert size == 3
        # CRC is computed over the full unescaped frame including sync_id and size.
        full_frame = (
            COMM_ID.to_bytes(2, "big")
            + size.to_bytes(2, "big")
            + bytes([Command.GET_VALUE])
            + address_bytes
        )
        expected_crc = calculate_crc(full_frame)
        assert crc == expected_crc

    def test_build_get_state_no_payload(self) -> None:
        """build_frame for GET_STATE with no payload."""
        frame = build_frame(Command.GET_STATE)
        sync, size, command, payload, crc = self._unpack_frame(frame)
        assert sync == COMM_ID
        assert command == Command.GET_STATE
        assert payload == b""
        assert size == 1

    def test_build_frame_starts_with_sync(self) -> None:
        """Every built frame must start with the 2-byte sync word 0x02FD."""
        frame = build_frame(Command.CHECK)
        assert frame[0] == 0x02
        assert frame[1] == 0xFD

    def test_build_frame_crc_is_correct(self) -> None:
        """CRC in built frame must match manual calculation over the full frame."""
        payload = b"\x01\x02\x03"
        frame = build_frame(Command.GET_VERSION, payload)
        _, size, command, parsed_payload, frame_crc = self._unpack_frame(frame)
        # CRC covers: sync_id(2) + size(2) + command(1) + payload
        full_frame = (
            COMM_ID.to_bytes(2, "big")
            + size.to_bytes(2, "big")
            + bytes([Command.GET_VERSION])
            + payload
        )
        expected_crc = calculate_crc(full_frame)
        assert frame_crc == expected_crc

    def test_crc_covers_full_frame_including_sync(self) -> None:
        """CRC must differ when computed over full frame vs command+payload only.

        This test guards against regressions where CRC is accidentally computed
        over only the command+payload portion instead of the full unescaped frame
        (sync_id + size + command + payload) as required by the linux-p4d spec.
        """
        payload = b"\xAB\xCD"
        frame = build_frame(Command.GET_VALUE, payload)
        _, size, command, parsed_payload, frame_crc = self._unpack_frame(frame)

        # CRC over full frame (correct scope)
        full_frame = (
            COMM_ID.to_bytes(2, "big")
            + size.to_bytes(2, "big")
            + bytes([Command.GET_VALUE])
            + payload
        )
        crc_full = calculate_crc(full_frame)

        # CRC over command+payload only (old/wrong scope)
        crc_cmd_payload_only = calculate_crc(bytes([Command.GET_VALUE]) + payload)

        # The frame should contain the full-frame CRC, not the shorter one.
        assert frame_crc == crc_full
        # The two CRC scopes must differ for this input (otherwise the test is vacuous).
        assert crc_full != crc_cmd_payload_only, (
            "Test input chose data where both CRC scopes happen to agree; "
            "pick a different payload to make this test meaningful."
        )

    def test_build_frame_payload_too_large_raises(self) -> None:
        """build_frame must raise ValueError when payload exceeds MAX_PAYLOAD_SIZE."""
        oversized = bytes(MAX_PAYLOAD_SIZE + 1)
        with pytest.raises(ValueError, match="Payload too large"):
            build_frame(Command.GET_VALUE, oversized)

    def test_build_frame_max_payload_accepted(self) -> None:
        """build_frame must accept a payload of exactly MAX_PAYLOAD_SIZE bytes."""
        max_payload = bytes(MAX_PAYLOAD_SIZE)
        # Should not raise
        frame = build_frame(Command.GET_VALUE, max_payload)
        assert frame is not None

    def test_build_frame_escape_applied(self) -> None:
        """If payload contains 0x02, the built frame must contain the escape sequence."""
        # Payload with 0x02 byte → must be escaped to 0x02 0x00
        payload = b"\x02"
        frame = build_frame(Command.GET_VALUE, payload)
        # The body (after the 2-byte sync word) must contain the escaped form.
        body = frame[2:]
        # 0x02 anywhere in body should only appear as part of an escape pair 0x02 0x00.
        # Quick check: the escaped form 0x02 0x00 should be present somewhere.
        assert b"\x02\x00" in body

    # --- parse_frame_header ---

    def test_parse_header_round_trip(self) -> None:
        """parse_frame_header on a built frame must return the original command."""
        frame = build_frame(Command.GET_STATE)
        # The header is the unescaped first 5 bytes.
        # frame[0:2] is the raw sync word (unescaped), frame[2:] is escaped body.
        unescaped_body = unescape_bytes(frame[2:])
        header = frame[0:2] + unescaped_body[0:3]  # sync(2) + size(2) + command(1)
        command, payload_size = parse_frame_header(header)
        assert command == Command.GET_STATE
        assert payload_size == 1  # no payload, just the CRC byte

    def test_parse_header_with_payload(self) -> None:
        """parse_frame_header returns the correct payload_size for frames with payload."""
        address_bytes = b"\x00\x01"
        frame = build_frame(Command.GET_VALUE, address_bytes)
        unescaped_body = unescape_bytes(frame[2:])
        header = frame[0:2] + unescaped_body[0:3]
        command, payload_size = parse_frame_header(header)
        assert command == Command.GET_VALUE
        # size = len(payload) + 1 = 2 + 1 = 3
        assert payload_size == 3

    def test_parse_header_wrong_length_raises(self) -> None:
        """parse_frame_header must raise ValueError if not exactly 5 bytes."""
        with pytest.raises(ValueError, match="exactly"):
            parse_frame_header(b"\x02\xFD\x00\x01")  # only 4 bytes

    def test_parse_header_wrong_sync_raises(self) -> None:
        """parse_frame_header must raise ValueError on wrong sync word."""
        bad_header = b"\x02\xFE\x00\x01\x22"  # sync 0x02FE instead of 0x02FD
        with pytest.raises(ValueError, match="sync"):
            parse_frame_header(bad_header)

    def test_parse_header_unknown_command_raises(self) -> None:
        """parse_frame_header must raise ValueError for an unknown command byte."""
        # 0x99 is not in the Command enum
        bad_header = b"\x02\xFD\x00\x01\x99"
        with pytest.raises(ValueError, match="command"):
            parse_frame_header(bad_header)
