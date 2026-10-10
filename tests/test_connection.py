"""Async tests for the FroelingConnection TCP layer (Task 5).

These tests use unittest.mock to mock asyncio.open_connection so that no
real TCP connections are made.  Each test constructs a fake reader/writer
pair to exercise the connection lifecycle and byte-reading logic.

Test classes:
  - TestConnectionLifecycle  : connect / disconnect / abort / is_connected
  - TestWriteFrame           : write_frame forwards bytes to the writer
  - TestReadAndUnescape      : _read_and_unescape correctly un-escapes bytes
  - TestReadResponse         : read_response orchestrates sync + header + body
  - TestDiscardInput         : discard_input / drain_input drop stale bytes
"""

from __future__ import annotations

import asyncio
import socket
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.froeling.pyfroeling.connection import (
    ConnectionError,
    FroelingConnection,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_writer() -> MagicMock:
    """Create a mock asyncio.StreamWriter with the minimum interface."""
    writer = MagicMock()
    # is_closing() must return False by default (connection open).
    writer.is_closing.return_value = False
    # write() and drain() are the I/O methods.
    writer.write = MagicMock()
    writer.drain = AsyncMock()
    # close() and wait_closed() for teardown.
    writer.close = MagicMock()
    writer.wait_closed = AsyncMock()
    # The transport is open until a test says otherwise.
    writer.transport.is_closing.return_value = False
    return writer


def _make_mock_reader(data_sequence: list[bytes], eof: bool = True) -> asyncio.StreamReader:
    """Create a real asyncio.StreamReader pre-loaded with ``data_sequence``.

    The chunks are concatenated into one byte stream, exactly as they would
    arrive from a socket, so the reader under test may consume them in any
    chunk size.  With ``eof=True`` the stream ends after the data (further
    reads raise IncompleteReadError); otherwise it stays open.
    """
    reader = asyncio.StreamReader()
    reader.feed_data(b"".join(data_sequence))
    if eof:
        reader.feed_eof()
    return reader


# Patch target for faking the TCP connection.
_OPEN_CONNECTION = "custom_components.froeling.pyfroeling.connection.asyncio.open_connection"


async def _connected(
    reader: asyncio.StreamReader | None = None,
    writer: MagicMock | None = None,
    timeout: float = 1.0,
) -> FroelingConnection:
    """Return a FroelingConnection "connected" to ``reader`` / ``writer``.

    Defaults: an open, empty StreamReader and a fresh mock writer.
    """
    conn = FroelingConnection(timeout=timeout)
    with patch(_OPEN_CONNECTION, new=AsyncMock(return_value=(
        reader or asyncio.StreamReader(), writer or _make_mock_writer(),
    ))):
        await conn.connect("localhost", 9000)
    return conn


async def _connected_to(data_sequence: list[bytes]) -> FroelingConnection:
    """Return a FroelingConnection reading ``data_sequence``, then EOF."""
    return await _connected(_make_mock_reader(data_sequence))


# ===========================================================================
# TestConnectionLifecycle
# ===========================================================================

class TestConnectionLifecycle:
    """Verify connect / disconnect / is_connected behaviour."""

    @pytest.mark.asyncio
    async def test_is_connected_false_before_connect(self) -> None:
        """is_connected must be False when no connect() call has been made."""
        conn = FroelingConnection()
        assert conn.is_connected is False

    @pytest.mark.asyncio
    async def test_connect_stores_host_port(self) -> None:
        """After connect(), the writer is stored and is_connected returns True."""
        conn = FroelingConnection()
        mock_reader = asyncio.StreamReader()
        mock_writer = _make_mock_writer()

        with patch(
            _OPEN_CONNECTION,
            new=AsyncMock(return_value=(mock_reader, mock_writer)),
        ) as mock_open:
            await conn.connect("192.168.1.10", 8899)

            # asyncio.open_connection must have been called with host and port.
            mock_open.assert_called_once_with("192.168.1.10", 8899)

        assert conn.is_connected is True

    @pytest.mark.asyncio
    async def test_connect_raises_connection_error_on_oserror(self) -> None:
        """connect() must raise ConnectionError when asyncio.open_connection fails."""
        conn = FroelingConnection()

        with patch(
            _OPEN_CONNECTION,
            new=AsyncMock(side_effect=OSError("refused")),
        ):
            with pytest.raises(ConnectionError, match="Cannot connect"):
                await conn.connect("192.168.1.10", 8899)

        assert conn.is_connected is False

    @pytest.mark.asyncio
    async def test_disconnect_closes_writer(self) -> None:
        """disconnect() calls writer.close() and wait_closed()."""
        mock_writer = _make_mock_writer()
        conn = await _connected(writer=mock_writer)

        assert conn.is_connected is True
        await conn.disconnect()

        # After disconnect, is_connected must return False.
        assert conn.is_connected is False
        mock_writer.close.assert_called_once()
        mock_writer.wait_closed.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_disconnect_safe_when_not_connected(self) -> None:
        """disconnect() must not raise when called without a prior connect()."""
        conn = FroelingConnection()
        # Should complete without exception.
        await conn.disconnect()

    @pytest.mark.asyncio
    async def test_is_connected_false_after_disconnect(self) -> None:
        """is_connected must be False after disconnect()."""
        conn = await _connected()

        await conn.disconnect()
        assert conn.is_connected is False

    @pytest.mark.asyncio
    async def test_is_connected_false_after_peer_eof(self) -> None:
        """is_connected returns False once the bridge closed its side (FIN).

        asyncio keeps a half-closed transport open (is_closing() stays False),
        so without this check the coordinator would never reconnect.
        """
        mock_reader = asyncio.StreamReader()
        conn = await _connected(mock_reader)

        assert conn.is_connected is True
        mock_reader.feed_eof()
        assert conn.is_connected is False

    @pytest.mark.asyncio
    async def test_abort_resets_transport(self) -> None:
        """abort() aborts the transport immediately and clears the stream."""
        mock_writer = _make_mock_writer()
        conn = await _connected(writer=mock_writer)

        conn.abort()
        mock_writer.transport.abort.assert_called_once()
        assert conn.is_connected is False
        conn.abort()  # safe when already disconnected

    @pytest.mark.asyncio
    async def test_abort_skips_transport_that_is_already_closing(self) -> None:
        """abort() must not abort a transport that is already tearing itself down.

        serial_asyncio_fast's SerialTransport would otherwise run its
        connection-lost callback twice and crash in a background task.
        """
        mock_writer = _make_mock_writer()
        conn = await _connected(writer=mock_writer)

        mock_writer.transport.is_closing.return_value = True
        conn.abort()
        mock_writer.transport.abort.assert_not_called()
        assert conn.is_connected is False

    @pytest.mark.asyncio
    async def test_connect_times_out(self) -> None:
        """A TCP connect that never completes fails after the timeout."""
        conn = FroelingConnection(timeout=0.1)

        async def never_connects(*args, **kwargs):
            await asyncio.sleep(10)

        with patch(
            _OPEN_CONNECTION,
            new=never_connects,
        ):
            with pytest.raises(ConnectionError, match="Cannot connect"):
                await conn.connect("192.0.2.1", 8899)

        assert conn.is_connected is False

    @pytest.mark.asyncio
    async def test_connect_releases_previous_transport(self) -> None:
        """A second connect() aborts the old transport instead of leaking it."""
        conn = FroelingConnection()
        old_writer = _make_mock_writer()
        new_writer = _make_mock_writer()

        with patch(
            _OPEN_CONNECTION,
            new=AsyncMock(side_effect=[
                (asyncio.StreamReader(), old_writer),
                (asyncio.StreamReader(), new_writer),
            ]),
        ):
            await conn.connect("localhost", 9000)
            await conn.connect("localhost", 9000)

        old_writer.transport.abort.assert_called_once()
        new_writer.transport.abort.assert_not_called()
        assert conn.is_connected is True

    @pytest.mark.asyncio
    async def test_is_connected_false_when_writer_is_closing(self) -> None:
        """is_connected returns False when the writer reports it is closing."""
        mock_writer = _make_mock_writer()
        mock_writer.is_closing.return_value = True  # simulate closing state
        conn = await _connected(writer=mock_writer)

        # Even though we "connected", is_closing=True means is_connected=False.
        assert conn.is_connected is False


# ===========================================================================
# TestWriteFrame
# ===========================================================================

class TestWriteFrame:
    """Verify that write_frame forwards bytes to the underlying writer."""

    @pytest.mark.asyncio
    async def test_write_frame_calls_writer_write(self) -> None:
        """write_frame must call writer.write with the exact frame bytes."""
        mock_writer = _make_mock_writer()
        conn = await _connected(writer=mock_writer)

        test_frame = b"\x02\xFD\x00\x01\x22\x66"
        await conn.write_frame(test_frame)

        mock_writer.write.assert_called_once_with(test_frame)
        mock_writer.drain.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_write_frame_raises_when_not_connected(self) -> None:
        """write_frame must raise ConnectionError when not connected."""
        conn = FroelingConnection()
        with pytest.raises(ConnectionError, match="Not connected"):
            await conn.write_frame(b"\x02\xFD")

    @pytest.mark.asyncio
    async def test_write_frame_wraps_os_error(self) -> None:
        """An OSError from drain() (e.g. connection reset) becomes ConnectionError."""
        mock_writer = _make_mock_writer()
        mock_writer.drain = AsyncMock(side_effect=ConnectionResetError("reset"))
        conn = await _connected(writer=mock_writer)

        with pytest.raises(ConnectionError, match="Connection lost"):
            await conn.write_frame(b"\x02\xFD")

    @pytest.mark.asyncio
    async def test_lock_property_returns_asyncio_lock(self) -> None:
        """The lock property must return an asyncio.Lock instance."""
        conn = FroelingConnection()
        assert isinstance(conn.lock, asyncio.Lock)


# ===========================================================================
# TestReadAndUnescape
# ===========================================================================

class TestReadAndUnescape:
    """Verify the _read_and_unescape helper handles all escape sequences."""

    @pytest.mark.asyncio
    async def test_passthrough_bytes_unchanged(self) -> None:
        """Regular (non-escape) bytes pass through un-modified."""
        # Feed three plain bytes; expect exactly three unescaped bytes back.
        conn = await _connected_to([b"\x30", b"\x41", b"\x51"])
        result = await conn._read_and_unescape(3)
        assert result == bytearray([0x30, 0x41, 0x51])

    @pytest.mark.asyncio
    async def test_unescape_0x02_0x00(self) -> None:
        """0x02 0x00 on the wire must decode to a single 0x02 byte."""
        conn = await _connected_to([b"\x02", b"\x00"])
        result = await conn._read_and_unescape(1)
        assert result == bytearray([0x02])

    @pytest.mark.asyncio
    async def test_unescape_0x2b_0x00(self) -> None:
        """0x2B 0x00 on the wire must decode to a single 0x2B byte."""
        conn = await _connected_to([b"\x2B", b"\x00"])
        result = await conn._read_and_unescape(1)
        assert result == bytearray([0x2B])

    @pytest.mark.asyncio
    async def test_unescape_0xfe_0x00(self) -> None:
        """0xFE 0x00 on the wire must decode to a single 0xFE byte."""
        conn = await _connected_to([b"\xFE", b"\x00"])
        result = await conn._read_and_unescape(1)
        assert result == bytearray([0xFE])

    @pytest.mark.asyncio
    async def test_unescape_0xfe_0x12_gives_xon(self) -> None:
        """0xFE 0x12 must decode to 0x11 (XON)."""
        conn = await _connected_to([b"\xFE", b"\x12"])
        result = await conn._read_and_unescape(1)
        assert result == bytearray([0x11])

    @pytest.mark.asyncio
    async def test_unescape_0xfe_0x14_gives_xoff(self) -> None:
        """0xFE 0x14 must decode to 0x13 (XOFF)."""
        conn = await _connected_to([b"\xFE", b"\x14"])
        result = await conn._read_and_unescape(1)
        assert result == bytearray([0x13])

    @pytest.mark.asyncio
    async def test_unescape_mixed_sequence(self) -> None:
        """Mixed escaped and plain bytes are all handled in one call."""
        # Wire: 0x30 (plain) | 0x02 0x00 (→ 0x02) | 0xFE 0x12 (→ 0x11)
        # Expected output: [0x30, 0x02, 0x11]
        conn = await _connected_to([
            b"\x30", b"\x02", b"\x00", b"\xFE", b"\x12"
        ])
        result = await conn._read_and_unescape(3)
        assert result == bytearray([0x30, 0x02, 0x11])

    @pytest.mark.asyncio
    async def test_unknown_fe_escape_raises_value_error(self) -> None:
        """0xFE followed by an unrecognised byte must raise ValueError."""
        conn = await _connected_to([b"\xFE", b"\x99"])
        with pytest.raises(ValueError, match="Unknown escape sequence"):
            await conn._read_and_unescape(1)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("wire", [b"\x02\x01", b"\x02\xFD", b"\x2B\x05"])
    async def test_invalid_self_escape_raises_value_error(self, wire: bytes) -> None:
        """0x02 / 0x2B not followed by 0x00 is a framing error (as in linux-p4d).

        ``02 FD`` in particular means a new frame started inside this one.
        """
        conn = await _connected_to([wire, b"\x41\x42"])
        with pytest.raises(ValueError, match="Invalid escape sequence"):
            await conn._read_and_unescape(3)

    @pytest.mark.asyncio
    async def test_reset_while_reading_raises_connection_error(self) -> None:
        """An OSError stored on the reader (connection reset) becomes ConnectionError."""
        conn = await _connected_to([])
        conn._reader = asyncio.StreamReader()
        conn._reader.set_exception(ConnectionResetError("reset by peer"))
        with pytest.raises(ConnectionError, match="Connection lost"):
            await conn._read_and_unescape(1)


# ===========================================================================
# TestReadResponse
# ===========================================================================

class TestReadResponse:
    """Verify that read_response correctly parses a full response frame."""

    @pytest.mark.asyncio
    async def test_read_response_raises_when_not_connected(self) -> None:
        """read_response must raise ConnectionError when not connected."""
        conn = FroelingConnection()
        with pytest.raises(ConnectionError, match="Not connected"):
            await conn.read_response()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("garbage", [
        b"\x00",                 # the byte a heater power cycle leaves behind
        b"\x00\x00\xFF",
        b"\x02",                 # "02 02 FD": sliding search, not 2-byte steps
        b"\x02\xFE\xFD",
    ])
    async def test_read_response_skips_garbage_before_sync(self, garbage: bytes) -> None:
        """Bytes in front of the sync word are skipped, not fatal.

        This is the reported bug: a single 0x00 before "02 FD" used to raise
        "Bad sync word: expected 0x02FD, got 0x0002".
        """
        conn = await _connected_to([
            garbage,
            b"\x02\xFD",                 # sync
            b"\x00\x03\x30",             # size=3, command=GET_VALUE
            b"\x07\xD1\xAA",             # payload 0x07D1 + CRC (not checked here)
        ])
        command, size, payload_with_crc = await conn.read_response()
        assert command == 0x30
        assert size == 3
        assert payload_with_crc == b"\x07\xD1\xAA"

    @pytest.mark.asyncio
    async def test_read_response_gives_up_without_sync(self) -> None:
        """A stream without any sync word raises ValueError after a bounded search."""
        conn = await _connected_to([b"\x00" * 2000])
        with pytest.raises(ValueError, match="No sync word"):
            await conn.read_response()

    @pytest.mark.asyncio
    async def test_read_response_sync_search_is_time_bounded(self) -> None:
        """A slow trickle of non-sync bytes cannot keep read_response busy for long."""
        reader = asyncio.StreamReader()
        conn = await _connected(reader, timeout=0.1)
        loop = asyncio.get_running_loop()
        handles = [loop.call_later(0.02 * i, reader.feed_data, b"\x00") for i in range(200)]

        start = loop.time()
        with pytest.raises(ValueError, match="No sync word"):
            await conn.read_response()
        assert loop.time() - start < 0.5
        for handle in handles:
            handle.cancel()

    @pytest.mark.asyncio
    async def test_cut_off_frame_does_not_swallow_next_sync(self) -> None:
        """An invalid escape ``02 FD`` is the next frame's sync word: keep it.

        A frame cut off after its CRC-less body runs into the next frame's
        sync; reading that frame must fail, but the following read must
        still find the next frame intact.
        """
        conn = await _connected_to([
            b"\x02\xFD\x00\x05\x30\xAA",        # cut off: 1 of 5 body bytes
            b"\x02\xFD\x00\x03\x30\x07\xD1\xAA",  # the next, complete frame
        ])
        with pytest.raises(ValueError, match="Invalid escape"):
            await conn.read_response()
        assert await conn.read_response() == (0x30, 3, b"\x07\xD1\xAA")

    @pytest.mark.asyncio
    async def test_bytes_kept_after_invalid_escape_are_discarded_as_stale(self) -> None:
        """Bytes kept for the next read count as stale input for discard_input."""
        conn = await _connected(_make_mock_reader([b"\x02\xFD\x00\x05\x30\xAA\x02\xFD"],
                                                  eof=False))
        with pytest.raises(ValueError, match="Invalid escape"):
            await conn.read_response()
        assert await conn.discard_input() == b"\x02\xFD"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("size_bytes", [b"\x00\x00", b"\x7F\xFF"])
    async def test_read_response_rejects_implausible_size(self, size_bytes: bytes) -> None:
        """A zero or huge size field (noise that looked like a sync) is rejected."""
        conn = await _connected_to([b"\x02\xFD", size_bytes, b"\x30", b"\x41" * 8])
        with pytest.raises(ValueError, match="Implausible frame size"):
            await conn.read_response()

    @pytest.mark.asyncio
    async def test_read_response_simple_frame(self) -> None:
        """read_response correctly parses a simple un-escaped CHECK response.

        We simulate a CHECK response with payload b"Tescht ;-)" + CRC 0x00
        (CRC value doesn't matter here – connection layer doesn't validate it).

        Wire layout (unescaped values, no special bytes so no escaping needed):
          Sync:     0x02 0xFD                       (2 bytes raw)
          Size:     0x00 0x0B  (= 10 payload + 1 CRC = 11)
          Command:  0x22       (CHECK)
          Payload:  b"Tescht ;-)" (10 bytes)
          CRC:      0xAB       (arbitrary – not validated by connection layer)
        """
        check_payload = b"Tescht ;-)"
        crc_byte = b"\xAB"
        size = len(check_payload) + 1  # +1 for CRC

        # Feed byte-by-byte (readexactly is called one byte at a time inside
        # _read_and_unescape).  We provide multi-byte chunks for the raw reads.
        raw_bytes: list[bytes] = []
        # Sync (2 bytes, raw)
        raw_bytes.append(b"\x02\xFD")
        # Remaining header: size (2 bytes) + command (1 byte) – unescaped/plain
        for b in [size >> 8, size & 0xFF, 0x22]:
            raw_bytes.append(bytes([b]))
        # Payload + CRC: each as individual byte (no escape sequences here)
        for b in check_payload + crc_byte:
            raw_bytes.append(bytes([b]))

        conn = await _connected_to(raw_bytes)
        command, s, payload_with_crc = await conn.read_response()

        assert command == 0x22  # CHECK
        assert s == size
        assert payload_with_crc == check_payload + crc_byte

    @pytest.mark.asyncio
    async def test_read_response_with_escape_in_payload(self) -> None:
        """read_response correctly un-escapes 0x02 0x00 in the payload.

        We build a minimal frame where the payload is the single byte 0x02,
        which on the wire is encoded as the two bytes 0x02 0x00.

        Unescaped frame:
          Sync:     0x02 0xFD
          Size:     0x00 0x02  (1 payload byte + 1 CRC byte)
          Command:  0x30       (GET_VALUE)
          Payload:  0x02       (raw; wire form: 0x02 0x00)
          CRC:      0x00

        Wire bytes fed to reader:
          [0x02 0xFD]  raw sync
          [0x00]       size byte 0 (unescaped)
          [0x02 0x00]  size byte 1 – but wait, 0x02 is an escape prefix!
          ...
        """
        # Let's use a simpler payload that avoids escaping in the header.
        # Size = 2 (1 payload + 1 CRC) → bytes 0x00 0x02 – 0x02 is escaped!
        # Easier: make size = 0x00 0x05 so header bytes are plain.
        # Payload = [0x30, 0x02 (escaped→0x02 0x00), 0x41] + CRC = 4 plain + CRC
        # size = 4 (3 payload bytes + 1 CRC)
        # Unescaped: cmd=0x30, payload=[0x30,0x02,0x41], crc=0xAA

        raw_bytes: list[bytes] = []
        raw_bytes.append(b"\x02\xFD")          # sync

        # Header (size=4, cmd=0x30) – individual bytes for _read_and_unescape
        for b in [0x00, 0x04, 0x30]:
            raw_bytes.append(bytes([b]))

        # Payload: 0x30 (plain), 0x02 0x00 (escaped 0x02), 0x41 (plain)
        for b in [0x30, 0x02, 0x00, 0x41]:
            raw_bytes.append(bytes([b]))

        # CRC byte (plain)
        raw_bytes.append(b"\xAA")

        conn = await _connected_to(raw_bytes)
        command, size, payload_with_crc = await conn.read_response()

        assert command == 0x30  # GET_VALUE
        assert size == 4
        # Payload (3 bytes) + CRC (1 byte): 0x30, 0x02 (unescaped), 0x41, 0xAA
        assert payload_with_crc == bytes([0x30, 0x02, 0x41, 0xAA])


# ===========================================================================
# TestDiscardInput
# ===========================================================================

class TestDiscardInput:
    """Verify that stale input is dropped before a request / during resync."""

    @pytest.mark.asyncio
    async def test_discard_input_drops_buffered_bytes(self) -> None:
        """Everything already buffered is returned and removed from the stream."""
        reader = asyncio.StreamReader()
        reader.feed_data(b"\x00\x02\xFD\x00")
        conn = await _connected(reader)

        assert await conn.discard_input() == b"\x00\x02\xFD\x00"
        assert await conn.discard_input() == b""

    @pytest.mark.asyncio
    async def test_discard_input_does_not_wait(self) -> None:
        """With nothing buffered, discard_input returns immediately."""
        conn = await _connected(asyncio.StreamReader())
        loop = asyncio.get_running_loop()
        start = loop.time()
        assert await conn.discard_input() == b""
        assert loop.time() - start < 0.05

    @pytest.mark.asyncio
    async def test_discard_input_raises_at_eof(self) -> None:
        """A closed stream is reported instead of silently returning nothing."""
        reader = asyncio.StreamReader()
        reader.feed_eof()
        conn = await _connected(reader)
        with pytest.raises(ConnectionError, match="closed"):
            await conn.discard_input()

    @pytest.mark.asyncio
    async def test_drain_input_waits_for_late_bytes(self) -> None:
        """drain_input also drops bytes that arrive within the quiet window."""
        reader = asyncio.StreamReader()
        conn = await _connected(reader)
        loop = asyncio.get_running_loop()
        loop.call_later(0.02, reader.feed_data, b"\x02\xFD")
        loop.call_later(0.04, reader.feed_data, b"\x00\x03")

        assert await conn.drain_input(quiet=0.1, max_duration=1.0) == b"\x02\xFD\x00\x03"

    @pytest.mark.asyncio
    async def test_drain_input_is_bounded(self) -> None:
        """A line that never goes quiet cannot block the drain forever."""
        reader = asyncio.StreamReader()
        conn = await _connected(reader)
        loop = asyncio.get_running_loop()
        handles = [loop.call_later(0.01 * i, reader.feed_data, b"\x00") for i in range(100)]

        start = loop.time()
        await conn.drain_input(quiet=0.05, max_duration=0.2)
        assert loop.time() - start < 0.4
        for handle in handles:
            handle.cancel()

    @pytest.mark.asyncio
    async def test_drain_input_survives_loop_stall(self) -> None:
        """Bytes that arrive just as the quiet timer expires are still drained.

        If the event loop stalls past the quiet deadline while data arrives,
        asyncio handles the socket and the expired timer in the same
        iteration and the read is cancelled with the data already buffered.
        Those bytes must not be left behind as "quiet".
        """
        rsock, wsock = socket.socketpair()
        reader, writer = await asyncio.open_connection(sock=rsock)
        conn = FroelingConnection(timeout=1.0)
        conn._reader, conn._writer = reader, writer

        def stall_then_send() -> None:
            time.sleep(0.08)                 # loop blocked past the 0.05 s deadline
            wsock.send(b"\x02\xFD\x00\x0B")

        asyncio.get_running_loop().call_later(0.01, stall_then_send)
        try:
            dropped = await conn.drain_input(quiet=0.05, max_duration=1.0)
            assert dropped == b"\x02\xFD\x00\x0B"
            assert await conn.discard_input() == b""
        finally:
            writer.close()
            wsock.close()

    @pytest.mark.asyncio
    async def test_drain_input_survives_concurrent_disconnect(self) -> None:
        """Unloading the integration mid-drain must not raise AttributeError."""
        conn = await _connected(asyncio.StreamReader())
        drain = asyncio.create_task(conn.drain_input(quiet=0.1, max_duration=1.0))
        await asyncio.sleep(0.02)
        await conn.disconnect()                   # clears conn._reader meanwhile

        assert await drain == b""
