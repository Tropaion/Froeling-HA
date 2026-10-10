"""Async TCP connection layer for the Fröling Lambdatronic P/S 3200 protocol.

This module handles the raw TCP byte stream to/from the serial bridge
that is connected to the controller's COM1 service port.

Responsibilities:
  - Open / close / abort an asyncio TCP stream to the bridge.
  - Write complete escaped frames to the stream.
  - Read a response frame back, byte-by-byte un-escaping on the fly.
  - Throw away stale input (bytes nobody asked for) so that the next
    response read starts on a frame boundary.
  - Serialise all communication through an asyncio.Lock so callers can
    safely share a single connection from multiple coroutines.

Frame structure (unescaped) reminder:
    [sync_hi][sync_lo] [size_hi][size_lo] [cmd] [payload...] [crc]

The sync word (0x02FD) is sent RAW – it is never escaped.  Everything from
the size field onward is subject to the escape rules in const.ESCAPE_TABLE.
When reading a response we therefore:
  1. Search for the 2 raw sync bytes, skipping anything in front of them.
  2. Read (and un-escape) the next 3 bytes to recover size + command.
  3. Read (and un-escape) ``size`` more bytes to recover payload + CRC.

Why the sync search matters: the bridge keeps its TCP session open while the
heater restarts.  The heater's UART glitches during a power cycle and the
bridge forwards that as stray byte(s), usually a single 0x00, into the open
stream.  Because 0x02 is always escaped as ``02 00`` inside a frame, the raw
pair ``02 FD`` can only appear at a real frame start, so skipping to it is
safe.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from .const import COMM_ID, MAX_PAYLOAD_SIZE, SIZE_CRC

_log = logging.getLogger(__name__)

# Maximum number of unexpected bytes skipped while searching for the sync word
# before giving up on a response.  A maximum-size frame is ~520 wire bytes when
# every byte is escaped, so this comfortably covers a stale frame plus noise.
_MAX_SYNC_SEARCH: int = 1024

# Plausibility bound for the size field of a response (payload + CRC).  Real
# responses carry at most MAX_PAYLOAD_SIZE payload bytes; twice that leaves
# generous headroom while still rejecting a size read from line noise.
_MAX_RESPONSE_SIZE: int = 2 * (MAX_PAYLOAD_SIZE + SIZE_CRC)

# Bytes requested per read() call when discarding stale input.
_DISCARD_CHUNK: int = 4096


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class ConnectionError(Exception):
    """Raised when a TCP connection to the heater bridge cannot be established
    or when the connection drops unexpectedly mid-operation."""


class TimeoutError(Exception):
    """Raised when a read operation does not complete within the configured
    timeout period.  This usually indicates the bridge is unresponsive."""


# ---------------------------------------------------------------------------
# FroelingConnection
# ---------------------------------------------------------------------------

class FroelingConnection:
    """Manages a single async TCP connection to the Fröling serial bridge.

    The bridge is a TCP-to-serial adapter that forwards
    bytes between a TCP socket and the controller's COM1 service port.

    Usage pattern::

        conn = FroelingConnection(timeout=5.0)
        await conn.connect("192.168.1.100", 8899)
        async with conn.lock:
            await conn.write_frame(frame_bytes)
            command, size, payload_crc = await conn.read_response()
        await conn.disconnect()

    Thread safety:
        All send/receive operations must be protected by ``conn.lock``.
        The lock is deliberately *not* acquired inside write_frame /
        read_response so that callers can atomically pair a write with its
        corresponding read without the lock being released in between.

    Parameters
    ----------
    timeout:
        Seconds to wait for each individual asyncio read before raising
        :exc:`TimeoutError`.  Defaults to 5.0 s.
    """

    def __init__(self, timeout: float = 5.0) -> None:
        # Configurable per-read timeout in seconds.
        self._timeout: float = timeout

        # asyncio StreamReader / StreamWriter set by connect().
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None

        # Mutex so the high-level client can lock a request/response pair.
        self._lock: asyncio.Lock = asyncio.Lock()

        # Bytes read ahead and handed back to the stream: when an invalid
        # escape turns out to be the next frame's sync word (a cut-off frame
        # ran into it), those bytes are returned here so the next read still
        # finds that frame.  Served before the reader by _read_one_byte.
        self._pushback = bytearray()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        """Return True when the underlying stream appears to be open.

        Checks that a writer exists, that it has not been closed
        (asyncio.StreamWriter.is_closing() returns True once close() has been
        called or the peer reset the connection) and that the peer has not
        closed its side.  The last check matters for a bridge that ends the
        TCP session gracefully (FIN): asyncio keeps such a half-closed
        transport open, so is_closing() alone would keep reporting True and
        the coordinator would never reconnect.
        """
        if self._writer is None or self._reader is None:
            return False
        return not self._writer.is_closing() and not self._reader.at_eof()

    @property
    def lock(self) -> asyncio.Lock:
        """Return the asyncio.Lock used to serialise request/response pairs."""
        return self._lock

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self, host: str, port: int) -> None:
        """Open a TCP connection to the serial bridge.

        Parameters
        ----------
        host:
            Hostname or IP address of the TCP-to-serial bridge.
        port:
            TCP port the bridge listens on (commonly 8899).

        Raises
        ------
        ConnectionError
            If the TCP handshake fails (OSError / ConnectionRefusedError from
            asyncio, re-raised as our own ConnectionError for clean API) or
            does not complete within the timeout.
        """
        _log.debug("Connecting to %s:%d (timeout=%.1f s)", host, port, self._timeout)
        # Never leak a previous (e.g. half-closed) transport: release it first.
        self.abort()
        try:
            # Bounded: an unreachable bridge whose SYNs are silently dropped
            # would otherwise block for the OS retry period (~2 minutes),
            # holding the client's lock the whole time.
            async with asyncio.timeout(self._timeout):
                self._reader, self._writer = await asyncio.open_connection(host, port)
        except asyncio.TimeoutError as exc:
            raise ConnectionError(
                f"Cannot connect to Fröling bridge at {host}:{port}: "
                f"no answer within {self._timeout:.1f} s"
            ) from exc
        except OSError as exc:
            raise ConnectionError(
                f"Cannot connect to Fröling bridge at {host}:{port}: {exc}"
            ) from exc
        _log.info("Connected to Fröling bridge at %s:%d", host, port)

    async def connect_serial(self, device: str, baudrate: int = 57600) -> None:
        """Open a direct USB serial connection to the heater's COM1 port.

        Uses pyserial-asyncio to open the serial device. The returned
        reader/writer streams are API-compatible with asyncio TCP streams,
        so the rest of the protocol code works unchanged.

        Parameters
        ----------
        device:
            Serial device path (e.g., /dev/ttyUSB0, COM3).
        baudrate:
            Baud rate -- always 57600 for the Lambdatronic protocol.

        Raises
        ------
        ConnectionError
            If the serial port cannot be opened.
        """
        _log.debug("Opening serial port %s at %d baud", device, baudrate)
        # Never leak a previous transport: release it first.
        self.abort()
        try:
            import serial_asyncio_fast as serial_asyncio
        except ImportError:
            try:
                # Optional legacy fallback (same API); not a requirement, so
                # type checkers usually cannot resolve it.
                import serial_asyncio  # type: ignore[import-not-found, no-redef]
            except ImportError:
                raise ConnectionError(
                    "pyserial-asyncio is required for USB serial connections. "
                    "Install it with: pip install pyserial-asyncio"
                )
        try:
            self._reader, self._writer = await serial_asyncio.open_serial_connection(
                url=device,
                baudrate=baudrate,
                bytesize=8,       # 8 data bits
                parity="N",       # No parity
                stopbits=1,       # 1 stop bit
                xonxoff=False,    # No software flow control
                rtscts=False,     # No hardware flow control
            )
        except Exception as exc:
            raise ConnectionError(
                f"Cannot open serial port {device}: {exc}"
            ) from exc
        _log.info("Connected to Fröling heater via serial port %s", device)

    async def disconnect(self) -> None:
        """Close the TCP connection gracefully.

        Calls writer.close() and awaits writer.wait_closed() so that the OS
        socket is fully released before this coroutine returns.  Safe to call
        even when already disconnected.
        """
        writer = self._writer
        if writer is None:
            return
        _log.debug("Disconnecting from Fröling bridge")
        # Clear the references BEFORE awaiting, so is_connected is False at
        # once and a connection opened meanwhile is never cleared by mistake.
        self._writer = None
        self._reader = None
        self._pushback.clear()
        try:
            writer.close()
            await writer.wait_closed()
        except OSError:
            # Ignore errors during teardown – the connection may already be gone.
            pass
        _log.info("Disconnected from Fröling bridge")

    def abort(self) -> None:
        """Tear the connection down immediately, discarding all buffered data.

        Used when the stream can no longer be trusted (the peer went away or
        the client could not re-synchronise).  Unlike :meth:`disconnect`, it
        does not wait for pending output to be flushed, which could take a
        long time on a dead TCP session.  Afterwards :attr:`is_connected` is
        False, so the next poll reconnects with a fresh, empty stream.  Safe
        to call when already disconnected.
        """
        if self._writer is None:
            return
        transport = self._writer.transport
        try:
            # A transport that is already closing (reset by the peer, or a
            # serial device error) is tearing itself down; aborting it again
            # would make serial_asyncio_fast run its connection-lost callback
            # twice and crash in a background task.
            if not transport.is_closing():
                transport.abort()
        except Exception as exc:  # teardown must never raise
            _log.debug("Error while aborting connection: %s", exc)
        finally:
            self._writer = None
            self._reader = None
            self._pushback.clear()
        _log.info("Connection to Fröling bridge reset")

    # ------------------------------------------------------------------
    # Frame I/O
    # ------------------------------------------------------------------

    async def write_frame(self, frame: bytes) -> None:
        """Write a complete escaped frame to the TCP stream.

        Parameters
        ----------
        frame:
            A fully-built, escaped wire frame as returned by
            :func:`~pyfroeling.protocol.build_frame`.

        Raises
        ------
        ConnectionError
            If the writer is not available (not connected), or the OS reports
            the connection as broken (e.g. reset by the bridge).
        """
        if self._writer is None:
            raise ConnectionError("Not connected – call connect() first.")
        _log.debug("TX %d bytes: %s", len(frame), frame.hex())
        try:
            self._writer.write(frame)
            # drain() flushes the write buffer and yields to the event loop,
            # ensuring the bytes are handed off to the OS network stack.
            await self._writer.drain()
        except OSError as exc:
            # e.g. ConnectionResetError / BrokenPipeError after the bridge
            # dropped the session.  Re-raise as our own ConnectionError so
            # callers only have to handle one exception type.
            raise ConnectionError(f"Connection lost while sending: {exc}") from exc

    async def read_response(self) -> tuple[int, int, bytes]:
        """Read one complete response frame from the stream.

        The Fröling protocol frames start with a fixed 2-byte sync word
        (0x02, 0xFD) that is transmitted WITHOUT escaping.  Everything after
        the sync word is escaped on the wire and must be un-escaped as we read.

        Reading strategy:
          1. Search for the raw sync word, skipping (and logging) any bytes
             in front of it – e.g. the 0x00 a heater power cycle leaves on the
             line.  Gives up after ``_MAX_SYNC_SEARCH`` skipped bytes or two
             timeouts in total, so a slow trickle of noise cannot keep the
             caller (and the client lock) busy indefinitely.
          2. Un-escape-read 3 bytes → size_hi, size_lo, command_byte, and
             reject an implausible size before trying to read that much.
          3. Un-escape-read ``size`` bytes → payload bytes + 1 CRC byte.

        Returns
        -------
        tuple[int, int, bytes]
            ``(command, size, payload_with_crc)`` where:
            - ``command`` is the raw command byte value (int).
            - ``size`` is the value from the size field (= len(payload)+1).
            - ``payload_with_crc`` is the un-escaped payload bytes followed by
              the single CRC byte at the end.

        Raises
        ------
        ConnectionError
            If the reader is not available or the stream closes.
        TimeoutError
            If any individual byte read exceeds the configured timeout.
        ValueError
            If no sync word is found within the search budget, the size field
            is implausible, or the frame contains an invalid escape sequence.
            The stream is then misaligned and must be re-synchronised by the
            caller before it is used again.
        """
        if self._reader is None:
            raise ConnectionError("Not connected – call connect() first.")

        # --- Step 1: find the 2-byte sync word (raw, no un-escaping) ---
        # Slide one byte at a time so that e.g. "02 02 FD" is still found.
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 2 * self._timeout
        skipped = bytearray()
        previous: int = await self._read_one_byte()
        while True:
            current: int = await self._read_one_byte()
            if (previous << 8 | current) == COMM_ID:
                break
            skipped.append(previous)
            if len(skipped) >= _MAX_SYNC_SEARCH or loop.time() > deadline:
                raise ValueError(
                    f"No sync word 0x{COMM_ID:04X} found in {len(skipped)} bytes "
                    f"(starting {bytes(skipped[:16]).hex()})"
                )
            previous = current
        if skipped:
            _log.warning(
                "Skipped %d unexpected byte(s) before a response frame: %s",
                len(skipped), bytes(skipped[:32]).hex(),
            )

        # --- Step 2: un-escape-read 3 bytes: size(2) + command(1) ---
        header_body: bytearray = await self._read_and_unescape(3)
        size: int = int.from_bytes(header_body[0:2], "big")
        command: int = header_body[2]

        # Every frame carries at least the CRC byte.  A zero or huge size means
        # we locked onto noise; reading that many bytes would swallow the
        # following frames, so fail fast and let the caller re-synchronise.
        if not 1 <= size <= _MAX_RESPONSE_SIZE:
            raise ValueError(f"Implausible frame size {size} (command 0x{command:02X})")

        # --- Step 3: un-escape-read ``size`` bytes: payload + CRC ---
        payload_with_crc: bytearray = await self._read_and_unescape(size)

        _log.debug(
            "RX cmd=0x%02X size=%d payload+crc=%s",
            command, size, bytes(payload_with_crc).hex(),
        )
        return command, size, bytes(payload_with_crc)

    async def discard_input(self) -> bytes:
        """Throw away every byte that is already buffered, without waiting.

        The protocol is strictly request/response, so nothing is legitimately
        pending before a request is sent.  Anything already received at that
        point is stale (line noise from a heater restart, a reply that arrived
        after its timeout, data from another client of the bridge) and would
        otherwise be read as the response to the next request.

        Costs a single event-loop iteration when the buffer is empty.  Bytes
        that are still in the kernel's socket buffer at that moment are not
        seen; the sync-word search and the client's re-synchronisation cover
        those.

        Returns
        -------
        bytes
            The discarded bytes (empty when the stream was clean).

        Raises
        ------
        ConnectionError
            If not connected, or the peer has closed the connection.
        """
        if self._reader is None:
            raise ConnectionError("Not connected – call connect() first.")
        return self._take_pushback() + await self._read_buffered(self._reader)

    async def drain_input(self, quiet: float, max_duration: float) -> bytes:
        """Discard incoming bytes until the line has been quiet for ``quiet`` s.

        Used to re-synchronise after an error: unlike :meth:`discard_input`
        it also waits for bytes that are still on their way, such as the rest
        of a frame that was cut off by a timeout, or a late reply.  Mirrors
        linux-p4d's RequestClean / Serial::flush (read until a quiet gap).

        Parameters
        ----------
        quiet:
            Length of silence (seconds) that ends the drain.
        max_duration:
            Upper bound (seconds) on the whole drain, so a continuously
            chattering line cannot block the caller forever.

        Returns
        -------
        bytes
            The discarded bytes.

        Raises
        ------
        ConnectionError
            If not connected, or the peer has closed the connection.
        """
        # Keep a local reference: disconnect() (e.g. on unload) may clear
        # self._reader while we wait; the closed stream then ends the drain.
        reader = self._reader
        if reader is None:
            raise ConnectionError("Not connected – call connect() first.")

        loop = asyncio.get_running_loop()
        deadline = loop.time() + max_duration
        dropped = bytearray(self._take_pushback())

        while loop.time() < deadline:
            # If the loop ran late, data arriving just as the quiet timer
            # expires is handled in the same iteration (I/O first): the read
            # is then cancelled with the bytes already buffered.  Pick those
            # up too; if there were any, the line was not quiet.
            chunk = await self._read_chunk(reader, quiet) or await self._read_buffered(reader)
            if not chunk:
                break                           # quiet for ``quiet`` seconds
            dropped += chunk

        return bytes(dropped)

    def _take_pushback(self) -> bytes:
        """Remove and return the bytes handed back by an invalid escape."""
        data = bytes(self._pushback)
        self._pushback.clear()
        return data

    @classmethod
    async def _read_buffered(cls, reader: asyncio.StreamReader) -> bytes:
        """Return (and remove) whatever ``reader`` has buffered, without waiting.

        Raises
        ------
        ConnectionError
            If the peer has closed the connection or it was reset.
        """
        data = bytearray()
        while chunk := await cls._read_chunk(reader, 0):
            data += chunk
        return bytes(data)

    @staticmethod
    async def _read_chunk(reader: asyncio.StreamReader, timeout: float) -> bytes:
        """Read up to one chunk, waiting at most ``timeout`` s; ``b""`` if none arrived.

        ``asyncio.timeout`` is used instead of ``asyncio.wait_for`` on
        purpose: it runs ``read()`` inline in the current task, so buffered
        data is returned immediately even with a zero timeout, while
        ``wait_for(..., 0)`` would cancel the read before it ever ran.

        Raises
        ------
        ConnectionError
            If the peer has closed the connection or it was reset.
        """
        try:
            async with asyncio.timeout(timeout):
                chunk = await reader.read(_DISCARD_CHUNK)
        except asyncio.TimeoutError:
            return b""
        except OSError as exc:
            raise ConnectionError(f"Connection lost while discarding input: {exc}") from exc
        if not chunk:
            # read() returns b"" only at EOF: the peer closed the session.
            raise ConnectionError("Connection closed by the bridge.")
        return chunk

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _read_and_unescape(self, count: int) -> bytearray:
        """Read exactly ``count`` *unescaped* bytes from the TCP stream.

        Reads wire bytes one-at-a-time, expanding two-byte escape sequences
        into the original single byte.  Continues until ``count`` unescaped
        bytes have been accumulated.

        Escape rules (mirrors ESCAPE_TABLE in const.py):
          - 0x02 followed by 0x00  → 0x02
          - 0x2B followed by 0x00  → 0x2B
          - 0xFE followed by 0x00  → 0xFE
          - 0xFE followed by 0x12  → 0x11  (XON)
          - 0xFE followed by 0x14  → 0x13  (XOFF)

        Any other byte passes through unchanged (one wire byte = one output
        byte).  Like linux-p4d (P4Request::readByte) any other second byte is
        a framing error: in particular ``02 FD`` here means a new frame
        started inside this one (the previous frame was truncated).

        Parameters
        ----------
        count:
            Number of *unescaped* bytes to return.

        Returns
        -------
        bytearray
            Exactly ``count`` unescaped bytes.

        Raises
        ------
        TimeoutError
            If any individual byte read exceeds the timeout.
        ConnectionError
            If the stream closes prematurely.
        ValueError
            If 0x02 / 0x2B is not followed by 0x00, or 0xFE is followed by an
            unrecognised second byte.
        """
        # Bytes that indicate the start of a two-byte escape sequence.
        # 0x02 and 0x2B each escape themselves (followed by 0x00).
        # 0xFE is the escape prefix for XON, XOFF, and itself.
        SELF_ESCAPE_FIRST: frozenset[int] = frozenset({0x02, 0x2B})

        result = bytearray()
        while len(result) < count:
            # Read one wire byte.
            b: int = await self._read_one_byte()

            if b in SELF_ESCAPE_FIRST:
                # These bytes escape themselves: the next byte must be 0x00.
                # The decoded value is the first byte itself.
                second: int = await self._read_one_byte()
                if second != 0x00:
                    # Not a valid escape – the frame is corrupt or truncated.
                    # Guessing here would silently shift every following byte,
                    # so report it and let the caller re-synchronise.
                    self._push_back_invalid_escape(b, second)
                    raise ValueError(
                        f"Invalid escape sequence: 0x{b:02X} 0x{second:02X}"
                    )
                # Valid self-escape: decoded value = first byte.
                result.append(b)

            elif b == 0xFE:
                # 0xFE is the generic escape prefix; the second byte determines
                # which original byte is being represented.
                second = await self._read_one_byte()
                if second == 0x00:
                    result.append(0xFE)       # 0xFE 0x00 → 0xFE
                elif second == 0x12:
                    result.append(0x11)       # 0xFE 0x12 → 0x11 (XON)
                elif second == 0x14:
                    result.append(0x13)       # 0xFE 0x14 → 0x13 (XOFF)
                else:
                    self._push_back_invalid_escape(b, second)
                    raise ValueError(
                        f"Unknown escape sequence: 0xFE 0x{second:02X}"
                    )

            else:
                # Regular byte: one wire byte → one output byte.
                result.append(b)

        return result

    def _push_back_invalid_escape(self, first: int, second: int) -> None:
        """Hand an invalid escape's bytes back so a frame start in them survives.

        Inside a frame 0x02 is always escaped as ``02 00``, so an invalid pair
        starting with 0x02 (typically ``02 FD``) is where the next frame
        begins: keep both bytes.  After 0x2B / 0xFE only the second byte can
        begin a frame.
        """
        self._pushback[:0] = bytes((first, second) if first == 0x02 else (second,))

    async def _read_one_byte(self) -> int:
        """Read a single raw byte from the stream.

        Serves bytes handed back by an invalid escape first, then wraps
        asyncio.StreamReader.readexactly(1) with timeout handling.

        Returns
        -------
        int
            The byte value in range [0, 255].

        Raises
        ------
        TimeoutError
            If the byte does not arrive within the configured timeout.
        ConnectionError
            If the stream closes or is reset before delivering the byte.
        """
        if self._pushback:
            return self._pushback.pop(0)
        if self._reader is None:
            raise ConnectionError("Not connected – call connect() first.")
        try:
            data = await asyncio.wait_for(
                self._reader.readexactly(1),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                "Timeout waiting for byte from Fröling bridge."
            ) from exc
        except asyncio.IncompleteReadError as exc:
            raise ConnectionError(
                "Connection closed while waiting for byte."
            ) from exc
        except OSError as exc:
            # e.g. ConnectionResetError stored on the reader by the transport.
            raise ConnectionError(
                f"Connection lost while waiting for byte: {exc}"
            ) from exc
        return data[0]
