"""Fake TCP-to-serial bridge + Lambdatronic controller for client-level tests.

The fake listens on an ephemeral localhost port and speaks the real wire
protocol: it parses escaped request frames with its own (independent) decoder
and answers with escaped, CRC-valid frames built by
:func:`pyfroeling.protocol.build_frame`.  Like the real controller it handles
requests strictly one at a time, in arrival order (FIFO).

Fault hooks let tests reproduce what a heater power cycle or a slow controller
does to the byte stream:

  - :meth:`FakeHeater.inject`   – write raw garbage into the open stream now.
  - :attr:`FakeHeater.silent`   – heater "powered off": requests are dropped.
  - :meth:`FakeHeater.on_next`  – one-shot fault for the next matching request
    (delay, prefix garbage, insert a byte mid-frame, split the frame, send a
    bogus but CRC-valid payload, or die and go silent).
  - :meth:`FakeHeater.close_fin` / :meth:`FakeHeater.close_rst` – drop the TCP
    session gracefully (FIN) or hard (RST), as a rebooting bridge would.

Timing knobs: :attr:`FakeHeater.latency` delays every response frame (a real
controller + bridge take tens to hundreds of milliseconds), and
:attr:`FakeHeater.echo2_delay` is the gap between the two frames the
controller sends after SET_PARAMETER.

Sensor values are ``address * 1000 + poll_no`` so that a value shifted from
another sensor or from an older poll is immediately detectable.  ``poll_no``
increments on every GET_STATE request (the first request of every poll).
"""

from __future__ import annotations

import asyncio
import socket
import struct
from dataclasses import dataclass
from typing import Callable, Optional

from custom_components.froeling.pyfroeling.const import (
    COMM_ID,
    UNESCAPE_FIRST_BYTES,
    UNESCAPE_TABLE,
    Command,
)
from custom_components.froeling.pyfroeling.protocol import build_frame


def expected_value(address: int, poll_no: int) -> int:
    """Return the raw value the fake reports for ``address`` during ``poll_no``."""
    return address * 1000 + poll_no


@dataclass
class _Fault:
    """A one-shot fault applied to the response of the next matching request."""

    match: Callable[[int, bytes], bool]   # (command, request payload) -> bool
    delay: float = 0.0                    # seconds to stall before answering
    prefix: bytes = b""                   # raw garbage written before the frame
    insert_at: Optional[int] = None       # wire offset at which to insert 0x00
    split_at: Optional[int] = None        # send this many wire bytes, then stall
    split_delay: float = 0.0              # stall length between the two halves
    payload: Optional[bytes] = None       # replace the response payload
    go_silent: bool = False               # heater dies: drop this and all later requests
    drop_session: bool = False            # bridge drops the TCP session right after
                                          # forwarding this request (heater still answers)


class FakeHeater:
    """asyncio TCP server that behaves like a serial bridge + heater."""

    def __init__(self, state_text: bytes = b"Automatik;Heizen") -> None:
        self.state_text = state_text
        self.poll_no = 0
        self.silent = False
        # Seconds before each response frame is sent (processing + bridge latency).
        self.latency: float = 0.0
        # Seconds between SET_PARAMETER's first and second echo frame.
        self.echo2_delay: float = 0.03
        # Number of TCP sessions accepted so far / currently open.
        self.connections: int = 0
        self.open_sessions: int = 0
        # True: like a real bridge, the serial line outlives TCP sessions, so
        # a reply still in flight is delivered to whichever client is
        # connected when it is sent.  False: replies only go to the session
        # the request came from.
        self.shared_line: bool = False
        # Command byte used in replies, per request command (default: same
        # code).  The real controller does not always echo the request code.
        self.reply_cmd: dict[int, int] = {}
        # Reply payload used instead of the normal one, per request command
        # (e.g. a device that is not a Fröling heater).
        self.reply_payload: dict[int, bytes] = {}
        # Sensors listed by GET_VALUE_LIST_FIRST / NEXT, and the list cursor.
        self.value_list: list[int] = [0x0001, 0x0002, 0x0003]
        self._value_list_pos = 0
        # Writable parameters listed by GET_MENU_LIST_FIRST / NEXT (address ->
        # title, all of type 0x07 PAR), and that list's cursor.
        self.menu: dict[int, str] = {0x0100: "Kesselsolltemperatur", 0x0101: "Betriebsart"}
        self._menu_pos = 0
        # Writable parameters: address -> raw value.
        self.params: dict[int, int] = {0x0100: 650, 0x0101: 1}
        # Their GET_PARAMETER metadata: address -> (digits, factor, raw min,
        # raw max).  0x0100 is a 0.0-90.0 setpoint (a number entity in HA),
        # 0x0101 a 0-3 choice (a select entity).
        self.param_meta: dict[int, tuple[int, int, int, int]] = {
            0x0100: (1, 10, 0, 900),
            0x0101: (0, 1, 0, 3),
        }
        self.requests: list[int] = []        # command codes, in arrival order
        # Raw bytes sent as soon as a client connects (bridges that replay
        # serial data buffered while no TCP client was connected).
        self.greeting: bytes = b""
        self._faults: list[_Fault] = []
        self._writer: Optional[asyncio.StreamWriter] = None
        self._server: Optional[asyncio.base_events.Server] = None
        self.port = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start listening on an ephemeral localhost port."""
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        """Abort the active session and stop the server."""
        if self._writer is not None:
            self._writer.transport.abort()
        if self._server is not None:
            self._server.close()
            try:
                await asyncio.wait_for(self._server.wait_closed(), 1.0)
            except (asyncio.TimeoutError, TimeoutError):
                pass

    # ------------------------------------------------------------------
    # Fault hooks
    # ------------------------------------------------------------------

    async def inject(self, data: bytes) -> None:
        """Write raw bytes into the open TCP stream immediately."""
        assert self._writer is not None, "no client connected"
        self._writer.write(data)
        await self._writer.drain()
        # Give the client's event loop a moment to receive the bytes.
        await asyncio.sleep(0.02)

    def on_next(self, command: int, address: Optional[int] = None, **fault) -> None:
        """Register a one-shot fault for the next request with ``command``.

        ``address`` (optional) additionally matches the first two request
        payload bytes, e.g. the sensor address of a GET_VALUE.
        """
        def match(cmd: int, req: bytes) -> bool:
            if cmd != command:
                return False
            return address is None or req[:2] == struct.pack(">H", address)

        self._faults.append(_Fault(match=match, **fault))

    async def wait_for_request(self, command: int, timeout: float = 2.0) -> None:
        """Wait until a request with ``command`` has been received."""
        async with asyncio.timeout(timeout):
            while command not in self.requests:
                await asyncio.sleep(0.001)

    async def close_fin(self) -> None:
        """Close the TCP session gracefully (FIN), like a bridge idle timeout."""
        assert self._writer is not None
        self._writer.close()
        self._writer = None
        await asyncio.sleep(0.05)

    async def close_rst(self) -> None:
        """Abort the TCP session with a RST, like a rebooting bridge."""
        assert self._writer is not None
        sock = self._writer.get_extra_info("socket")
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        self._writer.transport.abort()
        self._writer = None
        await asyncio.sleep(0.05)

    # ------------------------------------------------------------------
    # Controller behaviour
    # ------------------------------------------------------------------

    def _responses(self, cmd: int, req: bytes) -> list[tuple[int, bytes]]:
        """Return the (command, payload) frames the controller sends for a request."""
        if cmd == Command.CHECK:
            return [(cmd, req)]
        if cmd == Command.GET_STATE:
            self.poll_no += 1
            return [(cmd, bytes([0x01, 0x03]) + self.state_text)]
        if cmd == Command.GET_VERSION:
            # Version 50.02.13.08; clock 03:09:<poll> 10.10.2026 (0x02 / 0x13 get escaped).
            return [(cmd, bytes([0x50, 0x02, 0x13, 0x08,
                                 self.poll_no % 60, 9, 3, 10, 10, 6, 26]))]
        if cmd == Command.GET_VALUE:
            (address,) = struct.unpack(">H", req[:2])
            return [(cmd, struct.pack(">h", expected_value(address, self.poll_no)))]
        if cmd in (Command.GET_ERROR_FIRST, Command.GET_ERROR_NEXT):
            return [(cmd, b"\x00")]                       # empty error log
        if cmd in (Command.GET_VALUE_LIST_FIRST, Command.GET_VALUE_LIST_NEXT):
            if cmd == Command.GET_VALUE_LIST_FIRST:
                self._value_list_pos = 0
            if self._value_list_pos >= len(self.value_list):
                return [(cmd, b"\x00")]                   # end of list
            address = self.value_list[self._value_list_pos]
            self._value_list_pos += 1
            # more=1, factor=1, menu_type=3 (measured value), unit "°C", address, title
            return [(cmd, b"\x01" + struct.pack(">HH", 1, 3) + "°C".encode("latin-1")
                     + struct.pack(">H", address) + f"Sensor {address:02X}".encode() + b"\x00")]
        if cmd in (Command.GET_MENU_LIST_FIRST, Command.GET_MENU_LIST_NEXT):
            if cmd == Command.GET_MENU_LIST_FIRST:
                self._menu_pos = 0
            if self._menu_pos >= len(self.menu):
                return [(cmd, b"\x00")]                   # end of list
            address, title = list(self.menu.items())[self._menu_pos]
            self._menu_pos += 1
            # more=1, type=0x07 (PAR), spare, parent, child, 18 spare bytes,
            # address, 2 spare bytes, title (see parse_menu_entry_response)
            return [(cmd, b"\x01\x07\x00" + struct.pack(">HH", 0, self._menu_pos)
                     + bytes(18) + struct.pack(">H", address) + b"\x00\x00"
                     + title.encode("latin-1") + b"\x00")]
        if cmd == Command.GET_PARAMETER:
            (address,) = struct.unpack(">H", req[:2])
            raw = self.params.get(address, 0)
            digits, factor, low, high = self.param_meta.get(address, (1, 10, 0, 900))
            payload = (b"\x00" + struct.pack(">H", address) + b"\xB0"
                       + bytes([digits, 0, factor]) + struct.pack(">hhhh", raw, low, high, low)
                       + b"\x00\x00\x00")
            return [(cmd, payload)]
        if cmd == Command.SET_PARAMETER:
            address, raw = struct.unpack(">Hh", req[:4])
            self.params[address] = raw
            # Two echo frames, both carrying address + new value (linux-p4d
            # reads echo 2 as address word + value word, p4io.c setParameter).
            return [(cmd, req[:4]), (cmd, req[:4])]
        return []

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Serve one client connection: read a request, answer it, repeat."""
        self._writer = writer
        self.connections += 1
        self.open_sessions += 1
        try:
            if self.greeting:
                writer.write(self.greeting)
                await writer.drain()
            while True:
                cmd, req = await self._read_request(reader)
                self.requests.append(cmd)
                if self.silent:
                    continue                              # heater off: request lost
                fault = next((f for f in self._faults if f.match(cmd, req)), None)
                if fault is not None:
                    self._faults.remove(fault)
                    if fault.go_silent:
                        self.silent = True
                        continue
                    if fault.drop_session:
                        writer.close()
                        if self._writer is writer:
                            self._writer = None
                for index, (resp_cmd, payload) in enumerate(self._responses(cmd, req)):
                    if fault is not None and fault.payload is not None:
                        payload = fault.payload
                    resp_cmd = self.reply_cmd.get(cmd, resp_cmd)
                    payload = self.reply_payload.get(cmd, payload)
                    if index > 0:
                        await asyncio.sleep(self.echo2_delay)
                    if self.latency:
                        await asyncio.sleep(self.latency)
                    await self._send(writer, build_frame(resp_cmd, payload), fault)
                    fault = None                          # fault applies to the first frame only
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            pass
        finally:
            self.open_sessions -= 1

    async def _send(self, writer: asyncio.StreamWriter, wire: bytes,
                    fault: Optional[_Fault]) -> None:
        """Write one response frame, applying ``fault`` if given."""
        if self.shared_line:
            # Deliver to whichever session is connected now (if any).
            if self._writer is None:
                return
            writer = self._writer
        if fault is not None:
            if fault.delay:
                await asyncio.sleep(fault.delay)
            if fault.insert_at is not None:
                wire = wire[:fault.insert_at] + b"\x00" + wire[fault.insert_at:]
            wire = fault.prefix + wire
            if fault.split_at is not None:
                writer.write(wire[:fault.split_at])
                await writer.drain()
                await asyncio.sleep(fault.split_delay)
                wire = wire[fault.split_at:]
        writer.write(wire)
        await writer.drain()

    @staticmethod
    async def _read_request(reader: asyncio.StreamReader) -> tuple[int, bytes]:
        """Hunt for a request frame and return its (command, payload)."""
        prev = (await reader.readexactly(1))[0]
        while True:
            cur = (await reader.readexactly(1))[0]
            if (prev << 8 | cur) == COMM_ID:
                break
            prev = cur

        async def unescaped(count: int) -> bytes:
            out = bytearray()
            while len(out) < count:
                b = (await reader.readexactly(1))[0]
                if b in UNESCAPE_FIRST_BYTES:
                    b2 = (await reader.readexactly(1))[0]
                    out.append(UNESCAPE_TABLE[bytes([b, b2])])
                else:
                    out.append(b)
            return bytes(out)

        header = await unescaped(3)
        size = int.from_bytes(header[:2], "big")
        body = await unescaped(size)
        return header[2], body[:-1]                       # strip CRC
