"""Client-level regression tests: recovering from a desynchronised byte stream.

Background (the "unavailable after heater restart" bug)
------------------------------------------------------
When the heater is power-cycled while the TCP-to-serial bridge stays up, the
UART glitch reaches the still-open TCP stream as stray byte(s), typically a
single 0x00.  The next response then reads as ``00 02 | FD ...`` and the client
used to fail with "Bad sync word: expected 0x02FD, got 0x0002".  Because the
client never re-synchronised the stream (no sync-word search, no discarding of
stale input, no reset after a protocol error), every later poll failed too and
only an integration reload (fresh socket) recovered.

These tests drive the REAL :class:`FroelingClient` over a REAL localhost TCP
socket against :class:`tests.fake_heater.FakeHeater` and assert that every
fault costs at most the request it hits, and that the following poll returns
correct, un-shifted values without a reload.
"""

from __future__ import annotations

import asyncio

import pytest
import pytest_asyncio

from custom_components.froeling.pyfroeling import (
    FroelingClient,
    FroelingConnectionError,
    FroelingError,
    FroelingProtocolError,
    ValueSpec,
)
from custom_components.froeling.pyfroeling.commands import build_check_request
from custom_components.froeling.pyfroeling.connection import ConnectionError as WireConnectionError
from custom_components.froeling.pyfroeling.const import Command
from custom_components.froeling.pyfroeling.protocol import build_frame

from .fake_heater import FakeHeater, expected_value

# The CHECK echo frame; its payload read as a GET_VALUE is 0x5465 ("Te").
CHECK_ECHO = build_frame(*build_check_request())

# Short timeouts keep the suite fast; the production defaults are larger.
TIMEOUT = 0.3
RESYNC_QUIET = 0.05

# 0x0011 forces escaping in the request payload (0x11 -> FE 12).
ADDRESSES = [0x0001, 0x0002, 0x0003, 0x0011, 0x0005]
SPECS = [
    ValueSpec(address=a, factor=1, unit="°C", title=f"Sensor {a:02X}", menu_type=3)
    for a in ADDRESSES
]


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def client(heater):
    """A connected client talking to the fake heater."""
    c = FroelingClient(
        host="127.0.0.1", port=heater.port, timeout=TIMEOUT, resync_quiet=RESYNC_QUIET,
    )
    await c.connect()
    yield c
    await c.disconnect()


async def poll(client: FroelingClient):
    """Mirror FroelingCoordinator._async_update_data (read-only mode)."""
    await client.ensure_connected()
    status = await client.get_status()
    values = await client.get_all_values(SPECS)
    errors = await client.get_errors()
    return status, values, errors


def raw_values(values) -> dict[int, int]:
    """Reduce a get_all_values() result to {address: raw_value}."""
    return {address: sv.raw_value for address, sv in values.items()}


def expected(heater: FakeHeater, skip: tuple[int, ...] = ()) -> dict[int, int]:
    """The values a correct poll must return for the heater's current poll."""
    return {a: expected_value(a, heater.poll_no) for a in ADDRESSES if a not in skip}


async def cancel_during_resync(
    client: FroelingClient, heater: FakeHeater, echo_delay: float
) -> None:
    """Leave a resync CHECK unanswered by cancelling its caller.

    Dirties the stream with a corrupted reply, delays the next CHECK echo by
    ``echo_delay`` and cancels the request whose resync sent that CHECK, once
    the drain is done and the CHECK is on the wire.  Replies are slower than
    the resync quiet window throughout.
    """
    heater.latency = RESYNC_QUIET + 0.03
    await client.get_status()
    heater.on_next(Command.GET_VALUE, address=0x0002, insert_at=6)   # dirty the stream
    with pytest.raises(FroelingProtocolError):
        await client.get_value(0x0002, SPECS[1])

    heater.on_next(Command.CHECK, delay=echo_delay)                   # slow resync echo
    task = asyncio.create_task(client.get_value(0x0003, SPECS[2]))
    await asyncio.sleep(RESYNC_QUIET + 0.05)                          # drain done, CHECK sent
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def assert_good_poll(client: FroelingClient, heater: FakeHeater) -> None:
    """Run one poll and assert that every value is correct and current."""
    status, values, errors = await poll(client)
    assert status.state == 3
    assert status.mode == 1
    assert status.datetime.second == heater.poll_no % 60
    assert raw_values(values) == expected(heater)
    assert errors == []


# ---------------------------------------------------------------------------
# The reported bug: stray bytes from a heater power cycle
# ---------------------------------------------------------------------------

class TestStrayBytes:
    """Stray bytes on the open stream must not break the following poll."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("state_text", [b"Automatik;Heizen", b"Automatik;Heizen!"])
    @pytest.mark.parametrize("garbage", [b"\x00", b"\x00\x00", b"\x00\xff\x00", b"\x02", b"\xfe"])
    async def test_garbage_between_polls(self, heater, client, state_text, garbage) -> None:
        """Garbage arriving while idle (the user's case) is discarded.

        Both state texts are tested: an even- and an odd-length GET_STATE
        frame used to fail in different ways (endless bad sync vs. permanent
        one-frame shift with cross-wired values).
        """
        heater.state_text = state_text
        await assert_good_poll(client, heater)

        await heater.inject(garbage)

        await assert_good_poll(client, heater)
        await assert_good_poll(client, heater)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("garbage", [b"\x00", b"\x00\x00", b"\x02", b"\xfe\x13", b"\x02\x02"])
    async def test_garbage_prefixing_a_response(self, heater, client, garbage) -> None:
        """Garbage arriving right before a response is skipped by the sync search."""
        await assert_good_poll(client, heater)

        heater.on_next(Command.GET_STATE, prefix=garbage)

        await assert_good_poll(client, heater)
        await assert_good_poll(client, heater)

    @pytest.mark.asyncio
    async def test_stale_frame_between_polls(self, heater, client) -> None:
        """A complete, CRC-valid frame nobody asked for is discarded, not used."""
        await assert_good_poll(client, heater)

        await heater.inject(build_frame(Command.GET_VALUE, b"\x7f\xff"))

        await assert_good_poll(client, heater)

    @pytest.mark.asyncio
    async def test_check_echo_is_never_a_value(self, heater, client) -> None:
        """A stray CHECK echo in front of a reply is rejected, not read as 0x5465."""
        await assert_good_poll(client, heater)

        heater.on_next(Command.GET_VALUE, address=0x0002, prefix=CHECK_ECHO)
        _, values, _ = await poll(client)
        assert raw_values(values) == expected(heater, skip=(0x0002,))

        await assert_good_poll(client, heater)

    @pytest.mark.asyncio
    async def test_exact_user_scenario_never_reports_bad_sync(self, heater, client) -> None:
        """One 0x00 before GET_STATE used to raise 'got 0x0002' forever."""
        await assert_good_poll(client, heater)
        await heater.inject(b"\x00")
        for _ in range(5):
            await assert_good_poll(client, heater)


# ---------------------------------------------------------------------------
# Heater off / late responses / partial frames
# ---------------------------------------------------------------------------

class TestTimeouts:
    """A timed-out request must not shift the pairing of later requests."""

    @pytest.mark.asyncio
    async def test_short_outage_with_glitches(self, heater, client) -> None:
        """Power loss (0x00, silence) and boot glitch on the same TCP session."""
        await assert_good_poll(client, heater)

        await heater.inject(b"\x00")          # TX line drops at power loss
        heater.silent = True
        with pytest.raises(FroelingConnectionError):
            await poll(client)

        heater.silent = False
        await heater.inject(b"\x00")          # boot glitch
        await assert_good_poll(client, heater)
        await assert_good_poll(client, heater)

    @pytest.mark.asyncio
    async def test_long_outage_reconnects(self, heater, client) -> None:
        """Several failed polls (connection reset in between) recover once it is back."""
        await assert_good_poll(client, heater)

        await heater.inject(b"\x00")
        heater.silent = True
        for _ in range(3):
            with pytest.raises(FroelingConnectionError):
                await poll(client)

        heater.silent = False
        await assert_good_poll(client, heater)
        await assert_good_poll(client, heater)

    @pytest.mark.asyncio
    async def test_late_response_does_not_shift_values(self, heater, client) -> None:
        """A reply that arrives after the timeout must not be used for the next sensor."""
        await assert_good_poll(client, heater)

        heater.on_next(Command.GET_VALUE, address=0x0002, delay=TIMEOUT + 0.2)
        _, values, _ = await poll(client)
        assert raw_values(values) == expected(heater, skip=(0x0002,))

        await assert_good_poll(client, heater)

    @pytest.mark.asyncio
    async def test_late_response_with_slow_heater(self, heater, client) -> None:
        """Same as above, but every reply takes longer than the resync quiet window.

        The resync must not take an older CHECK echo for its own: with a
        retry that sent a second identical CHECK, the first CHECK's echo
        "proved" alignment while the second was still in flight, and the rest
        of the poll got the previous request's reply (e.g. 0x5465 = "Te").
        """
        heater.latency = RESYNC_QUIET + 0.03
        await assert_good_poll(client, heater)

        heater.on_next(Command.GET_VALUE, address=0x0002, delay=TIMEOUT + 0.2)
        _, values, _ = await poll(client)
        assert raw_values(values) == expected(heater, skip=(0x0002,))

        await assert_good_poll(client, heater)

    @pytest.mark.asyncio
    async def test_late_reply_after_resync_drain(self, heater, client) -> None:
        """A late reply that lands after the drain is skipped while waiting for the echo."""
        await assert_good_poll(client, heater)

        heater.on_next(Command.GET_VALUE, address=0x0002,
                       delay=TIMEOUT + RESYNC_QUIET + 0.1)
        heater.on_next(Command.CHECK, delay=0.02)
        _, values, _ = await poll(client)
        assert raw_values(values) == expected(heater, skip=(0x0002,))

        await assert_good_poll(client, heater)

    @pytest.mark.asyncio
    async def test_frame_split_across_timeout(self, heater, client) -> None:
        """A partial frame left behind by a timeout is discarded, not misread."""
        await assert_good_poll(client, heater)

        heater.on_next(Command.GET_VALUE, address=0x0002, split_at=4,
                       split_delay=TIMEOUT + 0.2)
        _, values, _ = await poll(client)
        assert raw_values(values) == expected(heater, skip=(0x0002,))

        await assert_good_poll(client, heater)

    @pytest.mark.asyncio
    async def test_sync_byte_split_across_timeout(self, heater, client) -> None:
        """Only the first sync byte arrives before the timeout."""
        await assert_good_poll(client, heater)

        heater.on_next(Command.GET_STATE, split_at=1, split_delay=TIMEOUT + 0.2)
        with pytest.raises(FroelingConnectionError):
            await poll(client)

        await assert_good_poll(client, heater)

    @pytest.mark.asyncio
    async def test_poll_aborts_quickly_when_heater_dies_mid_poll(self, heater, client) -> None:
        """A dead heater fails the poll instead of timing out every sensor."""
        await assert_good_poll(client, heater)

        heater.on_next(Command.GET_VALUE, address=0x0002, go_silent=True)
        loop = asyncio.get_running_loop()
        start = loop.time()
        with pytest.raises(FroelingConnectionError):
            await poll(client)
        # Old behaviour: 4 sensors + the error log each waited a full timeout.
        assert loop.time() - start < 3.5 * TIMEOUT


# ---------------------------------------------------------------------------
# Corrupted frames and unparseable payloads
# ---------------------------------------------------------------------------

class TestCorruptFrames:
    """A corrupted frame costs exactly that request."""

    @pytest.mark.asyncio
    async def test_glitch_inside_a_value_frame(self, heater, client) -> None:
        """A byte inserted mid-frame fails that sensor; the others stay correct."""
        await assert_good_poll(client, heater)

        heater.on_next(Command.GET_VALUE, address=0x0002, insert_at=6)
        _, values, _ = await poll(client)
        assert raw_values(values) == expected(heater, skip=(0x0002,))

        await assert_good_poll(client, heater)

    @pytest.mark.asyncio
    async def test_glitch_inside_state_frame(self, heater, client) -> None:
        """A byte inserted in the GET_STATE text fails one poll only."""
        await assert_good_poll(client, heater)

        heater.on_next(Command.GET_STATE, insert_at=10)
        with pytest.raises(FroelingProtocolError):
            await poll(client)

        await assert_good_poll(client, heater)

    @pytest.mark.asyncio
    async def test_unparseable_payload_is_wrapped(self, heater, client) -> None:
        """A CRC-valid but nonsensical payload raises FroelingProtocolError, not ValueError."""
        await assert_good_poll(client, heater)

        # Month 0 makes datetime() raise ValueError inside the version parser.
        heater.on_next(Command.GET_VERSION, payload=bytes([0x50, 0x02, 0x13, 0x08,
                                                           0, 0, 0, 0, 0, 0, 0]))
        with pytest.raises(FroelingProtocolError):
            await poll(client)

        await assert_good_poll(client, heater)

    @pytest.mark.asyncio
    async def test_cut_off_frame_before_check_echo(self, heater, client) -> None:
        """A cut-off stale frame right before the CHECK echo must not hide the echo."""
        await assert_good_poll(client, heater)

        heater.on_next(Command.CHECK, prefix=b"\x02\xFD\x00\x05\x30\xAA")
        assert await client.check_connection() is True
        await assert_good_poll(client, heater)

    @pytest.mark.asyncio
    async def test_resync_does_not_rely_on_check_reply_command(self, heater, client) -> None:
        """The CHECK echo is recognised by its text, not by the reply's command byte.

        The controller does not always answer with the request's command code
        (e.g. GET_VALUE_LIST_NEXT is answered with 0x31), and linux-p4d never
        checks it, so nothing guarantees CHECK replies carry 0x22.
        """
        heater.reply_cmd[Command.CHECK] = 0x23
        await assert_good_poll(client, heater)

        heater.on_next(Command.GET_VALUE, address=0x0002, insert_at=6)
        _, values, _ = await poll(client)
        assert raw_values(values) == expected(heater, skip=(0x0002,))
        await assert_good_poll(client, heater)

    @pytest.mark.asyncio
    async def test_short_value_payload_is_wrapped(self, heater, client) -> None:
        """A 1-byte GET_VALUE payload (struct.error) only fails that sensor."""
        await assert_good_poll(client, heater)

        heater.on_next(Command.GET_VALUE, address=0x0003, payload=b"\x01")
        _, values, _ = await poll(client)
        assert raw_values(values) == expected(heater, skip=(0x0003,))


# ---------------------------------------------------------------------------
# TCP session loss
# ---------------------------------------------------------------------------

class TestSessionLoss:
    """Lost TCP sessions are detected and reconnected without raw exceptions."""

    @pytest.mark.asyncio
    async def test_fin_between_polls_reconnects(self, heater, client) -> None:
        """A bridge closing the socket (FIN) while idle is detected before the next poll."""
        await assert_good_poll(client, heater)

        await heater.close_fin()
        assert client.is_connected is False

        await assert_good_poll(client, heater)

    @pytest.mark.asyncio
    async def test_fin_behind_stray_byte_reconnects(self, heater, client) -> None:
        """A FIN that arrives behind unread stray bytes is still noticed before the poll."""
        await assert_good_poll(client, heater)

        await heater.inject(b"\x00")
        await heater.close_fin()

        await assert_good_poll(client, heater)
        assert heater.connections == 2

    @pytest.mark.asyncio
    async def test_rst_between_polls_reconnects(self, heater, client) -> None:
        """A bridge reset (RST) while idle is detected before the next poll."""
        await assert_good_poll(client, heater)

        await heater.close_rst()
        assert client.is_connected is False

        await assert_good_poll(client, heater)

    @pytest.mark.asyncio
    async def test_ensure_connected_reconnects_only_when_needed(self, heater, client) -> None:
        """ensure_connected() is a no-op on a live session and reconnects a dead one."""
        await assert_good_poll(client, heater)
        await client.ensure_connected()
        assert heater.connections == 1

        await heater.close_fin()
        await asyncio.gather(client.ensure_connected(), client.ensure_connected())
        assert heater.connections == 2
        await assert_good_poll(client, heater)

    @pytest.mark.asyncio
    async def test_reconnect_while_check_echo_in_flight(self, heater, client) -> None:
        """An unanswered CHECK's echo can arrive on the NEXT TCP session.

        The bridge's serial line outlives the TCP session, so the first
        resync after reconnecting must not take that echo for its own.
        """
        heater.shared_line = True
        heater.latency = RESYNC_QUIET + 0.05
        await assert_good_poll(client, heater)

        # Dirty the stream, then let the bridge drop the session right after
        # it forwarded the resync's CHECK; the heater still answers it.
        heater.on_next(Command.GET_VALUE, address=0x0002, insert_at=6)
        heater.on_next(Command.CHECK, drop_session=True)
        with pytest.raises(FroelingError):
            await client.get_value(0x0002, SPECS[1])
        with pytest.raises(FroelingConnectionError):
            await client.get_value(0x0003, SPECS[2])

        await client.ensure_connected()                  # e.g. a write reconnecting
        for spec in (SPECS[0], SPECS[4], SPECS[2], SPECS[3]):
            value = await client.get_value(spec.address, spec)
            assert value.raw_value == expected_value(spec.address, heater.poll_no)

    @pytest.mark.asyncio
    async def test_disconnect_wins_over_concurrent_reconnect(self, heater, client) -> None:
        """A reconnect racing disconnect() (e.g. a write during unload) must not reopen."""
        await assert_good_poll(client, heater)

        await asyncio.gather(client.disconnect(), client.ensure_connected(),
                             return_exceptions=True)
        await asyncio.sleep(0.05)

        assert client.is_connected is False
        assert heater.open_sessions == 0
        with pytest.raises(FroelingConnectionError):
            await client.ensure_connected()
        await client.connect()                        # an explicit connect reopens
        await assert_good_poll(client, heater)

    @pytest.mark.asyncio
    async def test_disconnect_during_reconnect_closes_new_session(self, heater, client) -> None:
        """disconnect() while a reconnect is still opening must not leave it open."""
        await assert_good_poll(client, heater)
        await heater.close_fin()                      # next ensure_connected reconnects

        open_session = client._conn.connect

        async def slow_open(host: str, port: int) -> None:
            await asyncio.sleep(0.05)
            await open_session(host, port)

        client._conn.connect = slow_open
        reconnect = asyncio.create_task(client.ensure_connected())
        await asyncio.sleep(0.01)                     # reconnect is opening now
        await client.disconnect()

        with pytest.raises(FroelingConnectionError):
            await reconnect
        await asyncio.sleep(0.05)
        assert client.is_connected is False
        assert heater.open_sessions == 0

    @pytest.mark.asyncio
    async def test_failed_check_send_leaves_no_pending_check(self, heater, client) -> None:
        """A resync CHECK that never went out must not delay the next resync."""
        await assert_good_poll(client, heater)

        write_frame = client._conn.write_frame

        async def fail_once(frame: bytes) -> None:
            client._conn.write_frame = write_frame
            raise WireConnectionError("Connection lost while sending: reset")

        client._needs_resync = True
        client._conn.write_frame = fail_once
        with pytest.raises(FroelingConnectionError):
            await client.get_value(0x0001, SPECS[0])

        await client.ensure_connected()
        loop = asyncio.get_running_loop()
        start = loop.time()
        value = await client.get_value(0x0001, SPECS[0])
        assert value.raw_value == expected_value(0x0001, heater.poll_no)
        assert loop.time() - start < TIMEOUT          # no 2 x timeout drain

    @pytest.mark.asyncio
    async def test_reconnect_discards_replayed_bridge_data(self, heater, client) -> None:
        """Data a bridge replays right after (re)connecting is not taken as a response."""
        heater.greeting = build_frame(Command.GET_VALUE, b"\x7f\xff")
        await client.disconnect()
        await client.connect()

        await assert_good_poll(client, heater)


# ---------------------------------------------------------------------------
# Concurrency: writes and cancellation
# ---------------------------------------------------------------------------

class TestConcurrency:
    """Concurrent callers must never receive each other's responses."""

    @pytest.mark.asyncio
    async def test_write_during_poll(self, heater, client) -> None:
        """set_parameter's second echo must not be read by a concurrent poll."""
        await assert_good_poll(client, heater)

        # Keep the poll busy inside get_all_values while the write is issued.
        # With the lock released between the echoes, the poll's next GET_VALUE
        # would read echo 2 (0x0100 = 256) as the value of sensor 0x0002.
        heater.on_next(Command.GET_VALUE, address=0x0001, delay=0.05)
        heater.requests.clear()
        poll_task = asyncio.create_task(poll(client))
        await heater.wait_for_request(Command.GET_VALUE)
        confirmed = await client.set_parameter(0x0100, 70.0, 10)
        status, values, errors = await poll_task

        assert confirmed == 70.0
        assert raw_values(values) == expected(heater)
        # The write really happened in the middle of the poll's sensor reads.
        set_index = heater.requests.index(Command.SET_PARAMETER)
        assert Command.GET_VALUE in heater.requests[:set_index]
        assert Command.GET_VALUE in heater.requests[set_index:]
        await assert_good_poll(client, heater)

    @pytest.mark.asyncio
    async def test_write_rounds_to_nearest_raw_value(self, heater, client) -> None:
        """0.29 at factor 100 is 28.999999999999996 in floating point: write 29, not 28."""
        await client.set_parameter(0x0100, 0.29, 100)
        assert heater.params[0x0100] == 29

    @pytest.mark.asyncio
    async def test_write_never_lands_inside_a_paginated_read(self, heater, client) -> None:
        """A write waits until a FIRST/NEXT list walk is complete."""
        heater.latency = 0.01
        heater.requests.clear()
        discovery = asyncio.create_task(client.discover_sensors())
        await heater.wait_for_request(Command.GET_VALUE_LIST_FIRST)
        await client.set_parameter(0x0100, 70.0, 10)
        assert [spec.address for spec in await discovery] == heater.value_list

        last_page = max(i for i, c in enumerate(heater.requests)
                        if c == Command.GET_VALUE_LIST_NEXT)
        assert heater.requests.index(Command.SET_PARAMETER) > last_page

    @pytest.mark.asyncio
    async def test_write_with_late_second_echo(self, heater, client) -> None:
        """Echo 2 arriving after its timeout must not shift the confirmation read."""
        await assert_good_poll(client, heater)

        heater.echo2_delay = TIMEOUT + 0.1
        confirmed = await client.set_parameter(0x0100, 70.0, 10)

        assert confirmed == 70.0
        await assert_good_poll(client, heater)

    @pytest.mark.asyncio
    async def test_check_connection_skips_stale_frame_before_echo(self, heater, client) -> None:
        """A stale frame ahead of the CHECK echo neither fails the check nor shifts replies.

        The stale frame arrives first and the real echo only 0.15 s later;
        check_connection (a forced resync) must skip the former, wait for the
        latter, and leave the stream aligned for the following polls.
        """
        heater.latency = 0.02
        await assert_good_poll(client, heater)

        stale = build_frame(Command.GET_VALUE, b"\x7f\xff")
        heater.on_next(Command.CHECK, prefix=stale, split_at=len(stale), split_delay=0.15)
        assert await client.check_connection() is True

        await assert_good_poll(client, heater)
        await assert_good_poll(client, heater)

    @pytest.mark.asyncio
    async def test_get_status_is_not_interleaved(self, heater, client) -> None:
        """A write queued during GET_STATE runs after GET_VERSION, not between."""
        heater.latency = 0.02
        await assert_good_poll(client, heater)
        heater.requests.clear()

        status_task = asyncio.create_task(client.get_status())
        await heater.wait_for_request(Command.GET_STATE)
        write_task = asyncio.create_task(client.set_parameter(0x0100, 70.0, 10))
        await status_task
        assert await write_task == 70.0

        assert heater.requests[:3] == [Command.GET_STATE, Command.GET_VERSION,
                                       Command.SET_PARAMETER]

    @pytest.mark.asyncio
    async def test_cancel_during_resync_keeps_pairing(self, heater, client) -> None:
        """Cancelling a request while its resync waits for the CHECK echo.

        The echo is then still in flight; the next resync must not take it
        for its own CHECK's echo.
        """
        await cancel_during_resync(client, heater, echo_delay=0.2)

        for spec in (SPECS[4], SPECS[0], SPECS[2], SPECS[1]):             # back-to-back reads
            value = await client.get_value(spec.address, spec)
            assert value.raw_value == expected_value(spec.address, heater.poll_no)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("echo_delay", [1.5 * TIMEOUT, 3 * TIMEOUT])
    async def test_very_late_check_echo_never_becomes_a_value(
        self, heater, client, echo_delay
    ) -> None:
        """A cancelled resync's CHECK echo arriving (very) late never yields wrong data.

        At 1.5 x timeout the echo still falls inside the unanswered-CHECK
        guard (two timeouts from sending) and is drained.  At 3 x timeout the
        heater looks unresponsive and the connection is reset.  Requests may
        fail, but every value that IS returned is correct and reads recover
        (reconnecting first, as the coordinator does).  A stray echo that
        slips past both is rejected (see test_check_echo_is_never_a_value).
        """
        await cancel_during_resync(client, heater, echo_delay=echo_delay)

        succeeded = []
        for spec in SPECS + SPECS:
            try:
                await client.ensure_connected()
                value = await client.get_value(spec.address, spec)
            except FroelingError:
                continue
            assert value.raw_value == expected_value(spec.address, heater.poll_no)
            succeeded.append(spec.address)
        assert succeeded[-len(SPECS):] == ADDRESSES

    @pytest.mark.asyncio
    async def test_cancelled_request_does_not_leak_its_reply(self, heater, client) -> None:
        """A request cancelled after sending leaves no stale reply for the next one."""
        await assert_good_poll(client, heater)

        heater.on_next(Command.GET_VALUE, address=0x0001, delay=0.1)
        task = asyncio.create_task(client.get_value(0x0001, SPECS[0]))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.1)               # the late reply lands in the buffer

        value = await client.get_value(0x0003, SPECS[2])
        assert value.raw_value == expected_value(0x0003, heater.poll_no)

    @pytest.mark.asyncio
    async def test_cancelled_request_reply_arrives_after_next_request(self, heater, client) -> None:
        """Like above, but the next request is sent before the late reply arrives."""
        await assert_good_poll(client, heater)

        heater.on_next(Command.GET_VALUE, address=0x0001, delay=0.1)
        task = asyncio.create_task(client.get_value(0x0001, SPECS[0]))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        value = await client.get_value(0x0003, SPECS[2])
        assert value.raw_value == expected_value(0x0003, heater.poll_no)
