"""Home Assistant-level tests: config/options flows and the coordinator.

These run the real integration inside a real Home Assistant instance (via
pytest-homeassistant-custom-component) against :class:`FakeHeater` over
localhost TCP.  They are skipped when Home Assistant is not installed.

Run with ``python -m pytest tests/test_ha_integration.py`` (pytest.ini sets
the ``asyncio_mode = auto`` that pytest-homeassistant-custom-component needs).

What they pin down:
  - One connection per heater: while the entry is loaded, the options flow
    and reconfigure borrow the coordinator's live connection instead of
    opening a second one (which bridges allowing one client answer by
    kicking the live session, and bridges forwarding to all clients answer
    by crossing the two sessions' replies).
  - Adding an already configured heater aborts before any discovery traffic.
  - Discovery and polls never interleave on the wire.
  - The original bug end to end: a stray byte after a heater restart no
    longer leaves the integration unavailable until a reload.
"""

from __future__ import annotations

import asyncio
import functools

import pytest

pytest.importorskip("pytest_homeassistant_custom_component")

from homeassistant.config_entries import SOURCE_USER, ConfigEntryState  # noqa: E402
from homeassistant.core import HomeAssistant  # noqa: E402
from homeassistant.data_entry_flow import FlowResultType  # noqa: E402
from pytest_homeassistant_custom_component.common import MockConfigEntry  # noqa: E402

from custom_components.froeling.const import (  # noqa: E402
    CONF_CONNECTION_TYPE,
    CONF_DEVICE_NAME,
    CONF_HOST,
    CONF_PARAMETER_TITLES,
    CONF_PORT,
    CONF_SELECTED_PARAMETERS,
    CONF_SELECTED_SENSORS,
    CONF_SENSOR_SPECS,
    CONF_WRITE_ENABLED,
    CONN_TYPE_NETWORK,
    DOMAIN,
)
from custom_components.froeling.pyfroeling import FroelingClient  # noqa: E402
from custom_components.froeling.pyfroeling.const import Command  # noqa: E402

from .fake_heater import FakeHeater, expected_value  # noqa: E402

ADDRESSES = [0x0001, 0x0002, 0x0003]


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations):
    """Let Home Assistant load custom_components/froeling from this repo."""
    yield


@pytest.fixture(autouse=True)
def fast_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give the integration's clients short protocol timeouts.

    The production defaults (5 s timeout, 0.25 s resync drain) would make
    every failure path in these tests wait seconds.  All clients are built by
    coordinator.create_client, so patching the name there covers setup and
    the config/options flows.
    """
    monkeypatch.setattr(
        "custom_components.froeling.coordinator.FroelingClient",
        functools.partial(FroelingClient, timeout=0.5, resync_quiet=0.05),
    )
    # No real wait before retrying a fresh session's unanswered CHECK.
    monkeypatch.setattr("custom_components.froeling.config_flow._BRIDGE_SETTLE", 0)


def _entry_for(
    heater: FakeHeater, write: bool = False, cached_specs: bool = True
) -> MockConfigEntry:
    """A config entry for the fake heater, with cached sensor specs.

    With ``write=True`` it also enables write mode for the fake's two
    parameters (a numeric setpoint and a 0-3 choice).  With
    ``cached_specs=False`` it looks like an entry from before the spec cache,
    so setup and the options flow have to walk the sensor list.
    """
    params = {"0x0100": "Kesselsolltemperatur", "0x0101": "Betriebsart"} if write else {}
    return MockConfigEntry(
        domain=DOMAIN,
        unique_id=f"127.0.0.1:{heater.port}",
        title="Fröling",
        data={
            CONF_DEVICE_NAME: "Fröling",
            CONF_CONNECTION_TYPE: CONN_TYPE_NETWORK,
            CONF_HOST: "127.0.0.1",
            CONF_PORT: heater.port,
            CONF_SELECTED_SENSORS: [f"0x{a:04X}" for a in ADDRESSES],
            CONF_WRITE_ENABLED: write,
            CONF_SELECTED_PARAMETERS: list(params),
            CONF_PARAMETER_TITLES: params,
            CONF_SENSOR_SPECS: [
                {"address": a, "factor": 1, "unit": "°C",
                 "title": f"Sensor {a:02X}", "menu_type": 3}
                for a in ADDRESSES
            ] if cached_specs else [],
        },
    )


@pytest.fixture
async def setup_entry(hass: HomeAssistant, heater: FakeHeater):
    """Factory: create and set up a config entry, unloaded after the test.

    Call as ``await setup_entry(fake=None, **kwargs)``; ``fake`` defaults to
    the ``heater`` fixture and ``kwargs`` go to :func:`_entry_for`.  Entries
    are unloaded at teardown even if the test failed half-way.
    """
    entries: list[MockConfigEntry] = []

    async def _setup(fake: FakeHeater | None = None, **kwargs) -> MockConfigEntry:
        entry = _entry_for(fake or heater, **kwargs)
        entry.add_to_hass(hass)
        entries.append(entry)
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
        assert entry.state is ConfigEntryState.LOADED
        return entry

    yield _setup
    for entry in entries:
        if entry.state is ConfigEntryState.LOADED:
            await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()


@pytest.fixture
async def loaded_entry(setup_entry, heater: FakeHeater) -> MockConfigEntry:
    """The default config entry, set up (one live connection)."""
    entry = await setup_entry()
    assert heater.connections == 1
    return entry


def _schema_keys(result) -> set[str]:
    """Names of the fields in a flow result's form."""
    return {str(key) for key in result["data_schema"].schema}


def _assert_values_current(entry: MockConfigEntry, heater: FakeHeater) -> None:
    """The coordinator's data holds every sensor's value from the latest poll."""
    data = entry.runtime_data.data
    assert {a: sv.raw_value for a, sv in data.values.items()} == {
        a: expected_value(a, heater.poll_no) for a in ADDRESSES
    }


# ---------------------------------------------------------------------------
# The reported bug, end to end
# ---------------------------------------------------------------------------

async def test_stray_byte_after_heater_restart_recovers(
    hass: HomeAssistant, heater: FakeHeater, loaded_entry: MockConfigEntry
) -> None:
    """A stray 0x00 on the open stream no longer makes every later poll fail."""
    coordinator = loaded_entry.runtime_data
    _assert_values_current(loaded_entry, heater)

    await heater.inject(b"\x00")
    for _ in range(3):
        await coordinator.async_refresh()
        assert coordinator.last_update_success
        _assert_values_current(loaded_entry, heater)
    assert heater.connections == 1


# ---------------------------------------------------------------------------
# One connection per heater
# ---------------------------------------------------------------------------

async def test_options_flow_borrows_live_connection(
    hass: HomeAssistant, heater: FakeHeater, loaded_entry: MockConfigEntry
) -> None:
    """Options-flow discovery runs over the coordinator's connection."""
    result = await hass.config_entries.options.async_init(loaded_entry.entry_id)

    assert result["type"] is FlowResultType.FORM
    assert CONF_SELECTED_SENSORS in _schema_keys(result)   # discovery succeeded
    assert heater.connections == 1                     # ...without a 2nd session
    assert loaded_entry.runtime_data.client.is_connected


async def test_options_flow_uses_cached_specs_and_titles(
    hass: HomeAssistant, heater: FakeHeater, setup_entry
) -> None:
    """With cached specs and parameter titles, no list walk blocks the connection."""
    entry = await setup_entry(write=True)
    heater.requests.clear()

    result = await hass.config_entries.options.async_init(entry.entry_id)

    assert {CONF_SELECTED_SENSORS, CONF_SELECTED_PARAMETERS} <= _schema_keys(result)
    assert Command.GET_VALUE_LIST_FIRST not in heater.requests
    assert Command.GET_MENU_LIST_FIRST not in heater.requests


async def test_options_flow_discovery_is_not_interleaved_with_polls(
    hass: HomeAssistant, heater: FakeHeater, setup_entry
) -> None:
    """Without cached specs the list is walked; a poll waits until it is done."""
    entry = await setup_entry(cached_specs=False)
    heater.latency = 0.01
    heater.requests.clear()
    options_task = hass.async_create_task(
        hass.config_entries.options.async_init(entry.entry_id)
    )
    await heater.wait_for_request(Command.GET_VALUE_LIST_FIRST)
    poll_task = hass.async_create_task(entry.runtime_data.async_refresh())
    await options_task
    await poll_task

    first = heater.requests.index(Command.GET_VALUE_LIST_FIRST)
    state = heater.requests.index(Command.GET_STATE)
    last_list = max(i for i, c in enumerate(heater.requests)
                    if c == Command.GET_VALUE_LIST_NEXT)
    assert state > last_list > first
    assert entry.runtime_data.last_update_success


async def test_options_flow_without_loaded_entry_uses_own_connection(
    hass: HomeAssistant, heater: FakeHeater
) -> None:
    """With nothing connected, the options flow opens (and closes) its own session."""
    entry = _entry_for(heater)
    entry.add_to_hass(hass)

    result = await hass.config_entries.options.async_init(entry.entry_id)

    assert result["type"] is FlowResultType.FORM
    assert CONF_SELECTED_SENSORS in _schema_keys(result)
    assert heater.connections == 1


async def test_reconfigure_unchanged_settings_borrows_live_connection(
    hass: HomeAssistant, heater: FakeHeater, loaded_entry: MockConfigEntry
) -> None:
    """Validating unchanged settings does not open a second session."""
    result = await loaded_entry.start_reconfigure_flow(hass)
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_HOST: "127.0.0.1", CONF_PORT: heater.port}
    )
    connections_during_validation = heater.connections
    await hass.async_block_till_done()                 # the reload reconnects

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    assert connections_during_validation == 1


async def test_reconfigure_rejects_device_that_fails_check(
    hass: HomeAssistant, heater: FakeHeater, loaded_entry: MockConfigEntry
) -> None:
    """A device that accepts TCP but does not echo CHECK is not accepted."""
    other = FakeHeater()
    other.reply_payload[Command.CHECK] = b"not a heater"
    await other.start()
    try:
        result = await loaded_entry.start_reconfigure_flow(hass)
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {CONF_HOST: "127.0.0.1", CONF_PORT: other.port}
        )
    finally:
        await other.stop()

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": "cannot_connect"}
    assert loaded_entry.data[CONF_PORT] == heater.port


async def test_adding_configured_heater_aborts_before_discovery(
    hass: HomeAssistant, heater: FakeHeater, loaded_entry: MockConfigEntry
) -> None:
    """Re-adding a configured heater aborts without touching its bridge."""
    result = await hass.config_entries.flow.async_init(DOMAIN, context={"source": SOURCE_USER})
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {"next_step_id": "network"}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {CONF_DEVICE_NAME: "Again", CONF_HOST: "127.0.0.1", CONF_PORT: heater.port},
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "already_configured"
    assert heater.connections == 1


async def test_options_flow_reconnects_lost_session(
    hass: HomeAssistant, heater: FakeHeater, loaded_entry: MockConfigEntry
) -> None:
    """Borrowing the client after the bridge dropped the session reconnects it once."""
    await heater.close_fin()

    result = await hass.config_entries.options.async_init(loaded_entry.entry_id)

    assert CONF_SELECTED_SENSORS in _schema_keys(result)
    assert heater.connections == 2
    assert heater.open_sessions == 1


@pytest.mark.parametrize(
    "state", [ConfigEntryState.SETUP_IN_PROGRESS, ConfigEntryState.SETUP_RETRY]
)
async def test_options_flow_during_setup_does_not_connect(
    hass: HomeAssistant, heater: FakeHeater, state: ConfigEntryState
) -> None:
    """While setup owns (or is about to retry) the connection, do not open another."""
    entry = _entry_for(heater)
    entry.add_to_hass(hass)
    entry.mock_state(hass, state)

    result = await hass.config_entries.options.async_init(entry.entry_id)

    assert result["type"] is FlowResultType.FORM
    assert CONF_SELECTED_SENSORS not in _schema_keys(result)
    assert heater.connections == 0


async def test_failed_setup_closes_its_connection(
    hass: HomeAssistant, heater: FakeHeater
) -> None:
    """A heater that does not answer during setup must not leave a session open."""
    heater.silent = True
    entry = _entry_for(heater)
    entry.add_to_hass(hass)

    assert not await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    await asyncio.sleep(0.05)                          # let the fake see the close

    assert entry.state is ConfigEntryState.SETUP_RETRY
    assert heater.connections == 1
    assert heater.open_sessions == 0


# ---------------------------------------------------------------------------
# Write mode
# ---------------------------------------------------------------------------

async def test_number_and_select_write_to_heater(
    hass: HomeAssistant, heater: FakeHeater, setup_entry
) -> None:
    """Number and select entities show the heater's values and write new ones."""
    await setup_entry(write=True)

    (number,) = hass.states.async_all("number")
    (select,) = hass.states.async_all("select")
    assert float(number.state) == 65.0
    assert select.state == "1"

    await hass.services.async_call(
        "number", "set_value", {"entity_id": number.entity_id, "value": 70.5},
        blocking=True,
    )
    await hass.services.async_call(
        "select", "select_option", {"entity_id": select.entity_id, "option": "3"},
        blocking=True,
    )

    assert heater.params == {0x0100: 705, 0x0101: 3}
    assert float(hass.states.get(number.entity_id).state) == 70.5
    assert hass.states.get(select.entity_id).state == "3"


async def test_reconfigure_to_other_configured_heater_aborts(
    hass: HomeAssistant, heater: FakeHeater, loaded_entry: MockConfigEntry, setup_entry
) -> None:
    """Pointing an entry at a heater another entry owns aborts without connecting."""
    other = FakeHeater()
    await other.start()
    try:
        await setup_entry(other)
        result = await loaded_entry.start_reconfigure_flow(hass)
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {CONF_HOST: "127.0.0.1", CONF_PORT: other.port}
        )

        assert result["type"] is FlowResultType.ABORT
        assert result["reason"] == "already_configured"
        assert other.connections == 1                  # only its own entry's
        assert loaded_entry.data[CONF_PORT] == heater.port
    finally:
        await other.stop()


async def _advance(hass: HomeAssistant, result):
    """Drive a config flow through its progress (spinner) steps."""
    while result["type"] in (FlowResultType.SHOW_PROGRESS, FlowResultType.SHOW_PROGRESS_DONE):
        await hass.async_block_till_done()
        result = await hass.config_entries.flow.async_configure(result["flow_id"])
    return result


async def test_user_flow_with_write_mode_retries_unanswered_check(
    hass: HomeAssistant, heater: FakeHeater
) -> None:
    """Full setup with write mode; parameter discovery survives one missed CHECK.

    The bridge may still be recovering from the sensor-discovery session when
    parameter discovery connects.  An unanswered CHECK must lead to a retry,
    not to write mode being switched off silently.
    """
    result = await hass.config_entries.flow.async_init(DOMAIN, context={"source": SOURCE_USER})
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {"next_step_id": "network"}
    )
    result = await _advance(hass, await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {CONF_DEVICE_NAME: "Fröling", CONF_HOST: "127.0.0.1", CONF_PORT: heater.port},
    ))
    assert result["step_id"] == "sensors"
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_SELECTED_SENSORS: ["0x0001"]}
    )
    assert result["step_id"] == "access_mode"

    heater.on_next(Command.CHECK, delay=1.0)           # longer than the client timeout
    result = await _advance(hass, await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_WRITE_ENABLED: True}
    ))

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "parameters"
    # The fake's addresses are not in the known "basic" table, so they are
    # offered on the expert page.
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_SELECTED_PARAMETERS: []}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {"next_step_id": "show_expert"}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {"selected_expert_parameters": ["0x0100"]}
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"][CONF_WRITE_ENABLED] is True
    assert result["data"][CONF_SELECTED_PARAMETERS] == ["0x0100"]
    assert result["data"][CONF_PARAMETER_TITLES]["0x0100"] == "Kesselsolltemperatur"

    entry = result["result"]
    await hass.async_block_till_done()
    await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()
