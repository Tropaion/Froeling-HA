"""Shared test fixtures for Fröling integration tests."""

import pytest
import pytest_asyncio

from .fake_heater import FakeHeater


@pytest.fixture(autouse=True)
def _allow_localhost_sockets(request: pytest.FixtureRequest) -> None:
    """Let the protocol tests reach the fake heater over localhost TCP.

    When the Home Assistant test plugin is installed, it blocks sockets in
    every test via pytest-socket; opt back in (it still only allows
    127.0.0.1).  Without that plugin this fixture does nothing.
    """
    if request.config.pluginmanager.hasplugin("socket"):
        request.getfixturevalue("socket_enabled")


@pytest_asyncio.fixture
async def heater():
    """A running fake bridge + heater (see fake_heater.py), stopped afterwards."""
    fake = FakeHeater()
    await fake.start()
    yield fake
    await fake.stop()
