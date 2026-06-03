# Copyright AGNTCY Contributors (https://github.com/agntcy)
# SPDX-License-Identifier: Apache-2.0

"""Regression test for concurrent JSON-RPC id collisions in MCPProtocol.

The server keyed its response futures by the client-supplied JSON-RPC id.
That id is only unique within a single client session and resets per
session, so concurrent clients reusing the same id overwrote each other's
futures - all but the last writer timed out. The fix remaps every inbound
request to a process-unique id and restores the original on the reply.

This test drives the protocol directly over its in-memory streams (no
transport / SLIM / NATS), so it is deterministic and infra-free.
"""

import asyncio
import json

import pytest
from mcp.server.fastmcp import FastMCP

from agntcy_app_sdk.semantic.mcp.protocol import MCPProtocol
from agntcy_app_sdk.semantic.message import Message

pytest_plugins = "pytest_asyncio"


def _rpc(method: str, req_id, params=None) -> Message:
    payload = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params is not None:
        payload["params"] = params
    return Message(type="JSONRPCMessage", payload=json.dumps(payload).encode())


def _decode(resp: Message) -> dict:
    payload = resp.payload
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode()
    return json.loads(payload)


async def _make_protocol() -> MCPProtocol:
    mcp = FastMCP()

    @mcp.tool()
    async def get_forecast(location: str) -> str:
        return f"Sunny in {location}"

    protocol = MCPProtocol()
    protocol.bind_server(mcp)
    await protocol.setup()

    # setup() launches the server loop as a background task; wait until it has
    # wired up the inbound stream before sending any request.
    for _ in range(100):
        if getattr(protocol, "read_stream_writer", None) is not None:
            break
        await asyncio.sleep(0.01)
    else:
        raise RuntimeError("MCP server stream was not ready in time")

    # Drive a single initialize so the low-level server is ready to serve.
    init_params = {
        "protocolVersion": "2025-03-26",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "1.0.0"},
    }
    await protocol.handle_message(_rpc("initialize", 0, init_params), timeout=10)
    return protocol


@pytest.mark.asyncio
async def test_concurrent_requests_with_colliding_ids():
    """Many concurrent requests reusing the same JSON-RPC id must each get
    their own correct response with the original id restored."""
    protocol = await _make_protocol()

    async def list_tools(req_id):
        resp = await protocol.handle_message(_rpc("tools/list", req_id), timeout=10)
        data = _decode(resp)
        assert data["id"] == req_id, f"id mismatch: got {data['id']} want {req_id}"
        assert "result" in data, f"no result for id={req_id}: {data}"
        return data

    # Every "client" independently uses id=1 -> collision pre-fix.
    results = await asyncio.gather(*[list_tools(1) for _ in range(8)])
    assert len(results) == 8
    for data in results:
        assert "tools" in data["result"]


@pytest.mark.asyncio
async def test_response_futures_cleaned_up():
    """Completed requests must not leak entries in the response-future map."""
    protocol = await _make_protocol()

    await asyncio.gather(
        *[protocol.handle_message(_rpc("tools/list", 1), timeout=10) for _ in range(5)]
    )

    assert protocol._response_futures == {}
