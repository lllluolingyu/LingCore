"""Search provider protocols, credential isolation, and daily-profile behavior."""

from __future__ import annotations

import asyncio
import json
import sys

import httpx
import pytest
from pydantic import ValidationError

from lingcore.agent import Agent
from lingcore.config import AgentProfile
from lingcore.errors import ToolError
from lingcore.plugins.discovery import discover_plugins
from lingcore.profiles import initialize_profile
from lingcore.skills import _load_tool_module
from lingcore.tools import ToolContext
from lingcore.tools.builtin import web
from tests.fakes import FakeLLMClient, ScriptedTurn


def _load_search():
    plugin = discover_plugins(None)["search"]
    _load_tool_module(
        "search",
        plugin.root / plugin.manifest.module,
        plugin.manifest.provides,
        prefix="search",
    )
    return next(
        module
        for name, module in sys.modules.items()
        if name.startswith("lingcore_skill_tools.search.")
    )


search = _load_search()


@pytest.fixture
def ctx(tmp_path):
    return ToolContext(
        workspace=tmp_path,
        environment={"SEARXNG_URL": "https://search.example/searx/"},
    )


@pytest.fixture(autouse=True)
def fake_dns(monkeypatch):
    async def resolve(host, port):
        return ["127.0.0.1" if host == "private.example" else "93.184.216.34"]

    monkeypatch.setattr(web, "_resolve_ips", resolve)


@pytest.fixture
def serve(monkeypatch):
    real_client = httpx.AsyncClient
    requests = []

    def make(data=None, *, status=200, content=None, headers=None, error=None):
        def respond(request):
            requests.append(request)
            if error is not None:
                raise error
            return httpx.Response(
                status,
                json=data if content is None else None,
                content=content,
                headers=headers,
            )

        def client(**kwargs):
            return real_client(transport=httpx.MockTransport(respond), **kwargs)

        monkeypatch.setattr(search.httpx, "AsyncClient", client)
        return requests

    return make


async def test_searxng_encodes_query_and_returns_direct_bounded_sources(ctx, serve):
    rows = [
        {
            "title": "<b>Opening hours</b>",
            "url": "https://example.com/#hours",
            "content": "Weekdays &amp; weekends",
            "publishedDate": "2026-10-19",
        },
        {"title": "Duplicate", "url": "https://example.com/"},
        {"title": "Unsafe", "url": "javascript:alert(1)"},
        {"title": "Credential URL", "url": "https://user:secret@example.com/"},
        {"title": "Next", "url": "https://other.example/", "content": "x" * 10_000},
        {"title": "Beyond the requested limit", "url": "https://third.example/"},
    ]
    requests = serve({"results": rows})
    query = "上海图书馆 & opening hours"
    result = json.loads(
        await search.search_web(search.SearchArgs(query=query, max_results=2), ctx)
    )

    assert len(requests) == 1  # Discovered URLs are never fetched automatically.
    request = requests[0]
    assert request.method == "GET"
    assert request.url.path == "/searx/search"
    assert request.url.params["q"] == query
    assert request.url.params["format"] == "json"
    assert request.url.host == "93.184.216.34"
    assert request.headers["host"] == "search.example"
    assert request.extensions["sni_hostname"] == "search.example"
    assert "authorization" not in request.headers and "x-api-key" not in request.headers
    assert result["results"][0] == {
        "title": "Opening hours",
        "url": "https://example.com/",
        "snippet": "Weekdays & weekends",
        "published": "2026-10-19",
    }
    assert len(result["results"]) == 2
    assert len(result["results"][1]["snippet"]) == 1200


@pytest.mark.parametrize(
    "provider,key_env", [("exa", "EXA_API_KEY"), ("deepseek", "DEEPSEEK_API_KEY")]
)
async def test_api_protocol_and_profile_credentials(
    ctx, serve, monkeypatch, provider, key_env
):
    monkeypatch.setenv(key_env, "ambient-secret")
    ctx.environment = {key_env: "profile-secret"}
    ctx.options = {"search_web": {"provider": provider}}
    requests = serve(
        {
            "results": [
                {
                    "title": "Source",
                    "url": "https://example.com/",
                    "highlights": ["Exa passage"],
                    "summary": "DeepSeek summary",
                }
            ]
        }
    )
    result = json.loads(
        await search.search_web(
            search.SearchArgs(query="bus timetable", max_results=3), ctx
        )
    )
    request = requests[0]
    assert request.method == "POST"
    assert (
        request.headers["host"]
        == {"exa": "api.exa.ai", "deepseek": "api.deepseek.com"}[provider]
    )
    assert request.url.host == "93.184.216.34"
    body = json.loads(request.content)
    assert body["query"] == "bus timetable"
    if provider == "exa":
        assert request.url.path == "/search"
        assert request.headers["x-api-key"] == "profile-secret"
        assert body["numResults"] == 3 and body["contents"] == {"highlights": True}
    else:
        assert request.url.path == "/v1/tools/search"
        assert request.headers["authorization"] == "Bearer profile-secret"
        assert body["count"] == 3 and body["need_content"] is False
    assert result["provider"] == provider
    assert result["results"][0]["snippet"] == "DeepSeek summary"
    assert "secret" not in json.dumps(result)


async def test_exa_highlights_are_used_for_snippets(ctx, serve):
    ctx.options = {"search_web": {"provider": "exa", "api_key_env": "CUSTOM_KEY"}}
    ctx.environment = {"CUSTOM_KEY": "custom-secret"}
    serve(
        {
            "results": [
                {
                    "url": "https://example.com/",
                    "highlights": ["First passage", "Second passage"],
                }
            ]
        }
    )
    result = json.loads(
        await search.search_web(search.SearchArgs(query="recipes"), ctx)
    )
    assert result["results"][0]["snippet"] == "First passage Second passage"


@pytest.mark.parametrize(
    "provider,key_env", [("exa", "EXA_API_KEY"), ("deepseek", "DEEPSEEK_API_KEY")]
)
async def test_empty_profile_key_blocks_exported_key(
    ctx, serve, monkeypatch, provider, key_env
):
    ctx.options = {"search_web": {"provider": provider}}
    ctx.environment = {key_env: ""}
    monkeypatch.setenv(key_env, "ambient-secret")
    requests = serve({"results": []})
    with pytest.raises(ToolError, match=key_env):
        await search.search_web(search.SearchArgs(query="recipes"), ctx)
    assert requests == []


async def test_missing_searxng_instance_is_actionable(ctx, monkeypatch):
    ctx.environment = {"SEARXNG_URL": ""}
    with pytest.raises(ToolError, match="set SEARXNG_URL"):
        await search.search_web(search.SearchArgs(query="recipes"), ctx)


@pytest.mark.parametrize("suffix", ["?", "#", "/?", "/#", "/search?", "/search#"])
async def test_empty_endpoint_delimiters_keep_the_search_path_and_query(
    ctx, serve, suffix
):
    ctx.options = {
        "search_web": {"searxng_url": "https://search.example/searx" + suffix}
    }
    requests = serve({"results": []})
    await search.search_web(search.SearchArgs(query="opening hours"), ctx)
    assert requests[0].url.path == "/searx/search"
    assert requests[0].url.params["q"] == "opening hours"
    assert requests[0].url.params["format"] == "json"
    assert not requests[0].url.fragment


async def test_provider_surrogates_are_replaced_in_text_and_rejected_in_urls(
    ctx, serve
):
    data = {
        "results": [
            {
                "url": "https://example.com/",
                "title": "Museum \ud800",
                "snippet": "Hours \udfff",
                "date": "2026-10-19\ud800",
            },
            {"url": "https://example.com/\ud800", "title": "Invalid URL"},
            {"url": "https://other.example/", "title": "Another source"},
        ]
    }
    # Escaped surrogates are valid JSON, even though their decoded strings
    # cannot be encoded into UTF-8 tool results, session rows, or UI frames.
    serve(content=json.dumps(data, ensure_ascii=True).encode("ascii"))
    output = await search.search_web(search.SearchArgs(query="museums"), ctx)
    output.encode("utf-8")
    results = json.loads(output)["results"]
    assert len(results) == 2
    assert results[0]["title"] == "Museum ?"
    assert results[0]["snippet"] == "Hours ?"
    assert results[0]["published"] == "2026-10-19?"
    assert results[1]["url"] == "https://other.example/"


@pytest.mark.parametrize("key", ["", "malformed-secret\nvalue"])
async def test_identifier_shaped_credential_in_key_option_is_never_echoed(
    ctx, serve, key
):
    sentinel = "sk_live_ABC123"
    ctx.options = {"search_web": {"provider": "exa", "api_key_env": sentinel}}
    ctx.environment = {sentinel: key}
    requests = serve({"results": []})
    with pytest.raises(ToolError, match="tool_options.search_web.api_key_env") as error:
        await search.search_web(search.SearchArgs(query="recipes"), ctx)
    assert sentinel not in str(error.value)
    assert key == "" or key not in str(error.value)
    assert requests == []


@pytest.mark.parametrize(
    "url",
    [
        "https://user:secret@example.com/",
        "file:///secret",
        "https://example.com/?key=secret",
        "http://example.com:99999/",
    ],
)
async def test_invalid_endpoint_never_exposes_its_values(ctx, serve, url):
    ctx.options = {"search_web": {"searxng_url": url}}
    requests = serve({"results": []})
    with pytest.raises(ToolError, match="SearXNG URL") as error:
        await search.search_web(search.SearchArgs(query="recipes"), ctx)
    assert "secret" not in str(error.value)
    assert requests == []


async def test_private_instance_requires_approval_and_pins_approved_address(ctx, serve):
    ctx.environment = {"SEARXNG_URL": "http://private.example:8080"}
    requests = serve({"results": []})
    with pytest.raises(ToolError, match="private/local"):
        await search.search_web(search.SearchArgs(query="recipes"), ctx)
    assert requests == []
    prompts = []

    async def approve(prompt):
        prompts.append(prompt)
        return True

    ctx.confirm = approve
    result = json.loads(
        await search.search_web(search.SearchArgs(query="recipes"), ctx)
    )
    assert result["results"] == []
    assert len(prompts) == 1 and "Allow search_web" in prompts[0]
    assert requests[0].url.host == "127.0.0.1"
    assert requests[0].headers["host"] == "private.example:8080"


async def test_private_instance_respects_refusal_policy(ctx, serve):
    ctx.environment = {"SEARXNG_URL": "http://private.example/"}
    ctx.options = {"fetch_url": {"confirm_private_hosts": False}}
    ctx.confirm = lambda _: pytest.fail("must not ask when disabled")
    requests = serve({"results": []})
    with pytest.raises(ToolError, match="private/local"):
        await search.search_web(search.SearchArgs(query="recipes"), ctx)
    assert requests == []


async def test_narrow_network_exemption_allows_repeated_private_searches(ctx, serve):
    ctx.environment = {"SEARXNG_URL": "http://private.example:8080"}
    ctx.options = {"fetch_url": {"allowed_networks": ["127.0.0.1/32"]}}
    ctx.confirm = lambda _: pytest.fail("the configured address needs no approval")
    requests = serve({"results": []})
    for _ in range(2):
        await search.search_web(search.SearchArgs(query="recipes"), ctx)
    assert len(requests) == 2
    assert all(request.url.host == "127.0.0.1" for request in requests)


@pytest.mark.parametrize("status,match", [(403, "enable json"), (429, "rate limit")])
async def test_provider_errors_are_actionable_and_do_not_echo_response(
    ctx, serve, status, match
):
    requests = serve({"error": "secret-sentinel"}, status=status)
    with pytest.raises(ToolError, match=match) as error:
        await search.search_web(search.SearchArgs(query="recipes"), ctx)
    assert "secret-sentinel" not in str(error.value)
    assert len(requests) == 1


async def test_api_redirect_never_forwards_credentials(ctx, serve):
    ctx.options = {"search_web": {"provider": "deepseek"}}
    ctx.environment = {"DEEPSEEK_API_KEY": "profile-secret"}
    requests = serve(status=307, headers={"location": "https://other.example/leak"})
    with pytest.raises(ToolError, match="redirected"):
        await search.search_web(search.SearchArgs(query="recipes"), ctx)
    assert len(requests) == 1


@pytest.mark.parametrize(
    "content,match",
    [
        (b"<html>captcha</html>", "valid JSON"),
        (b'{"results": {}}', "invalid results"),
        (b"x" * 2000, "byte limit"),
    ],
)
async def test_invalid_and_oversized_payloads_fail_clearly(ctx, serve, content, match):
    ctx.options = {"search_web": {"max_bytes": 1024}}
    serve(content=content)
    with pytest.raises(ToolError, match=match):
        await search.search_web(search.SearchArgs(query="recipes"), ctx)


async def test_failed_engines_and_invalid_result_urls_are_not_empty_matches(ctx, serve):
    serve({"results": [], "unresponsive_engines": [["engine", "error"]]})
    with pytest.raises(ToolError, match="engine failures"):
        await search.search_web(search.SearchArgs(query="recipes"), ctx)
    serve({"results": [{"url": "javascript:alert(1)"}]})
    with pytest.raises(ToolError, match="no usable HTTP"):
        await search.search_web(search.SearchArgs(query="recipes"), ctx)


async def test_malformed_api_key_is_rejected_without_echoing_it(ctx, serve):
    ctx.options = {"search_web": {"provider": "exa"}}
    ctx.environment = {"EXA_API_KEY": "secret-sentinel\ninvalid"}
    requests = serve({"results": []})
    with pytest.raises(ToolError, match="valid key in EXA_API_KEY") as error:
        await search.search_web(search.SearchArgs(query="recipes"), ctx)
    assert "secret-sentinel" not in str(error.value)
    assert requests == []


async def test_transport_errors_are_sanitized_and_cancellation_propagates(ctx, serve):
    serve(error=httpx.ConnectError("secret-sentinel"))
    with pytest.raises(ToolError, match="ConnectError") as error:
        await search.search_web(search.SearchArgs(query="recipes"), ctx)
    assert "secret-sentinel" not in str(error.value)
    serve(error=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await search.search_web(search.SearchArgs(query="recipes"), ctx)


@pytest.mark.parametrize(
    "options",
    [
        {"provider": "unknown-secret"},
        {"api_key": "secret-sentinel"},
        {"api_key_env": "secret-sentinel"},
    ],
)
async def test_invalid_options_do_not_echo_values(ctx, options):
    ctx.options = {"search_web": options}
    with pytest.raises(ToolError, match="tool_options.search_web") as error:
        await search.search_web(search.SearchArgs(query="recipes"), ctx)
    assert "secret" not in str(error.value)


@pytest.mark.parametrize(
    "args",
    [
        {"query": " "},
        {"query": "x" * 129},
        {"query": "\ud800"},
        {"query": "hello", "max_results": 11},
        {"query": "hello", "max_results": True},
    ],
)
def test_query_and_result_limits(args):
    with pytest.raises(ValidationError):
        search.SearchArgs(**args)


async def test_daily_has_search_immediately_without_code_review(tmp_path, monkeypatch):
    monkeypatch.delenv("LINGCORE_SEARCH_PROVIDER", raising=False)
    destination = initialize_profile("daily", destination=tmp_path / "daily")
    profile = AgentProfile.load(destination)
    assert profile.plugins == ["search"]
    assert profile.tool_options["search_web"]["provider"] == "searxng"
    assert {"search", "search_web", "fetch_url"} <= profile.initial_tool_set()
    async with Agent.from_profile(
        profile, llm=FakeLLMClient([ScriptedTurn(text="hello")])
    ) as agent:
        names = {
            schema["function"]["name"] for schema in agent._effective_tool_schemas()
        }
        assert "search_web" in names
        assert "activate_skill" not in names
        assert "run_shell" not in names
        assert agent.skill_state is None
