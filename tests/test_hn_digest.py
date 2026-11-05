import base64
import json

import mistralai.workflows as workflows
from worker.workflows import hn_digest
from worker.workflows.hn_digest import (
    Digest,
    DigestItem,
    Story,
    build_digest_email,
    build_tech_digest,
    extract_json_object,
    fetch_front_page_stories,
    generate_podcast_audio,
    send_digest_email,
    send_via_gmail,
)


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


HN_ITEMS = {
    1: {"id": 1, "type": "story", "title": "Big AI lab ships new model", "url": "https://ai-lab.example.com/model", "score": 500, "by": "founder", "descendants": 120},
    2: {"id": 2, "type": "story", "title": "My personal morning routine", "url": "https://blog.example.com/routine", "score": 300, "by": "blogger", "descendants": 40},
    3: {"id": 3, "type": "comment", "title": "not a story"},
    4: {"id": 4, "type": "story", "title": "Chipmaker unveils 2nm process", "url": "https://chips.example.com/2nm", "score": 450, "by": "reporter", "descendants": 90},
}


class FakeAsyncClient:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def get(self, url):
        if url.endswith("/frontpage.json"):
            return FakeResponse([1, 2, 3, 4])
        item_id = int(url.rsplit("/", 1)[1].removesuffix(".json"))
        return FakeResponse(HN_ITEMS[item_id])


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeChatResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeChat:
    async def complete_async(self, model, messages, response_format=None):
        payload = {
            "items": [
                {"id": 1, "title": "Big AI lab ships new model", "summary": "A new frontier model with stronger coding scores."},
                {"id": 4, "title": "Chipmaker unveils 2nm process", "summary": "Mass production is slated for next year."},
                {"id": 999, "title": "hallucinated", "summary": "should be dropped"},
            ],
            "podcast_script": "Welcome to the HackerNews tech digest. Today: a new AI model and a 2nm chip process.",
        }
        return FakeChatResponse(json.dumps(payload))


class FakeSpeechResponse:
    audio_data = base64.b64encode(b"fake mp3 bytes").decode()


class FakeSpeech:
    async def complete_async(self, model, input, voice_id, response_format):
        assert response_format == "mp3"
        assert input
        return FakeSpeechResponse()


class FakeAudio:
    speech = FakeSpeech()


class FakeMistralClient:
    chat = FakeChat()
    audio = FakeAudio()


class FakeToolCallClient:
    def __init__(self):
        self.calls = []

    async def call_tool(self, tool_name, arguments=None):
        self.calls.append((tool_name, arguments))
        return {"id": "msg-1", "threadId": "thread-1"}


async def test_fetch_front_page_stories_filters_to_stories(monkeypatch):
    monkeypatch.setattr(hn_digest.httpx, "AsyncClient", FakeAsyncClient)
    with workflows.run_activities_locally():
        fetched_at, stories = await fetch_front_page_stories(30)
    assert len(stories) == 3
    assert all(isinstance(s, Story) for s in stories)
    assert {s.id for s in stories} == {1, 2, 4}
    assert stories[0].hn_url == "https://news.ycombinator.com/item?id=1"
    assert stories[0].domain == "ai-lab.example.com"
    assert fetched_at.endswith("+00:00")


async def test_build_tech_digest_parses_and_filters(monkeypatch):
    monkeypatch.setattr(hn_digest, "get_mistral_client", lambda **kwargs: FakeMistralClient())
    stories = [Story(id=i, title=t, url=u) for i, t, u in [(1, "a", "https://x.example.com/a"), (4, "b", None)]]
    with workflows.run_activities_locally():
        digest = await build_tech_digest("2026-01-01T00:00:00+00:00", stories, 8)
    assert isinstance(digest, Digest)
    assert [item.id for item in digest.items] == [1, 4]
    assert all(item.summary for item in digest.items)
    assert "digest" in digest.podcast_script.lower()


async def test_generate_podcast_audio_returns_base64(monkeypatch):
    monkeypatch.setattr(hn_digest, "get_mistral_client", lambda **kwargs: FakeMistralClient())
    with workflows.run_activities_locally():
        audio = await generate_podcast_audio("Hello headlines.")
    assert base64.b64decode(audio) == b"fake mp3 bytes"


async def test_send_digest_email_attaches_mp3():
    gmail = FakeToolCallClient()
    audio = base64.b64encode(b"fake mp3 bytes").decode()
    result = await send_via_gmail(
        gmail, "iqbal.adudu@gmail.com", "subject", "plain", "<p>html</p>", audio
    )
    tool_name, args = gmail.calls[0]
    assert tool_name == "send_email"
    assert args["to"] == ["iqbal.adudu@gmail.com"]
    assert args["attachments"][0]["mimeType"] == "audio/mpeg"
    assert args["attachments"][0]["filename"] == "hn-tech-digest.mp3"
    assert args["attachments"][0]["content"] == audio
    assert result == {"id": "msg-1", "threadId": "thread-1"}


def test_extract_json_object_handles_code_fences():
    text = 'Here you go:\n```json\n{"a": 1, "b": {"c": 2}}\n```'
    assert extract_json_object(text) == {"a": 1, "b": {"c": 2}}


def test_build_digest_email_contents():
    digest = Digest(
        fetched_at="2026-01-01T12:00:00+00:00",
        items=[
            DigestItem(
                id=1,
                title="Big AI lab ships new model",
                url="https://ai-lab.example.com/model",
                hn_url="https://news.ycombinator.com/item?id=1",
                domain="ai-lab.example.com",
                summary="A new frontier model.",
            )
        ],
        podcast_script="script",
    )
    subject, plain, html = build_digest_email(digest)
    assert "2026-01-01" in subject
    assert "1 stories" in subject
    assert "Big AI lab ships new model" in plain
    assert "hn-tech-digest.mp3" in plain
    assert 'href="https://ai-lab.example.com/model"' in html
    assert "<ol>" in html
