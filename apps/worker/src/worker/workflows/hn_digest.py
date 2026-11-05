import asyncio
import json
from datetime import timedelta, datetime, timezone
from typing import Any
from urllib.parse import urlparse

import httpx
import pydantic

import mistralai.workflows as workflows
from mistralai.workflows import Depends
from mistralai.workflows.client import get_mistral_client
from mistralai.workflows.plugins.mistralai.connectors import (
    ToolCallClient,
    connector,
    uses_connectors,
)

gmail_connector = connector("gmail")

HN_API_BASE = "https://hacker-news.firebaseio.com/v1"
CHAT_MODEL = "mistral-medium-latest"
TTS_MODEL = "voxtral-mini-tts-3-0"
TTS_VOICE = "gb_jane_neutral"
TTS_FORMAT = "mp3"
DEFAULT_RECIPIENT = "iqbal.adudu@gmail.com"
MAX_CANDIDATES = 30


class HnDigestInput(pydantic.BaseModel):
    recipient: str = DEFAULT_RECIPIENT
    max_items: int = 8


class Story(pydantic.BaseModel):
    id: int
    title: str
    url: str | None = None
    score: int = 0
    by: str = ""
    comments: int = 0

    @property
    def hn_url(self) -> str:
        return f"https://news.ycombinator.com/item?id={self.id}"

    @property
    def domain(self) -> str:
        if not self.url:
            return "news.ycombinator.com"
        return urlparse(self.url).netloc.removeprefix("www.")


class DigestItem(pydantic.BaseModel):
    id: int
    title: str
    url: str | None = None
    hn_url: str
    domain: str
    summary: str


class Digest(pydantic.BaseModel):
    fetched_at: str
    items: list[DigestItem]
    podcast_script: str


@workflows.activity(start_to_close_timeout=timedelta(seconds=90), retry_policy_max_attempts=3)
async def fetch_front_page_stories(max_candidates: int) -> tuple[str, list[Story]]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{HN_API_BASE}/frontpage.json")
        response.raise_for_status()
        ids = response.json()[:max_candidates]

        async def fetch_item(item_id: int) -> dict[str, Any]:
            item_response = await client.get(f"{HN_API_BASE}/item/{item_id}.json")
            item_response.raise_for_status()
            return item_response.json()

        raw_items = await asyncio.gather(*(fetch_item(i) for i in ids))

    fetched_at = datetime.now(timezone.utc).isoformat()
    stories = [
        Story(
            id=item["id"],
            title=item.get("title", ""),
            url=item.get("url"),
            score=item.get("score", 0),
            by=item.get("by", ""),
            comments=item.get("descendants", 0),
        )
        for item in raw_items
        if item.get("type") == "story" and item.get("title")
    ]
    return fetched_at, stories


@workflows.activity(start_to_close_timeout=timedelta(seconds=180), retry_policy_max_attempts=2)
async def build_tech_digest(fetched_at: str, stories: list[Story], max_items: int) -> Digest:
    client = get_mistral_client(use_executor_credentials=True)
    listing = "\n".join(
        f"- id={s.id} | score={s.score} | comments={s.comments} | domain={s.domain} | by={s.by} | {s.title}"
        for s in stories
    )
    prompt = f"""You are a tech-news editor. Below are the current HackerNews front-page stories.

Select at most {max_items} stories that are TECH INDUSTRY NEWS: AI/ML products and research releases, semiconductors and hardware, big-tech product launches or policy moves, cybersecurity incidents, developer tools and infrastructure, major open-source releases, startup funding and acquisitions.

Exclude personal blogs, individual opinions with no news value, Ask HN / Show HN posts, and anything not about the tech industry.

For each selected story write a 1-2 sentence summary a busy engineer can act on.

Also write a "podcast_script": a spoken-style monologue of 180-260 words (about 1 to 2 minutes at a normal speaking pace) that introduces the digest and reads through the headlines with the key takeaway of each. Plain text only, no markdown, no sound-effect directions.

Respond with JSON only, in this exact shape:
{{
  "items": [
    {{"id": <story id>, "title": "<title>", "summary": "<1-2 sentence summary>"}}
  ],
  "podcast_script": "<180-260 word monologue>"
}}

Stories:
{listing}"""
    response = await client.chat.complete_async(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    payload = extract_json_object(response.choices[0].message.content or "")
    by_id = {s.id: s for s in stories}
    items = []
    for raw in payload.get("items", []):
        story = by_id.get(raw.get("id"))
        if story is None:
            continue
        items.append(
            DigestItem(
                id=story.id,
                title=story.title,
                url=story.url,
                hn_url=story.hn_url,
                domain=story.domain,
                summary=str(raw.get("summary", "")).strip(),
            )
        )
        if len(items) >= max_items:
            break
    script = str(payload.get("podcast_script", "")).strip()
    if not items or not script:
        raise ValueError("LLM returned an empty digest")
    return Digest(fetched_at=fetched_at, items=items, podcast_script=script)


@workflows.activity(start_to_close_timeout=timedelta(seconds=180), retry_policy_max_attempts=2)
async def generate_podcast_audio(podcast_script: str) -> str:
    client = get_mistral_client(use_executor_credentials=True)
    response = await client.audio.speech.complete_async(
        model=TTS_MODEL,
        input=podcast_script,
        voice_id=TTS_VOICE,
        response_format=TTS_FORMAT,
    )
    return response.audio_data


def extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in LLM response")
    return json.loads(text[start : end + 1])


def build_digest_email(digest: Digest) -> tuple[str, str, str]:
    date_label = digest.fetched_at[:10]
    subject = f"HN Tech Digest — {date_label} ({len(digest.items)} stories)"
    lines = [
        f"HackerNews tech digest for {date_label}",
        "",
        "Top tech-industry headlines from the HN front page:",
        "",
    ]
    html_items = []
    for rank, item in enumerate(digest.items, start=1):
        link = item.url or item.hn_url
        lines.append(f"{rank}. {item.title} ({item.domain}) — {item.summary}")
        lines.append(f"   {link} | discussion: {item.hn_url}")
        lines.append("")
        html_items.append(
            f"<li><a href=\"{link}\">{item.title}</a> "
            f"<em>({item.domain})</em> — {item.summary} "
            f"[<a href=\"{item.hn_url}\">HN discussion</a>]</li>"
        )
    lines.append("A 1-2 minute audio briefing of these headlines is attached as hn-tech-digest.mp3.")
    plain_body = "\n".join(lines)
    html_body = (
        f"<h2>HackerNews tech digest — {date_label}</h2>"
        f"<p>Top tech-industry headlines from the HN front page:</p>"
        f"<ol>{''.join(html_items)}</ol>"
        "<p>A 1-2 minute audio briefing of these headlines is attached as "
        "<strong>hn-tech-digest.mp3</strong>.</p>"
    )
    return subject, plain_body, html_body


async def send_via_gmail(
    gmail: ToolCallClient,
    recipient: str,
    subject: str,
    plain_body: str,
    html_body: str,
    audio_base64: str,
) -> Any:
    return await gmail.call_tool(
        tool_name="send_email",
        arguments={
            "to": [recipient],
            "subject": subject,
            "body": plain_body,
            "htmlBody": html_body,
            "attachments": [
                {
                    "content": audio_base64,
                    "filename": "hn-tech-digest.mp3",
                    "mimeType": "audio/mpeg",
                }
            ],
        },
    )


@workflows.activity(start_to_close_timeout=timedelta(seconds=60), retry_policy_max_attempts=3)
async def send_digest_email(
    recipient: str,
    subject: str,
    plain_body: str,
    html_body: str,
    audio_base64: str,
    gmail: ToolCallClient = Depends(gmail_connector),
) -> Any:
    return await send_via_gmail(gmail, recipient, subject, plain_body, html_body, audio_base64)


@workflows.workflow.define(
    name="hn-tech-digest",
    workflow_display_name="HN Tech Digest",
    workflow_description=(
        "One-shot run: fetches the HackerNews front page, selects tech-industry news, "
        "summarizes it, generates a 1-2 minute MP3 audio briefing, and emails the digest "
        "with the audio attached to the triggering user's Gmail inbox."
    ),
    on_behalf_of=True,
)
@uses_connectors(gmail_connector)
class HnTechDigestWorkflow:
    @workflows.workflow.entrypoint
    async def run(self, input: HnDigestInput) -> dict[str, Any]:
        fetched_at, stories = await fetch_front_page_stories(MAX_CANDIDATES)
        digest = await build_tech_digest(fetched_at, stories, input.max_items)
        audio_base64 = await generate_podcast_audio(digest.podcast_script)
        subject, plain_body, html_body = build_digest_email(digest)
        send_result = await send_digest_email(
            input.recipient, subject, plain_body, html_body, audio_base64
        )
        return {
            "status": "sent",
            "recipient": input.recipient,
            "subject": subject,
            "story_count": len(digest.items),
            "titles": [item.title for item in digest.items],
            "audio_filename": "hn-tech-digest.mp3",
            "gmail_response": send_result,
        }
