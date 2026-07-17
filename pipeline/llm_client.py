import json
import urllib.error
import urllib.request
from pathlib import Path

LLM_CHUNK_SIZE = 40


def format_sent_content(sent: str | list[dict]) -> str:
    if isinstance(sent, str):
        return f"```\n{sent}\n```"
    parts = []
    frame_num = 0
    for block in sent:
        if block.get("type") == "text":
            parts.append(f"```\n{block['text']}\n```")
        elif block.get("type") == "image_url":
            frame_num += 1
            url = block["image_url"]["url"]
            parts.append(f"Frame {frame_num}:\n\n![frame {frame_num}]({url})")
    return "\n\n".join(parts)


def write_raw_log_entry(
    raw_log_path: Path, first_cue: str, last_cue: str, sent: str | list[dict], message: dict,
) -> None:
    with open(raw_log_path, "a", encoding="utf-8") as f:
        f.write(f"## Cues {first_cue}-{last_cue}\n\n")
        f.write(f"### Sent\n\n{format_sent_content(sent)}\n\n")
        f.write(f"### Reasoning\n\n{message.get('reasoning_content') or '*(none)*'}\n\n")
        f.write(f"### Output\n\n```\n{message.get('content') or '(empty)'}\n```\n\n")
        f.write("---\n\n")


def call_llm(
    endpoint: str, model: str, content: str | list[dict], frequency_penalty: float | None = 0.4,
    presence_penalty: float | None = 0.3, temperature: float = 1.0, top_p: float | None = 0.95,
    top_k: int | None = 64, max_tokens: int = 20000, timeout: int = 600,
) -> dict:
    """POST a chat completion request, return the response message dict. Raises
    urllib.error.URLError / OSError / TimeoutError on connection failure.

    temperature/top_p/top_k default to Google's recommended sampling config for Gemma
    (https://ai.google.dev/gemma/docs/core/model_card_4) rather than greedy decoding
    (temperature=0) - greedy decoding is more prone to the kind of deterministic
    repetition/oscillation loops we hit repeatedly with this model. frequency_penalty and
    presence_penalty are kept on top as an extra safety net, not a replacement.
    Trade-off: no longer fully deterministic run-to-run."""
    body = {"model": model, "messages": [{"role": "user", "content": content}],
            "temperature": temperature, "max_tokens": max_tokens}
    if top_p is not None:
        body["top_p"] = top_p
    if top_k is not None:
        body["top_k"] = top_k
    if frequency_penalty is not None:
        body["frequency_penalty"] = frequency_penalty
    if presence_penalty is not None:
        body["presence_penalty"] = presence_penalty
    req = urllib.request.Request(
        f"{endpoint}/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read())
    return result["choices"][0]["message"]
