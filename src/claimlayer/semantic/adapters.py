"""OpenAI-compatible HTTP transport for the live derivation planner."""

from __future__ import annotations


from dataclasses import dataclass, field


import copy


import json


import time


from typing import Any, Mapping


from urllib.error import HTTPError, URLError


from urllib.parse import urlsplit


from urllib.request import Request, urlopen


from .models import CONTRACT_VERSION, canonical, digest


def _json_object(raw: str, *, allow_code_fence: bool = False) -> dict[str, Any]:
    text = raw.strip()
    if allow_code_fence and text.startswith('```'):
        lines = text.splitlines()
        if lines and lines[0].startswith('```'):
            lines = lines[1:]
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        text = '\n'.join(lines).strip()

    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError('Duplicate JSON field')
            out[key] = value
        return out
    value = json.loads(text, object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
    if type(value) is not dict:
        raise ValueError('Model must return a JSON object')
    return value


@dataclass(frozen=True)
class OpenAICompatibleModel:
    base_url: str
    model: str
    api_key: str = field(repr=False)
    provider: str = 'openai-compatible'
    timeout: float = 45.0
    max_tokens: int = 4096
    retries: int = 1
    extra_body: Mapping[str, Any] = field(default_factory=dict)
    response_format: Mapping[str, Any] | None = field(default_factory=lambda: {'type': 'json_object'})

    def __post_init__(self):
        url = urlsplit(self.base_url)
        if url.scheme not in {'http', 'https'} or not url.netloc or url.username or url.password or url.query or url.fragment:
            raise ValueError('Use an HTTP(S) base URL without credentials or query parameters')
        if not 0 < self.timeout <= 600 or not 0 <= self.retries <= 3 or not 1 <= self.max_tokens <= 32768:
            raise ValueError('Invalid request limits')
        if set(self.extra_body) - {'chat_template_kwargs', 'reasoning_effort'}:
            raise ValueError('Only chat_template_kwargs and reasoning_effort are allowed in extra_body')

    @property
    def identity(self):
        return {'provider': self.provider, 'model': self.model, 'endpoint': self.base_url.rstrip('/'),
                'parameters': {'temperature': 0, 'max_tokens': self.max_tokens,
                               'extra_body': dict(self.extra_body)}}

    def complete(self, system: str, payload: Mapping[str, Any]):
        body: dict[str, Any] = {
            'model': self.model,
            'temperature': 0,
            'max_tokens': self.max_tokens,
            'messages': [{'role': 'system', 'content': system},
                         {'role': 'user', 'content': canonical(payload)}],
            **dict(self.extra_body),
        }
        if self.response_format is not None:
            body['response_format'] = dict(self.response_format)
        request = Request(self.base_url.rstrip('/') + '/chat/completions',
                          data=canonical(body).encode(), method='POST',
                          headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.api_key})
        attempts = []
        for attempt in range(self.retries + 1):
            raw = None
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    raw_bytes = response.read(2_000_001)
                if len(raw_bytes) > 2_000_000:
                    raise ValueError('Provider response exceeds limit')
                raw = _json_object(raw_bytes.decode())
                choices = raw.get('choices', [])
                if len(choices) != 1 or choices[0].get('finish_reason') != 'stop':
                    raise ValueError('Missing or truncated model response')
                parsed = _json_object(choices[0]['message']['content'], allow_code_fence=self.response_format is None)
                return {'output': parsed, 'usage': raw.get('usage', {}),
                        'response_model': raw.get('model'), 'provider_response': raw,
                        'transport_attempts': attempts + [{'attempt': attempt + 1, 'status': 'success'}]}
            except (HTTPError, URLError, TimeoutError) as exc:
                diagnostic = {'attempt': attempt + 1, 'error_type': type(exc).__name__}
                if isinstance(exc, HTTPError):
                    diagnostic['http_status'] = exc.code
                attempts.append(diagnostic)
                exc.diagnostics = {'attempts': copy.deepcopy(attempts)}
                if attempt == self.retries or (isinstance(exc, HTTPError) and exc.code not in {408, 429, 500, 502, 503, 504}):
                    raise
                delay = min(2 ** attempt, 4)
                if isinstance(exc, HTTPError) and exc.code == 429:
                    delay = 60
                    retry_after = exc.headers.get('Retry-After', '') if exc.headers else ''
                    if retry_after.isdigit():
                        delay = min(60, max(1, int(retry_after)))
                time.sleep(delay)
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                diagnostic = {'attempt': attempt + 1, 'error_type': type(exc).__name__,
                              'stage': 'response_contract', 'detail': 'invalid_or_truncated_response'}
                if isinstance(raw, dict):
                    choices = raw.get('choices')
                    if isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict):
                        diagnostic['finish_reason'] = choices[0].get('finish_reason')
                attempts.append(diagnostic)
                exc.diagnostics = {'attempts': copy.deepcopy(attempts), 'provider_response': copy.deepcopy(raw)}
                raise
