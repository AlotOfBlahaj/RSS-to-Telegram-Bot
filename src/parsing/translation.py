#  RSS to Telegram Bot
#  Copyright (C) 2021-2024  Rongrong <i@rong.moe>
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as
#  published by the Free Software Foundation, either version 3 of the
#  License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations
from typing import Optional

import asyncio
import os
from html import escape

from bs4 import BeautifulSoup

from .. import env, log

logger = log.getLogger('RSStT.translation')

ID_ATTR = 'data-t'
PROMPT = (
    'You are a translation engine. Translate the human-readable text in the HTML sent by the user into {lang}. '
    f'Keep every tag and its {ID_ATTR} attribute exactly as given, in the same order and nesting; '
    'do not add, remove or merge tags. Do not translate code inside <pre> or <code>. '
    'Keep proper nouns such as product, brand, project and person names in their original form. '
    'Keep text that is already in {lang} unchanged. '
    'Output only the resulting HTML, without code fences or explanations.'
)

_semaphore: Optional[asyncio.Semaphore] = None


def _strip_attrs(html: str) -> tuple[str, list[dict]]:
    # URLs never reach the model, so it cannot break or hallucinate them
    soup = BeautifulSoup(html, 'html.parser')
    attrs = []
    for tag in soup.find_all(True):
        attrs.append(tag.attrs)
        tag.attrs = {ID_ATTR: str(len(attrs) - 1)}
    return str(soup), attrs


def _restore_attrs(html: str, attrs: list[dict]) -> Optional[str]:
    soup = BeautifulSoup(html, 'html.parser')
    if not attrs:
        return escape(soup.get_text(), quote=False)
    seen = set()
    for tag in soup.find_all(True):
        idx = tag.attrs.get(ID_ATTR)
        if not (isinstance(idx, str) and idx.isdigit() and int(idx) < len(attrs)) or idx in seen:
            return None
        seen.add(idx)
        tag.attrs = attrs[int(idx)]
    return str(soup) if len(seen) == len(attrs) else None


async def _complete(html: str) -> str:
    global _semaphore
    os.environ.setdefault('LITELLM_LOCAL_MODEL_COST_MAP', 'True')
    import litellm  # lazy: costs ~200MB RAM, only paid when translation is enabled

    if _semaphore is None:
        log.getLogger('LiteLLM').setLevel(log.WARNING)
        _semaphore = asyncio.Semaphore(env.TRANSLATION_CONCURRENCY)
    async with _semaphore:
        response = await litellm.acompletion(
            model=env.TRANSLATION_MODEL,
            messages=[
                {'role': 'system', 'content': PROMPT.format(lang=env.TRANSLATION_TARGET_LANG)},
                {'role': 'user', 'content': html},
            ],
            api_key=env.TRANSLATION_API_KEY,
            api_base=env.TRANSLATION_API_BASE,
            extra_body=env.TRANSLATION_EXTRA_BODY or None,
            timeout=60,
            num_retries=1,
        )
    text = (response.choices[0].message.content or '').strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
    return text


async def _translate_html(html: str, link: Optional[str]) -> Optional[str]:
    stripped, attrs = _strip_attrs(html)
    try:
        translated = _restore_attrs(await _complete(stripped), attrs)
    except Exception as e:
        logger.warning(f'Translation failed, sending the original: {link}', exc_info=e)
        return None
    if translated is None:
        logger.warning(f'Translation broke the HTML structure, sending the original: {link}')
    return translated


async def translate_entry(title: Optional[str], content: str, link: Optional[str]) -> tuple[Optional[str], str]:
    if not env.TRANSLATION_MODEL:
        return title, content
    title_t, content_t = await asyncio.gather(
        _translate_html(escape(title), link) if title else asyncio.sleep(0),
        _translate_html(content, link) if content.strip() else asyncio.sleep(0),
    )
    return (
        BeautifulSoup(title_t, 'html.parser').get_text() if title_t else title,
        content_t or content,
    )
