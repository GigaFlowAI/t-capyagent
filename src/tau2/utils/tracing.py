"""
Braintrust tracing.

Optional: without BRAINTRUST_API_KEY every helper here is a no-op, and a tracing
failure never breaks a benchmark run.
"""

import os
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

DEFAULT_BRAINTRUST_PROJECT = "capyagent"

_configured = False
_enabled = False


def configure_tracing() -> bool:
    """Configure Braintrust once. Returns True when traces are being sent.

    Must run before ``from litellm import completion`` so the imported name is the
    patched one.
    """
    global _configured, _enabled
    if _configured:
        return _enabled
    _configured = True
    if not os.environ.get("BRAINTRUST_API_KEY"):
        logger.info(
            "BRAINTRUST_API_KEY is not set: no traces will be sent to Braintrust."
        )
        return False
    try:
        import braintrust
        from braintrust.wrappers.litellm import patch_litellm

        braintrust.init_logger(
            project=os.environ.get("BRAINTRUST_PROJECT") or DEFAULT_BRAINTRUST_PROJECT
        )
        patch_litellm()
        _enabled = True
    except Exception as e:
        logger.warning(f"Braintrust tracing is disabled: {e}")
    return _enabled


class _NoopSpan:
    def log(self, **event: Any) -> None:
        pass


class _SafeSpan:
    """A span whose logging never raises into the run."""

    def __init__(self, span: Any) -> None:
        self._span = span

    def log(self, **event: Any) -> None:
        try:
            self._span.log(**event)
        except Exception as e:
            logger.warning(f"Braintrust span log failed: {e}")


@contextmanager
def traced_span(name: str, type: Optional[str] = None, **event: Any) -> Iterator[Any]:
    """Open a Braintrust span under the current one; a no-op when tracing is off."""
    if not configure_tracing():
        yield _NoopSpan()
        return
    import braintrust

    with braintrust.start_span(name=name, type=type, **event) as span:
        yield _SafeSpan(span)


def flush_tracing() -> None:
    """Send buffered spans now. batch.py force-exits, which skips atexit hooks."""
    if not _enabled:
        return
    try:
        import braintrust

        braintrust.flush()
    except Exception as e:
        logger.warning(f"Braintrust flush failed: {e}")
