"""Public chat events shared by the Muse transport and HTTP protocol."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class MuseMessage:
    kind: Literal["progress", "reply", "cancel"]
    text: str = ""
    message_id: str | None = None
    activity_code: str | None = None
    superseded_by: str | None = None
