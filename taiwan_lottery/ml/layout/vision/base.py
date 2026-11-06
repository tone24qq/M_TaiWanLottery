"""One vote per engine, not per image augmentation."""
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import re


def numeric(text, maximum):
    # No stripping arbitrary text into a plausible number.
    text = str(text).strip()
    return int(text) if re.fullmatch(r"[0-9]{1,3}", text) and 1 <= int(text) <= maximum else None


@dataclass
class EngineResult:
    engine: str
    candidates: list[dict] = field(default_factory=list)
    raw_text: list[str] = field(default_factory=list)
    quality: dict = field(default_factory=dict)
    status: str = "ready"
    reason: str | None = None

    def as_dict(self):
        from dataclasses import asdict
        return asdict(self)


class BaseCellRecognizer(ABC):
    name = "base"
    family = "base"
    status = "not_ready"
    reason = None
    load_seconds = 0.0
    artifact_bytes = 0

    @abstractmethod
    def recognize(self, variants, maximum, top_k=5): ...

    def fingerprint(self):
        import hashlib,json
        return hashlib.sha256(json.dumps([self.name,getattr(self,"version",None),
            getattr(self,"artifact_digest",None)],sort_keys=True).encode()).hexdigest()

    def metadata(self):
        return {"fingerprint":self.fingerprint(),"version":getattr(self,"version",None),"engine":self.name,"family":self.family,"status":self.status,
                "reason":self.reason,"load_seconds":self.load_seconds,
                "artifact_bytes":self.artifact_bytes}
