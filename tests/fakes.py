"""Deterministic stand-ins for the embedding model, cross-encoder and LLM."""
import hashlib
import re

import numpy as np

from ragagent import agent as agent_mod

_WORD = re.compile(r"[a-z0-9]+")
DIM = 128


def _vec(text):
    v = np.zeros(DIM, dtype=np.float32)
    for w in _WORD.findall(text.lower()):
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v


class FakeEncoder:
    def encode(self, texts, **kwargs):
        return np.stack([_vec(t) for t in texts])


class FakeCrossEncoder:
    def predict(self, pairs, **kwargs):
        return np.array([float(_vec(q) @ _vec(d)) for q, d in pairs], dtype=np.float32)


CORPUS = [
    ("Statins and breast cancer", "Statin use was not associated with higher breast cancer risk."),
    ("Statin survival", "Breast cancer patients using statins showed improved survival."),
    ("Vitamin D and bones", "Vitamin D deficiency causes bone loss and fractures."),
    ("Vitamin D immunity", "Low vitamin D impairs immune function."),
    ("Fibre and colon", "Dietary fibre lowers colon cancer risk."),
    ("Coffee and liver", "Coffee drinking is linked to lower liver enzyme levels."),
]


class FakeLLM:
    """Replies according to which agent prompt it receives; records every call."""

    def __init__(self, route="retrieve", aspects=None, relevant=None, rewrite="vitamin D bones",
                 answer="Statins do not raise breast cancer risk [1]. They may improve survival [2].",
                 unsupported=()):
        self.route, self.aspects, self.relevant = route, aspects, relevant
        self.rewrite_query, self.answer, self.unsupported = rewrite, answer, list(unsupported)
        self.calls = 0
        self.log = []

    def json(self, system, user):
        self.calls += 1
        self.log.append(system[:20])
        if system == agent_mod.PLAN_SYSTEM:
            return {"route": self.route, "aspects": self.aspects or [], "reply": "Hi! Ask me about nutrition."}
        if system == agent_mod.GRADE_SYSTEM:
            n = user.count("\n[") + 1
            return {"relevant": self.relevant if self.relevant is not None else list(range(1, n + 1))}
        if system == agent_mod.REWRITE_SYSTEM:
            return {"query": self.rewrite_query}
        if system == agent_mod.VERIFY_SYSTEM:
            return {"unsupported": self.unsupported}
        raise AssertionError("unexpected prompt")

    def text(self, system, user):
        self.calls += 1
        self.log.append(system[:20])
        return self.answer
