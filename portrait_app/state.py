from __future__ import annotations

import json
import math
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PortraitState:
    def __init__(self, data_dir: Path):
        self.path = data_dir / "portrait.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        if not self.path.exists():
            self._write({"people": [], "relations": [], "generation_history": [], "face_descriptors": {}})

    def _read(self) -> dict:
        with self._lock:
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError):
                return {"people": [], "relations": [], "generation_history": [], "face_descriptors": {}}

    def _write(self, data: dict) -> None:
        with self._lock:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.path)

    def snapshot(self) -> dict:
        return self._read()

    def create_person(self, name: str, image_id: str | None = None, descriptor: list[float] | None = None) -> dict:
        data = self._read()
        if image_id and not descriptor:
            descriptor = data.setdefault("face_descriptors", {}).get(image_id)
        person = {
            "id": uuid.uuid4().hex,
            "name": name.strip(),
            "aliases": [],
            "notes": "",
            "photo_ids": [image_id] if image_id else [],
            "descriptors": [descriptor] if descriptor else [],
            "created_at": _now(),
            "updated_at": _now(),
        }
        data["people"].append(person)
        self._write(data)
        return person

    def remember_face(self, image_id: str, descriptor: list[float]) -> None:
        data = self._read()
        descriptors = data.setdefault("face_descriptors", {})
        descriptors[image_id] = descriptor
        # Bound metadata growth while preserving every image attached to a
        # named person. Old unreviewed samples are the first to go.
        protected = {photo for person in data["people"] for photo in person.get("photo_ids", [])}
        while len(descriptors) > 1000:
            removable = next((key for key in descriptors if key not in protected), None)
            if removable is None:
                break
            descriptors.pop(removable, None)
        self._write(data)

    def update_person(self, person_id: str, changes: dict) -> dict | None:
        data = self._read()
        for person in data["people"]:
            if person["id"] != person_id:
                continue
            for key in ("name", "aliases", "notes"):
                if key in changes:
                    person[key] = changes[key]
            person["updated_at"] = _now()
            self._write(data)
            return person
        return None

    def add_sample(self, person_id: str, image_id: str, descriptor: list[float] | None) -> dict | None:
        data = self._read()
        if not descriptor:
            descriptor = data.setdefault("face_descriptors", {}).get(image_id)
        for person in data["people"]:
            if person["id"] != person_id:
                continue
            if image_id not in person["photo_ids"]:
                person["photo_ids"].append(image_id)
            if descriptor:
                person["descriptors"] = (person.get("descriptors", []) + [descriptor])[-12:]
            person["updated_at"] = _now()
            self._write(data)
            return person
        return None

    def match(self, descriptor: list[float], threshold: float = 0.91) -> tuple[dict | None, float]:
        if not descriptor:
            return None, 0.0
        best, best_score = None, -1.0
        for person in self._read()["people"]:
            for known in person.get("descriptors", []):
                if len(known) != len(descriptor):
                    continue
                dot = sum(a * b for a, b in zip(known, descriptor))
                norm = math.sqrt(sum(a * a for a in known) * sum(b * b for b in descriptor))
                score = dot / norm if norm else 0.0
                if score > best_score:
                    best, best_score = person, score
        return (best if best_score >= threshold else None), max(0.0, best_score)

    def save_relation(self, source_id: str, target_id: str, kind: str, notes: str = "") -> dict:
        data = self._read()
        existing = next((r for r in data["relations"] if r["source_id"] == source_id and r["target_id"] == target_id), None)
        if existing:
            existing.update(kind=kind.strip(), notes=notes.strip(), updated_at=_now())
            relation = existing
        else:
            relation = {"id": uuid.uuid4().hex, "source_id": source_id, "target_id": target_id, "kind": kind.strip(), "notes": notes.strip(), "created_at": _now(), "updated_at": _now()}
            data["relations"].append(relation)
        self._write(data)
        return relation

    def log_generation(self, entry: dict) -> None:
        data = self._read()
        data["generation_history"] = ([{"id": uuid.uuid4().hex, "created_at": _now(), **entry}] + data["generation_history"])[:100]
        self._write(data)
