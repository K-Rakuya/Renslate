"""Directed relationship knowledge graph with bilateral O(1) lookup and dual persistence."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from collections import deque
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from renslate.core.models import RelationshipEdge
from renslate.core.textio import write_atomic

if TYPE_CHECKING:
    from renslate.core.db import Database

_LOG = logging.getLogger("renslate.engine.graph")


class RelationshipGraph:
    """Directed graph representing character relationships, pronoun mappings, and interactions.

    Internal Invariants:
    - `_out_edges[src][tgt]` stores the edge directed from `src` to `tgt`.
    - `_in_edges[tgt][src]` stores the exact same `RelationshipEdge` instance for
      O(1) reverse lookup.
    - `_nodes` contains all registered character tags (including isolated nodes).
    - `_interactions[spk][lis]` stores the directed dialogue turn count from `spk` to `lis`.
    - All character tags are normalized (whitespace-stripped).
    """

    def __init__(self) -> None:
        self._out_edges: dict[str, dict[str, RelationshipEdge]] = {}
        self._in_edges: dict[str, dict[str, RelationshipEdge]] = {}
        self._nodes: set[str] = set()
        self._interactions: dict[str, dict[str, int]] = {}

    def add_node(self, tag: str) -> None:
        """Register a character tag as a node in the graph."""
        t = tag.strip()
        if not t:
            return
        self._nodes.add(t)
        self._out_edges.setdefault(t, {})
        self._in_edges.setdefault(t, {})

    def has_node(self, tag: str) -> bool:
        """Check if character tag is registered in the graph."""
        return tag.strip() in self._nodes

    def remove_node(self, tag: str) -> bool:
        """Remove a character node and cascade removal of all connected edges and interactions."""
        t = tag.strip()
        if t not in self._nodes:
            return False

        # Cascade remove outgoing edges
        for tgt in list(self._out_edges.get(t, {}).keys()):
            if tgt in self._in_edges and t in self._in_edges[tgt]:
                del self._in_edges[tgt][t]
        self._out_edges.pop(t, None)

        # Cascade remove incoming edges
        for src in list(self._in_edges.get(t, {}).keys()):
            if src in self._out_edges and t in self._out_edges[src]:
                del self._out_edges[src][t]
        self._in_edges.pop(t, None)

        # Clean interactions
        self._interactions.pop(t, None)
        for spk in self._interactions:
            self._interactions[spk].pop(t, None)

        self._nodes.remove(t)
        return True

    def add_edge(self, edge: RelationshipEdge, overwrite_manual: bool = False) -> bool:
        """Add or update a directed relationship edge in the graph.

        If an existing edge has source_type == 'manual' and incoming edge has
        source_type != 'manual', the update is rejected unless overwrite_manual=True.
        """
        if not isinstance(edge, RelationshipEdge):
            msg = f"Expected RelationshipEdge, got {type(edge).__name__}"
            raise TypeError(msg)

        src = edge.source_tag.strip()
        tgt = edge.target_tag.strip()
        if not src or not tgt:
            msg = "RelationshipEdge source_tag and target_tag must not be empty."
            raise ValueError(msg)

        existing = self.get_edge(src, tgt)
        if (
            existing is not None
            and existing.source_type == "manual"
            and edge.source_type != "manual"
            and not overwrite_manual
        ):
            _LOG.debug("Preserving manual edge %s -> %s against automated overwrite", src, tgt)
            return False

        self.add_node(src)
        self.add_node(tgt)

        if src != edge.source_tag or tgt != edge.target_tag:
            edge = dataclasses.replace(edge, source_tag=src, target_tag=tgt)

        self._out_edges.setdefault(src, {})[tgt] = edge
        self._in_edges.setdefault(tgt, {})[src] = edge
        return True

    def remove_edge(self, source_tag: str, target_tag: str) -> bool:
        """Remove a directed edge from source_tag to target_tag."""
        src = source_tag.strip()
        tgt = target_tag.strip()
        if not self.has_edge(src, tgt):
            return False

        del self._out_edges[src][tgt]
        del self._in_edges[tgt][src]
        return True

    def has_edge(self, source_tag: str, target_tag: str) -> bool:
        """Check if a directed edge exists from source_tag to target_tag."""
        src = source_tag.strip()
        tgt = target_tag.strip()
        return src in self._out_edges and tgt in self._out_edges[src]

    def get_edge(self, source_tag: str, target_tag: str) -> RelationshipEdge | None:
        """Retrieve the directed edge from source_tag to target_tag, or None."""
        return self._out_edges.get(source_tag.strip(), {}).get(target_tag.strip())

    def get_bilateral(
        self, a: str, b: str
    ) -> tuple[RelationshipEdge | None, RelationshipEdge | None]:
        """Retrieve bilateral edges (a -> b, b -> a) in O(1) time."""
        return (self.get_edge(a, b), self.get_edge(b, a))

    def get_neighbors(self, tag: str, direction: str = "both") -> set[str]:
        """Retrieve neighboring character tags connected by relationship edges."""
        t = tag.strip()
        if direction == "out":
            return set(self._out_edges.get(t, {}).keys())
        if direction == "in":
            return set(self._in_edges.get(t, {}).keys())
        if direction == "both":
            return set(self._out_edges.get(t, {}).keys()) | set(self._in_edges.get(t, {}).keys())
        msg = f"Invalid direction: '{direction}'. Must be 'out', 'in', or 'both'."
        raise ValueError(msg)

    def all_edges(self) -> list[RelationshipEdge]:
        """Return a flat list of all directed relationship edges in the graph."""
        return [edge for out_map in self._out_edges.values() for edge in out_map.values()]

    @property
    def edges(self) -> list[RelationshipEdge]:
        """Property returning all directed relationship edges."""
        return self.all_edges()

    @property
    def nodes(self) -> frozenset[str]:
        """Immutable view of all registered character nodes."""
        return frozenset(self._nodes)

    @property
    def node_count(self) -> int:
        """Total number of registered character nodes."""
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        """Total number of directed relationship edges."""
        return sum(len(m) for m in self._out_edges.values())

    def record_interaction(self, speaker: str, listener: str, count: int = 1) -> None:
        """Record turn-taking dialogue interactions between speaker and listener."""
        spk = speaker.strip()
        lis = listener.strip()
        if not spk or not lis or count <= 0:
            return

        self.add_node(spk)
        self.add_node(lis)

        spk_map = self._interactions.setdefault(spk, {})
        spk_map[lis] = spk_map.get(lis, 0) + count

    def get_interaction_count(self, speaker: str, listener: str) -> int:
        """Get the directed number of turns where speaker addressed listener."""
        return self._interactions.get(speaker.strip(), {}).get(listener.strip(), 0)

    def get_bilateral_interaction_count(self, a: str, b: str) -> int:
        """Get the mutual dialogue turn count between a and b."""
        return self.get_interaction_count(a, b) + self.get_interaction_count(b, a)

    def get_total_interactions(self, tag: str) -> int:
        """Return the total dialogue turns involving tag as speaker or listener."""
        t = tag.strip()
        out_turns = sum(self._interactions.get(t, {}).values())
        in_turns = sum(
            targets.get(t, 0)
            for spk, targets in self._interactions.items()
            if spk != t
        )
        return out_turns + in_turns

    def top_interlocutors(self, tag: str, limit: int = 5) -> list[tuple[str, int]]:
        """Return top dialogue partners for tag sorted by bilateral turn frequency."""
        if limit <= 0:
            return []
        t = tag.strip()
        partners: dict[str, int] = {}
        for other in self._nodes:
            if other == t:
                continue
            cnt = self.get_bilateral_interaction_count(t, other)
            if cnt > 0:
                partners[other] = cnt
        sorted_partners = sorted(partners.items(), key=lambda kv: kv[1], reverse=True)
        return sorted_partners[:limit]

    @staticmethod
    def _backtrack_path(
        parent_map: dict[str, tuple[str, RelationshipEdge]],
        start: str,
        target: str,
    ) -> list[RelationshipEdge]:
        path: list[RelationshipEdge] = []
        curr = target
        while curr != start:
            prev, edge = parent_map[curr]
            path.append(edge)
            curr = prev
        path.reverse()
        return path

    def find_path(
        self,
        src: str,
        tgt: str,
        max_depth: int | None = None,
        directed: bool = True,
    ) -> list[RelationshipEdge]:
        """Find the shortest relational path from src to tgt using BFS."""
        s = src.strip()
        t = tgt.strip()
        if s == t or s not in self._nodes or t not in self._nodes:
            return []
        if max_depth is not None and max_depth <= 0:
            return []

        queue: deque[tuple[str, int]] = deque([(s, 0)])
        parent_map: dict[str, tuple[str, RelationshipEdge]] = {}

        while queue:
            curr, depth = queue.popleft()
            if max_depth is not None and depth >= max_depth:
                continue

            for neighbor, edge in self._out_edges.get(curr, {}).items():
                if neighbor not in parent_map and neighbor != s:
                    parent_map[neighbor] = (curr, edge)
                    if neighbor == t:
                        return self._backtrack_path(parent_map, s, t)
                    queue.append((neighbor, depth + 1))

            if not directed:
                for neighbor, edge in self._in_edges.get(curr, {}).items():
                    if neighbor not in parent_map and neighbor != s:
                        parent_map[neighbor] = (curr, edge)
                        if neighbor == t:
                            return self._backtrack_path(parent_map, s, t)
                        queue.append((neighbor, depth + 1))

        return []

    def subgraph(self, tags: Iterable[str]) -> RelationshipGraph:
        """Extract an induced subgraph containing only the specified character tags."""
        sub = RelationshipGraph()
        tag_set = {t.strip() for t in tags if t.strip() in self._nodes}

        for t in tag_set:
            sub.add_node(t)

        for src in tag_set:
            for tgt, edge in self._out_edges.get(src, {}).items():
                if tgt in tag_set:
                    sub.add_edge(edge)

        for spk in tag_set:
            for lis, cnt in self._interactions.get(spk, {}).items():
                if lis in tag_set:
                    sub.record_interaction(spk, lis, cnt)

        return sub

    def to_dict(self) -> dict[str, Any]:
        """Serialize graph to dictionary for JSON persistence."""
        return {
            "version": 1,
            "nodes": sorted(self._nodes),
            "relationships": [
                edge.to_dict()
                for edge in sorted(self.all_edges(), key=lambda e: (e.source_tag, e.target_tag))
            ],
            "interactions": {
                spk: dict(sorted(targets.items()))
                for spk, targets in sorted(self._interactions.items())
                if targets
            },
        }

    @classmethod
    def from_dict(cls, data: Any) -> RelationshipGraph:
        """Deserialize graph from dictionary or list."""
        graph = cls()

        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    msg = f"Expected dict for relationship item, got {type(item).__name__}"
                    raise TypeError(msg)
                graph.add_edge(RelationshipEdge.from_dict(item), overwrite_manual=True)
            return graph

        if isinstance(data, dict):
            # Rich format with nodes, relationships, and/or interactions
            if "nodes" in data and isinstance(data["nodes"], list):
                for node in data["nodes"]:
                    graph.add_node(str(node))

            if "relationships" in data and isinstance(data["relationships"], list):
                for rel in data["relationships"]:
                    if not isinstance(rel, dict):
                        msg = f"Expected dict for relationship item, got {type(rel).__name__}"
                        raise TypeError(msg)
                    graph.add_edge(RelationshipEdge.from_dict(rel), overwrite_manual=True)

            if "interactions" in data and isinstance(data["interactions"], dict):
                for spk, targets in data["interactions"].items():
                    if isinstance(targets, dict):
                        for lis, cnt in targets.items():
                            if isinstance(cnt, (int, float)) and cnt > 0:
                                graph.record_interaction(str(spk), str(lis), int(cnt))

            # Legacy / flat adjacency map format: {"mc": {"liana": {"kind": ...}}}
            if "relationships" not in data and "nodes" not in data and "interactions" not in data:
                for src, targets in data.items():
                    if isinstance(targets, dict):
                        for tgt, fields in targets.items():
                            if isinstance(fields, dict):
                                d = {"source_tag": str(src), "target_tag": str(tgt), **fields}
                                graph.add_edge(RelationshipEdge.from_dict(d), overwrite_manual=True)
            return graph

        msg = f"Expected dict or list for RelationshipGraph, got {type(data).__name__}"
        raise TypeError(msg)

    def save_to_json_file(self, path: Path | str, indent: int = 2) -> None:
        """Save graph to JSON file atomically."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        payload = (json.dumps(data, ensure_ascii=False, indent=indent) + "\n").encode("utf-8")
        write_atomic(p, payload)

    @classmethod
    def from_json_file(cls, path: Path | str) -> RelationshipGraph:
        """Load graph from JSON file."""
        p = Path(path)
        if not p.is_file():
            return cls()
        content = p.read_text(encoding="utf-8").strip()
        if not content:
            return cls()
        data = json.loads(content)
        return cls.from_dict(data)

    def save_to_db(self, db: Database, *, clear_existing: bool = False) -> None:
        """Save relationship edges to SQLite database."""
        edges: Sequence[RelationshipEdge] = self.all_edges()
        if clear_existing:
            with db.transaction():
                db.clear_relationships()
                db.save_relationships_bulk(edges)
        else:
            db.save_relationships_bulk(edges)

    @classmethod
    def from_db(cls, db: Database) -> RelationshipGraph:
        """Reconstruct graph from SQLite database."""
        graph = cls()
        for edge in db.get_relationships():
            graph.add_edge(edge, overwrite_manual=True)
        return graph

    @classmethod
    def sync(
        cls,
        db: Database,
        json_path: Path | str,
        *,
        direction: str = "auto",
    ) -> RelationshipGraph:
        """Bidirectionally synchronize relationships between SQLite and JSON file.

        Supported directions:
        - "auto": Intelligent two-way merge using content hash and timestamps.
        - "db_to_json": Explicit unidirectional sync from SQLite DB to JSON file.
        - "json_to_db": Explicit unidirectional sync from JSON file to SQLite DB.

        Raises:
            ValueError: If direction is unrecognized.
            FileNotFoundError: If direction is "json_to_db" and json_path does not exist.
        """
        p = Path(json_path)
        if direction not in ("auto", "db_to_json", "json_to_db"):
            msg = (
                f"Invalid direction: '{direction}'. "
                "Must be 'auto', 'db_to_json', or 'json_to_db'."
            )
            raise ValueError(msg)

        if direction == "db_to_json":
            db_graph = cls.from_db(db)
            if p.is_file():
                json_graph = cls.from_json_file(p)
                for spk, targets in json_graph._interactions.items():
                    for lis, cnt in targets.items():
                        if cnt > 0:
                            db_graph.record_interaction(spk, lis, cnt)
                for node in json_graph.nodes:
                    db_graph.add_node(node)
            db_graph.save_to_json_file(p)
            content = p.read_bytes()
            db.set_meta("relationships_json_hash", hashlib.sha256(content).hexdigest())
            db.set_meta("relationships_updated_at", datetime.now(timezone.utc).isoformat())
            return db_graph

        if direction == "json_to_db":
            if not p.is_file():
                msg = f"Cannot sync json_to_db: JSON file not found at {p}"
                raise FileNotFoundError(msg)
            json_graph = cls.from_json_file(p)
            json_graph.save_to_db(db, clear_existing=True)
            content = p.read_bytes()
            db.set_meta("relationships_json_hash", hashlib.sha256(content).hexdigest())
            db.set_meta("relationships_updated_at", datetime.now(timezone.utc).isoformat())
            return json_graph

        # direction == "auto"
        if not p.is_file():
            db_graph = cls.from_db(db)
            if db_graph.edge_count > 0:
                db_graph.save_to_json_file(p)
                content = p.read_bytes()
                db.set_meta("relationships_json_hash", hashlib.sha256(content).hexdigest())
                db.set_meta("relationships_updated_at", datetime.now(timezone.utc).isoformat())
            return db_graph

        json_graph = cls.from_json_file(p)
        db_graph = cls.from_db(db)

        if db_graph.edge_count == 0 and json_graph.edge_count > 0:
            json_graph.save_to_db(db)
            content = p.read_bytes()
            db.set_meta("relationships_json_hash", hashlib.sha256(content).hexdigest())
            db.set_meta("relationships_updated_at", datetime.now(timezone.utc).isoformat())
            return json_graph

        if json_graph.edge_count == 0 and db_graph.edge_count > 0:
            db_graph.save_to_json_file(p)
            content = p.read_bytes()
            db.set_meta("relationships_json_hash", hashlib.sha256(content).hexdigest())
            db.set_meta("relationships_updated_at", datetime.now(timezone.utc).isoformat())
            return db_graph

        merged = cls()

        # Merge nodes
        for node in json_graph.nodes | db_graph.nodes:
            merged.add_node(node)

        # Detect whether JSON on disk was modified externally or DB was updated
        stored_hash = db.get_meta("relationships_json_hash")
        current_content = p.read_bytes()
        current_hash = hashlib.sha256(current_content).hexdigest()

        db_updated_meta = db.get_meta("relationships_updated_at")
        try:
            db_ts = (
                datetime.fromisoformat(db_updated_meta.replace("Z", "+00:00")).timestamp()
                if db_updated_meta
                else 0.0
            )
        except Exception:
            db_ts = 0.0
        json_ts = p.stat().st_mtime

        if stored_hash is not None:
            # If JSON on disk unchanged since last sync, DB wins; else check recency
            json_wins_tie = False if current_hash == stored_hash else db_ts <= json_ts + 0.005
        else:
            # Fall back to timestamp comparison
            json_wins_tie = json_ts > db_ts + 0.005

        # Merge edges with symmetric human curation protection & conflict resolution
        all_edge_pairs = {
            (e.source_tag, e.target_tag)
            for e in json_graph.all_edges() + db_graph.all_edges()
        }
        for src, tgt in all_edge_pairs:
            j_edge = json_graph.get_edge(src, tgt)
            d_edge = db_graph.get_edge(src, tgt)

            if j_edge is not None and d_edge is None:
                merged.add_edge(j_edge, overwrite_manual=True)
            elif d_edge is not None and j_edge is None:
                merged.add_edge(d_edge, overwrite_manual=True)
            elif j_edge is not None and d_edge is not None:
                # 1. Human manual curation protection (symmetric) & confidence precedence
                if j_edge.source_type == "manual" and d_edge.source_type != "manual":
                    merged.add_edge(j_edge, overwrite_manual=True)
                elif (
                    (d_edge.source_type == "manual" and j_edge.source_type != "manual")
                    or d_edge.confidence > j_edge.confidence
                ):
                    merged.add_edge(d_edge, overwrite_manual=True)
                elif j_edge.confidence > d_edge.confidence or json_wins_tie:
                    merged.add_edge(j_edge, overwrite_manual=True)
                else:
                    merged.add_edge(d_edge, overwrite_manual=True)

        # Merge interaction frequencies
        all_speakers = set(json_graph._interactions.keys()) | set(db_graph._interactions.keys())
        for spk in all_speakers:
            j_targets = json_graph._interactions.get(spk, {})
            d_targets = db_graph._interactions.get(spk, {})
            all_listeners = set(j_targets.keys()) | set(d_targets.keys())
            for lis in all_listeners:
                cnt = max(j_targets.get(lis, 0), d_targets.get(lis, 0))
                if cnt > 0:
                    merged.record_interaction(spk, lis, cnt)

        merged.save_to_db(db)
        merged.save_to_json_file(p)
        new_content = p.read_bytes()
        db.set_meta("relationships_json_hash", hashlib.sha256(new_content).hexdigest())
        db.set_meta("relationships_updated_at", datetime.now(timezone.utc).isoformat())
        return merged
