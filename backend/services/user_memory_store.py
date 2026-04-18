"""
Per-user in-process memory for LangGraph / backend use.

Database-backed persistence is owned by a separate integration; this registry
keeps conversation history, preferences, and privacy fields in memory so the
agent and services can run without SQLAlchemy or a live database.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from backend.services.memory_errors import UserNotFoundError

logger = logging.getLogger(__name__)


@dataclass
class StoredInteraction:
    interaction_id: str
    user_id: str
    session_id: Optional[str]
    query: str
    query_metadata: Dict[str, Any]
    results: List[Dict[str, Any]]
    feedback: Optional[Dict[str, Any]]
    reasoning_trace: Optional[List[str]]
    created_at: datetime
    processing_time_ms: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "interaction_id": self.interaction_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "query": self.query,
            "query_metadata": self.query_metadata,
            "results": self.results,
            "feedback": self.feedback,
            "reasoning_trace": self.reasoning_trace,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "processing_time_ms": self.processing_time_ms,
        }


@dataclass
class PreferencePatternRecord:
    preference_type: str
    preference_value: str
    confidence_score: float
    last_updated: datetime
    user_id_ref: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id_ref,
            "preference_type": self.preference_type,
            "preference_value": self.preference_value,
            "confidence_score": self.confidence_score,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }


@dataclass
class UserProfileRecord:
    user_id: str
    google_id: str = ""
    email: str = ""
    display_name: str = ""
    avatar_url: Optional[str] = None
    preferences: Dict[str, Any] = field(default_factory=dict)
    privacy_settings: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_login: Optional[datetime] = None
    is_active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "google_id": self.google_id,
            "email": self.email,
            "display_name": self.display_name,
            "avatar_url": self.avatar_url,
            "preferences": self.preferences,
            "privacy_settings": self.privacy_settings,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "is_active": self.is_active,
        }


@dataclass
class UserBucket:
    profile: UserProfileRecord
    interactions: List[StoredInteraction] = field(default_factory=list)
    patterns: Dict[Tuple[str, str, str], PreferencePatternRecord] = field(default_factory=dict)


_default_registry: Optional["UserMemoryRegistry"] = None


class UserMemoryRegistry:
    """
    One logical store per user_id (opaque string, e.g. UUID or OAuth-derived id).
    All methods are async and guarded by a lock for safe concurrent use.
    """

    def __init__(self) -> None:
        self._users: Dict[str, UserBucket] = {}
        self._lock = asyncio.Lock()

    async def ensure_user(
        self,
        user_id: str,
        *,
        google_id: str = "",
        email: str = "",
        display_name: str = "",
        avatar_url: Optional[str] = None,
    ) -> UserBucket:
        async with self._lock:
            if user_id not in self._users:
                now = datetime.now(timezone.utc)
                self._users[user_id] = UserBucket(
                    profile=UserProfileRecord(
                        user_id=user_id,
                        google_id=google_id,
                        email=email,
                        display_name=display_name,
                        avatar_url=avatar_url,
                        last_login=now,
                        created_at=now,
                        updated_at=now,
                    )
                )
                logger.debug("Created in-memory user bucket: %s", user_id)
            return self._users[user_id]

    async def get_user_profile(self, user_id: str) -> Optional[UserProfileRecord]:
        async with self._lock:
            b = self._users.get(user_id)
            return b.profile if b else None

    async def update_user_profile(self, user_id: str, updates: Dict[str, Any]) -> bool:
        async with self._lock:
            b = self._users.get(user_id)
            if not b:
                return False
            p = b.profile
            for key, val in updates.items():
                if hasattr(p, key):
                    setattr(p, key, val)
            p.updated_at = datetime.now(timezone.utc)
            return True

    async def delete_user_profile(self, user_id: str) -> bool:
        async with self._lock:
            if user_id not in self._users:
                return False
            del self._users[user_id]
            logger.info("Removed in-memory user bucket: %s", user_id)
            return True

    async def store_interaction(
        self,
        user_id: str,
        query: str,
        results: List[Dict[str, Any]],
        session_id: Optional[str] = None,
        query_metadata: Optional[Dict[str, Any]] = None,
        feedback: Optional[Dict[str, Any]] = None,
        reasoning_trace: Optional[List[str]] = None,
        processing_time_ms: Optional[int] = None,
    ) -> StoredInteraction:
        async with self._lock:
            b = self._get_or_create_bucket_locked(user_id)

            rec = StoredInteraction(
                interaction_id=str(uuid4()),
                user_id=user_id,
                session_id=session_id,
                query=query,
                query_metadata=query_metadata or {},
                results=results,
                feedback=feedback,
                reasoning_trace=reasoning_trace,
                created_at=datetime.now(timezone.utc),
                processing_time_ms=processing_time_ms,
            )
            b.interactions.insert(0, rec)
            return rec

    async def get_user_interactions(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> List[StoredInteraction]:
        async with self._lock:
            b = self._users.get(user_id)
            if not b:
                return []
            return b.interactions[offset : offset + limit]

    async def get_interaction_count(self, user_id: str) -> int:
        async with self._lock:
            b = self._users.get(user_id)
            return len(b.interactions) if b else 0

    async def update_interaction_feedback(
        self,
        interaction_id: str,
        feedback: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> bool:
        async with self._lock:
            buckets = [self._users[uid]] if user_id and user_id in self._users else list(self._users.values())
            for b in buckets:
                for inter in b.interactions:
                    if inter.interaction_id == interaction_id:
                        inter.feedback = feedback
                        return True
            return False

    async def store_preference_pattern(
        self,
        user_id: str,
        preference_type: str,
        preference_value: str,
        confidence_score: float,
    ) -> PreferencePatternRecord:
        async with self._lock:
            b = self._get_or_create_bucket_locked(user_id)
            key = (preference_type, preference_value, user_id)
            now = datetime.now(timezone.utc)
            if key in b.patterns:
                pr = b.patterns[key]
                pr.confidence_score = confidence_score
                pr.last_updated = now
            else:
                pr = PreferencePatternRecord(
                    preference_type=preference_type,
                    preference_value=preference_value,
                    confidence_score=confidence_score,
                    last_updated=now,
                    user_id_ref=user_id,
                )
                b.patterns[key] = pr
            pr.user_id_ref = user_id
            return pr

    async def get_user_preference_patterns(
        self,
        user_id: str,
        preference_type: Optional[str] = None,
    ) -> List[PreferencePatternRecord]:
        async with self._lock:
            b = self._users.get(user_id)
            if not b:
                return []
            out: List[PreferencePatternRecord] = []
            for (pt, pv, uid), pr in b.patterns.items():
                if uid != user_id:
                    continue
                if preference_type and pt != preference_type:
                    continue
                copy = PreferencePatternRecord(
                    preference_type=pr.preference_type,
                    preference_value=pr.preference_value,
                    confidence_score=pr.confidence_score,
                    last_updated=pr.last_updated,
                    user_id_ref=user_id,
                )
                out.append(copy)
            out.sort(key=lambda x: x.confidence_score, reverse=True)
            return out

    async def delete_interactions_before(
        self,
        cutoff_date: datetime,
        user_id: Optional[str] = None,
    ) -> int:
        async with self._lock:
            deleted = 0
            targets = [user_id] if user_id else list(self._users.keys())
            for uid in targets:
                b = self._users.get(uid)
                if not b:
                    continue
                kept: List[StoredInteraction] = []
                for inter in b.interactions:
                    if inter.created_at < cutoff_date:
                        deleted += 1
                    else:
                        kept.append(inter)
                b.interactions = kept
            return deleted

    async def get_user_data_export(self, user_id: str) -> Dict[str, Any]:
        async with self._lock:
            b = self._users.get(user_id)
            if not b:
                raise UserNotFoundError(f"User not found: {user_id}")
            patterns = self._patterns_list_for_user_locked(user_id, b)
            return {
                "user_profile": b.profile.to_dict(),
                "sessions": [],
                "interactions": [i.to_dict() for i in b.interactions],
                "preference_patterns": [p.to_dict() for p in patterns],
            }

    async def list_user_ids(self) -> List[str]:
        async with self._lock:
            return list(self._users.keys())

    async def get_users_scheduled_for_deletion(self) -> List[UserProfileRecord]:
        async with self._lock:
            now = datetime.now(timezone.utc)
            out: List[UserProfileRecord] = []
            for b in self._users.values():
                p = b.profile
                if p.is_active:
                    continue
                raw = (p.privacy_settings or {}).get("scheduled_deletion")
                if not raw:
                    continue
                try:
                    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    if dt <= now:
                        out.append(p)
                except (ValueError, TypeError):
                    continue
            return out

    def _get_or_create_bucket_locked(self, user_id: str) -> UserBucket:
        if user_id not in self._users:
            now = datetime.now(timezone.utc)
            self._users[user_id] = UserBucket(
                profile=UserProfileRecord(
                    user_id=user_id,
                    created_at=now,
                    updated_at=now,
                )
            )
        return self._users[user_id]

    def _patterns_list_for_user_locked(
        self,
        user_id: str,
        b: UserBucket,
    ) -> List[PreferencePatternRecord]:
        out: List[PreferencePatternRecord] = []
        for (pt, pv, uid), pr in b.patterns.items():
            if uid != user_id:
                continue
            out.append(
                PreferencePatternRecord(
                    preference_type=pr.preference_type,
                    preference_value=pr.preference_value,
                    confidence_score=pr.confidence_score,
                    last_updated=pr.last_updated,
                    user_id_ref=user_id,
                )
            )
        return out


def get_default_memory_registry() -> UserMemoryRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = UserMemoryRegistry()
    return _default_registry
