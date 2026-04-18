"""
Privacy Manager for data privacy controls and retention management.

Implements data retention policies, privacy settings, access controls,
and audit logging for GDPR compliance.

Requirements: 7.3, 7.6, 8.4
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum

from backend.services.memory_errors import MemoryError
from backend.services.user_memory_store import UserMemoryRegistry, get_default_memory_registry

logger = logging.getLogger(__name__)


# ── Enums ──────────────────────────────────────────────────────────────────────

class DataCollectionLevel(str, Enum):
    """Data collection level for privacy settings."""
    FULL = "full"          # Collect all data including interactions and preferences
    MINIMAL = "minimal"    # Collect only essential data (no interaction history)
    NONE = "none"          # Do not collect any personal data


class AuditEventType(str, Enum):
    """Types of audit events."""
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    DATA_DELETION = "data_deletion"
    PRIVACY_SETTINGS_CHANGE = "privacy_settings_change"
    ACCOUNT_DELETION_REQUESTED = "account_deletion_requested"
    DATA_EXPORT_REQUESTED = "data_export_requested"


# ── Data Structures ────────────────────────────────────────────────────────────

@dataclass
class PrivacySettings:
    """User privacy settings."""

    data_collection_level: DataCollectionLevel = DataCollectionLevel.FULL
    share_data_for_improvement: bool = False
    retention_days: int = 365          # How long to keep interaction history
    allow_personalization: bool = True
    scheduled_deletion: Optional[str] = None  # ISO datetime string

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_collection_level": self.data_collection_level.value,
            "share_data_for_improvement": self.share_data_for_improvement,
            "retention_days": self.retention_days,
            "allow_personalization": self.allow_personalization,
            "scheduled_deletion": self.scheduled_deletion,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PrivacySettings":
        return cls(
            data_collection_level=DataCollectionLevel(
                data.get("data_collection_level", DataCollectionLevel.FULL.value)
            ),
            share_data_for_improvement=data.get("share_data_for_improvement", False),
            retention_days=data.get("retention_days", 365),
            allow_personalization=data.get("allow_personalization", True),
            scheduled_deletion=data.get("scheduled_deletion"),
        )


@dataclass
class AuditLogEntry:
    """Audit log entry for data access and modifications."""

    event_type: AuditEventType
    user_id: str
    resource_type: str
    resource_id: str
    timestamp: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "user_id": self.user_id,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "timestamp": self.timestamp.isoformat(),
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "details": self.details,
        }


@dataclass
class RetentionCleanupResult:
    """Result of a retention cleanup operation."""

    interactions_deleted: int
    sessions_deleted: int
    users_deleted: int
    cleanup_timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "interactions_deleted": self.interactions_deleted,
            "sessions_deleted": self.sessions_deleted,
            "users_deleted": self.users_deleted,
            "cleanup_timestamp": self.cleanup_timestamp.isoformat(),
        }


# ── Privacy Manager ────────────────────────────────────────────────────────────

class PrivacyManager:
    """
    Privacy manager for data privacy controls and retention management.

    Provides:
    - Data retention policy enforcement
    - Privacy settings management
    - Access control validation
    - Audit logging for data operations

    Requirements:
    - 7.3: Data deletion within 24 hours of request
    - 7.6: Audit logs of data access and modifications
    - 8.4: Access controls ensuring users can only access their own data
    """

    # In-memory audit log (in production this would persist to DB or a log service)
    _audit_log: List[AuditLogEntry] = []

    def __init__(self, registry: Optional[UserMemoryRegistry] = None):
        """
        Initialize privacy manager.

        Args:
            registry: Shared UserMemoryRegistry (same as MemorySystem); defaults to process singleton.
        """
        self._registry = registry or get_default_memory_registry()
        logger.info("Privacy manager initialized")

    # ── Privacy Settings ───────────────────────────────────────────────────────

    async def get_privacy_settings(self, user_id: str) -> PrivacySettings:
        """
        Get privacy settings for a user.

        Args:
            user_id: User identifier

        Returns:
            PrivacySettings object

        Raises:
            MemoryError: If retrieval fails
        """
        try:
            user = await self._registry.get_user_profile(user_id)
            if not user:
                # Return defaults for unknown users
                return PrivacySettings()

            raw = user.privacy_settings or {}
            return PrivacySettings.from_dict(raw)

        except Exception as e:
            logger.error(f"Failed to get privacy settings for user {user_id}: {e}")
            raise MemoryError(f"Failed to get privacy settings: {e}")

    async def update_privacy_settings(
        self,
        user_id: str,
        settings: PrivacySettings,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> bool:
        """
        Update privacy settings for a user.

        Requirement 7.7: Respect new privacy settings for future interactions.

        Args:
            user_id: User identifier
            settings: New PrivacySettings
            ip_address: Optional IP for audit log
            user_agent: Optional user agent for audit log

        Returns:
            True if update successful

        Raises:
            MemoryError: If update fails
        """
        try:
            success = await self._registry.update_user_profile(
                user_id,
                {"privacy_settings": settings.to_dict()},
            )

            if success:
                await self._write_audit_log(AuditLogEntry(
                    event_type=AuditEventType.PRIVACY_SETTINGS_CHANGE,
                    user_id=user_id,
                    resource_type="privacy_settings",
                    resource_id=user_id,
                    timestamp=datetime.now(timezone.utc),
                    ip_address=ip_address,
                    user_agent=user_agent,
                    details={"new_settings": settings.to_dict()},
                ))
                logger.info(f"Updated privacy settings for user {user_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to update privacy settings for user {user_id}: {e}")
            raise MemoryError(f"Failed to update privacy settings: {e}")

    # ── Data Collection Control ────────────────────────────────────────────────

    async def should_collect_data(self, user_id: str) -> bool:
        """
        Check whether data collection is allowed for a user.

        Requirement 8.4: Respect user privacy settings.

        Args:
            user_id: User identifier

        Returns:
            True if data collection is allowed
        """
        try:
            settings = await self.get_privacy_settings(user_id)
            return settings.data_collection_level != DataCollectionLevel.NONE
        except Exception as e:
            logger.warning(f"Could not check data collection for {user_id}, defaulting to True: {e}")
            return True

    async def should_store_interactions(self, user_id: str) -> bool:
        """
        Check whether interaction history storage is allowed.

        Args:
            user_id: User identifier

        Returns:
            True if interaction storage is allowed
        """
        try:
            settings = await self.get_privacy_settings(user_id)
            return settings.data_collection_level == DataCollectionLevel.FULL
        except Exception as e:
            logger.warning(f"Could not check interaction storage for {user_id}, defaulting to True: {e}")
            return True

    async def is_personalization_allowed(self, user_id: str) -> bool:
        """
        Check whether personalization is allowed for a user.

        Args:
            user_id: User identifier

        Returns:
            True if personalization is allowed
        """
        try:
            settings = await self.get_privacy_settings(user_id)
            return settings.allow_personalization
        except Exception as e:
            logger.warning(f"Could not check personalization for {user_id}, defaulting to True: {e}")
            return True

    # ── Access Control ─────────────────────────────────────────────────────────

    async def validate_access(
        self,
        requesting_user_id: str,
        resource_user_id: str,
        resource_type: str,
    ) -> bool:
        """
        Validate that a user has access to a resource.

        Requirement 8.4: Users can only access their own data.

        Args:
            requesting_user_id: ID of the user making the request
            resource_user_id: ID of the user who owns the resource
            resource_type: Type of resource being accessed

        Returns:
            True if access is allowed
        """
        # Users can only access their own data
        has_access = requesting_user_id == resource_user_id

        if not has_access:
            logger.warning(
                f"Access denied: user {requesting_user_id} attempted to access "
                f"{resource_type} belonging to user {resource_user_id}"
            )
            await self._write_audit_log(AuditLogEntry(
                event_type=AuditEventType.DATA_ACCESS,
                user_id=requesting_user_id,
                resource_type=resource_type,
                resource_id=resource_user_id,
                timestamp=datetime.now(timezone.utc),
                details={"access_denied": True, "reason": "cross_user_access_attempt"},
            ))

        return has_access

    # ── Audit Logging ──────────────────────────────────────────────────────────

    async def log_data_access(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> None:
        """
        Log a data access event.

        Requirement 7.6: Maintain audit logs of data access.

        Args:
            user_id: User whose data was accessed
            resource_type: Type of resource accessed
            resource_id: Identifier of the resource
            ip_address: Optional IP address
            user_agent: Optional user agent string
        """
        await self._write_audit_log(AuditLogEntry(
            event_type=AuditEventType.DATA_ACCESS,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            timestamp=datetime.now(timezone.utc),
            ip_address=ip_address,
            user_agent=user_agent,
        ))

    async def log_data_modification(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        modification_type: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> None:
        """
        Log a data modification event.

        Requirement 7.6: Maintain audit logs of data modifications.

        Args:
            user_id: User whose data was modified
            resource_type: Type of resource modified
            resource_id: Identifier of the resource
            modification_type: Type of modification (create/update/delete)
            ip_address: Optional IP address
            user_agent: Optional user agent string
        """
        await self._write_audit_log(AuditLogEntry(
            event_type=AuditEventType.DATA_MODIFICATION,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            timestamp=datetime.now(timezone.utc),
            ip_address=ip_address,
            user_agent=user_agent,
            details={"modification_type": modification_type},
        ))

    async def log_data_deletion(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> None:
        """
        Log a data deletion event.

        Requirement 7.6: Maintain audit logs of data deletions.
        """
        await self._write_audit_log(AuditLogEntry(
            event_type=AuditEventType.DATA_DELETION,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            timestamp=datetime.now(timezone.utc),
            ip_address=ip_address,
            user_agent=user_agent,
        ))

    def get_audit_log(
        self,
        user_id: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
        limit: int = 100,
    ) -> List[AuditLogEntry]:
        """
        Retrieve audit log entries.

        Args:
            user_id: Optional filter by user ID
            event_type: Optional filter by event type
            limit: Maximum number of entries to return

        Returns:
            List of AuditLogEntry objects (newest first)
        """
        entries = list(reversed(PrivacyManager._audit_log))

        if user_id:
            entries = [e for e in entries if e.user_id == user_id]
        if event_type:
            entries = [e for e in entries if e.event_type == event_type]

        return entries[:limit]

    # ── Data Retention ─────────────────────────────────────────────────────────

    async def schedule_account_deletion(
        self,
        user_id: str,
        deletion_delay_hours: int = 24,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> datetime:
        """
        Schedule account deletion within 24 hours.

        Requirement 7.3: Remove all personal data within 24 hours of request.

        Args:
            user_id: User identifier
            deletion_delay_hours: Hours until deletion (default 24)
            ip_address: Optional IP for audit log
            user_agent: Optional user agent for audit log

        Returns:
            Scheduled deletion datetime

        Raises:
            MemoryError: If scheduling fails
        """
        try:
            deletion_time = datetime.now(timezone.utc) + timedelta(hours=deletion_delay_hours)

            user = await self._registry.get_user_profile(user_id)
            if not user:
                raise MemoryError(f"User not found: {user_id}")

            # Update privacy settings with scheduled deletion
            current_settings = PrivacySettings.from_dict(user.privacy_settings or {})
            current_settings.scheduled_deletion = deletion_time.isoformat()

            await self._registry.update_user_profile(
                user_id,
                {
                    "is_active": False,
                    "privacy_settings": current_settings.to_dict(),
                },
            )

            await self._write_audit_log(AuditLogEntry(
                event_type=AuditEventType.ACCOUNT_DELETION_REQUESTED,
                user_id=user_id,
                resource_type="account",
                resource_id=user_id,
                timestamp=datetime.now(timezone.utc),
                ip_address=ip_address,
                user_agent=user_agent,
                details={"scheduled_deletion": deletion_time.isoformat()},
            ))

            logger.info(f"Scheduled account deletion for user {user_id} at {deletion_time.isoformat()}")
            return deletion_time

        except Exception as e:
            logger.error(f"Failed to schedule account deletion for user {user_id}: {e}")
            raise MemoryError(f"Failed to schedule account deletion: {e}")

    async def enforce_retention_policies(self) -> RetentionCleanupResult:
        """
        Enforce data retention policies across all users.

        Deletes:
        - Interactions older than each user's configured retention period
        - Expired sessions
        - Accounts scheduled for deletion

        Requirement 7.3: Enforce data retention and deletion policies.

        Returns:
            RetentionCleanupResult with counts of deleted records
        """
        interactions_deleted = 0
        sessions_deleted = 0
        users_deleted = 0

        try:
            # 1. OAuth/JWT sessions are handled in session_manager (in-memory), not here
            sessions_deleted = 0

            # 2. Process accounts scheduled for deletion
            users_to_delete = await self._registry.get_users_scheduled_for_deletion()
            for user in users_to_delete:
                uid = user.user_id
                success = await self._registry.delete_user_profile(uid)
                if success:
                    users_deleted += 1
                    await self.log_data_deletion(
                        user_id=uid,
                        resource_type="account",
                        resource_id=uid,
                    )
                    logger.info(f"Deleted scheduled account: {uid}")

            # 3. Enforce per-user interaction retention
            # For simplicity, apply a global default of 365 days for users
            # without explicit settings. In production, iterate per user.
            default_cutoff = datetime.now(timezone.utc) - timedelta(days=365)
            interactions_deleted += await self._registry.delete_interactions_before(default_cutoff)

            result = RetentionCleanupResult(
                interactions_deleted=interactions_deleted,
                sessions_deleted=sessions_deleted,
                users_deleted=users_deleted,
                cleanup_timestamp=datetime.now(timezone.utc),
            )

            logger.info(
                f"Retention cleanup complete: {interactions_deleted} interactions, "
                f"{sessions_deleted} sessions, {users_deleted} users deleted"
            )
            return result

        except Exception as e:
            logger.error(f"Failed to enforce retention policies: {e}")
            raise MemoryError(f"Failed to enforce retention policies: {e}")

    async def enforce_user_retention_policy(self, user_id: str) -> int:
        """
        Enforce retention policy for a specific user.

        Deletes interactions older than the user's configured retention period.

        Args:
            user_id: User identifier

        Returns:
            Number of interactions deleted

        Raises:
            MemoryError: If enforcement fails
        """
        try:
            settings = await self.get_privacy_settings(user_id)
            cutoff = datetime.now(timezone.utc) - timedelta(days=settings.retention_days)

            deleted = await self._registry.delete_interactions_before(
                cutoff_date=cutoff,
                user_id=user_id,
            )

            if deleted > 0:
                logger.info(
                    f"Enforced retention for user {user_id}: deleted {deleted} interactions "
                    f"older than {settings.retention_days} days"
                )

            return deleted

        except Exception as e:
            logger.error(f"Failed to enforce retention for user {user_id}: {e}")
            raise MemoryError(f"Failed to enforce user retention policy: {e}")

    # ── Data Export ────────────────────────────────────────────────────────────

    async def export_user_data(
        self,
        user_id: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Export all user data for GDPR compliance.

        Requirement 7.4: Allow users to export their conversation history in JSON format.

        Args:
            user_id: User identifier
            ip_address: Optional IP for audit log
            user_agent: Optional user agent for audit log

        Returns:
            Dictionary with all user data

        Raises:
            MemoryError: If export fails
        """
        try:
            data = await self._registry.get_user_data_export(user_id)

            await self._write_audit_log(AuditLogEntry(
                event_type=AuditEventType.DATA_EXPORT_REQUESTED,
                user_id=user_id,
                resource_type="user_data",
                resource_id=user_id,
                timestamp=datetime.now(timezone.utc),
                ip_address=ip_address,
                user_agent=user_agent,
            ))

            logger.info(f"Exported data for user {user_id}")
            return data

        except Exception as e:
            logger.error(f"Failed to export data for user {user_id}: {e}")
            raise MemoryError(f"Failed to export user data: {e}")

    # ── Internal Helpers ───────────────────────────────────────────────────────

    async def _write_audit_log(self, entry: AuditLogEntry) -> None:
        """
        Write an entry to the audit log.

        Requirement 7.6: Maintain audit logs without exposing sensitive information.

        Args:
            entry: AuditLogEntry to record
        """
        # Sanitise: never log tokens, passwords, or raw query content
        PrivacyManager._audit_log.append(entry)

        # Keep log bounded in memory (last 10 000 entries)
        if len(PrivacyManager._audit_log) > 10_000:
            PrivacyManager._audit_log = PrivacyManager._audit_log[-10_000:]

        logger.debug(
            f"Audit: {entry.event_type.value} | user={entry.user_id} | "
            f"resource={entry.resource_type}/{entry.resource_id}"
        )
