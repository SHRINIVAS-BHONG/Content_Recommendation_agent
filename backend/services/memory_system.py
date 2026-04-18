"""
Memory System for conversation history management and user preferences.

Provides high-level operations for storing and retrieving user interactions,
managing conversation history, and analyzing user preference patterns.

Requirements: 3.1, 3.2, 3.3, 3.6, 7.3, 7.6, 8.4
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from dataclasses import dataclass

from backend.services.memory_errors import MemoryError, UserNotFoundError
from backend.services.user_memory_store import (
    UserMemoryRegistry,
    StoredInteraction,
    get_default_memory_registry,
)

logger = logging.getLogger(__name__)


@dataclass
class Interaction:
    """User interaction data structure."""
    
    interaction_id: str
    user_id: str
    query: str
    results: List[Dict[str, Any]]
    feedback: Optional[Dict[str, Any]]
    timestamp: datetime
    session_id: Optional[str]
    reasoning_trace: Optional[List[str]]
    query_metadata: Optional[Dict[str, Any]]
    processing_time_ms: Optional[int]
    
    @classmethod
    def from_stored(cls, stored: StoredInteraction) -> "Interaction":
        """Create Interaction from an in-memory stored record."""
        return cls(
            interaction_id=stored.interaction_id,
            user_id=stored.user_id,
            query=stored.query,
            results=stored.results,
            feedback=stored.feedback,
            timestamp=stored.created_at,
            session_id=stored.session_id,
            reasoning_trace=stored.reasoning_trace,
            query_metadata=stored.query_metadata,
            processing_time_ms=stored.processing_time_ms,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert interaction to dictionary."""
        return {
            "interaction_id": self.interaction_id,
            "user_id": self.user_id,
            "query": self.query,
            "results": self.results,
            "feedback": self.feedback,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "session_id": self.session_id,
            "reasoning_trace": self.reasoning_trace,
            "query_metadata": self.query_metadata,
            "processing_time_ms": self.processing_time_ms,
        }


@dataclass
class UserPreferences:
    """User preference data structure."""
    
    preferred_genres: List[str]
    avoided_genres: List[str]
    content_types: List[str]
    personalization_level: str
    privacy_settings: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert preferences to dictionary."""
        return {
            "preferred_genres": self.preferred_genres,
            "avoided_genres": self.avoided_genres,
            "content_types": self.content_types,
            "personalization_level": self.personalization_level,
            "privacy_settings": self.privacy_settings,
        }


@dataclass
class ConversationContext:
    """Conversation context for personalization."""
    
    recent_queries: List[str]
    recent_results: List[Dict[str, Any]]
    interaction_count: int
    last_interaction_time: Optional[datetime]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "recent_queries": self.recent_queries,
            "recent_results": self.recent_results,
            "interaction_count": self.interaction_count,
            "last_interaction_time": self.last_interaction_time.isoformat() if self.last_interaction_time else None,
        }


class MemorySystem:
    """
    Memory system for conversation history management.
    
    Provides methods for storing and retrieving user interactions,
    managing conversation history with pagination, and analyzing
    user preference patterns.
    
    Requirements:
    - 3.1: Retrieve conversation history for authenticated users
    - 3.2: Store user queries, recommendations, and feedback
    - 3.3: Include timestamps and query metadata
    - 3.6: Support conversation history pagination
    """
    
    def __init__(self, registry: Optional[UserMemoryRegistry] = None):
        """
        Initialize memory system with a per-user in-process store.

        Args:
            registry: Optional shared UserMemoryRegistry; defaults to a process-wide singleton.
        """
        self._registry = registry or get_default_memory_registry()
        self.privacy_manager = None  # Will be set externally if needed
        logger.info("Memory system initialized")
    
    def set_privacy_manager(self, privacy_manager):
        """
        Set privacy manager for access control and audit logging.
        
        Args:
            privacy_manager: PrivacyManager instance
        """
        self.privacy_manager = privacy_manager
        logger.info("Privacy manager attached to memory system")
    
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
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> Interaction:
        """
        Store a user interaction with timestamps and metadata.
        
        Requirement 3.2: Store each user query, generated recommendations, and user feedback
        Requirement 3.3: Include timestamps and query metadata
        Requirement 7.6: Audit logging for data modifications
        Requirement 8.4: Access control validation
        
        Args:
            user_id: User identifier
            query: User's query text
            results: List of recommendation results
            session_id: Optional session identifier
            query_metadata: Optional metadata about the query
            feedback: Optional user feedback
            reasoning_trace: Optional reasoning steps
            processing_time_ms: Optional processing time in milliseconds
            ip_address: Optional IP address for audit logging
            user_agent: Optional user agent for audit logging
        
        Returns:
            Interaction object with stored data
        
        Raises:
            MemoryError: If storage operation fails
        """
        try:
            # Check privacy settings if privacy manager is available
            if self.privacy_manager:
                should_collect = await self.privacy_manager.should_collect_data(user_id)
                if not should_collect:
                    logger.info(f"Data collection disabled for user {user_id}, skipping interaction storage")
                    # Return a dummy interaction without storing
                    return Interaction(
                        interaction_id="00000000-0000-0000-0000-000000000000",
                        user_id=user_id,
                        query=query,
                        results=results,
                        feedback=feedback,
                        timestamp=datetime.now(timezone.utc),
                        session_id=session_id,
                        reasoning_trace=reasoning_trace,
                        query_metadata=query_metadata,
                        processing_time_ms=processing_time_ms,
                    )
            
            stored = await self._registry.store_interaction(
                user_id=user_id,
                query=query,
                results=results,
                session_id=session_id,
                query_metadata=query_metadata,
                feedback=feedback,
                reasoning_trace=reasoning_trace,
                processing_time_ms=processing_time_ms,
            )
            
            interaction = Interaction.from_stored(stored)
            
            # Log the data modification if privacy manager is available
            if self.privacy_manager:
                await self.privacy_manager.log_data_modification(
                    user_id=user_id,
                    resource_type="interaction",
                    resource_id=interaction.interaction_id,
                    modification_type="create",
                    ip_address=ip_address,
                    user_agent=user_agent,
                )
            
            logger.info(f"Stored interaction for user {user_id}: {interaction.interaction_id}")
            return interaction
        
        except Exception as e:
            logger.error(f"Failed to store interaction for user {user_id}: {e}")
            raise MemoryError(f"Failed to store interaction: {e}")
    
    async def get_conversation_history(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
        requesting_user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> List[Interaction]:
        """
        Retrieve conversation history with pagination.
        
        Requirement 3.1: Retrieve conversation history for authenticated users
        Requirement 3.6: Support conversation history pagination with 50 items per page
        Requirement 7.6: Audit logging for data access
        Requirement 8.4: Access control validation
        
        Args:
            user_id: User identifier
            limit: Maximum number of interactions to retrieve (default: 50)
            offset: Number of interactions to skip (default: 0)
            requesting_user_id: Optional user making the request (for access control)
            ip_address: Optional IP address for audit logging
            user_agent: Optional user agent for audit logging
        
        Returns:
            List of Interaction objects in reverse chronological order
        
        Raises:
            MemoryError: If retrieval operation fails
            PermissionError: If access control validation fails
        """
        try:
            # Validate access control if privacy manager is available
            if self.privacy_manager and requesting_user_id:
                has_access = await self.privacy_manager.validate_access(
                    requesting_user_id=requesting_user_id,
                    resource_user_id=user_id,
                    resource_type="conversation_history",
                )
                if not has_access:
                    raise PermissionError(f"User {requesting_user_id} does not have access to user {user_id}'s data")
            
            stored_list = await self._registry.get_user_interactions(
                user_id=user_id,
                limit=limit,
                offset=offset,
            )
            
            interactions = [Interaction.from_stored(s) for s in stored_list]
            
            # Log the data access if privacy manager is available
            if self.privacy_manager:
                await self.privacy_manager.log_data_access(
                    user_id=user_id,
                    resource_type="conversation_history",
                    resource_id=user_id,
                    ip_address=ip_address,
                    user_agent=user_agent,
                )
            
            logger.info(f"Retrieved {len(interactions)} interactions for user {user_id}")
            return interactions
        
        except PermissionError:
            raise
        except Exception as e:
            logger.error(f"Failed to retrieve conversation history for user {user_id}: {e}")
            raise MemoryError(f"Failed to retrieve conversation history: {e}")
    
    async def get_interaction_count(self, user_id: str) -> int:
        """
        Get total interaction count for a user.
        
        Args:
            user_id: User identifier
        
        Returns:
            Total number of interactions
        
        Raises:
            MemoryError: If count operation fails
        """
        try:
            count = await self._registry.get_interaction_count(user_id)
            logger.debug(f"User {user_id} has {count} interactions")
            return count
        
        except Exception as e:
            logger.error(f"Failed to get interaction count for user {user_id}: {e}")
            raise MemoryError(f"Failed to get interaction count: {e}")
    
    async def update_interaction_feedback(
        self,
        interaction_id: str,
        feedback: Dict[str, Any],
    ) -> bool:
        """
        Update feedback for a specific interaction.
        
        Args:
            interaction_id: Interaction identifier
            feedback: Feedback data to store
        
        Returns:
            True if update successful, False otherwise
        
        Raises:
            MemoryError: If update operation fails
        """
        try:
            success = await self._registry.update_interaction_feedback(
                interaction_id=interaction_id,
                feedback=feedback,
            )
            
            if success:
                logger.info(f"Updated feedback for interaction {interaction_id}")
            else:
                logger.warning(f"Interaction not found: {interaction_id}")
            
            return success
        
        except Exception as e:
            logger.error(f"Failed to update feedback for interaction {interaction_id}: {e}")
            raise MemoryError(f"Failed to update feedback: {e}")
    
    async def get_user_preferences(self, user_id: str) -> UserPreferences:
        """
        Get user preferences from profile and learned patterns.
        
        Args:
            user_id: User identifier
        
        Returns:
            UserPreferences object
        
        Raises:
            MemoryError: If retrieval operation fails
        """
        try:
            # Get user profile
            user = await self._registry.get_user_profile(user_id)
            if not user:
                raise UserNotFoundError(f"User not found: {user_id}")
            
            # Get preference patterns
            patterns = await self._registry.get_user_preference_patterns(user_id)
            
            # Extract preferences from profile
            profile_prefs = user.preferences or {}
            
            # Extract learned genre preferences
            genre_patterns = [p for p in patterns if p.preference_type == "genre"]
            preferred_genres = [p.preference_value for p in genre_patterns if float(p.confidence_score) > 0.5]
            avoided_genres = [p.preference_value for p in genre_patterns if float(p.confidence_score) < -0.5]
            
            preferences = UserPreferences(
                preferred_genres=profile_prefs.get("preferred_genres", preferred_genres),
                avoided_genres=profile_prefs.get("avoided_genres", avoided_genres),
                content_types=profile_prefs.get("content_types", ["anime", "manga"]),
                personalization_level=profile_prefs.get("personalization_level", "medium"),
                privacy_settings=dict(user.privacy_settings or {}),
            )
            
            logger.debug(f"Retrieved preferences for user {user_id}")
            return preferences
        
        except Exception as e:
            logger.error(f"Failed to get preferences for user {user_id}: {e}")
            raise MemoryError(f"Failed to get user preferences: {e}")
    
    async def update_preferences(
        self,
        user_id: str,
        preferences: UserPreferences,
    ) -> bool:
        """
        Update user preferences in profile.
        
        Args:
            user_id: User identifier
            preferences: UserPreferences object with updated values
        
        Returns:
            True if update successful, False otherwise
        
        Raises:
            MemoryError: If update operation fails
        """
        try:
            updates = {
                "preferences": preferences.to_dict(),
            }
            
            success = await self._registry.update_user_profile(user_id, updates)
            
            if success:
                logger.info(f"Updated preferences for user {user_id}")
            else:
                logger.warning(f"User not found: {user_id}")
            
            return success
        
        except Exception as e:
            logger.error(f"Failed to update preferences for user {user_id}: {e}")
            raise MemoryError(f"Failed to update preferences: {e}")
    
    async def get_conversation_context(
        self,
        user_id: str,
        recent_limit: int = 5,
    ) -> ConversationContext:
        """
        Get conversation context for personalization.
        
        Args:
            user_id: User identifier
            recent_limit: Number of recent interactions to include
        
        Returns:
            ConversationContext object
        
        Raises:
            MemoryError: If retrieval operation fails
        """
        try:
            # Get recent interactions
            recent_interactions = await self.get_conversation_history(
                user_id=user_id,
                limit=recent_limit,
                offset=0,
            )
            
            # Get total interaction count
            total_count = await self.get_interaction_count(user_id)
            
            # Extract recent queries and results
            recent_queries = [i.query for i in recent_interactions]
            recent_results = []
            for interaction in recent_interactions:
                recent_results.extend(interaction.results)
            
            # Get last interaction time
            last_interaction_time = recent_interactions[0].timestamp if recent_interactions else None
            
            context = ConversationContext(
                recent_queries=recent_queries,
                recent_results=recent_results[:10],  # Limit to 10 most recent results
                interaction_count=total_count,
                last_interaction_time=last_interaction_time,
            )
            
            logger.debug(f"Retrieved conversation context for user {user_id}")
            return context
        
        except Exception as e:
            logger.error(f"Failed to get conversation context for user {user_id}: {e}")
            raise MemoryError(f"Failed to get conversation context: {e}")
    
    async def learn_preferences_from_interactions(
        self,
        user_id: str,
        min_interactions: int = 5,
    ) -> Dict[str, Any]:
        """
        Learn user preferences from interaction history and update preference patterns.
        
        Analyzes user interactions, feedback, and result patterns to extract
        genre preferences, content type preferences, and other patterns.
        Updates confidence scores based on interaction frequency and feedback.
        
        Requirements: 3.4, 5.6
        
        Args:
            user_id: User identifier
            min_interactions: Minimum interactions required for learning (default: 5)
        
        Returns:
            Dictionary with learned preferences and confidence scores
        
        Raises:
            MemoryError: If learning operation fails
        """
        try:
            # Get all interactions
            all_interactions = await self.get_conversation_history(
                user_id=user_id,
                limit=1000,
                offset=0,
            )
            
            if len(all_interactions) < min_interactions:
                logger.info(f"Insufficient interactions for learning: {len(all_interactions)} < {min_interactions}")
                return {
                    "learned": False,
                    "reason": "insufficient_interactions",
                    "interaction_count": len(all_interactions),
                }
            
            # Extract genre preferences from results and feedback
            genre_scores = {}
            content_type_scores = {}
            
            for interaction in all_interactions:
                feedback_weight = self._calculate_feedback_weight(interaction.feedback)
                
                # Extract genres from results
                for result in interaction.results:
                    # Extract genres from result
                    genres = result.get("genres", [])
                    if isinstance(genres, str):
                        genres = [genres]
                    
                    for genre in genres:
                        if genre:
                            genre_scores[genre] = genre_scores.get(genre, 0.0) + feedback_weight
                    
                    # Extract content type
                    content_type = result.get("type", "")
                    if content_type:
                        content_type_scores[content_type] = content_type_scores.get(content_type, 0.0) + feedback_weight
            
            # Normalize scores to confidence range [-1.0, 1.0]
            max_genre_score = max(genre_scores.values()) if genre_scores else 1.0
            normalized_genres = {
                genre: min(score / max_genre_score, 1.0)
                for genre, score in genre_scores.items()
            }
            
            max_content_score = max(content_type_scores.values()) if content_type_scores else 1.0
            normalized_content_types = {
                content_type: min(score / max_content_score, 1.0)
                for content_type, score in content_type_scores.items()
            }
            
            # Store learned patterns in the registry
            stored_patterns = []
            
            # Store genre preferences
            for genre, confidence in normalized_genres.items():
                if confidence >= 0.3:  # Only store significant preferences
                    pattern = await self._registry.store_preference_pattern(
                        user_id=user_id,
                        preference_type="genre",
                        preference_value=genre,
                        confidence_score=confidence,
                    )
                    stored_patterns.append(pattern.to_dict())
            
            # Store content type preferences
            for content_type, confidence in normalized_content_types.items():
                if confidence >= 0.3:
                    pattern = await self._registry.store_preference_pattern(
                        user_id=user_id,
                        preference_type="content_type",
                        preference_value=content_type,
                        confidence_score=confidence,
                    )
                    stored_patterns.append(pattern.to_dict())
            
            logger.info(f"Learned {len(stored_patterns)} preference patterns for user {user_id}")
            
            return {
                "learned": True,
                "interaction_count": len(all_interactions),
                "patterns_stored": len(stored_patterns),
                "genre_preferences": normalized_genres,
                "content_type_preferences": normalized_content_types,
                "stored_patterns": stored_patterns,
            }
        
        except Exception as e:
            logger.error(f"Failed to learn preferences for user {user_id}: {e}")
            raise MemoryError(f"Failed to learn preferences: {e}")
    
    def _calculate_feedback_weight(self, feedback: Optional[Dict[str, Any]]) -> float:
        """
        Calculate feedback weight for preference learning.
        
        Positive feedback increases weight, negative feedback decreases it.
        No feedback results in neutral weight.
        
        Args:
            feedback: User feedback dictionary
        
        Returns:
            Weight value between -1.0 and 1.0
        """
        if not feedback:
            return 0.5  # Neutral weight for no feedback (user saw it)
        
        rating = feedback.get("rating", 0)
        
        if rating >= 4:
            return 1.0  # Strong positive
        elif rating == 3:
            return 0.7  # Moderate positive
        elif rating == 2:
            return 0.3  # Slight negative
        else:
            return -0.5  # Strong negative (avoid this)
    
    async def update_preference_confidence(
        self,
        user_id: str,
        preference_type: str,
        preference_value: str,
        feedback_weight: float,
    ) -> bool:
        """
        Update preference confidence score based on new feedback.
        
        Uses exponential moving average to update confidence scores,
        giving more weight to recent interactions.
        
        Requirements: 3.4, 5.6
        
        Args:
            user_id: User identifier
            preference_type: Type of preference (e.g., "genre", "content_type")
            preference_value: Value of preference (e.g., "action", "anime")
            feedback_weight: Weight from feedback (-1.0 to 1.0)
        
        Returns:
            True if update successful, False otherwise
        
        Raises:
            MemoryError: If update operation fails
        """
        try:
            # Get existing pattern
            patterns = await self._registry.get_user_preference_patterns(
                user_id=user_id,
                preference_type=preference_type,
            )
            
            existing_pattern = next(
                (p for p in patterns if p.preference_value == preference_value),
                None
            )
            
            if existing_pattern:
                # Update using exponential moving average (alpha = 0.3)
                old_confidence = float(existing_pattern.confidence_score)
                new_confidence = 0.7 * old_confidence + 0.3 * feedback_weight
                new_confidence = max(-1.0, min(1.0, new_confidence))  # Clamp to [-1, 1]
            else:
                # New pattern
                new_confidence = feedback_weight * 0.5  # Start conservative
            
            # Store updated pattern
            await self._registry.store_preference_pattern(
                user_id=user_id,
                preference_type=preference_type,
                preference_value=preference_value,
                confidence_score=new_confidence,
            )
            
            logger.debug(f"Updated preference confidence: {user_id}/{preference_type}/{preference_value} = {new_confidence}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to update preference confidence: {e}")
            raise MemoryError(f"Failed to update preference confidence: {e}")
    
    async def analyze_patterns(self, user_id: str) -> Dict[str, Any]:
        """
        Analyze user interaction patterns for preference learning.
        
        Performs comprehensive pattern analysis including:
        - Interaction frequency and feedback patterns
        - Query type distribution
        - Genre and content preferences with confidence scores
        - Recommendation personalization insights
        
        Requirements: 3.4, 5.6
        
        Args:
            user_id: User identifier
        
        Returns:
            Dictionary with pattern analysis results
        
        Raises:
            MemoryError: If analysis operation fails
        """
        try:
            # Get all interactions
            all_interactions = await self.get_conversation_history(
                user_id=user_id,
                limit=1000,
                offset=0,
            )
            
            # Get preference patterns
            patterns = await self._registry.get_user_preference_patterns(user_id)
            
            # Analyze interaction frequency
            total_interactions = len(all_interactions)
            
            # Analyze feedback patterns
            positive_feedback = sum(
                1 for i in all_interactions
                if i.feedback and i.feedback.get("rating", 0) >= 4
            )
            negative_feedback = sum(
                1 for i in all_interactions
                if i.feedback and i.feedback.get("rating", 0) <= 2
            )
            
            # Analyze query patterns
            query_types = {}
            for interaction in all_interactions:
                metadata = interaction.query_metadata or {}
                query_type = metadata.get("intent", "unknown")
                query_types[query_type] = query_types.get(query_type, 0) + 1
            
            # Extract genre preferences from patterns
            genre_patterns = [p for p in patterns if p.preference_type == "genre"]
            preferred_genres = [
                {"genre": p.preference_value, "confidence": float(p.confidence_score)}
                for p in genre_patterns
                if float(p.confidence_score) > 0.5
            ]
            avoided_genres = [
                {"genre": p.preference_value, "confidence": abs(float(p.confidence_score))}
                for p in genre_patterns
                if float(p.confidence_score) < -0.3
            ]
            
            # Extract content type preferences
            content_patterns = [p for p in patterns if p.preference_type == "content_type"]
            content_preferences = [
                {"type": p.preference_value, "confidence": float(p.confidence_score)}
                for p in content_patterns
            ]
            
            # Calculate personalization readiness
            personalization_ready = (
                total_interactions >= 5 and
                len(preferred_genres) > 0
            )
            
            analysis = {
                "total_interactions": total_interactions,
                "positive_feedback_count": positive_feedback,
                "negative_feedback_count": negative_feedback,
                "feedback_ratio": positive_feedback / max(positive_feedback + negative_feedback, 1),
                "query_type_distribution": query_types,
                "learned_patterns": [p.to_dict() for p in patterns],
                "preferred_genres": preferred_genres,
                "avoided_genres": avoided_genres,
                "content_preferences": content_preferences,
                "personalization_ready": personalization_ready,
            }
            
            logger.info(f"Analyzed patterns for user {user_id}: {len(patterns)} patterns, personalization_ready={personalization_ready}")
            return analysis
        
        except Exception as e:
            logger.error(f"Failed to analyze patterns for user {user_id}: {e}")
            raise MemoryError(f"Failed to analyze patterns: {e}")
    
    async def delete_user_data(self, user_id: str) -> bool:
        """
        Delete all user data from memory system.
        
        Args:
            user_id: User identifier
        
        Returns:
            True if deletion successful, False otherwise
        
        Raises:
            MemoryError: If deletion operation fails
        """
        try:
            # Delete user bucket (interactions and patterns)
            success = await self._registry.delete_user_profile(user_id)
            
            if success:
                logger.info(f"Deleted all data for user {user_id}")
            else:
                logger.warning(f"User not found: {user_id}")
            
            return success
        
        except Exception as e:
            logger.error(f"Failed to delete data for user {user_id}: {e}")
            raise MemoryError(f"Failed to delete user data: {e}")
