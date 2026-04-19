"""
models.py — Recommendation model classes for the system
"""

class RecommendationModel:
    """Simple recommendation model that works with the existing system"""
    
    def __init__(self, data, content_type):
        self.data = data
        self.content_type = content_type
    
    def recommend(self, model_input):
        """
        Recommend items based on input parameters
        
        Args:
            model_input (dict): Input parameters from the agent
        
        Returns:
            list[dict]: List of recommended items
        """
        # Simple recommendation: return top-rated items
        # In a real system, this would use the tags, semantic_hints, etc.
        
        # Sort by score and return top 5
        sorted_items = sorted(
            self.data, 
            key=lambda x: x.get('score', 0), 
            reverse=True
        )[:5]
        
        # Format results according to the expected schema
        results = []
        for item in sorted_items:
            result = {
                'title': item.get('title', 'Unknown'),
                'image': item.get('image', ''),
                'synopsis': item.get('synopsis', 'No synopsis available.')[:200] + '...',
                'score': float(item.get('score', 0.0)),
                'genres': item.get('genres', [])
            }
            results.append(result)
        
        return results