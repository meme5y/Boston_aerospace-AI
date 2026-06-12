"""Core/Exceptions.py — Excepções personalizadas"""

class BostonAIError(Exception):
    """Base exception for Boston Aerospace AI"""
    pass

class ModelNotLoadedError(BostonAIError):
    """Raised when ML models are not loaded"""
    pass

class RAGNotReadyError(BostonAIError):
    """Raised when RAG/Ollama is not initialised"""
    pass

class InvalidSensorDataError(BostonAIError):
    """Raised when sensor data is invalid"""
    pass

class DatabaseError(BostonAIError):
    """Raised on database errors"""
    pass
