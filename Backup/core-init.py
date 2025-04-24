"""
Core functionality for the ML Platform.
"""

from core.session import initialize_session, get_session_state, update_session_state
from core.constants import *

__all__ = [
    'initialize_session',
    'get_session_state',
    'update_session_state'
]
