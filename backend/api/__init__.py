from .routes_stream import router as stream_router
from .routes_cameras import router as camera_router
from .routes_analytics import router as analytics_router

__all__ = ["stream_router", "camera_router", "analytics_router"]
