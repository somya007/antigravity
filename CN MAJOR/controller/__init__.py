"""
Intelligent CRC Controller — Adaptive Error Control Protocol.

Provides real-time channel sensing, proactive CRC/ORBGRAND switching,
energy-efficiency mode, and a live decision dashboard.
"""

from controller.intelligent_controller import (
    ChannelStateMonitor,
    AliasingPredictor,
    AdaptiveProtocolController,
    ControllerDashboard,
    run_controller_demo,
)
