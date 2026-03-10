"""
AEGIS-Cloud — Edge / Sensors / Driver Interface
=================================================
Module:   edge.src.sensors.driver_interface
Platform: NVIDIA Jetson Nano (JetPack 5.x)

PURPOSE
-------
Hardware Abstraction Layer (HAL) providing a unified API for all
physical sensors on the AEGIS guardian platform.  Currently supports:

  • **ICM-20948** — 9-DoF IMU (accelerometer, gyroscope, magnetometer)
  • **VL6180X**  — Time-of-Flight proximity / ambient light sensor

The HAL isolates physical I²C/SPI communication from the business logic
in ``FusionEngine``, enabling transparent swap-out of sensor hardware
and, critically, transparent simulation-mode substitution.

DESIGN PRINCIPLES
-----------------
1. **Simulation-First**
   Each driver method checks ``SIMULATION_MODE`` from configuration.
   When True, the driver returns synthetic sensor data read from
   CSV/JSON trace files in ``data/sim_samples/``, enabling full
   end-to-end pipeline testing without any connected hardware.

   This is critical for CI/CD: the GitHub Actions pipeline
   (``mlops/pipelines/github_actions_ci.yml``) runs the entire Edge
   stack in simulation mode inside a container.

2. **AIOps Integration**
   - Each driver tracks consecutive read failures and publishes a
     ``sensor_health`` metric to the telemetry buffer.
   - Failure counts are consumed by the Cloud self-healing loop  which
     may issue recalibration or failover commands.

3. **NATO-Standard Logging**
   Hardware faults (I²C NACK, range-sensor timeout) are logged as
   Incident Reports:
       Timestamp | Sensor_ID | Fault_Code | Recovery_Action

INTERFACES
----------
- Consumed by: ``FusionEngine``
- Configuration: ``edge/config/edge_settings.yaml``

SPRINT ASSIGNMENT
-----------------
Day 1:   Define abstract ``SensorDriver`` protocol and I²C helpers.
Day 2:   Implement ICM-20948 read loop (real + simulation stubs).
Day 3:   Implement VL6180X read loop (real + simulation stubs).
Day 4:   Integration test with ``FusionEngine`` in simulation mode.
"""

from abc import ABC, abstractmethod


class SensorDriver(ABC):
    """
    Abstract base class for all AEGIS sensor drivers.

    Subclasses must implement ``read()`` and ``self_test()``.
    """

    @abstractmethod
    def read(self) -> dict:
        """Return the latest sensor reading as a dict."""
        ...

    @abstractmethod
    def self_test(self) -> bool:
        """Run a hardware self-test.  Return True if healthy."""
        ...


class ICM20948Driver(SensorDriver):
    """
    ICM-20948 9-DoF IMU driver.

    Returns
    -------
    dict
        {
            "accel": (ax, ay, az),  # m/s²
            "gyro":  (gx, gy, gz),  # deg/s
            "mag":   (mx, my, mz),  # µT
        }
    """

    def __init__(self, config: dict):
        """Initialise I²C bus or load simulation trace."""
        ...

    def read(self) -> dict:
        ...

    def self_test(self) -> bool:
        ...


class VL6180XDriver(SensorDriver):
    """
    VL6180X Time-of-Flight range / ambient light sensor driver.

    Returns
    -------
    dict
        {
            "range_mm": int,      # Measured distance in mm
            "ambient_lux": float  # Ambient light level in lux
        }
    """

    def __init__(self, config: dict):
        """Initialise I²C bus or load simulation trace."""
        ...

    def read(self) -> dict:
        ...

    def self_test(self) -> bool:
        ...
