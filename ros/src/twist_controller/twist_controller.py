import time

from pid import PID
from yaw_controller import YawController

GAS_DENSITY = 2.858
ONE_MPH = 0.44704

class Controller(object):

    def __init__(self,
                 vehicle_mass,
                 fuel_capacity,
                 brake_deadband,
                 decel_limit,
                 accel_limit,
                 wheel_base,
                 wheel_radius,
                 steer_ratio,
                 max_lat_accel,
                 max_steer_angle,
                 min_speed=0.1):
        # PID controller for throttle and braking
        self.tb_pid = PID(kp=0.1, ki=0.004, kd=2.0, mn=decel_limit, mx=accel_limit)
        # Constants that are relevant to computing the final throttle/brake value
        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = -decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius
        # Compute the braking conversion constant according to the formula `braking_torque = total_mass * max_deceleration * wheel_radius`.
        # `self.tb_pid` outputs barking values in `[-1,0]`, which will then be scaled by the constant below.
        self.braking_constant = (self.vehicle_mass + self.fuel_capacity * GAS_DENSITY) * self.decel_limit * self.wheel_radius
        # Record the time of the last control computation.
        self.last_time = None
        # Controller for steering angle
        self.yaw_controller = YawController(wheel_base=wheel_base,
                                            steer_ratio=steer_ratio,
                                            min_speed=min_speed,
                                            max_lat_accel=max_lat_accel,
                                            max_steer_angle=max_steer_angle)

    def control(self,
                target_linear_velocity,
                target_angular_velocity,
                current_linear_velocity,
                dbw_enabled):

        if dbw_enabled:
            if self.last_time is None:
                sample_time = 0.02
                self.last_time = time.time()
            else:
                this_time = time.time()
                sample_time = this_time - self.last_time
                self.last_time = this_time
            # Compute the raw throttle/braking value.
            tb = self.tb_pid.step(error=(current_linear_velocity - target_linear_velocity), sample_time=sample_time)
            # Scale the raw braking value.
            if tb < 0: # We're braking.
                # Convert the raw braking value to torque in Nm (Newton-meters).
                brake = self.braking_constant * tb
                if brake < -self.brake_deadband:
                    brake = 0
                throttle = 0
            else: # We're accelerating.
                throttle = tb
                brake = 0
            # Compute the steering value.
            steer = self.yaw_controller.get_steering(linear_velocity=target_linear_velocity,
                                                     angular_velocity=target_angular_velocity,
                                                     current_velocity=current_linear_velocity)

            return throttle, brake, steer

        else: # If `dbw_enabled` is False.
            self.tb_pid.reset()
            return 0, 0, 0
