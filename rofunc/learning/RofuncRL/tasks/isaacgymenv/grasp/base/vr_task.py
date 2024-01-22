import functools
from isaacgym.torch_utils import *
import numpy as np
import time
import torch
from typing import *
from senseglove_teleoperation import SenseGlove
from vrviewer import VRViewer


def step_simulation(gym, sim, vr_viewer, senseglove_operator, compute_haptics, headless, wait_time) -> None:
    gym.simulate(sim)
    if headless:
        gym.step_graphics(sim)
    vr_viewer.step()
    buzz, force_feedback = compute_haptics()
    senseglove_operator.haptic_feedback(buzz, force_feedback)
    wait_time()


class FPSLimit:
    def __init__(self, fps: int, control_freq_inv: int,
                 tau: float = 0.2) -> None:
        self.target_dt = control_freq_inv / fps
        self.control_freq_inv = control_freq_inv
        self.exp_avg_runtime = control_freq_inv / fps
        self.tau = tau

    def __enter__(self) -> None:
        self.loop_start = time.time()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.loop_end = time.time()
        runtime = max(self.loop_end - self.loop_start -
                      (self.prev_wait_time * self.control_freq_inv), 0)

        self.exp_avg_runtime = (1 - self.tau) * self.exp_avg_runtime + \
                                self.tau * runtime

    def wait_time(self) -> None:
        self.prev_wait_time = max(
            (self.target_dt - self.exp_avg_runtime) / self.control_freq_inv, 0)
        time.sleep(self.prev_wait_time)


class VRTask:
    def __init__(self, wrapped_task, vr_viewer_cfg: Dict[str, Any],
                 senseglove_cfg: Dict[str, Any], headless,
                 control_freq_inv: int, target_fps: int) -> None:
        self._wrapped_task = wrapped_task
        self._senseglove_operator = SenseGlove(senseglove_cfg)
        self._vr_viewer = VRViewer(wrapped_task.gym, wrapped_task.envs[0],
                                   wrapped_task.sim, vr_viewer_cfg)

        self.fps_limit = FPSLimit(target_fps, control_freq_inv)

        self._wrapped_task.step_simulation = functools.partial(
            step_simulation, vr_viewer=self._vr_viewer,
            senseglove_operator=self._senseglove_operator,
            compute_haptics=self.compute_haptics, headless=headless, wait_time=self.fps_limit.wait_time)

    @property
    def wrapped_task(self):
        return self._wrapped_task

    @property
    def vr_viewer(self):
        return self._vr_viewer

    @property
    def operator(self):
        return self._senseglove_operator

    def __getattr__(self, item):
        return getattr(self._wrapped_task, item)

    def compute_haptics(self):
        # Compute vibration. Vibration depends on the absolute fingertip contact
        # force independent of direction. As such it is also activated when
        # impact is made with an object with the outside of the fingertip
        # for example.
        if not self.cfg["haptics"]["enable"]:
            return np.zeros(5), np.zeros(5)


        self.gym.refresh_net_contact_force_tensor(self.sim)

        ft_contact_force = self.contact_forces[:, self.fingertip_actor_rb_handle]
        force_magnitude = torch.norm(ft_contact_force[0], dim=1)
        force_direction = torch.zeros_like(ft_contact_force[0])
        for f in range(5):
            force_direction[f] = ft_contact_force[0, f] / (force_magnitude[f] + 1e-5)
        if self.cfg["haptics"]["buzz"]["proportional_to"] == "force":
            buzz = 100 * torch.clamp(
                force_magnitude / self.cfg["haptics"]["buzz"]["full_at"],
                0., 1.)
        elif self.cfg["haptics"]["buzz"]["proportional_to"] == "force_increase":
            force_increase = torch.abs(
                force_magnitude - self.prev_force_magnitude)
            smoothed_force_increase = torch.where(
                force_increase > 0.5 * self.cfg["haptics"]["buzz"]["full_at"],
                force_increase,
                0.2 * force_increase + 0.8 * self.prev_smoothed_force_increase)

            self.prev_smoothed_force_increase = smoothed_force_increase
            buzz = 100 * torch.clamp(
                smoothed_force_increase[0] / self.cfg["haptics"]["buzz"]["full_at"], 0., 1.)
            self.prev_force_magnitude = force_magnitude
        elif self.cfg["haptics"]["buzz"]["proportional_to"] == "collision":
            self.finger_collision = torch.where(
                self.finger_in_contact_for <
                self.collision_vibration_time_steps,
                force_magnitude, torch.zeros_like(force_magnitude))
            self.finger_in_contact_for = torch.where(
                force_magnitude < 0.1, 0, self.finger_in_contact_for + 1)
            buzz = 100 * torch.clamp(self.finger_collision[0] / self.cfg["haptics"]["buzz"]["full_at"], 0., 1.)
        elif self.cfg["haptics"]["buzz"]["proportional_to"] == "high_freq":
            low_pass_force_magnitude = (1 / self.low_pass_horizon_len) * force_magnitude
            for prev in range(self.low_pass_horizon_len - 1):
                low_pass_force_magnitude += (1 / self.low_pass_horizon_len) * self.past_force_magnitudes[:, prev]
            self.past_force_magnitudes = torch.roll(self.past_force_magnitudes, 1, 1)
            self.past_force_magnitudes[:, 0] = force_magnitude

            self.past_force_magnitudes = torch.where(force_magnitude.unsqueeze(-1).repeat(1, self.low_pass_horizon_len - 1) < 0.1,
                                                     torch.zeros_like(self.past_force_magnitudes),
                                                     self.past_force_magnitudes)

            low_pass_force_direction = (1 / self.low_pass_horizon_len) * force_direction
            for prev in range(self.low_pass_horizon_len - 1):
                low_pass_force_direction += (1 / self.low_pass_horizon_len) * self.past_force_directions[:, :, prev]
            self.past_force_directions = torch.roll(self.past_force_directions, 1, 2)
            self.past_force_directions[:, :, 0] = force_direction

            high_freq_force_magnitude = (force_magnitude / (low_pass_force_magnitude + 1)) - 1

            high_freq_angle_change = torch.arccos((force_direction * low_pass_force_direction).sum(1))
            high_freq_angle_change = torch.where(force_magnitude > 0.1, high_freq_angle_change, torch.zeros_like(high_freq_angle_change))

            buzz = high_freq_force_magnitude
            buzz = 100 * torch.clamp(buzz / self.cfg["haptics"]["buzz"]["full_at"], 0., 1.)

        else:
            assert False
        self.extras["buzz"] = buzz.cpu().numpy()

        # Compute force-feedback. The force-feedback depends on the direction of
        # the forces, as it stops the fingers of the SenseGlove from closing.
        # Hence, the contact forces are projected into the rigid body coordinate
        # system of the fingertips and only the components acting 'through' the
        # fingertips are relevant.
        ft_contact_force_in_rb_coordinates = quat_apply(
            quat_conjugate(self.fingertip_rot), ft_contact_force)
        eff_dir = to_torch([2, 1, 1, 1, 1], device=self.device,
                           dtype=torch.long).unsqueeze(-1)
        effective_force = ft_contact_force_in_rb_coordinates[0].gather(
            1, eff_dir).squeeze()
        effective_force = -effective_force
        if self.cfg["haptics"]['force_feedback']['proportional_to'] == 'force':
            force_feedback = 100 * torch.clamp(
                effective_force / self.cfg["haptics"]["force_feedback"][
                    "full_at"], 0., 1.)
        else:
            assert False

        if self.cfg["haptics"]["force_feedback"]["binary"]:
            force_feedback = torch.where(force_feedback > 0.99, force_feedback,
                                         torch.zeros_like(force_feedback))
        self.extras["force_feedback"] = force_feedback.cpu().numpy()

        return buzz.cpu().numpy(), force_feedback.cpu().numpy()
