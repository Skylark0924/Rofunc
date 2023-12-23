# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from rofunc.learning.RofuncRL.tasks.isaacgymenv.base.vec_task import VecTask


# Base class for RL tasks
class BaseTask(VecTask):

    def __init__(self, config, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture,
                 force_render, is_meta=False, task_num=0):
        self.cfg = config

        if is_meta:
            self.cfg["env"]["numEnvs"] *= task_num

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device,
                         graphics_device_id=graphics_device_id, headless=headless,
                         virtual_screen_capture=virtual_screen_capture, force_render=force_render)

    # def get_states(self):
    #     return self.states_buf

    # Apply randomizations only on resets, due to current PhysX limitations
    # def apply_randomizations(self, dr_params):
    #     # If we don't have a randomization frequency, randomize every step
    #     rand_freq = dr_params.get("frequency", 1)
    #
    #     # First, determine what to randomize:
    #     #   - non-environment parameters when > frequency steps have passed since the last non-environment
    #     #   - physical environments in the reset buffer, which have exceeded the randomization frequency threshold
    #     #   - on the first call, randomize everything
    #     self.last_step = self.gym.get_frame_count(self.sim)
    #     if self.first_randomization:
    #         do_nonenv_randomize = True
    #         env_ids = list(range(self.num_envs))
    #     else:
    #         do_nonenv_randomize = (self.last_step - self.last_rand_step) >= rand_freq
    #         rand_envs = torch.where(self.randomize_buf >= rand_freq, torch.ones_like(self.randomize_buf),
    #                                 torch.zeros_like(self.randomize_buf))
    #         rand_envs = torch.logical_and(rand_envs, self.reset_buf)
    #         env_ids = torch.nonzero(rand_envs, as_tuple=False).squeeze(-1).tolist()
    #         self.randomize_buf[rand_envs] = 0
    #
    #     if do_nonenv_randomize:
    #         self.last_rand_step = self.last_step
    #
    #     param_setters_map = get_property_setter_map(self.gym)
    #     param_setter_defaults_map = get_default_setter_args(self.gym)
    #     param_getters_map = get_property_getter_map(self.gym)
    #
    #     # On first iteration, check the number of buckets
    #     if self.first_randomization:
    #         check_buckets(self.gym, self.envs, dr_params)
    #
    #     for nonphysical_param in ["observations", "actions"]:
    #         if nonphysical_param in dr_params and do_nonenv_randomize:
    #             dist = dr_params[nonphysical_param]["distribution"]
    #             op_type = dr_params[nonphysical_param]["operation"]
    #             sched_type = dr_params[nonphysical_param]["schedule"] if "schedule" in dr_params[
    #                 nonphysical_param] else None
    #             sched_step = dr_params[nonphysical_param]["schedule_steps"] if "schedule" in dr_params[
    #                 nonphysical_param] else None
    #             op = operator.add if op_type == 'additive' else operator.mul
    #
    #             if sched_type == 'linear':
    #                 sched_scaling = 1.0 / sched_step * \
    #                                 min(self.last_step, sched_step)
    #             elif sched_type == 'constant':
    #                 sched_scaling = 0 if self.last_step < sched_step else 1
    #             else:
    #                 sched_scaling = 1
    #
    #             if dist == 'gaussian':
    #                 mu, var = dr_params[nonphysical_param]["range"]
    #                 mu_corr, var_corr = dr_params[nonphysical_param].get("range_correlated", [0., 0.])
    #
    #                 if op_type == 'additive':
    #                     mu *= sched_scaling
    #                     var *= sched_scaling
    #                     mu_corr *= sched_scaling
    #                     var_corr *= sched_scaling
    #                 elif op_type == 'scaling':
    #                     var = var * sched_scaling  # scale up var over time
    #                     mu = mu * sched_scaling + 1.0 * \
    #                          (1.0 - sched_scaling)  # linearly interpolate
    #
    #                     var_corr = var_corr * sched_scaling  # scale up var over time
    #                     mu_corr = mu_corr * sched_scaling + 1.0 * \
    #                               (1.0 - sched_scaling)  # linearly interpolate
    #
    #                 def noise_lambda(tensor, param_name=nonphysical_param):
    #                     params = self.dr_randomizations[param_name]
    #                     corr = params.get('corr', None)
    #                     if corr is None:
    #                         corr = torch.randn_like(tensor)
    #                         params['corr'] = corr
    #                     corr = corr * params['var_corr'] + params['mu_corr']
    #                     return op(
    #                         tensor, corr + torch.randn_like(tensor) * params['var'] + params['mu'])
    #
    #                 self.dr_randomizations[nonphysical_param] = {'mu': mu, 'var': var, 'mu_corr': mu_corr,
    #                                                              'var_corr': var_corr, 'noise_lambda': noise_lambda}
    #
    #             elif dist == 'uniform':
    #                 lo, hi = dr_params[nonphysical_param]["range"]
    #                 lo_corr, hi_corr = dr_params[nonphysical_param].get("range_correlated", [0., 0.])
    #
    #                 if op_type == 'additive':
    #                     lo *= sched_scaling
    #                     hi *= sched_scaling
    #                     lo_corr *= sched_scaling
    #                     hi_corr *= sched_scaling
    #                 elif op_type == 'scaling':
    #                     lo = lo * sched_scaling + 1.0 * (1.0 - sched_scaling)
    #                     hi = hi * sched_scaling + 1.0 * (1.0 - sched_scaling)
    #                     lo_corr = lo_corr * sched_scaling + 1.0 * (1.0 - sched_scaling)
    #                     hi_corr = hi_corr * sched_scaling + 1.0 * (1.0 - sched_scaling)
    #
    #                 def noise_lambda(tensor, param_name=nonphysical_param):
    #                     params = self.dr_randomizations[param_name]
    #                     corr = params.get('corr', None)
    #                     if corr is None:
    #                         corr = torch.randn_like(tensor)
    #                         params['corr'] = corr
    #                     corr = corr * (params['hi_corr'] - params['lo_corr']) + params['lo_corr']
    #                     return op(tensor, corr + torch.rand_like(tensor) * (params['hi'] - params['lo']) + params['lo'])
    #
    #                 self.dr_randomizations[nonphysical_param] = {'lo': lo, 'hi': hi, 'lo_corr': lo_corr,
    #                                                              'hi_corr': hi_corr, 'noise_lambda': noise_lambda}
    #
    #     if "sim_params" in dr_params and do_nonenv_randomize:
    #         prop_attrs = dr_params["sim_params"]
    #         prop = self.gym.get_sim_params(self.sim)
    #
    #         if self.first_randomization:
    #             self.original_props["sim_params"] = {
    #                 attr: getattr(prop, attr) for attr in dir(prop)}
    #
    #         for attr, attr_randomization_params in prop_attrs.items():
    #             apply_random_samples(
    #                 prop, self.original_props["sim_params"], attr, attr_randomization_params, self.last_step)
    #
    #         self.gym.set_sim_params(self.sim, prop)
    #
    #     # If self.actor_params_generator is initialized: use it to
    #     # sample actor simulation params. This gives users the
    #     # freedom to generate samples from arbitrary distributions,
    #     # e.g. use full-covariance distributions instead of the DR's
    #     # default of treating each simulation parameter independently.
    #     extern_offsets = {}
    #     if self.actor_params_generator is not None:
    #         for env_id in env_ids:
    #             self.extern_actor_params[env_id] = \
    #                 self.actor_params_generator.sample()
    #             extern_offsets[env_id] = 0
    #
    #     for actor, actor_properties in dr_params["actor_params"].items():
    #         for env_id in env_ids:
    #             env = self.envs[env_id]
    #             handle = self.gym.find_actor_handle(env, actor)
    #             extern_sample = self.extern_actor_params[env_id]
    #
    #             for prop_name, prop_attrs in actor_properties.items():
    #                 if prop_name == 'color':
    #                     num_bodies = self.gym.get_actor_rigid_body_count(
    #                         env, handle)
    #                     for n in range(num_bodies):
    #                         self.gym.set_rigid_body_color(env, handle, n, gymapi.MESH_VISUAL,
    #                                                       gymapi.Vec3(random.uniform(0, 1), random.uniform(0, 1),
    #                                                                   random.uniform(0, 1)))
    #                     continue
    #                 if prop_name == 'scale':
    #                     attr_randomization_params = prop_attrs
    #                     sample = generate_random_samples(attr_randomization_params, 1,
    #                                                      self.last_step, None)
    #                     og_scale = 1
    #                     if attr_randomization_params['operation'] == 'scaling':
    #                         new_scale = og_scale * sample
    #                     elif attr_randomization_params['operation'] == 'additive':
    #                         new_scale = og_scale + sample
    #                     self.gym.set_actor_scale(env, handle, new_scale)
    #                     continue
    #
    #                 prop = param_getters_map[prop_name](env, handle)
    #                 if isinstance(prop, list):
    #                     if self.first_randomization:
    #                         self.original_props[prop_name] = [
    #                             {attr: getattr(p, attr) for attr in dir(p)} for p in prop]
    #                     for p, og_p in zip(prop, self.original_props[prop_name]):
    #                         for attr, attr_randomization_params in prop_attrs.items():
    #                             smpl = None
    #                             if self.actor_params_generator is not None:
    #                                 smpl, extern_offsets[env_id] = get_attr_val_from_sample(
    #                                     extern_sample, extern_offsets[env_id], p, attr)
    #                             apply_random_samples(
    #                                 p, og_p, attr, attr_randomization_params,
    #                                 self.last_step, smpl)
    #                 else:
    #                     if self.first_randomization:
    #                         self.original_props[prop_name] = deepcopy(prop)
    #                     for attr, attr_randomization_params in prop_attrs.items():
    #                         smpl = None
    #                         if self.actor_params_generator is not None:
    #                             smpl, extern_offsets[env_id] = get_attr_val_from_sample(
    #                                 extern_sample, extern_offsets[env_id], prop, attr)
    #                         apply_random_samples(
    #                             prop, self.original_props[prop_name], attr,
    #                             attr_randomization_params, self.last_step, smpl)
    #
    #                 setter = param_setters_map[prop_name]
    #                 default_args = param_setter_defaults_map[prop_name]
    #                 setter(env, handle, prop, *default_args)
    #
    #     if self.actor_params_generator is not None:
    #         for env_id in env_ids:  # check that we used all dims in sample
    #             if extern_offsets[env_id] > 0:
    #                 extern_sample = self.extern_actor_params[env_id]
    #                 if extern_offsets[env_id] != extern_sample.shape[0]:
    #                     print('env_id', env_id,
    #                           'extern_offset', extern_offsets[env_id],
    #                           'vs extern_sample.shape', extern_sample.shape)
    #                     raise Exception("Invalid extern_sample size")
    #
    #     self.first_randomization = False
