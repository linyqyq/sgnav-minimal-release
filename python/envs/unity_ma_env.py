import numpy as np
import time
import os

from dataclasses import dataclass
from typing import Optional, Tuple

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel


@dataclass
class EnvObs:
    image: np.ndarray  # (H, W, C), float32
    vector: np.ndarray # (V,), float32


class UnitySingleAgentEnv:
    """
    Minimal single-agent wrapper for Unity ML-Agents.
    Exposes reset() and step(action) similar to gym.
    """
    def __init__(
        self,
        file_name: Optional[str],
        behavior_name: str,
        seed: int = 1,
        no_graphics: bool = True,
        worker_id: int = 0,
        time_scale: float = 20.0,
        base_port: int = 5005,
        timeout_wait: int = 120,   # ✅ 新增：避免启动慢就超时
        env_params: Optional[dict] = None,
    ):
        self.behavior_name = behavior_name
        self.env_parameters_channel = EnvironmentParametersChannel()
        self.env = UnityEnvironment(
            file_name=file_name,
            seed=seed,
            no_graphics=no_graphics,
            worker_id=worker_id,
            base_port=base_port,
            timeout_wait=timeout_wait,  # ✅ 新增
            additional_args=["-screen-width", "1280", "-screen-height", "720", "-screen-fullscreen", "0"],
            side_channels=[self.env_parameters_channel],
        )
        for key, value in (env_params or {}).items():
            self.env_parameters_channel.set_float_parameter(key, float(value))
        self.env.reset()

        # Set time scale if possible (Unity side must have Academy settings)
        try:
            self.env.set_configuration_parameters(time_scale=time_scale)
        except Exception:
            pass

        self._agent_id = None

        # ✅ 关键：先 step 一帧，让 spawner/agent 注册完成
        self.env.step()

        # ✅ 关键：等待 behavior_specs 里出现对应 behavior（否则 KeyError）
        wait_s = 30.0
        t0 = time.time()
        while self.behavior_name not in self.env.behavior_specs:
            if time.time() - t0 > wait_s:
                print("Available behaviors:", list(self.env.behavior_specs.keys()))
                raise KeyError(f"Behavior '{self.behavior_name}' not found after {wait_s}s.")
            time.sleep(0.1)

        # Inspect specs
        spec = self.env.behavior_specs[self.behavior_name]
        self.action_size = spec.action_spec.continuous_size
        assert self.action_size > 0, "This wrapper assumes continuous action space."


    def close(self):
        self.env.close()

    def _get_single_obs(self) -> Tuple[EnvObs, float, bool]:
        """
        Returns (obs, reward, done) for the first available agent.
        If agent terminated, done=True and obs is still returned from terminal steps.
        """
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)

        # Terminal takes precedence
        if len(terminal_steps) > 0:
            agent_id = terminal_steps.agent_id[0]
            self._agent_id = agent_id
            obs_list = terminal_steps.obs
            reward = float(terminal_steps.reward[0])
            done = True
        else:
            assert len(decision_steps) > 0, "No agents found in DecisionSteps."
            agent_id = decision_steps.agent_id[0]
            self._agent_id = agent_id
            obs_list = decision_steps.obs
            reward = float(decision_steps.reward[0])
            done = False

        # Expecting: obs_list[0]=image (H,W,C), obs_list[1]=vector (V,)
        image = obs_list[0][0].astype(np.float32)   # shape: (H,W,C)
        vector = obs_list[1][0].astype(np.float32)  # shape: (V,)

        return EnvObs(image=image, vector=vector), reward, done

    def reset(self) -> EnvObs:
        self.env.reset()
        self.env.step()
        obs, _, done = self._get_single_obs()
        # reset should not return done, but Unity might terminate immediately in edge cases
        if done:
            # do one more step to get a non-terminal decision if possible
            self.env.reset()
            self.env.step()
            obs, _, _ = self._get_single_obs()
        return obs

    # def step(self, action: np.ndarray) -> Tuple[EnvObs, float, bool, dict]:
    #     """
    #     action: np.ndarray shape (action_size,), range assumed [-1,1]
    #     """
    #     action = np.asarray(action, dtype=np.float32).reshape(1, -1)
    #     # clip to safe range
    #     action = np.clip(action, -1.0, 1.0)

    #     action_tuple = ActionTuple(continuous=action)
    #     self.env.set_actions(self.behavior_name, action_tuple)
    #     self.env.step()

    #     obs, reward, done = self._get_single_obs()
    #     info = {"agent_id": self._agent_id}
    #     return obs, reward, done, info
    def step(self, action: np.ndarray) -> Tuple[EnvObs, float, bool, dict]:
        """
        action:
        - shape (action_size,) for single action; if multiple agents are active,
            the same action will be broadcast to all agents
        - or shape (num_agents, action_size) for explicit per-agent actions

        range assumed [-1, 1]
        """
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        num_agents = len(decision_steps)

        if num_agents == 0:
            # 如果当前没有 decision agents，先推进一帧再取一次
            self.env.step()
            decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
            num_agents = len(decision_steps)

        assert num_agents > 0, "No agents found in DecisionSteps during step()."

        action = np.asarray(action, dtype=np.float32)

        # case 1: single action -> broadcast to all active agents
        if action.ndim == 1:
            action = np.tile(action.reshape(1, -1), (num_agents, 1))

        # case 2: already per-agent actions
        elif action.ndim == 2:
            if action.shape[0] != num_agents:
                raise ValueError(
                    f"Expected action first dim == num_agents ({num_agents}), "
                    f"but got shape {action.shape}"
                )
        else:
            raise ValueError(f"Unsupported action shape: {action.shape}")

        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        action_tuple = ActionTuple(continuous=action)
        if os.environ.get("UNITY_VERBOSE_ENV_STEP", "0").lower() in ("1", "true", "yes"):
            print(f"[Env.step] num_agents={num_agents}, action_shape={action.shape}, action={action}")
        self.env.set_actions(self.behavior_name, action_tuple)
        self.env.step()

        obs, reward, done = self._get_single_obs()
        info = {
            "agent_id": self._agent_id,
            "num_agents": num_agents,
            "applied_action_shape": action.shape,
        }
        return obs, reward, done, info
