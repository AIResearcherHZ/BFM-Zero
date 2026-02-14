# Taks_T1 评估配置，与 g1env.py 对应
import collections
import numbers
import time
import typing as tp

import numpy as np
import pydantic
from humanoidverse.envs.g1_env_helper.bench import RewardEvaluation, TrackingEvaluation
from humanoidverse.utils.taks_t1_env_config import TaksT1EnvConfig, TaksT1EnvRandConfig

from humanoidverse.agents.evaluations.base import BaseEvalConfig, extract_model
from humanoidverse.agents.wrappers.humenvbench import RewardWrapper, TrackingWrapper


class TaksT1TrackingEvaluationConfig(BaseEvalConfig):
    name: tp.Literal["taks_t1_tracking_eval"] = "taks_t1_tracking_eval"
    name_in_logs: str = "taks_t1_tracking_eval"
    motions: str
    motions_root: str
    tracking_env_cfg: TaksT1EnvConfig | TaksT1EnvRandConfig = pydantic.Field(TaksT1EnvConfig(), discriminator="name")
    num_envs: int = 50

    def build(self):
        return TaksT1TrackingEvaluation(self)


class TaksT1TrackingEvaluation:
    def __init__(self, config: TaksT1TrackingEvaluationConfig):
        self.cfg = config

    def run(self, *, timestep, agent_or_model, logger, **kwargs):
        model = extract_model(agent_or_model)
        eval_agent = TrackingWrapper(model=model)
        tracking_eval = TrackingEvaluation(
            motions=self.cfg.motions,
            motion_base_path=self.cfg.motions_root,
            env_config=self.cfg.tracking_env_cfg,
            num_envs=self.cfg.num_envs,
        )
        start_t = time.time()
        print(f"Taks_T1 Tracking started at {time.ctime(start_t)}", flush=True)
        tracking_metrics = tracking_eval.run(agent=eval_agent, disable_tqdm=True)
        duration = time.time() - start_t
        print(f"Taks_T1 Tracking eval time: {duration}")
        wandb_dict = {}
        aggregate = collections.defaultdict(list)
        for _, metr in tracking_metrics.items():
            for k, v in metr.items():
                if isinstance(v, numbers.Number):
                    aggregate[k].append(v)
        for k, v in aggregate.items():
            wandb_dict[k] = np.mean(v)
            wandb_dict[f"{k}#std"] = np.std(v)
        wandb_dict["time"] = duration

        if logger is not None:
            for k, v in tracking_metrics.items():
                v["motion_name"] = k
                v["timestep"] = timestep
                logger.log(v)

        return tracking_metrics, wandb_dict


class TaksT1RewardEvaluationConfig(BaseEvalConfig):
    name: tp.Literal["taks_t1_reward_eval"] = "taks_t1_reward_eval"
    name_in_logs: str = "taks_t1_reward_eval"
    tasks: list[str]
    reward_env_cfg: TaksT1EnvConfig | TaksT1EnvRandConfig = pydantic.Field(TaksT1EnvRandConfig(), discriminator="name")
    num_episodes: int = 10
    max_workers: int = 12
    process_executor: bool = True
    num_inference_workers: int = 1
    num_inference_samples: int = 50_000

    def build(self):
        return TaksT1RewardEvaluation(self)

    @classmethod
    def requires_replay_buffer(self):
        return True


class TaksT1RewardEvaluation:
    def __init__(self, config: TaksT1RewardEvaluationConfig):
        self.cfg = config

    def run(self, *, timestep, agent_or_model, replay_buffer, logger, **kwargs):
        inference_function: str = "reward_wr_inference"
        model = extract_model(agent_or_model)
        eval_agent = RewardWrapper(
            model=model,
            inference_dataset=replay_buffer["train"],
            num_samples_per_inference=self.cfg.num_inference_samples,
            inference_function=inference_function,
            max_workers=self.cfg.num_inference_workers,
            process_executor=False if self.cfg.num_inference_workers == 1 else True,
            make_env_fn=self.cfg.reward_env_cfg,
        )
        reward_eval = RewardEvaluation(
            tasks=self.cfg.tasks,
            num_episodes=self.cfg.num_episodes,
            env_config=self.cfg.reward_env_cfg,
        )
        start_t = time.time()
        reward_metrics = {}
        wandb_dict = {}
        if not replay_buffer["train"].empty():
            print(f"Taks_T1 Reward started at {time.ctime(start_t)}", flush=True)
            reward_metrics = reward_eval.run(agent=eval_agent, disable_tqdm=True)
            duration = time.time() - start_t
            print(f"Taks_T1 Reward eval time: {duration}")
            avg_return = []
            for task in reward_metrics.keys():
                wandb_dict[f"{task}/return"] = np.mean(reward_metrics[task]["reward"])
                wandb_dict[f"{task}/return#std"] = np.std(reward_metrics[task]["reward"])
                avg_return.append(reward_metrics[task]["reward"])
            wandb_dict["return"] = np.mean(avg_return)
            wandb_dict["return#std"] = np.std(avg_return)
            wandb_dict["time"] = duration

        if logger is not None:
            for k, v in reward_metrics.items():
                n = len(v[list(v.keys())[0]])
                v["task"] = [k] * n
                v["timestep"] = [timestep] * n
                logger.log(v)

        return reward_metrics, wandb_dict
