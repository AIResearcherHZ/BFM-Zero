# Taks_T1 env config and xml root for inference/bench (data/robots/Taks_T1).

import importlib.util
from pathlib import Path
from typing import Literal
import typing as tp

import pydantic
from pydantic import Field

from humanoidverse.utils.g1_env_config import (
    BaseConfig, G1EnvConfig, G1EnvRandConfig,
    StateInitConfig, NoiseConfig, PushConfig, AuxRewardConfig,
    DomainRandomizationConfig, flatten_frame_stack,
)


def get_taks_t1_robot_xml_root() -> Path:
    """Path to humanoidverse/data/robots/Taks_T1 (scene xmls, meshes, goal_frames)."""
    spec = importlib.util.find_spec("humanoidverse")
    if spec is not None and spec.origin is not None:
        pkg_dir = Path(spec.origin).resolve().parent
    else:
        pkg_dir = Path(__file__).resolve().parent.parent
    return pkg_dir / "data" / "robots" / "Taks_T1"


class TaksT1EnvConfig(G1EnvConfig):
    name: tp.Literal["taks_t1_env"] = "taks_t1_env"
    # 32 DOF velocity limits
    dof_vel_limit_list: tp.List[float] = Field(
        default_factory=lambda: [
            # left leg
            7.33, 4.19, 4.19, 7.33, 5.4454, 5.4454,
            # right leg
            7.33, 4.19, 4.19, 7.33, 5.4454, 5.4454,
            # waist
            4.19, 4.19, 4.19,
            # left arm
            5.4454, 5.4454, 5.4454, 5.4454, 20.944, 20.944, 20.944,
            # right arm
            5.4454, 5.4454, 5.4454, 5.4454, 20.944, 20.944, 20.944,
            # neck
            15.71, 15.71, 15.71,
        ]
    )

    @property
    def object_class(self):
        from humanoidverse.envs.g1_env_helper.robot import G1Env
        return G1Env


class TaksT1EnvRandConfig(TaksT1EnvConfig):
    name: tp.Literal["taks_t1_env_rand"] = "taks_t1_env_rand"
    domain_rand_config: DomainRandomizationConfig = DomainRandomizationConfig()

    @property
    def object_class(self):
        from humanoidverse.envs.g1_env_helper.robot_random import G1EnvRand
        return G1EnvRand


TaksT1EnvConfigsType = tp.Union[TaksT1EnvConfig, TaksT1EnvRandConfig]
