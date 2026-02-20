# Taks_T1 域随机化环境
from humanoidverse.envs.g1_env_helper.robot_random import G1EnvRand
from humanoidverse.utils.taks_t1_env_config import TaksT1EnvRandConfig
from humenv.misc.motionlib import MotionBuffer


class TaksT1EnvRand(G1EnvRand):
    """Taks_T1 32DOF 域随机化环境（task_to_xml 自动路由 Taks_T1 XML）"""

    def __init__(
        self,
        config: TaksT1EnvRandConfig = TaksT1EnvRandConfig(),
        shared_motion_lib: MotionBuffer | None = None,
    ):
        super().__init__(config=config, shared_motion_lib=shared_motion_lib)
