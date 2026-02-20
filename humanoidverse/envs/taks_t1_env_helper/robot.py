# Taks_T1 32DOF MuJoCo 环境
# G1Env.__init__ 中的 task_to_xml 已自动检测 config.name 含 'taks' 并路由到 Taks_T1 XML
from humanoidverse.envs.g1_env_helper.robot import G1Env, END_FREEJOINT_QPOS, END_FREEJOINT_QVEL
from humanoidverse.utils.taks_t1_env_config import TaksT1EnvConfig
from humenv.misc.motionlib import MotionBuffer


class TaksT1Env(G1Env):
    """Taks_T1 32DOF 运动环境（继承 G1Env，自动使用 Taks_T1 scene XML）"""

    def __init__(
        self,
        config: TaksT1EnvConfig = TaksT1EnvConfig(),
        shared_motion_lib: MotionBuffer | None = None,
    ):
        super().__init__(config=config, shared_motion_lib=shared_motion_lib)
