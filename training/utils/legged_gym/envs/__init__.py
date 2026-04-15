from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.utils.task_registry import task_registry
from .base.legged_robot import LeggedRobot

from legged_gym.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO
task_registry.register( "go2", LeggedRobot, GO2RoughCfg(), GO2RoughCfgPPO())

from legged_gym.envs.go2_stair.go2_terrain_config import GO2TerrainCfg, GO2TerrainCfgPPO
from legged_gym.envs.go2_stair.go2_terrain_env import Go2TerrainRobot
task_registry.register("go2_stair",Go2TerrainRobot,GO2TerrainCfg(),GO2TerrainCfgPPO())

from legged_gym.envs.h1.h1_config import H1RoughCfg, H1RoughCfgPPO
from legged_gym.envs.h1.h1_env import H1Robot
task_registry.register( "h1", H1Robot, H1RoughCfg(), H1RoughCfgPPO())

from legged_gym.envs.h1_2.h1_2_config import H1_2RoughCfg, H1_2RoughCfgPPO
from legged_gym.envs.h1_2.h1_2_env import H1_2Robot
task_registry.register( "h1_2", H1_2Robot, H1_2RoughCfg(), H1_2RoughCfgPPO())

from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO
from legged_gym.envs.g1.g1_env import G1Robot
task_registry.register( "g1", G1Robot, G1RoughCfg(), G1RoughCfgPPO())

from legged_gym.envs.g1_terrain.g1_terrain_config import G1TerrainCfg, G1TerrainCfgPPO
from legged_gym.envs.g1_terrain.g1_terrain_env import G1TerrainRobot
task_registry.register("g1_terrain", G1TerrainRobot, G1TerrainCfg(), G1TerrainCfgPPO())

from legged_gym.envs.go2_stair_navigation.go2_terrain_config import GO2TerrainCfg, GO2TerrainCfgPPO
from legged_gym.envs.go2_stair_navigation.go2_terrain_env import Go2TerrainRobot
task_registry.register("go2_terrain_navigation",Go2TerrainRobot,GO2TerrainCfg(),GO2TerrainCfgPPO())
