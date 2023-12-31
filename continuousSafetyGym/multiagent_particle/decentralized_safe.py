from .multiagent.environment import MultiAgentEnv
from .multiagent import scenarios



class DecentralizedSafe(MultiAgentEnv):

    def __init__(self, safe_init = False, render_mode='human'):
        
        scenario = scenarios.load("decentralized_safe.py").Scenario(safe_init)
        world    = scenario.make_world()
        
        self.scenario = scenario
        self.world = world

        super(DecentralizedSafe, self).__init__(
            world,
            render_mode,
            scenario.reset_world,
            scenario.reward,
            scenario.observation,
            info_callback=None,
            done_callback = scenario.done,
            constraint_callback = scenario.constraints,
            shared_viewer = True
            )
