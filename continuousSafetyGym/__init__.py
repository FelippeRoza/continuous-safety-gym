from gymnasium.envs.registration import register

# Multiagent envs
# ----------------------------------------

register(
    id='MultiagentDescentralizedSafe-v0',
    entry_point='continuousSafetyGym.multiagent_particle.decentralized_safe:DecentralizedSafe',
    # FIXME(cathywu) currently has to be exactly max_path_length parameters in
    # rllab run script
    max_episode_steps=300,
)


# ===== Safe Explorer =====
register(
    id='SpaceshipSafe-v0',
    entry_point='continuousSafetyGym.safe_explorer.spaceship:Spaceship',
    # FIXME(cathywu) currently has to be exactly max_path_length parameters in
    # rllab run script
    max_episode_steps=300,
)


# ===== Bullet Safety Gym =====
register(
    id='ContSafetyCarReach-v0',
    entry_point='continuousSafetyGym.bullet_safety_gym.envs.builder:EnvironmentBuilder',
    max_episode_steps=500,
    kwargs=dict(
        agent='Ball',
        task='ReachGoalTask',
        obstacles={'Pillar': {'number': 8, 'fixed_base': True,
                              'movement': 'static'}
                   },
        world={'name': 'SmallRoom'},
        # debug=True
    ),
)