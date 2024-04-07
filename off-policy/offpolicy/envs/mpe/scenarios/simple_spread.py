import numpy as np
from offpolicy.envs.mpe.core import World, Agent, Landmark
from offpolicy.envs.mpe.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, args):
        world = World()
        world.world_length = 25 # args.episode_length
        # set any world properties first
        world.dim_c = 2
        world.num_agents = 3 #args.num_agents
        world.num_landmarks = 5 #args.num_landmarks  # 3
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(world.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            # =================================== #
            # Set a smaller size for agent circle
            agent.size = 0.05
        # add landmarks
        world.landmarks = [Landmark() for i in range(world.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.movable = False
            # ========================================== #
            # Add 2 obstacles in landmark list
            if i < world.num_landmarks - 2:
                landmark.collide = False
            else:
                landmark.collide = True
            
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        world.assign_agent_colors()

        world.assign_landmark_colors()
        # =================================== #
        # Set color (red) and size for obstacles
        for l in world.landmarks:
            if l.collide == True:
                l.color = np.array([0.9, 0.1, 0.1])
                l.size = 0.15
        
        # =================================== #
        # Set the position of agent
        agent_p = np.array([[-0.5, 0.5], [0.3, -0.6], [0.7, -0.1]])
        for i, agent in enumerate(world.agents):
            agent.state.p_pos = agent_p[i]
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        # =================================== #
        # Set the position of obstacles and landmarks 
        landmark_p = np.array([[-0.2, -0.1], [0, 0], [0.2, 0.2], [-0.2, 0.2], [0.2, -0.2]])
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = landmark_p[i]
            landmark.state.p_vel = np.zeros(world.dim_p)
            
        

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            # ========================================== #
            # Only compute the dist to real landmarks
            if l.collide == False:
                dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
                        for a in world.agents]
                min_dists += min(dists)
                rew -= min(dists)
                if min(dists) < 0.1:
                    occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
            # ========================================== #
            # Counting for obstacle collision
            for l in world.landmarks:
                if l.collide == True:
                    if self.is_obstacle_collision(agent, l):
                        rew -= 1
                        collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False
    
    # ========================================== #
    # Add collision function for obstacle
    def is_obstacle_collision(self, agent, landmark):
        delta_pos = agent.state.p_pos - landmark.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent.size + landmark.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        for l in world.landmarks:
            # ========================================== #
            # Identify the real landmarks
            if l.collide == False:
                dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
                        for a in world.agents]
                rew -= min(dists)

        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
            # ========================================== #
            # Add penality for obstacle collision
            for l in world.landmarks:
                if l.collide == True:
                    if self.is_obstacle_collision(agent, l):
                        rew -= 1
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)


# import numpy as np
# from offpolicy.envs.mpe.core import World, Agent, Landmark
# from offpolicy.envs.mpe.scenario import BaseScenario


# class Scenario(BaseScenario):
#     def make_world(self, args):
#         world = World()
#         world.world_length = args.episode_length
#         # set any world properties first
#         world.dim_c = 2
#         world.num_agents = args.num_agents
#         world.num_landmarks = args.num_landmarks  # 3
#         world.collaborative = True
#         # add agents
#         world.agents = [Agent() for i in range(world.num_agents)]
#         for i, agent in enumerate(world.agents):
#             agent.name = 'agent %d' % i
#             agent.collide = True
#             agent.silent = True
#             agent.size = 0.15
#         # add landmarks
#         world.landmarks = [Landmark() for i in range(world.num_landmarks)]
#         for i, landmark in enumerate(world.landmarks):
#             landmark.name = 'landmark %d' % i
#             landmark.collide = False
#             landmark.movable = False
#         # make initial conditions
#         self.reset_world(world)
#         return world

#     def reset_world(self, world):
#         # random properties for agents
#         world.assign_agent_colors()

#         world.assign_landmark_colors()

#         # set random initial states
#         for agent in world.agents:
#             agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
#             agent.state.p_vel = np.zeros(world.dim_p)
#             agent.state.c = np.zeros(world.dim_c)
#         for i, landmark in enumerate(world.landmarks):
#             landmark.state.p_pos = 0.8 * np.random.uniform(-1, +1, world.dim_p)
#             landmark.state.p_vel = np.zeros(world.dim_p)

#     def benchmark_data(self, agent, world):
#         rew = 0
#         collisions = 0
#         occupied_landmarks = 0
#         min_dists = 0
#         for l in world.landmarks:
#             dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
#                      for a in world.agents]
#             min_dists += min(dists)
#             rew -= min(dists)
#             if min(dists) < 0.1:
#                 occupied_landmarks += 1
#         if agent.collide:
#             for a in world.agents:
#                 if self.is_collision(a, agent):
#                     rew -= 1
#                     collisions += 1
#         return (rew, collisions, min_dists, occupied_landmarks)

#     def is_collision(self, agent1, agent2):
#         delta_pos = agent1.state.p_pos - agent2.state.p_pos
#         dist = np.sqrt(np.sum(np.square(delta_pos)))
#         dist_min = agent1.size + agent2.size
#         return True if dist < dist_min else False

#     def reward(self, agent, world):
#         # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
#         rew = 0
#         for l in world.landmarks:
#             dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
#                      for a in world.agents]
#             rew -= min(dists)

#         if agent.collide:
#             for a in world.agents:
#                 if self.is_collision(a, agent):
#                     rew -= 1
#         return rew

#     def observation(self, agent, world):
#         # get positions of all entities in this agent's reference frame
#         entity_pos = []
#         for entity in world.landmarks:  # world.entities:
#             entity_pos.append(entity.state.p_pos - agent.state.p_pos)
#         # entity colors
#         entity_color = []
#         for entity in world.landmarks:  # world.entities:
#             entity_color.append(entity.color)
#         # communication of all other agents
#         comm = []
#         other_pos = []
#         for other in world.agents:
#             if other is agent:
#                 continue
#             comm.append(other.state.c)
#             other_pos.append(other.state.p_pos - agent.state.p_pos)
#         return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
