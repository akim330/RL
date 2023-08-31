import gymnasium as gym
import numpy as np

def basic_policy(obs):
    angle = obs[2]


class StaticActor:
    def __init__(self, action_type):
        self.learning_type = learning_type

    def action_callback(self, obs):
        angle = obs[2]

        if self.learning_type == 'random':
            return np.random.rand() < 0.5

        elif self.learning_type == 'bad':
            # Pole leaning to right
            if angle > 0:
                # Push cart to left
                return 0
            # Pole leaning to left
            else:
                # Push cart to right
                return 1

        elif self.learning_type == 'basic':
            # Pole leaning to right
            if angle > 0:
                # Push cart to right
                return 1
            # Pole leaning to left
            else:
                # Push cart to left
                return 0


class CartPole:
    def __init__(self, learner, render=True, verbose=False):
        self.env = gym.make("CartPole-v1", render_mode = "human")
        self.render = render
        self.last_obs = None

        self.verbose = verbose

        self.learner = learner

        self.__reset_env()

    def __reset_env(self):
        self.last_obs, _ = self.env.reset()
    def run(self, max_iters = 1000):
        self.__reset_env()

        for i in range(max_iters):
            if self.render:
                self.env.render()

            # observation: new observation, which is 1D array with 4 floats: [center horizontal position, velocity, angle of the pole, angular velocity of pole]
            # reward: the reward received at this step (for Cartpole-V1 you get a reward of 1.0 at every step so you try to keep it going as long as possible
            # terminated: agent reaches terminal state of the MDP
            # truncated: ends for some other reason than the MDP's terminal state, e.g. time limit
            # info: auxiliary information useful for debugging and monitoring

            action = self.learner.action_callback(self.last_obs)

            observation, reward, terminated, truncated, info = self.env.step(action)  # Taking a random action

            if terminated or truncated:
                if terminated:
                    s = 'Terminated'
                elif truncated:
                    s = 'Truncated'
                if verbose:
                    print(f"{s} on iter {i}")

                return i

        print(f"Finished max_iters {max_iters}")
        return max_iters






verbose = False
n_epochs = 5
render = True
max_iters = 1000

for learning_type in ["bad", "basic", "random"]:

    learner = StaticActor(action_type = 'bad')

    results = np.zeros(n_epochs)

    cart = CartPole(
        learner=learner,
        render=render,
        verbose=verbose
    )

    for epoch in range(n_epochs):
        print(f"Epoch {epoch}")

        iters_lasted = cart.run(max_iters)

        results[epoch] = iters_lasted

    cart.env.close()

    print(f"""
    Overall stats for policy "{learning_type}":
    - Average iters lasted: {results.mean()}
    - Max iters lasted: {results.max()}
    - Min iters lasted: {results.min()}
    """)
