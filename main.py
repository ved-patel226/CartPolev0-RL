import gym
import numpy as np
import tensorflow as tf
from dopamine.agents.dqn import dqn_agent
from dopamine.discrete_domains import gym_lib
from dopamine.discrete_domains import run_experiment


class DopamineDQN(dqn_agent.DQNAgent):
    def __init__(self, sess, num_actions):
        super().__init__(sess, num_actions)

    def _create_network(self, name):
        with tf.compat.v1.variable_scope(name):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Flatten(input_shape=(84, 84, 4)))
            model.add(tf.keras.layers.Dense(512, activation="relu"))
            model.add(tf.keras.layers.Dense(self.num_actions, activation=None))
        return model


def create_agent(sess, environment):
    return DopamineDQN(sess, num_actions=environment.action_space.n)


def main():
    env = gym_lib.create_gym_environment("CartPole")
    with tf.compat.v1.Session() as sess:
        agent = create_agent(sess, env)
        runner = run_experiment.TrainRunner(agent, environment=env)
        runner.run_experiment()


main()
