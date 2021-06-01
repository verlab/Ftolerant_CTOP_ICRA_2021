"""
Simulation of the execution of a mission.

Args:
    --config_file   Full path to the simulation config file;

    --plan_file     Full path to the file with the initial planning;

    --result_file   Full path to the result file to be created;

    --strategy      Strategy used when an agent fails: [non-adapt, ours].

"""

import simargs
from environment import Environment
import sys
import time
# import matplotlib
# matplotlib.use('Agg')
# # from agent import TspState
import matplotlib.pyplot as plt
from agent import Agent


def showSimulation(env, team, paths='pm', file='teste.png'):
    plt.clf()
    env.showEnv()
    if 'p' in paths:
        for agent in team:
            env.showPath(agent.planned_path, agent.color)

    if 'm' in paths:
        for agent in team:
            env.showPath(agent.moved_path, agent.color, 1.0, 3, failed=agent.failed())

    plt.savefig(file)


def LoadPlanFromFile(file):
    """
    Load the initial plan from file.

    Example file: initial_plan.example

    """
    sys.stdout.write("Loading initial plan from file")
    data = open(file, 'r')
    lines = data.readlines()
    data.close()

    plan = {}
    plan['num_nodes'] = int(lines[1].split(': ')[1])
    plan['num_agents'] = int(lines[2].split(': ')[1])
    plan['total_utility'] = float(lines[3].split(': ')[1])
    plan['total_time'] = float(lines[4].split(': ')[1])

    i = 6
    plan['agents'] = []
    for k in range(plan['num_agents']):
        plan['agents'].append({})
        plan['agents'][k]['utility'] = float(lines[i+2].split(': ')[1])
        plan['agents'][k]['cost'] = float(lines[i+3].split(': ')[1])
        plan['agents'][k]['budget'] = float(lines[i+4].split(': ')[1])
        strpath = lines[i+5].split(': ')[1][:-1]
        plan['agents'][k]['path'] = [int(node) for node in strpath.split(',')]
        i = i+6

    i += 3
    plan['nodes'] = []
    for k in range(plan['num_nodes']):
        node = lines[i].split(' ')
        plan['nodes'].append({})
        plan['nodes'][k]['x'] = float(node[0])
        plan['nodes'][k]['y'] = float(node[1])
        plan['nodes'][k]['r'] = float(node[2][:-1])
        i += 1

    sys.stdout.write(' ' + u'\u2713' + '\n')

    return plan


def LoadAgentsFromFile(file):
    """
    Load the agent settings from file.

    Example file: agents_config.example

    """
    sys.stdout.write("Loading agents configurations from file")
    data = open(file, 'r')
    lines = data.readlines()
    data.close()

    i = 0
    # consumes the comments at the beginning of the file
    while lines[i][0] == '#':
        i += 1

    agents = []
    while i != len(lines):
        if lines[i][0] == '\n':  # ignore empty lines at the end of the file
            i += 1
            continue
        agents.append({})
        settings = lines[i].split(' ')
        agents[-1]['id'] = int(settings[0])
        agents[-1]['color'] = settings[1]
        agents[-1]['nr'] = float(settings[2])
        agents[-1]['la'] = int(settings[3])
        agents[-1]['rem'] = float(settings[4])
        agents[-1]['pfails'] = float(settings[5])
        i += 1

    sys.stdout.write(' ' + u'\u2713' + '\n')
    return agents


def endOfMission(agents):
    for ag in agents:
        if ag.state == 'moving' or ag.state == 'planned':
            return False
    return True


if __name__ == "__main__":

    simulation = simargs.SimArgs()

    plan_info = LoadPlanFromFile(simulation.plan_file)
    agents_info = LoadAgentsFromFile(simulation.config_file)

    if plan_info['num_agents'] > len(agents_info):
        sys.stderr.write("Insufficient number of agents to execute the plan.\n")
        exit()

    # create the environment
    environment = Environment(plan_info['nodes'])

    # create the agents
    team = []
    for agent in agents_info:
        team.append(Agent(environment=environment,
                          id=agent['id'],
                          color=agent['color'],
                          budget=plan_info['agents'][agent['id']]['budget'],
                          nr=agent['nr'],
                          la=agent['la'],
                          rem=agent['rem'],
                          pfails=agent['pfails'],
                          path=plan_info['agents'][agent['id']]['path']))

    for agent in team:
        agent.setPool(environment)

    # start simulation
    print('Simulating: {:d} nodes, {:d} agents - {:s} strategy\
            '.format(plan_info['num_nodes'], plan_info['num_agents'], simulation.strategy))
    print("Total reward planned: {:f}".format(plan_info['total_utility']))

    start_time = time.time()
    if simulation.strategy == 'non-adapt':
        while not endOfMission(team):
            for agent in team:
                agent.move(environment, team)

    else:
        while not endOfMission(team):
            for agent in team:
                agent.move(environment, team)
                agent.adjustSegment(environment, team)

    end_time = time.time()

    final_gain = environment.getCurrentRewardGain()
    print("Total reward executed: {:f}".format(final_gain))

    total_time = end_time-start_time
    print('Total time: {:f}'.format(total_time))

    showSimulation(environment, team, 'pm', '{:s}.png'.format(simulation.result_file))

    # write the resul file
    sim_result = open(simulation.result_file, 'w')
    sim_result.write('######### S U M M A R Y ########\n')
    sim_result.write('Nodes: {:d}\n'.format(environment.num_nodes))
    sim_result.write('Agents: {:d}\n'.format(plan_info['num_agents']))
    sim_result.write('Utility Planned: {:f}\n'.format(plan_info['total_utility']))
    sim_result.write('Utility Executed: {:f}\n'.format(final_gain))
    sim_result.write('Total time: {:f}\n\n'.format(total_time))

    sim_result.write('######### EXECUTED ROUTES ########\n')
    for agent in team:
        sim_result.write("Agent: {:d}\n".format(agent.id))

        if agent.pfails == 100:
            failure = 'No'
        elif agent.pfails == 0:
            failure = 'at beggining'
        else:
            failure = 'at {:.2f}% of the planned path'.format(agent.pfails)

        sim_result.write("Fails: {:s}\n".format(failure))
        sim_result.write("Planned Reward: {:f}\n".format(plan_info['agents'][agent.id]['utility']))
        sim_result.write("Collected Reward: {:f}\n".format(agent.computePathReward(environment, agent.moved_path)))
        sim_result.write("Path cost: {:f}\n".format(agent.spended_budget))
        sim_result.write("Budget: {:f}\n".format(agent.budget))
        sim_result.write("Path executed: {:s}\n".format(str(agent.moved_path).strip('[]')))
        sim_result.write('--------------------------------\n')

    sim_result.close()
