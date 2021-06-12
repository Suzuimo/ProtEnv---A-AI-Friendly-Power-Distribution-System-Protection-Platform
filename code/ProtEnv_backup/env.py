import win32com.client
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .utils import *
from rl.core import Env

# load a DSS case into the dssCase object
# The dssCase object provide a easy interface with OpenDSS and
# utility functions
class dssCase():
    def __init__(self, case_path, ts):
        # initialize DSS interface objects
        self.dss_handle = win32com.client.Dispatch("OpenDSSEngine.DSS")
        self.txt = self.dss_handle.Text
        self.ckt = self.dss_handle.ActiveCircuit
        self.sol = self.ckt.Solution
        self.ActElmt = self.ckt.ActiveCktElement
        self.ActBus = self.ckt.ActiveBus
        
        # load the case passed through argument
        self.case_path = case_path
        self.load_case()
        self.solve_case()

        # examine the network info of the DSS case and create a graph
        self.get_network_info()
        self.create_graph()
        self.sort_edges()

        # simulation information
        self.ts = ts

    # clean DSS memory    
    def reset_dss(self):
        self.dss_handle.ClearAll()

    # load a local case from the case file
    def load_case(self):
        self.txt.Command = f"compile [{self.case_path}]"

    # solve the current loaded case
    def solve_case(self):
        self.sol.Solve()

    # process case and get network informations
    def get_network_info(self):
        # list of bus names
        self.busNames = self.ckt.AllBusNames
        self.busNum = len(self.busNames)
        # list of phases for each bus
        self.busPhases = []
        for n in self.busNames:
            self.ckt.SetActiveBus(n)
            self.busPhases.append(self.ActBus.Nodes)


        # list of lines
        self.lineNames = self.ckt.Lines.AllNames
        self.lineNum = self.ckt.Lines.Count

        self.lineT = []
        for n in self.lineNames:
            full_name = 'line.' + n
            self.ckt.SetActiveElement(full_name)
            F = self.ActElmt.Properties('Bus1').val.split('.')[0]
            T = self.ActElmt.Properties('Bus2').val.split('.')[0]
            
            # take only the 3-phase bus name 
            self.lineT.append((self.busNames.index(F),self.busNames.index(T)))

        # add transformers as lines (for graph making purpose)
        self.xfmrName = self.ckt.Transformers.AllNames
        self.xfmrNum = self.ckt.Transformers.Count
        self.xfmrT = []
        for tr in self.xfmrName:
            full_name = 'Transformer.' + tr
            self.ckt.SetActiveElement(full_name)
            F = self.busNames.index(self.ActElmt.busNames[0].split('.')[0])
            T = self.busNames.index(self.ActElmt.busNames[1].split('.')[0])

            self.xfmrT.append((F,T))
        self.xfmrT = list(set(self.xfmrT))

    # Read bus coordinates from an external file (low priority)
    def read_bus_coord(self, fp):
        pass

    ## GRAPH-RELATED FUNCTIONS
    # create a graph for the network using networkx
    def create_graph(self):
        # create new un-directed graph first
        self.graph = nx.Graph()
        
        # add lines as edges of this graph
        for l in self.lineT:
            self.graph.add_edge(l[0], l[1])

        # add transformers as edges of this graph
        for t in self.xfmrT:
            self.graph.add_edge(t[0], t[1])

        # change to directed graph and remove edges according to radial structure
        self.graph = self.graph.to_directed()

        ## for every line between bus, remove the backward edge that is not the assumed direction
        # compute the distance from source
        dist_from_source = nx.single_source_shortest_path_length(self.graph, 0) 

        # lines
        for l in self.lineT:
            # if bus1 is closer to souce than bus2
            if dist_from_source[l[0]] < dist_from_source[l[1]]:
                # remove the edge (bus2 -> bus1)
                self.graph.remove_edge(l[1], l[0])

        # transformers
        for t in self.xfmrT:
            # if bus1 is closer to souce than bus2
            if dist_from_source[t[0]] < dist_from_source[t[1]]:
                # remove the edge (bus2 -> bus1)
                self.graph.remove_edge(t[1], t[0])            
        

    # draw the network graph using matplotlib
    def draw_graph(self):
        plt.figure()
        nx.draw(self.graph, with_labels=True)
        plt.show()
        

    # sort nodes using DFS
    def sort_edges(self):
        self.edge_order = list(nx.dfs_edges(self.graph, source=0))
        
        
    # get line current measurement using line name
    def get_line_I(self, name, field, phase):
        full_name = 'line.' + name
        self.ckt.SetActiveElement(full_name)
        if phase == 3:
            if field == 'Iseq':
                res = self.ActElmt.SeqCurrents[0:3]
            elif field == 'IseqPU':
                res = self.ActElmt.SeqCurrents[0:3]
            elif field == 'Iph':
                res = cart_to_pol(self.ActElmt.Currents[0:6])
            else:
                raise ValueError(f'Please use a valid field name for Line measurement')
        # return single phase current if only 1 conductor
        elif phase == 1:
            res = cart_to_pol(self.ActElmt.Currents[0:2])
        else:
            raise ValueError(f'Phase input must be 1 or 3')
            
        return res
    
        
    # get bus voltage measurement using busname
    def get_bus_V(self, name, field, phase):
        self.ckt.SetActiveBus(name)
        if phase == 3:       
            if field == 'Vseq':
                res = self.ActBus.SeqVoltages
            elif field == 'VLN':
                mag, angle = cart_to_pol(self.ActBus.Voltages)
                res = [mag, angle]
            elif field == 'VLL':
                mag, angle = cart_to_pol(self.ActBus.VLL)
                res = [mag, angle]
            else:
                raise ValueError(f'Please use a valid field name for Bus measurement')
        elif phase == 1:
            if field == 'VLN':
                mag, angle = cart_to_pol(self.ActBus.Voltages)
                res = [mag, angle]
            elif field == 'VLL':
                mag, angle = cart_to_pol(self.ActBus.VLL)
                res = [mag, angle]
            else:
                raise ValueError(f'Please use a valid field name for Line measurement')   
        else:
            raise ValueError(f'Phase input must be 1 or 3')
        
        return res

    # edit a property, or a list of properties, of a DSS element
    def edit_elmt(self, name, fields, vals):
        cmd = f'Edit {name}'
        # if providing a list of properties, iterate through
        if isinstance(fields, list):
            for f, v in zip(fields, vals):
                cmd += f' {f}={v}'
        # if only one property, add and execute
        else:
            cmd += f'{fields}={vals}'

        self.txt.Command = cmd

    # trip an element in the netwrok
    def trip_elmt(self, elmt):
        self.txt.Command = f'open line.{elmt} term=1' 

    # create a random fault in this case
    def random_fault(self):
        randFault = fault(self.busNames, self.busPhases, self.ts)
        
        return randFault

# log class for storing episode
class log():
    def __init__(self):
        self.fault = None
        self.tripTimes = None
        self.times = None
        self.agents_waveforms = []

    # plot the record for agent a
    def visualize(self, a):
        all_waves = self.agents_waveforms[a]
        plt.figure()
        # plot waves
        for k in all_waves.keys():
            plt.plot(self.times, all_waves[k], label=k)
        # plot fault time and trip time
        plt.axvline(x = self.fault.T, color = 'r')
        if self.tripTimes[a] > 0:
            plt.axvline(x = self.tripTimes[a], color = 'b')
        plt.title(f'Waveform for Agent {a}')
        plt.legend(all_waves.keys())
        plt.show()

    # plot the record for all agents
    def visualize_all(self):
        if len(self.agents_waveforms) == 1:
            self.visualize(0)
            return
        agentNum = len(self.tripTimes)
        fig, axs = plt.subplots(agentNum)
        fig.suptitle('Waveform of All Agents')
        for a in range(agentNum):
            # waveforms for this agent
            for k in self.agents_waveforms[a].keys():
                axs[a].plot(self.times, self.agents_waveforms[a][k], label=k)
            # fault time and trip time
            axs[a].axvline(x = self.fault.T, color = 'r')
            if self.tripTimes[a] > 0:
                axs[a].axvline(x = self.tripTimes[a], color = 'b')
            fig.legend(self.agents_waveforms[a].keys())
        plt.show()
        
        

# main class for the relay environment
class rlEnv(Env):
    def __init__(self, case_path, agents, params=None):
        # unpack configuration dic
        self.ts = params['time_step']
        self.maxStep = params['max_step']
        self.case = dssCase(case_path, self.ts)
        self.caseName = case_path.split('\\')[-1].split('.')[0]
        
        # NEED TO SEPARATE PV and LOAD, LEAVE FOR NOW
        if not params['pv_profile'] == None:
            self.pvDF, self.loadDF = parse_profile(params['pv_profile'],params['load_profile'])

        # store all agents that need to be trained
        self.agents = agents
        self.agentNum = len(self.agents)
        self.calc_agent_successors()
        self.trainingAgent = None
        self.activeAgents = None

        # sort agents by location
        self.sort_agents()

        # required fields for gym
        self.svNum = None
        self.actNum = None
        self.acton_space = None
        self.observation_space = None

        # environment state and containers
        self.logs = []
        

    # calculate the order of training
    # return a list of intergers of agents to be sorted
    def sort_agents(self):
        # indicies of two terminals of the branch for each agent
        self.agentsBusIndex = [(self.case.busNames.index(a.bus1), self.case.busNames.index(a.bus2)) for a in self.agents]
    
        # find the index of line for each agent
        self.agentsLineIndex = []
        for a in self.agents:
            bus1_ind = self.case.busNames.index(a.bus1)
            bus2_ind = self.case.busNames.index(a.bus2)
            for l in self.case.lineT:
                if l[0] == bus1_ind and l[1] == bus2_ind:
                    self.agentsLineIndex.append(self.case.lineT.index(l))

        # find the position of each agent in the edge order from DFS
        # agent with the larger index is trained first
        agentPos = np.zeros(self.agentNum)
        for i in range(self.agentNum):
            agentPos[i] = self.case.edge_order.index(self.agentsBusIndex[i])
        sortedPos = -np.sort(-agentPos)

        self.train_order = [agentPos.tolist().index(i) for i in sortedPos]

    # parse the network and get the successors of each agent
    def calc_agent_successors(self):
        for a in self.agents:
            a_bus_idx = self.case.busNames.index(a.bus1)
            a.successors = list(nx.nodes(nx.dfs_tree(self.case.graph, a_bus_idx)))
            a.successors.remove(a_bus_idx)

    # get measurements needed for an agent, specified in the fields of the agent class
    def take_sample(self, idx):
        all_sample = {}
        for i in self.agents[idx].obs:
            if i in ['Vseq', 'VLN', 'VLL']:
                ob = self.case.get_bus_V(self.agents[idx].bus1, i, self.agents[idx].phases)
            elif i in ['Iseq', 'Iph']:
                lineName = self.case.lineNames[self.agentsLineIndex[idx]]
                ob = self.case.get_line_I(lineName, i, self.agents[idx].phases)
            else:
                raise ValueError(f'Observation type not supported for agent{i}!')
                
            all_sample[i] = ob
            
        return all_sample

    # this function is just for training, hense logic is tuned around the agent under training
    # next step giving the action of the relay under training
    def step(self, action):
        done = 0
        R = 0

        self.currStep += 1
        #print(self.case.sol.Seconds)

        # check for max simulation time
        if self.currStep == self.maxStep:
            done = 1

        # action of the current training agent
        train_trip = self.agents[self.trainingAgent].act(action)

        # compute reward for the training agent
        flags = self.assess_training_status()
        R = self.agents[self.trainingAgent].rewardFcn(flags, train_trip, self)
        
        if train_trip:
            self.agents[self.trainingAgent].open = True
            self.case.trip_elmt(self.agents[self.trainingAgent].line)

        # action of other active agents
        for a in self.activeAgents:
            act_temp = self.agents[a].process_state()
            a_trip = self.agents[a].act(act_temp)
            if a_trip:
                # if the tripping is successful
                #print(a, self.agents[a].triggerTime, self.agents[a].state)
                self.agents[a].open = True
                self.case.trip_elmt(self.agents[a].line)
 

        # solve this timestep
        self.case.solve_case()

        # get new observation for agents 
        for a in self.agents:
            a.observe(self.case)


        # DEBUG: print current and reward
        #print(self.agents[self.trainingAgent].state, train_trip, R, self.ts*self.currStep, self.agents[1].tripped)

        # return the observation of the agent under training
        ob_act = self.agents[self.trainingAgent].state
        
        
        return ob_act, R, done, {"Agent":self.trainingAgent}
                
        
    # clear DSS memory, reset the environment and start new episode
    def reset(self):
        # reset DSS
        self.case.reset_dss()
        self.case.load_case()

        # reset envrionment state
        self.tripTimes = np.zeros(self.agentNum)
        self.currStep = 1
        #print(self.case.sol.Seconds)

        # reset agent initial states
        for a in self.agents:
            a.reset()

        
        # add random fault
        self.fault = self.case.random_fault()
        self.case.txt.Command = self.fault.cmd

        # sample from load and DER profile


        # solve the initial power flow
        self.case.txt.Command = "set maxcontroliter=100"
        self.case.txt.Command = "set mode=snap"
        self.case.txt.Command = "Solve"

        # set dynamic mode
        self.case.txt.Command = "Solve mode=dynamics number=1 stepsize=" + str(self.ts)

        # get new observation for agents 
        for a in self.agents:
            a.observe(self.case)

        # return the observation of the agent under training
        if not self.trainingAgent == None:
            ob_act = self.agents[self.trainingAgent].state
        else:
            ob_act = None
            
        return ob_act
        
        
    # train all agents in this environment
    def train_agents(self):
        # activate not trainable agents
        self.activeAgents = []
        for a in range(self.agentNum):
            if not self.agents[a].trainable:
                self.activeAgents.append(a)
        
        # go through all agents in the list
        for a in self.train_order:
            # if a is not trainable, skip a
            if not self.agents[a].trainable:
                continue
            # else, train a
            else:
                # configure the env for this agent
                self.trainingAgent = a
                self.svNum = self.agents[a].svNum
                self.actNum = self.agents[a].actNum
                self.action_space = self.agents[a].action_space
                self.observation_space = self.agents[a].observation_space

                # train the agent
                self.agents[a].train(self)
                self.agents[a].save()
                # activate this trained agent
                if not a in self.activeAgents:
                    self.activeAgents.append(a)
                else:
                    raise ValueError(f' agent{a}!')
            
 

    # evaluate all agents by running random eposides
    # return a log object defined in utils.py
    def evaluate(self, epiNum, verbose = True):

        # activate all agents
        self.activeAgents = range(self.agentNum)
        # go through episodes
        for ep in range(epiNum):
            
            self.reset()
            print(ep)
            if verbose:
                print(f'================ Episode {ep} ================')
                print(self.fault.cmd)           

            # loop steps
            done = 0
            while self.currStep < self.maxStep and not done:
                
                self.currStep += 1
                # get new observation for agents 
                for a in self.agents:
                    a.observe(self.case)
                    
                # collect the action of all agents
                for a in self.activeAgents:
                    act_temp = self.agents[a].process_state()
                    a_trip = self.agents[a].act(act_temp)

                    # record and trip the line if an agent has not tripped already
                    if a_trip and self.tripTimes[a] == 0:
                        self.tripTimes[a] = self.case.sol.Seconds
                        if verbose:
                            print(f'Relay {a} at bus {self.agents[a].bus1}, tripped at time {self.tripTimes[a]}!')
                        #print('line.' + self.agents[a].line)
                        self.agents[a].open = True
                        self.case.trip_elmt(self.agents[a].line)

                # solve this timestep
                self.case.solve_case()

            agents_score = self.analyze_episode()
            # log only if incorrect operation happened
            if any([not i==1 for i in agents_score]):
                if verbose:
                    print('Wrong operation! Episode have been logged')
                self.log_episode()


    # log the current episode 
    def log_episode(self):
        newLog = log()
        newLog.fault = self.fault
        newLog.tripTimes = self.tripTimes
        newLog.times = np.linspace(0, self.maxStep*self.ts, self.maxStep)
        for a in self.agents:
            newLog.agents_waveforms.append(a.waveform)

        self.logs.append(newLog)


    # analyze trip oprations associated with a fault
    # return a vector of length self.agentNum telling whether an operation is wrong
    # 1 - correct; 0 - incorrect
    def analyze_episode(self):
        # self.fault
        # self.tripTimes
        res = np.zeros(self.agentNum)
        # analyze active agents one by one
        for a in self.activeAgents:
            
            # is the fault within its designated area?
            fault_bus_idx = self.case.busNames.index(self.fault.bus)

            if fault_bus_idx in self.agents[a].successors:
                area_flag = True
            else:
                area_flag = False                
            
            # if this agent has operated in the past episode
            if self.tripTimes[a] > 0:
                
                # is the tripping after the fault?
                time_flag = self.tripTimes[a] > self.fault.T

                # is the fault within its designated area?
                # that is, does the faulted bus belong to the children of a relay
                fault_bus_idx = self.case.busNames.index(self.fault.bus)

                if fault_bus_idx in self.agents[a].successors:
                    area_flag = True
                else:
                    area_flag = False
                    
                # did it trip before another agent with higher priority?
                miscoord_flag = False
                for b in self.activeAgents:
                    # check other agents
                    if b == a:
                        pass
                    else:
                        # if it tripped before agent b
                        if self.tripTimes[a] < self.tripTimes[b]:
                            # check if b has higher priority, that is:
                            # 1) fault is a successor of b
                            # 2) b is a successor of a
                            b_bus_idx = self.case.busNames.index(self.agents[b].bus1)
                            if fault_bus_idx in self.agents[b].successors and b_bus_idx in self.agents[a].successors:
                                miscoord_flag = True

                # assert the correctness of agent a
                res[a] = time_flag and area_flag and not miscoord_flag
            # if this agent have not operated in the past episode
            else:
                # only if fault is outside the area or a neighbour has operated
                miscoord_flag = True
                for b in self.activeAgents:
                    # check other agents
                    if b == a:
                        pass
                    else:
                        # if b tripped and b is a successor of a, and b tripped after the fault
                        b_bus_idx = self.case.busNames.index(self.agents[b].bus1)
                        if fault_bus_idx in self.agents[b].successors and b_bus_idx in self.agents[a].successors and self.tripTimes[b] > self.fault.T:
                            miscoord_flag = False

                # assert the correctness of agent a
                res[a] = not area_flag or (area_flag and not miscoord_flag)
                
        return res

    # check circuit and fault status and return the condition of the training agent
    def assess_training_status(self):
        fault_bus_idx = self.case.busNames.index(self.fault.bus)

        # compute flags first

        # is the fault within designated area?
        if fault_bus_idx in self.agents[self.trainingAgent].successors:
            area_flag = True
        else:
            area_flag = False

        # is the time after fault?
        time_past = self.currStep * self.ts - self.fault.T
        if time_past > 0:
            time_flag = True
        else:
            time_flag = False

        # has the fault been cleared by a downstream neighbour or itself?
        clear_flag = False
        for a in self.activeAgents:
            a_bus_idx = self.case.busNames.index(self.agents[a].bus1)
            if self.agents[a].open and fault_bus_idx in self.agents[a].successors and a_bus_idx in self.agents[self.trainingAgent].successors:
                clear_flag = True
        # if the training agent has tripped
        if self.agents[self.trainingAgent].open and time_flag and area_flag:
            clear_flag = True

        # if not, is there a downstream neighbour that has not operated yet?
        # if every neighour has sent trip signal (need a backup tripping), this flag is True
        coord_flag = True
        for a in self.activeAgents:
            a_bus_idx = self.case.busNames.index(self.agents[a].bus1)
            if not self.agents[a].tripped and fault_bus_idx in self.agents[a].successors and a_bus_idx in self.agents[self.trainingAgent].successors:
                coord_flag = False

        flags = {'area': area_flag,
                 'time': time_flag,
                 'cleared': clear_flag,
                 'coord': coord_flag}

        return flags


    # close DSS and quit
    def close(self):
        self.case.reset_dss()
        
