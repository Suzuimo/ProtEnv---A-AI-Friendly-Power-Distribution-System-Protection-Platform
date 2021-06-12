from .Agent_Template import agent
from gym import spaces
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

class DQNRelayAgent(agent):
    def __init__(self, bus1=None, bus2=None, line=None, weightPath=None):
        self.bus1 = bus1
        self.bus2 = bus2
        self.successors = None
        self.line = line
        self.tripped = False
        self.open = False
        self.waveform = None
        self.obs = ['Iseq']
        self.phases = 3
        self.svNum = None
        self.actNum = None
        self.trainable = True
        self.env = None

        # customizable parameters
        self.verbose = 2
        self.lr = 0.001
        self.trainingSteps = 40000
        self.windowLength = 10
        self.counterDelays = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1, 1.3, 1.5, -1]

        # learning parameters
        self.weightPath = weightPath
        self.bufferSize = 2048
        self.svNum = 3 * self.windowLength + 1 # seq current in past 6 steps + triggered Flag
        self.actNum = len(self.counterDelays)
        self.state = None
        self.action_space = spaces.Discrete(self.actNum)
        self.observation_space = spaces.Discrete(self.svNum)
        self.memory = SequentialMemory(limit=20000, window_length=1)
        self.policy = BoltzmannQPolicy()
        self.model = self.build_model()

        # relay parameters
        self.triggered = False
        self.triggerTime = None
        self.tripTime = None
        self.delay = None

    # build the DQN model for this agent
    def build_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=(1,self.svNum)))
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dense(self.actNum))
        model.add(Activation('linear'))

        return model

    
    # process state and get model output
    def process_state(self):
        action = np.argmax(self.model.predict(self.state.reshape((1,1,31))))

        return action
        
    # compute real relay action of the agent using model action
    def act(self, action):
        actVal = self.counterDelays[action]
        trip = 0
        
        # if already open, no need to move further 
        if self.open:
            return trip
        
        # check if counter is triggered
        if self.triggered:
            # reset
            if actVal == -1:
                self.triggered = False
            # check if need to trip (delay has passed)
            elif self.time - self.triggerTime >= self.delay:
                self.tripped = True
                self.triggered = False
                self.tripTime = self.time
                trip = 1
        # if not triggered, check if need to be
        else:
            # if positive, set the countdown
            if actVal > 0:
                self.triggered = True
                self.triggerTime = self.time
                self.delay = actVal

        return trip
                
                
        
    # observe the case and update state container
    def observe(self, case):
        I = case.get_line_I(self.line, 'Iseq', self.phases)
            
        self.time = case.sol.Seconds
        # store sequence current observation
        # add noise (assuming +-3% IT accuracy)
        self.waveform['I0'].append(I[0] * np.random.uniform(0.97, 1.03))
        self.waveform['I1'].append(I[1] * np.random.uniform(0.97, 1.03))
        self.waveform['I2'].append(I[2] * np.random.uniform(0.97, 1.03))

        # format into self.state
        # [I0, I1, I2]
        I0 = np.zeros(self.windowLength)
        I1 = np.zeros(self.windowLength)
        I2 = np.zeros(self.windowLength)
        
        # number of observations available
        stepNum = len(self.waveform['I0'])
        if stepNum < self.windowLength:
            # fill all with initial value
            I0[:] = self.waveform['I0'][0]
            I1[:] = self.waveform['I1'][0]
            I2[:] = self.waveform['I2'][0]

            # fill the latest will real value
            I0[-stepNum:] = self.waveform['I0'][-stepNum:]
            I1[-stepNum:] = self.waveform['I1'][-stepNum:]
            I2[-stepNum:] = self.waveform['I2'][-stepNum:]
        else:
            I0[:] = self.waveform['I0'][-self.windowLength:]
            I1[:] = self.waveform['I1'][-self.windowLength:]
            I2[:] = self.waveform['I2'][-self.windowLength:]
            
        # concatenate I012 to form state vector
        self.state = np.concatenate([I0, I1, I2, [self.triggered]], axis=0)
        

        
    

    # reset the agent's internal states
    def reset(self):
        self.tripped = False
        self.open = False
        self.waveform = {'I0':[], 'I1':[], 'I2':[]}
        self.state = None
        self.triggerTime = None
        self.triggered = False
        self.delay = None

        # load previous weights if provided
        if not self.weightPath == None:
            self.load()
    
    # save the trained weight into local folder
    def save(self):
        self.trainer.save_weights(self.weightPath, overwrite=True)

    # load
    def load(self):
        self.trainer = DQNAgent(self.model, \
                                nb_actions=self.actNum,
                                memory=self.memory, \
                                target_model_update=1e-3, \
                                policy=self.policy, \
                                nb_steps_warmup=1000, \
                                enable_double_dqn=True, \
                                enable_dueling_network=False)

        self.trainer.compile(Adam(learning_rate=self.lr), metrics=['mse'])
        self.trainer.load_weights(self.weightPath)

    # use keras-rl to train this agent
    def train(self, env):
        self.trainer = DQNAgent(self.model, \
                                nb_actions=self.actNum,
                                memory=self.memory, \
                                target_model_update=1e-3, \
                                policy=self.policy, \
                                nb_steps_warmup=1000, \
                                enable_double_dqn=True, \
                                enable_dueling_network=False)

        self.trainer.compile(Adam(learning_rate=self.lr), metrics=['mse'])
        self.hist = self.trainer.fit(env, nb_steps=self.trainingSteps, visualize=False, verbose=self.verbose)



        
    
