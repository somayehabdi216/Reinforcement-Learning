
import random
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal, getcontext
getcontext().prec = 10  # Set the precision to 10 decimal places

alpha=0.01
gamma=.9
Testsize=300

'''
n = 300
nfog = 30
ncloud=30
'''

'''
n = 300
nfog = 30
ncloud=70
'''

'''
n = 400
nfog = 50
ncloud=80
'''
'''
n = 400
nfog = 25
ncloud=50
'''

'''
n = 400
nfog = 35
ncloud=40
'''


'''
n = 500
nfog = 70
ncloud=100
'''

'''
n = 500
nfog = 30
ncloud=60
'''

n = 500
nfog = 45
ncloud=45


'''
n = 200
nfog = 20
ncloud=50
'''
'''
n = 200
nfog = 15
ncloud=30
'''
'''
n = 200
nfog = 22
ncloud=23
'''


'''
n = 150
nfog = 20
ncloud=30

'''

'''
n = 100
nfog = 10
ncloud=20
'''

'''
n = 100
nfog = 15 
ncloud=15
'''


'''
n = 100
nfog = 10
ncloud=20
'''
'''
n = 70
nfog = 5
ncloud=15
'''

totalnodes=nfog+ncloud

Episodes=100000
NumberofAvg = 100
episode_rewards=[]
episode_rewards_q=[]
episode_rewards_task=np.zeros([Episodes,n])
episode_positive_rewards_number=np.zeros([Episodes])

episode_deadline_hit=np.zeros(Episodes)

Test__deadline_hit=np.zeros([Testsize,n])
Test__violation_time=np.zeros([Testsize,n])

high_priority__violation_time=np.zeros([Testsize])
low_priority__violation_time=np.zeros([Testsize])

high_priority_miss_deadline=np.zeros([Testsize])
low_priority_miss_deadline=np.zeros([Testsize])

N_high_priority_task=np.zeros([Testsize])
N_low_priority_task=np.zeros([Testsize])



Generated_Tasks_size=np.zeros([Testsize,n])
Generated_Tasks_Data=np.zeros([Testsize,n])
Generated_Tasks_Deadline=np.zeros([Testsize,n])
Generated_Tasks_prio=np.zeros([Testsize,n])
Generated_Tasks_prio_Normalized=np.zeros([Testsize,n])

ResourcePerformance=np.zeros(totalnodes)


C1=n/100           # reward for tasks that meet  their deadline
C2=n/400
C3=n/50
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#------------------------------   Task specification-------------------------------------
#----------------------------------------------------------------------------------------

TaskPriority = np.zeros(n)
Normalized_TaskPriority = np.zeros(n)
TaskData= np.zeros(n)
TaskSize = np.zeros(n)
TaskDeadline = np.zeros(n)
MeetDealine=np.zeros(n)
Wait_queue=np.zeros(n)
RunTime=np.zeros(n)
DTT=np.zeros(n)
Makespan=np.zeros(n)
Assigned_Resource=np.zeros(n)
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#------------------------------   Resource specification---------------------------------
#----------------------------------------------------------------------------------------

#ResourcePerformance= [6,8,12,14,15,16,17]

a_size_fognode=3000
b_size_fognode=5000


a_size_cloudnode=7000
b_size_cloudnode=12000


for i in range(totalnodes):
  if i<nfog:
     ResourcePerformance[i]=random.randint(a_size_fognode,b_size_fognode)
  else:
     ResourcePerformance[i]=random.randint(a_size_cloudnode,b_size_cloudnode)



ResourceQueue=[[]]*(nfog+ncloud)
UsedResource = np.ones ([1,nfog+ncloud])
for i in range(nfog):
  UsedResource [0,i] =2

for i in range(nfog ,nfog+ncloud):
  UsedResource [0,i] =20
'''
Fog_bandwidth=4000
Cloud_bandwidth=1000
'''

Fog_bandwidth=1000
Cloud_bandwidth=100



def softmax(x):
    """
    Compute the softmax function for a vector x.

    Args:
        x (numpy.ndarray): Input vector.

    Returns:
        numpy.ndarray: Probability distribution.
    """
    exp_x = np.exp(x - np.max(x))  # Subtracting np.max(x) for numerical stability
    return exp_x / exp_x.sum(axis=0, keepdims=True)

#----------------------------------------------------------------------------------------
#-----------------------------    Create tasks              -----------------------
#----------------------------------------------------------------------------------------
low_a_pri=1
low_b_pri=2

high_a_pri=3
high_b_pri=5

'''
a_dl=20
b_dl=30
a_size_dataintensive=30
b_size_dataintensive=60
a_size_computeintensive=60
b_size_computeintensive=100

a_data_dataintensive=1000
b_data_dataintensive=1500
a_data_computeintensive=100
b_data_computeintensive=700
'''

'''
a_dl_dataintensive=100
b_dl_dataintensive=500

a_dl_computintensive=500
b_dl_computintensive=2500
'''
a_dl_dataintensive=10
b_dl_dataintensive=50

a_dl_computintensive=50
b_dl_computintensive=250

a_size_dataintensive=100
b_size_dataintensive=400

a_size_computeintensive=1000
b_size_computeintensive=4000

a_data_dataintensive=1000
b_data_dataintensive=5000

a_data_computeintensive=100
b_data_computeintensive=500


def Generat_Tasks():
  data_intensive_prob=.3
  for i in range(n):
    #if random.uniform(0, 1) < data_intensive_prob:
    if i <n/3-1:
     # TaskDeadline[i]=random.randint(a_dl_dataintensive, b_dl_dataintensive)
      TaskDeadline[i]=random.uniform(a_dl_dataintensive, b_dl_dataintensive)
      TaskPriority[i]= random.randint(high_a_pri, high_b_pri)
      TaskSize[i]=random.randint(a_size_dataintensive, b_size_dataintensive)
      TaskData[i]= random.randint(a_data_dataintensive,b_data_dataintensive)
    else:
      #TaskDeadline[i]=random.randint(a_dl_computintensive, b_dl_computintensive)
      TaskDeadline[i]=random.uniform(a_dl_computintensive, b_dl_computintensive)
      TaskPriority[i]= random.randint(low_a_pri, low_b_pri)
      TaskSize[i]=random.randint(a_size_computeintensive, b_size_computeintensive)
      TaskData[i]= random.randint(a_data_computeintensive,b_data_computeintensive)

#----------------------------------------------------------------------------------------
#-----------------------------                  -----------------------------------------
#-----------------------------                  -----------------------------------------
#-----------------------------    Class Task     -----------------------------------------
#-----------------------------                  -----------------------------------------
#-----------------------------                  -----------------------------------------
#----------------------------------------------------------------------------------------
class Task:
    def __init__(self, TaskId, TaskSize, TaskData, TaskDeadline, TaskPriority ):
        self.TaskId =TaskId
        self.TaskSize =TaskSize
        self.TaskData = TaskData
        self.TaskDeadline = TaskDeadline
        self.TaskPriority = TaskPriority
        self.runtime = 0
        self.starttime=0
        self.waitingtime=0
        self.meetdeadline=0
        self.makespan=0

    def set_waitingtime(self, waitingtime):
       self.waitingtime=waitingtime

    def set_starttime(self, starttime):
       self.starttime=starttime

    def set_meetdeadline(self, meetdeadline):
       self.meetdeadline=meetdeadline

    def set_makespan(self, makespan):
       self.makespan=makespan

    def set_runtime(self, runtime):
       self.runtime=runtime

    def get_makespan(self):
       return self.makespan


#----------------------------------------------------------------------------------------
#-----------------------------                  -----------------------------------------
#-----------------------------                  -----------------------------------------
#-----------------------------  Class Tasklist  -----------------------------------------
#-----------------------------                  -----------------------------------------
#-----------------------------                  -----------------------------------------
#----------------------------------------------------------------------------------------

class TaskList:
    def __init__(self, ntask):
        self.ntask = ntask
        self.tasklist = []

    def add_task(self, TaskId, TaskSize, TaskData, TaskDeadline, TaskPriority):
        task = Task(TaskId, TaskSize, TaskData, TaskDeadline, TaskPriority)
        self.tasklist.append(task)

    def sort_subtasks_by_priority(self):
        self.tasklist.sort(key=lambda x: x.TaskPriority, reverse=True)

    def get_tasklist(self):
        return self.tasklist

    def show_tasklist(self):
      for i, subtask in enumerate(self.tasklist):
        print(f"Subtask {i+1}: Size {subtask.size}, Priority {subtask.priority}")
        print()
#------------------------------------------------------------------------------------------------------------
#-----------------------------                                      -----------------------------------------
#-----------------------------                                      -----------------------------------------
#-----------------------------    Class TaskAssignmentEnvironment   -----------------------------------------
#-----------------------------                                      -----------------------------------------
#-----------------------------                                      -----------------------------------------
#------------------------------------------------------------------------------------------------------------


class TaskAssignmentEnvironment:
    def __init__(self, n, nfog, ncloud):
        self.n = n
        self.nfog = nfog
        self.ncloud = ncloud
        self.state_space = n
        self.action_space = nfog + ncloud
        self.queue=[[] for _ in range(nfog + ncloud)]


    def add_task_to_queue(self, action, state):
      if 0 <= action < len(self.queue):
          self.queue[action].append(state)
          return True
      else:
          print('out of index')
          return False


    def get_state(self):
        return random.randint(0, self.state_space - 1)

    def take_action1(self):
      reward=0
      for j in range(n):
        state=j
        #print(state)
        action = agent.choose_action(state)
        run_time = TaskSize[state] /  ResourcePerformance[action]
        node = action
        waiting_time=0
        for i in self.queue[action]:
          #waiting_time+=TaskSize[i] /  ResourcePerformance[action]
           waiting_time+=Makespan[i]
        self.add_task_to_queue( action, state)
        if node < self.nfog:
            response_time = waiting_time+ run_time + (TaskData[state] /Fog_bandwidth)  # consider data transfer time for fog response time
        else:
            response_time =waiting_time+ run_time + (TaskData[state] /Cloud_bandwidth)  #  consider data transfer time for cloud response time

        Wait_queue[state]= waiting_time
        Makespan[state]=response_time
        # Reward function (considering task deadline)
        if response_time<= TaskDeadline[state]:
           #reward = response_time *Normalized_TaskPriority[state]*C1
           reward += Normalized_TaskPriority[state]*C1
           #reward = C1*TaskPriority[state]
           #reward = response_time *TaskPriority[state]
           MeetDealine[state]=1
        else:
          reward -= (response_time-TaskDeadline[state])*Normalized_TaskPriority[state]*C2
          #reward = -(response_time-TaskDeadline[state])*TaskPriority[state]*C2
          #reward = -C2

        Assigned_Resource[state]=node
        return state, node, reward




    def take_action(self, state, action):
        run_time = TaskSize[state] /  ResourcePerformance[action]
        node = action
        waiting_time=0
        for i in self.queue[action]:
          #waiting_time+=TaskSize[i] /  ResourcePerformance[action]
           waiting_time+=Makespan[i]
        self.add_task_to_queue( action, state)
        if node < self.nfog:
            response_time = waiting_time+ run_time + (TaskData[state] /Fog_bandwidth)  # consider data transfer time for fog response time
        else:
            response_time =waiting_time+ run_time + (TaskData[state] /Cloud_bandwidth)  #  consider data transfer time for cloud response time

        Wait_queue[state]= waiting_time
        Makespan[state]=response_time
        # Reward function (considering task deadline)
        if response_time<= TaskDeadline[state]:
           #reward = response_time *Normalized_TaskPriority[state]*C1
           #reward = C1
           reward =C1*TaskPriority[state]
           #reward = C1*TaskPriority[state]
           #reward = response_time *TaskPriority[state]
           MeetDealine[state]=1
        else:
          reward = -(response_time-TaskDeadline[state])*TaskPriority[state]*C2
          #reward = -(response_time-TaskDeadline[state])
          #reward = -(response_time-TaskDeadline[state])*TaskPriority[state]*C2
          #reward = -C2*Normalized_TaskPriority[state]
        Assigned_Resource[state]=node
        return state, node, reward


    def Reset_ResourceQueue(self):
             self.queue=[[] for _ in range(nfog + ncloud)]


    def check_deadline(self,state,action,j):
         run_time = TaskSize[state] /  ResourcePerformance[action]
         node = action
         waiting_time=0
         for i in self.queue[action]:
             waiting_time+=Makespan[i]
             #print(f'waiting time',waiting_time )
         self.add_task_to_queue( action, state)

         Meet_Deadline=0
         if node < self.nfog:
              response_time =waiting_time+ run_time + (TaskData[state] /Fog_bandwidth) #  consider data transfer time for fog response time
              #print(f'waiting_time', waiting_time)
              #print(f'run_time', run_time )
              #print(f'DTT', (TaskData[state] /Fog_bandwidth)  )
              #print(f'makespan', response_time )
              #response_time = Decimal(run_time) + Decimal(TaskData[state] /Fog_bandwidth)  # consider data transfer time for fog response time
              DTT[state]=TaskData[state] /Fog_bandwidth
         else:
              response_time =waiting_time+ run_time + (TaskData[state] /Cloud_bandwidth)  #  consider data transfer time for cloud response time
              #response_time =Decimal(waiting_time)+Decimal(run_time) + Decimal(TaskData[state] /Cloud_bandwidth)
              #print(f'waiting_time', waiting_time)
              #print(f'run_time', run_time )
              #print(f'DTT', (TaskData[state] /Cloud_bandwidth)  )
              #print(f'makespan', response_time )
              DTT[state]=Decimal(TaskData[state] /Cloud_bandwidth)

         Wait_queue[state]= waiting_time
         RunTime[state]=run_time
         Makespan[state]= response_time                          #round(response_time,2)
         #print(f'makespan2', Makespan[state] )
         if TaskPriority[state]== 3 or TaskPriority[state]== 4 or TaskPriority[state]== 5:
           N_high_priority_task[j]=N_high_priority_task[j]+1
         else:
           N_low_priority_task[j]=N_low_priority_task[j]+1

         if Makespan[state]<= TaskDeadline[state]:
            MeetDealine[state]=1
            Test__deadline_hit[j][state]=1
         else:
          Test__violation_time[j][state]= Makespan[state]-TaskDeadline[state]
          if TaskPriority[state]== 3 or TaskPriority[state]== 4 or TaskPriority[state]== 5:
            high_priority__violation_time[j]=high_priority__violation_time[j]+(Makespan[state]-TaskDeadline[state] )
            high_priority_miss_deadline[j]=high_priority_miss_deadline[j]+1

          else:
            low_priority__violation_time[j]=low_priority__violation_time[j]+(Makespan[state]-TaskDeadline[state] )
            low_priority_miss_deadline[j]= low_priority_miss_deadline[j]+1

#----------------------------------------------------------------------------------------
#-----------------------------                                      -----------------------------------------
#-----------------------------                                      -----------------------------------------
#-----------------------------    Class QLearningAgent              -----------------------------------------
#-----------------------------                                      -----------------------------------------
#-----------------------------                                      -----------------------------------------
#----------------------------------------------------------------------------------------



class QLearningAgent:
    def __init__(self, state_space, action_space,  learning_rate=alpha, discount_factor=gamma, exploration_prob=.5):
        self.state_space = state_space
        self.action_space = action_space
        self.q_table = [[0 for _ in range(action_space)] for _ in range(state_space)]
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_prob:
            return random.randint(0, self.action_space - 1)
        else:
            return self.q_table[state].index(max(self.q_table[state]))

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (reward + self.discount_factor * best_next_action - self.q_table[state][action])

    def set_exploration_prob(self, exploration_prob):
        self.exploration_prob=exploration_prob


#----------------------------------------------------------------------------------------
#-----------------------------    Create environment              -----------------------
#----------------------------------------------------------------------------------------


env = TaskAssignmentEnvironment(n, nfog, ncloud)

#----------------------------------------------------------------------------------------
#-----------------------------    Create Q-learning agent              ------------------
#----------------------------------------------------------------------------------------

agent = QLearningAgent(env.state_space, env.action_space)


#----------------------------------------------------------------------------------------
#-----------------------------    Training loop             ------------------
#----------------------------------------------------------------------------------------
Episode_reward=[]
Episode_state= []
Episode_action= []
Average_rewards = []
nineteenPercentOfEpisodes=0.9 * Episodes
seventeenPercentOfEpisodes= 0.7 * Episodes
halfOfEpisodes= 0.5 * Episodes
ac_reward=np.zeros(Episodes)
previous_reward=0
for j in range(Episodes):
    Episode_reward=[]
    Episode_state= []
    Episode_action= []
    previous_reward=0
    all_deadline=True
    Generat_Tasks()
    Normalized_TaskPriority = softmax(TaskPriority)
    env.Reset_ResourceQueue()
    if  seventeenPercentOfEpisodes >j> halfOfEpisodes:
      agent.set_exploration_prob(0.3)
    elif j>seventeenPercentOfEpisodes:
      agent.set_exploration_prob(0.1)
    elif j>nineteenPercentOfEpisodes:
      agent.set_exploration_prob(0.05)
    for i in range(n):
      #state = env.get_state()
      state=i
      #print(state)
      Episode_state.append(state)
      action = agent.choose_action(state)
      Episode_action.append(action)
      task, node, reward = env.take_action(state, action)
      Episode_reward.append(reward)
      '''T=reward
      if previous_reward>0:
        reward+=previous_reward
      previous_reward+=T'''
      if reward<0:
        all_deadline=False
    for i in range(n):
      if all_deadline==True:
        Episode_reward[i]=C3
      agent.update_q_table(i, Episode_action[i], Episode_reward[i], env.get_state())
      ac_reward[j]+= Episode_reward[i]
      if reward>=0:
        episode_positive_rewards_number[j]+=1
      episode_rewards_task[j][i]=reward

    if j % NumberofAvg == 0 and j>NumberofAvg-1:
        Average_rewards.append(np.mean(ac_reward[j-NumberofAvg-1:j]))
    episode_rewards.append(ac_reward[j])
    #episode_rewards_task[j]=np.count_nonzero(MeetDealine)


# Test the trained agent
for j in range(Testsize):
  agent.set_exploration_prob(0)
  Generat_Tasks()
  Normalized_TaskPriority = softmax(TaskPriority)
  Generated_Tasks_size[j]=TaskSize
  Generated_Tasks_Data[j]=TaskData
  Generated_Tasks_Deadline[j]=TaskDeadline
  Generated_Tasks_prio[j]=TaskPriority

  Generated_Tasks_prio_Normalized[j]=Normalized_TaskPriority
  env.Reset_ResourceQueue()
  for task in range(n):
      state = task
      action = agent.choose_action(state)
      node = action
      env.check_deadline(state,action,j)
      '''
      if node < nfog:
          print(f"Task {task+1} scheduled on Fog Node {node+1}")
      else:
          print(f"Task {task+1} scheduled on Cloud Node {node+1}")
     '''