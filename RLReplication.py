#RL Replication 1.0, Phanish Puranam May 2018
#Python 2.7 Code for Replication Exercise for Reinforcement Learning 

#IMPORTING MODULES
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import datetime
import csv# for csv output, uncomment lines 110-115

STARTINGTIME = datetime.datetime.now().replace(microsecond=0)

#SIMULATION PARAMETERS
T=100 #number of periods to simulate the model
N=10000 #number of trials

# TASK ENVIRONMENT
S=10# size of task environment
delta=0.5# payoff on non-peak arlternatives range from zero to delta (peak is always 1)

# AGENT PARAMETERS
phi=0.1 #learning rate in EWRA learning 
tau=0.008 # exploration in softmax

#DEFINING RESULTS VECTORS (FOR STORAGE)
# we will produce per agent, one vector of length T and breadth=5 that reports average across Trials per period and cumulative performance  
org_perf=np.zeros((T,N)) # payoff in each period for each trial
org_cumperf=np.zeros((T,N))#cumulative payoffs by period and trial
spike=np.zeros((T,N)) # search success i.e. if the global peak was found
result=np.zeros((T,5)) # this stores the aggregated results which we will show on graphs
       
#DEFINING FUNCTIONS

def environment(d): #setting environment 
    if d==0:
       env=np.zeros((1,S))    
    else:
        env=0.01*(np.random.choice(int(100*d),(1,S),replace=True))
    col1=random.randint(0, S-1)
    env[0,col1]=1 # One global peak
    return env

def softmax(attraction): #softmax action selection with attraction vector as parameters
    prob=np.zeros((1,S))
    denom=0
    for i in range(S):
        denom=denom + math.exp((attraction[0,i])/tau)
    roulette=random.random()
    p=0
    for i in range(S):
        prob[0,i]=math.exp(attraction[0,i]/tau)/denom
        p= p+ prob[0,i]
        if p>roulette:
            choice= i
            return choice
            break #stops computing probability of action selection as soon as cumulative probability exceeds roulette

# running simulation here
for a in range(N): # for each trial
    Environment=environment(delta) #True payoffs to each state
    att=0.1*(np.random.choice(int(11),(1,S),replace=True))
    for t in range(T): # for each period
        choice=softmax(att) # agent  makes a choice for period t
        payoff=Environment[0,choice] # gets a payoff for this choice
        org_perf[t,a]=payoff # 
        att[0,choice]= att[0,choice]+ phi*(payoff-att[0,choice])
        if payoff==1:
            spike[t,a]=1
        else:
            spike[t,a]=0
        if t>0: 
            org_cumperf[t][a]=org_cumperf[t-1][a]+ payoff
        else:
            org_cumperf[t][a]=payoff

#PRODUCE RESULTS
for t in range(T):
    result[t][0]=t+1
    result[t][1]=float(np.sum(org_perf[t,:]))/N
    result[t][2]=float(np.sum(org_cumperf[t,:]))/N
    result[t][3]=float(np.std(org_perf[t,:]))
    result[t][4]=float(np.sum(spike[t,:]))/N

#MAKE GRAPHS
def t_by_t_plot(x, y, title, axe, fontsize=12):
    axe.plot(x, y, color='red')
    ymin = np.min(y)
    ymax = np.max(y)
    axe.set(ylim=(ymin - 0.05*(ymax-ymin), ymax + 0.05*(ymax-ymin)))
    axe.set_xlabel('t', fontsize=fontsize)
    axe.set_title(title, fontsize=fontsize, fontweight="bold")    
  
plt.style.use('ggplot') # Setting the plotting style

fig = plt.figure(figsize=(3*8.27, 3*11.69), dpi=250)
ax1 = plt.subplot2grid((6,5),(0,1), colspan = 3)
t_by_t_plot([x[0] for x in result], [x[1] for x in result], "Performance", ax1, fontsize=12)

fig = plt.figure(figsize=(3*8.27, 3*11.69), dpi=250)
ax1 = plt.subplot2grid((6,5),(0,1), colspan = 3)
t_by_t_plot([x[0] for x in result], [x[2] for x in result], "Cumulative Performance", ax1, fontsize=12)

fig = plt.figure(figsize=(3*8.27, 3*11.69), dpi=250)
ax1 = plt.subplot2grid((6,5),(0,1), colspan = 3)
t_by_t_plot([x[0] for x in result], [x[3] for x in result], "Std Dev of Performance", ax1, fontsize=12)

fig = plt.figure(figsize=(3*8.27, 3*11.69), dpi=250)
ax1 = plt.subplot2grid((6,5),(0,1), colspan = 3)
t_by_t_plot([x[0] for x in result], [x[4] for x in result], "Search Success", ax1, fontsize=12)

#EXCEL OUTPUT IF NEEDED
#results = open('RLReplication.csv', 'wb')
#writer = csv.writer(results)
#writer.writerow(['Period','Performance', 'Cumul Perf', 'Stdev perf','Search Success' ]) 
#for values in result:
#    writer.writerow(values)
#results.close()   

ENDINGTIME = datetime.datetime.now().replace(microsecond=0)
TIMEDIFFERENCE = ENDINGTIME - STARTINGTIME
#print 'Computation time:', TIMEDIFFERENCE
#print ('Computation time:', TIMEDIFFERENCE) #uncomment and use this in pythin 3.X 