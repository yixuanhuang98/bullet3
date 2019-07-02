import os,  inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)

import math
import gym
import time
from gym import spaces
from gym.utils import seeding
import numpy as np
import pybullet
from . import racecar
import random
from . import bullet_client
import pybullet_data
from pkg_resources import parse_version

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

class RacecarGymEnv(gym.Env):
  metadata = {
      'render.modes': ['human', 'rgb_array'],
      'video.frames_per_second' : 50
  }

  def __init__(self,
               urdfRoot=pybullet_data.getDataPath(),
               actionRepeat=50,
               isEnableSelfCollision=True,
               isDiscrete=False,
               renders=False):
    #print("init")
    self._timeStep = 0.01
    self._urdfRoot = urdfRoot
    self._actionRepeat = actionRepeat
    self._isEnableSelfCollision = isEnableSelfCollision
    self._observation = []
    self._ballUniqueId = -1
    self._envStepCounter = 0
    self._renders = renders
    self._isDiscrete = isDiscrete
    self.violation_flag = False
    self.step_violation = 0
    self.total_violation =0 
    self.total_perfect = 0
    self.total_finish_with_violation = 0
    if self._renders:
      self._p = bullet_client.BulletClient(
          connection_mode=pybullet.GUI)
    else:
      self._p = bullet_client.BulletClient()

    self.seed()
    #self.reset()
    observationDim = 2 #len(self.getExtendedObservation())
    #print("observationDim")
    #print(observationDim)
    # observation_high = np.array([np.finfo(np.float32).max] * observationDim)
    observation_high = np.ones(observationDim) * 1000 #np.inf
    if (isDiscrete):
      self.action_space = spaces.Discrete(9)
    else:
       action_dim = 2
       self._action_bound = 1
       action_high = np.array([self._action_bound] * action_dim)
       self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
    self.observation_space = spaces.Box(-observation_high, observation_high, dtype=np.float32)
    self.viewer = None

  def reset(self):
    #print('enter reset function')
    self.violation_flag = False
    if(self.step_violation != 0):
      print('step_violation')
      print(self.step_violation)
    # if(self.total_violation % 5 ==0 ):
    #   print('total_violation')
    #   print(self.total_violation)
    if(self.total_perfect %10 == 0 and self.total_perfect != 0):
      print('total_perfect')
      print(self.total_perfect)
      print('total_finished_with_violation')
      print(self.total_finish_with_violation)
      print('total_violation')
      print(self.total_violation)
    self.step_violation = 0
    self._p.resetSimulation()
    #p.setPhysicsEngineParameter(numSolverIterations=300)
    self._p.setTimeStep(self._timeStep)
    #self._p.loadURDF(os.path.join(self._urdfRoot,"plane.urdf"))
    stadiumobjects = self._p.loadSDF(os.path.join(self._urdfRoot,"stadium.sdf"))
    #move the stadium objects slightly above 0
    #for i in stadiumobjects:
    #	pos,orn = self._p.getBasePositionAndOrientation(i)
    #	newpos = [pos[0],pos[1],pos[2]-0.1]
    #	self._p.resetBasePositionAndOrientation(i,newpos,orn)

    dist = 5
    ang = -3.1415925438/2

    ballx = dist * math.sin(ang)
    bally = dist * math.cos(ang)
    ballz = 1

    # dist = 5 + 2. * random.random()
    # ang = 2. * 3.1415925438 * random.random()

    # ballx = dist * math.sin(ang)
    # bally = dist * math.cos(ang)
    # ballz = 1


    


    self._ballUniqueId = self._p.loadURDF(os.path.join(self._urdfRoot,"sphere2.urdf"),[ballx,2,ballz])
    self._ballUniqueId1 = self._p.loadURDF(os.path.join(self._urdfRoot,"sphere2.urdf"),[ballx,-2,ballz])
    self._ballUniqueId2 = self._p.loadURDF(os.path.join(self._urdfRoot,"sphere2.urdf"),[-ballx,2,ballz])
    self._ballUniqueId3 = self._p.loadURDF(os.path.join(self._urdfRoot,"sphere2.urdf"),[-ballx,-2,ballz])
    self.field1 = self._p.loadURDF(os.path.join(self._urdfRoot,"r2d2.urdf"),[ballx,1,ballz])
    self.field2 = self._p.loadURDF(os.path.join(self._urdfRoot,"r2d2.urdf"),[ballx,-1,ballz])
    self.field3 = self._p.loadURDF(os.path.join(self._urdfRoot,"r2d2.urdf"),[-ballx,1,ballz])
    self.field4 = self._p.loadURDF(os.path.join(self._urdfRoot,"r2d2.urdf"),[-ballx,-1,ballz])

    self._p.setGravity(0,0,-10)
    self._racecar = racecar.Racecar(self._p,urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
    self._envStepCounter = 0
    for i in range(100):
      self._p.stepSimulation()
    self._observation = self.getExtendedObservation()
    return np.array(self._observation)

  def __del__(self):
    self._p = 0

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def getExtendedObservation(self):
     self._observation = [] #self._racecar.getObservation()
     carpos,carorn = self._p.getBasePositionAndOrientation(self._racecar.racecarUniqueId)
     ballpos,ballorn = self._p.getBasePositionAndOrientation(self._ballUniqueId)
     invCarPos,invCarOrn = self._p.invertTransform(carpos,carorn)
     ballPosInCar,ballOrnInCar = self._p.multiplyTransforms(invCarPos,invCarOrn,ballpos,ballorn)

     self._observation.extend([carpos[0],carpos[1]])
     return self._observation

  def step(self, action):
    if (self._renders):
      basePos,orn = self._p.getBasePositionAndOrientation(self._racecar.racecarUniqueId)
      #self._p.resetDebugVisualizerCamera(1, 30, -40, basePos)

    if (self._isDiscrete):
	    fwd = [-1,-1,-1,0,0,0,1,1,1]
	    steerings = [-1,0,1,-1,0,1,-1,0,1] # previous : 0.6
	    forward = fwd[action]
	    steer = steerings[action]
	    realaction = [forward,steer]
    else:
      #print('not discrete')   # default for then pybullet is continuous
      realaction = action#action [1,0.6] right down [-1,0.6] upper right   [-1,-0.6] upper left

    self._racecar.applyAction(realaction)
    for i in range(self._actionRepeat):
      self._p.stepSimulation()
      if self._renders:
        time.sleep(self._timeStep)
      self._observation = self.getExtendedObservation()

      if self._termination():
        break
      self._envStepCounter += 1
    reward = self._reward()
    done = self._termination()
    carpos,carorn = self._p.getBasePositionAndOrientation(self._racecar.racecarUniqueId)
    ballpos,ballorn = self._p.getBasePositionAndOrientation(self._ballUniqueId)
    # print('carpos')    # right down is (+,+) 
    # print(carpos)
    
    if(carpos[1] > 1 or carpos[1] < -1):    #(left:-   right: +)
      reward += -3*(abs(carpos[1]))
    # print('carpos')
    # print(carpos)
    # print('action')
    # print(action)
    # print('reward')
    # print(reward)
    # if(carpos[1] > 2 or carpos[1] < -2):
    #   done = True
    if(carpos[1] > 3 or carpos[1] < -3):
      self.violation_flag = True
      self.step_violation += 1
      self.total_violation += 1
      #reward += -5
      print('violation')
    if(carpos[0] < -5 and self.violation_flag == False):
      print('success')
      self.total_perfect += 1
      done = True
    elif(carpos[0] < -5 and self.violation_flag == True):
      print('finished with violation')
      self.total_finish_with_violation += 1
      done = True
    
    #print("len=%r" % len(self._observation))
    # print('ob')
    # print(np.array(self._observation))
    return np.array(self._observation), reward, done, {}

  def render(self, mode='human', close=False):
    if mode != "rgb_array":
      return np.array([])
    base_pos,orn = self._p.getBasePositionAndOrientation(self._racecar.racecarUniqueId)
    view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=base_pos,
        distance=self._cam_dist,
        yaw=self._cam_yaw,
        pitch=self._cam_pitch,
        roll=0,
        upAxisIndex=2)
    proj_matrix = self._p.computeProjectionMatrixFOV(
        fov=60, aspect=float(RENDER_WIDTH)/RENDER_HEIGHT,
        nearVal=0.1, farVal=100.0)
    (_, _, px, _, _) = self._p.getCameraImage(
        width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix,
        projectionMatrix=proj_matrix, renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
    rgb_array = np.array(px)
    rgb_array = rgb_array[:, :, :3]
    return rgb_array


  def _termination(self):
    return self._envStepCounter > 1000

  def _reward(self):
    carpos,carorn = self._p.getBasePositionAndOrientation(self._racecar.racecarUniqueId)
    ballpos,ballorn = self._p.getBasePositionAndOrientation(self._ballUniqueId)
    
    closestPoints = self._p.getClosestPoints(self._racecar.racecarUniqueId,self._ballUniqueId,10000)

    numPt = len(closestPoints)
    # reward=-1000
    # #print(numPt)
    # if (numPt>0):
    #   #print("reward:")
    #   reward = -closestPoints[0][8]
    #   #print(reward)
    reward = (-5-carpos[0])
    # reward = 0
    return reward

  if parse_version(gym.__version__) < parse_version('0.9.6'):
    _render = render
    _reset = reset
    _seed = seed
    _step = step

