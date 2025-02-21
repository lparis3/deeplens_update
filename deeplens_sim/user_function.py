from deeplens_sim.simulation import dl_sim
import os
from datetime import date


def User_Inputs(DM_Type, Instrument, Sim_Number, Results_List = False):
  '''Only function user has to deal with when simulating.'''
    results_list = []
    os.mkdir(f'./{DM_Type}_{Instrument}_{Sim_Number}')
    for i in np.linspace(1,Sim_Number, num = Sim_Number):
      results = dl_sim(DM_Type, Instrument)
      if Results_List == True:
        results_list.append(results)
      
      
