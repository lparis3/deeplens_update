import os
import numpy as np
from datetime import date
from deeplens_sim.simulation import dl_sim


def User_Inputs(DM_Type, Instrument, Sim_Number, Results_List = False):
  '''Only function user has to deal with when simulating.'''
  results_list = []
  Date = date.today().strftime("%x")
  os.mkdir(f'./{DM_Type}_{Instrument}_{Sim_Number}_{Date}')
  for i in np.linspace(1,Sim_Number, num = Sim_Number):
    results = dl_sim(DM_Type, Instrument)
    if Results_List == True:
      results_list.append(results)
    np.save(f'{DM_Type}_{Instrument}_{Sim_Number}_{date.today().strftime("%x")}/{DM_Type}_{Instrument}_{i}')
    print('Simulation {i}/{Sim_Num} Done!')
  if not results_list:
    print(f'All {Sim_Number} Simulations Done!')
  else:
    print(f'All {Sim_Number} Simulations Done!')
    return results_list
    

    
      
      
      
