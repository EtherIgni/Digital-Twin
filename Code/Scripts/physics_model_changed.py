import numpy as np
from scipy.interpolate import CubicSpline
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
from physics_stuff.modules.update_temp_curve import update_Temperature
import global_run_number as g 






def calc_geometric_curves(segment_lengths,
                          num_nodes_in_segment,
                          segment_areas):
    point_curve_list = []
    area_curve_list  = []
    running_sum      = 0
    for i in range(len(segment_lengths)):
        point_curve  = np.linspace(running_sum,
                                   running_sum + segment_lengths[i],
                                   int(num_nodes_in_segment[i]),
                                   False)
        area_curve   = (point_curve*0+1) * segment_areas[i]
        
        running_sum += segment_lengths[i]
        
        point_curve_list.append(point_curve)
        area_curve_list.append(area_curve)
        
    point_curve      = np.concat(point_curve_list)
    
    area_curve       = np.concat(area_curve_list)
    
    return(point_curve,
           area_curve,
           running_sum)
    
    
    
    
    
    
def simulate_Data(data_folder_file_path,Heater,Pump1,Pump2,Wall):
    with open(os.path.dirname(os.path.realpath(__file__))+"/physics_stuff/calibrated_model_information/geometry.pkl","rb") as file:
            geometry_dict = pickle.load(file)
            
    with open(os.path.dirname(os.path.realpath(__file__))+"/physics_stuff/calibrated_model_information/constants.pkl","rb") as file:
            constants_dict = pickle.load(file)
            
    with open(os.path.dirname(os.path.realpath(__file__))+"/physics_stuff/calibrated_model_information/parameters.pkl","rb") as file:
            parameters_dict = pickle.load(file)

    num_to_pull                 = 10
    steps_per_pull              = 10
    set_100                     = 0
    Gal_min_to_Kg_s             = 0.0466
    
    physical_data_all           = pd.read_csv(data_folder_file_path+"filtered_data.csv", index_col=False, header=None)
    physical_data           = np.array(physical_data_all)[-(num_to_pull+1):]
    

    time_grid                   = physical_data[:,0]

    temperature_data            = physical_data[:,5:]

    mass_flow_rates             = np.zeros((num_to_pull+1,3))
    for i in range(3):
        if sum([Pump1,Pump2,Wall])==0: 
            print(np.shape(physical_data[:,i+1]))
            mass_flow_rates[:,i]      = physical_data[:,i+1] * Gal_min_to_Kg_s
        else:
            with open(data_folder_file_path + "control.txt",'r') as control:
                 L = control.readline()

            if L=="":
                with open(data_folder_file_path + "control.txt",'w') as control:
                     control.write(str(len(physical_data_all)))
            
            with open(data_folder_file_path + "control.txt",'r') as control:
                 L = int(control.readline())
            print(len(mass_flow_rates))
            mass_flow_rates[:,i]      = physical_data_all.to_numpy()[L-11:L,i+1] * Gal_min_to_Kg_s

                


    heater_flux                 = physical_data[:,4] * parameters_dict["Heater Conversion"]



    inlet_temps                 = physical_data[:,11]
    
    
    temp_curve                  = [None]*3
    simulated_data              = np.zeros((num_to_pull,12))
        
    if(os.path.isfile(data_folder_file_path+"model_save.csv")):
        temp_dataframe    = pd.read_csv(data_folder_file_path+"model_save.csv")
        for i in range(3):
            temp_curve[i] = temp_dataframe.loc[i].to_list()
            temp_curve[i] = [x for x in temp_curve[i] if str(x) != 'nan']
            temp_curve[i] = np.array(temp_curve[i])[1:]
            
    else:
        initial_temps              = [None]*3
        initial_temps[0]           = np.array([physical_data[0,8],  physical_data[0,7], physical_data[0,6], physical_data[0,5]])
        initial_temps[1]           = np.array([physical_data[0,9],  physical_data[0,10]])
        initial_temps[2]           = np.array([physical_data[0,11], physical_data[0,12]])
        
        print(initial_temps)
        for i in range(2):
            temp_curve[i]    = np.interp(geometry_dict["Node Positions"][i],
                                        geometry_dict["Thermometer Positions"][i],
                                        initial_temps[i],
                                        period=geometry_dict["Loop Lengths"][i])
        temp_curve[2]    = np.interp(geometry_dict["Node Positions"][2],
                                    np.concat([[0],geometry_dict["Thermometer Positions"][2]]),
                                    initial_temps[2])






    for step in range(num_to_pull):
        time_diff                   = time_grid[step+1] - time_grid[step]

        for i in range(steps_per_pull):
            temp_curve,probe_temps = update_Temperature(temp_curve,
                                                        geometry_dict,
                                                        constants_dict,
                                                        parameters_dict,
                                                        inlet_temps[step],
                                                        time_diff/steps_per_pull,
                                                        heater_flux[step],
                                                        mass_flow_rates[step])


        simulated_data[step,0] = time_grid[step+1]
        simulated_data[step,1] = probe_temps[0][2]
        simulated_data[step,2] = probe_temps[0][1]
        simulated_data[step,3] = probe_temps[0][0]
        simulated_data[step,4] = probe_temps[0][3]
        simulated_data[step,5] = probe_temps[1][0]
        simulated_data[step,6] = probe_temps[1][1]
        simulated_data[step,7] = probe_temps[2][0]
        






    data_frame = pd.DataFrame.from_records(simulated_data)
    data_frame.to_csv(data_folder_file_path+"simulated_data.csv",mode="a",header=False,index=False)

    data_frame = pd.DataFrame.from_records(temp_curve)
    data_frame.to_csv(data_folder_file_path+"model_save.csv")