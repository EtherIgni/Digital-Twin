import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.optimize import least_squares
from scipy.interpolate import RBFInterpolator, CubicSpline
import pandas as pd
from matplotlib.widgets import Button, Slider
import time
import os






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






specific_heat_water       = 4181   #J/Kg-K
specific_heat_exchanger   = 385    #J/Kg-K
density_water             = 997    #kg/m^3
C_to_K                    = 273.15 #K
Gal_min_to_Kg_s           = 0.0466 #(Kg/s)/(Gal/min)



data_folder_file_path     = "Code/Data/"
num_loops                 = 3
num_exchangers            = 2




def simulate_data(run_number):
    heater_conversion           = 100

    physical_data               = pd.read_csv(data_folder_file_path+"Run "+str(run_number)+"/filtered_data.csv",index_col=False,header=None)
    physical_data               = np.array(physical_data)

    time_grid                   = physical_data[-100:,0]/1000
    num_time_intervals          = len(time_grid)

    mass_flow_rates             = [None]*num_loops
    for i in range(num_loops):
        mass_flow_rates[i]      = physical_data[-100:,i+1] * Gal_min_to_Kg_s 

    heater_flux                 = physical_data[-100:,4] * heater_conversion

    temperature_data            = physical_data[-100:,5:]

    inlet_temps                 = temperature_data[:,6]






    heat_exchanger_positions   = np.array([[3.2512,0],
                                        [2.4257,0.3302]])
    heat_exchanger_main_length = 0.5207
    heat_exchanger_cap_length  = 0.0889
    heat_exchanger_areas       = np.array([0.00191598766,
                                        0.00191598766])
    heat_exchanger_cap_area    = 0.00383197531
    num_nodes_in_exchanger     = np.array([50,50])
    heat_exchange_coefficients = np.array([680,680])

    heater_length              = 1.143
    heater_area                = 0.004188254 

    large_tank_length          = 0.762
    large_tank_area            = 0.0791730436

    small_tank_length          = 0.6223
    small_tank_area            = 0.00809921437


    pipe_lengths_1             = [1.1176,1.3716,2.8448,1.27]
    pipe_lengths_2             = [0.6858,0.5969,2.7178]
    pipe_lengths_3             = [0.3302,0.762]
    pipe_area                  = 0.00053652265



    segment_lengths            = [None]*num_loops
    num_nodes_in_segment       = [None]*num_loops
    segment_areas              = [None]*num_loops
    thermo_probe_positions     = [None]*num_loops



    segment_lengths[0]         = np.array([heater_length,
                                        pipe_lengths_1[0],
                                        large_tank_length,
                                        pipe_lengths_1[1],
                                        heat_exchanger_cap_length,
                                        heat_exchanger_main_length,
                                        heat_exchanger_cap_length,
                                        pipe_lengths_1[2],
                                        large_tank_length,
                                        pipe_lengths_1[3]])
    segment_areas[0]           = np.array([heater_area,
                                        pipe_area,
                                        large_tank_area,
                                        pipe_area,
                                        heat_exchanger_cap_area,
                                        heat_exchanger_areas[0],
                                        heat_exchanger_cap_area,
                                        pipe_area,
                                        large_tank_area,
                                        pipe_area])
    num_nodes_in_segment[0]    = np.array([50,
                                        25,
                                        25,
                                        25,
                                        10,
                                        50,
                                        10,
                                        25,
                                        25,
                                        25])

    segment_lengths[1]         = np.array([heat_exchanger_main_length,
                                        pipe_lengths_2[0],
                                        small_tank_length,
                                        pipe_lengths_2[1],
                                        heat_exchanger_cap_length,
                                        heat_exchanger_main_length,
                                        heat_exchanger_cap_length,
                                        pipe_lengths_2[2]])
    segment_areas[1]           = np.array([heat_exchanger_areas[1],
                                        pipe_area,
                                        small_tank_area,
                                        pipe_area,
                                        heat_exchanger_cap_area,
                                        heat_exchanger_areas[0],
                                        heat_exchanger_cap_area,
                                        pipe_area])
    num_nodes_in_segment[1]    = np.array([50,
                                        25,
                                        25,
                                        25,
                                        10,
                                        50,
                                        10,
                                        25])

    segment_lengths[2]         = np.array([pipe_lengths_3[0],
                                        heat_exchanger_main_length,
                                        pipe_lengths_3[1]])
    segment_areas[2]           = np.array([pipe_area,
                                        heat_exchanger_areas[1],
                                        pipe_area])
    num_nodes_in_segment[2]    = np.array([25,
                                        50,
                                        25])

    thermo_probe_positions[0]  = np.array([1.2827,3.1369,4.0386,8.7249])
    thermo_probe_positions[1]  = np.array([2.2606,3.2131])
    thermo_probe_positions[2]  = np.array([0.0889,1.5494])






    flow_curves                                         = [None]*num_loops
    area_curves                                         = [None]*num_loops
    loop_lengths                                        = [None]*num_loops
    temp_curve                                          = [None]*num_loops

    for i in range(num_loops):
        flow_curves[i], area_curves[i], loop_lengths[i] = calc_geometric_curves(segment_lengths[i],
                                                                                num_nodes_in_segment[i],
                                                                                segment_areas[i])
        
    if(os.path.isfile(data_folder_file_path+"Run "+str(run_number)+"/model_save.csv")):
        temp_dataframe    = pd.read_csv(data_folder_file_path+"Run "+str(run_number)+"/model_save.csv")
        for i in range(num_loops):
            temp_curve[i] = temp_dataframe.loc[i].to_list()
            temp_curve[i] = [x for x in temp_curve[i] if str(x) != 'nan']
            temp_curve[i] = np.array(temp_curve[i])[1:]
            
    else:
        initial_temps              = [None]*num_loops
        initial_temps[0]           = np.array([temperature_data[0,1], temperature_data[0,2], temperature_data[0,3], temperature_data[0,0]])
        initial_temps[1]           = np.array([temperature_data[0,4], temperature_data[0,5]])
        initial_temps[2]           = np.array([temperature_data[0,6], temperature_data[0,7]])
        
        for i in range(num_loops):
            temp_curve[i]    = np.interp(flow_curves[i],
                                         thermo_probe_positions[i],
                                         initial_temps[i],
                                         period=loop_lengths[i])
            print(temp_curve[i].size)
        






    simulated_probe_temps           = [None]*num_loops
    for i in range(num_loops):
        simulated_probe_temps[i]    = np.zeros((num_time_intervals,thermo_probe_positions[i].size))
        simulated_probe_temps[i][0] = np.interp(thermo_probe_positions[i], flow_curves[i], temp_curve[i])

    comp_time_initial = time.time()

    for step in range(1, num_time_intervals):
        time_diff = time_grid[step] - time_grid[step-1]

        #Heat Shift
        for i in range(num_loops):
            #Calculate fluid flow distance
            travel_distances        = mass_flow_rates[i][step-1]*time_diff/(density_water*area_curves[i])
            
            #Rotate and re-interpolate temperature data
            new_positions           = flow_curves[i]+travel_distances
            if(i==2):
                keep_indices        = np.where(new_positions<=loop_lengths[i])[0]
                new_positions       = np.concat([[0],new_positions[keep_indices]])
                interp_temps        = np.concat([[inlet_temps[step-1]],temp_curve[i][keep_indices]])
                
                sort_indices        = np.argsort(new_positions)
                temp_curve[i]       = np.interp(flow_curves[i],
                                                new_positions[sort_indices],
                                                interp_temps[sort_indices])
            else:
                sort_indices        = np.argsort(new_positions)
                temp_curve[i]       = np.interp(flow_curves[i],
                                                new_positions[sort_indices],
                                                temp_curve[i][sort_indices],
                                                period=loop_lengths[i])
        
        
        
        #Heat Transfer
        for i in range(num_exchangers):
            exchanger_curve                             = np.linspace(0, heat_exchanger_main_length, num_nodes_in_exchanger[i])
            heat_flux                                   = heat_exchange_coefficients[i]*(np.interp(exchanger_curve+heat_exchanger_positions[i,0],                           flow_curves[i],   temp_curve[i])-
                                                                                        np.interp(heat_exchanger_positions[i,1]+heat_exchanger_main_length-exchanger_curve, flow_curves[i+1], temp_curve[i+1]))
            
            for j in range(2):
                sub_grid_indices                        = np.where(np.logical_and(flow_curves[j+i]>heat_exchanger_positions[i,j],
                                                                                flow_curves[j+i]<heat_exchanger_positions[i,j]+heat_exchanger_main_length))[0]
                next_index                              = np.max(sub_grid_indices)+1 
                
                sub_grid                                = np.ones(sub_grid_indices.size+2)
                sub_grid[0]                             = heat_exchanger_positions[i,j]
                sub_grid[-1]                            = heat_exchanger_positions[i,j]+heat_exchanger_main_length
                sub_grid[1:-1]                          = flow_curves[j+i][sub_grid_indices]
                
                matched_heat_flux                       = np.interp(sub_grid-heat_exchanger_positions[i,j], exchanger_curve, heat_flux)
                if(j>0):
                    matched_heat_flux                   = np.flip(matched_heat_flux)
                
                
                    exchanger_travel_distance           = np.ones(sub_grid.size-1)*time_diff/(density_water*heat_exchanger_areas[j])
                    if(mass_flow_rates[j+i][step-1]>0):
                        transfer_areas                  = np.min(np.stack((exchanger_travel_distance,
                                                                (sub_grid[1:]-sub_grid[:-1])/mass_flow_rates[j+i][step-1])),
                                                                0)
                    else:
                        transfer_areas                  = exchanger_travel_distance
                    
                    heat_transfer                       = ((matched_heat_flux[1:]+matched_heat_flux[:-1])/2) * transfer_areas * (j*2-1) / specific_heat_water
                    fill_indices                        = np.concat([sub_grid_indices,np.array([next_index])])
                    temp_curve[j+i][fill_indices]       = temp_curve[j+i][fill_indices] + heat_transfer
        
        
        
        #Heater Heat Input
        sub_grid_indices                    = np.where(np.logical_and(flow_curves[0]>0,
                                                                flow_curves[0]<heater_length))[0]
        next_index                          = np.max(sub_grid_indices)+1
        fill_indices                        = np.concat([sub_grid_indices,np.array([next_index])])
        
        heater_travel_distance              = np.ones(fill_indices.size)*time_diff/(density_water*heater_area)
        if(mass_flow_rates[0][step-1]>0):
            grid_sizes                      = (flow_curves[0][fill_indices]-np.concat([[0], flow_curves[0][fill_indices][:-1]]))/mass_flow_rates[0][step-1]
            heater_transfer_areas           = np.min(np.stack((heater_travel_distance,
                                                    grid_sizes)),
                                                    0)
        else:
            heater_transfer_areas           = heater_travel_distance
        
        heat_transfer                       = heater_flux[step-1] * transfer_areas / specific_heat_water
        temp_curve[0][fill_indices]         = temp_curve[0][fill_indices] + heat_transfer
        
        
        
        #Logs temperature at probs
        for i in range(num_loops):
            simulated_probe_temps[i][step] = np.interp(thermo_probe_positions[i], flow_curves[i], temp_curve[i])



    simulated_data=np.zeros((time_grid.size,9))
    simulated_data[:,0]=time_grid
    simulated_data[:,1]=simulated_probe_temps[0][:,3]
    simulated_data[:,2:5]=simulated_probe_temps[0][:,0:3]
    simulated_data[:,5:7]=simulated_probe_temps[1]
    simulated_data[:,7:9]=simulated_probe_temps[2]
    
    data_frame                          = pd.DataFrame.from_records(simulated_data)
    data_frame.to_csv(data_folder_file_path+"Run "+str(run_number)+"/simulated_data.csv",mode="a",header=False,index=False)
    
    data_frame =pd.DataFrame.from_records(temp_curve)
    data_frame.to_csv(data_folder_file_path+"Run "+str(run_number)+"/model_save.csv")