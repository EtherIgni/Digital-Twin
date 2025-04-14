import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.optimize import least_squares
from scipy.interpolate import RBFInterpolator, CubicSpline
import pandas as pd
from matplotlib.widgets import Button, Slider
import time
from scipy.ndimage import convolve1d






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






filtered_data_file_path  = "/home/aaron/Depo/DDT Project/Digital-Twin/Code/Data/Run 2/filtered_data.csv"






num_loops                   = 3
num_exchangers              = 2

pump_controller_conversions = [2.8, 3.6, 0.75]

physical_data               = pd.read_csv(filtered_data_file_path,index_col=False,header=None)
physical_data               = np.array(physical_data)

time_grid_real              = physical_data[:,0]
num_time_intervals          = len(time_grid_real)*10
time_grid, time_spacing     = np.linspace(np.min(time_grid_real),np.max(time_grid_real),num_time_intervals,retstep=True)

mass_flow_rates             = [None]*num_loops
for i in range(num_loops):
    mass_flow_rates[i]      = np.ones(num_time_intervals)*np.mean(physical_data[:,i+1]) * Gal_min_to_Kg_s

heater_flux_true            = np.ones(num_time_intervals)*np.max(physical_data[:,4])

temperature_data            = physical_data[:,5:]

avg_kernel                          = np.ones(100)/100
for i in range(temperature_data.shape[1]):
    for j in range(10):
        temperature_data[:,i]=convolve1d(temperature_data[:,i],  avg_kernel)

inlet_temps                 = np.interp(time_grid, time_grid_real, temperature_data[:,6])


true_temp_data              = np.zeros((num_time_intervals,7))
for i in range(7):
    if(i<6):
        j=i
    else:
        j=i+1
    true_temp_data[:,i]     = np.interp(time_grid,time_grid_real,temperature_data[:,j])


# plt.plot(time_grid, mass_flow_rates[0], label="Pump input 1", color="green")
# plt.plot(time_grid, mass_flow_rates[1], label="Pump input 2", color="blue")
# plt.plot(time_grid, mass_flow_rates[2], label="Pump input 3", color="red")
# plt.plot(time_grid, heater_flux[:], label="Heater input", color="orange")

# plt.legend()
# plt.xlabel("Time (s)")
# plt.ylabel("Input Voltage (V)")

# plt.show()





def simulate_temps(parameters):
    
    heater_conversion          = parameters[0]
    heater_flux                = heater_flux_true * heater_conversion
    
    heat_exchanger_positions   = np.array([[3.2512,0],
                                        [2.4257,0.2413]])
    heat_exchanger_main_length = 0.5207
    heat_exchanger_cap_length  = 0.0889
    heat_exchanger_area_ratio  = parameters[1]
    heat_exchanger_area_total  = parameters[2]
    heat_exchanger_areas       = np.array([heat_exchanger_area_total*heat_exchanger_area_ratio,
                                        heat_exchanger_area_total*(1-heat_exchanger_area_ratio)])
    heat_exchanger_cap_area    = parameters[3]
    num_nodes_in_exchanger     = np.array([50,50])
    heat_exchange_coefficients = np.array([parameters[4],parameters[5]])

    heater_length              = 1.143
    heater_area                = 0.004188254 

    large_tank_length          = 0.762
    large_tank_area            = 0.0791730436

    small_tank_length          = 0.6223
    small_tank_area            = 0.00809921437


    pipe_lengths_1             = [1.1176,1.3716,2.8448,1.27]
    pipe_lengths_2             = [0.6858,0.5969,2.7178]
    pipe_lengths_3             = [0.2413,0.762]
    pipe_area                  = 0.00053652265



    segment_lengths            = [None]*num_loops
    num_nodes_in_segment       = [None]*num_loops
    segment_areas              = [None]*num_loops
    initial_temps              = [None]*num_loops
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

    initial_temps[0]           = np.array([temperature_data[0,1], temperature_data[0,2], temperature_data[0,3], temperature_data[0,0]])
    initial_temps[1]           = np.array([temperature_data[0,4], temperature_data[0,5]])
    initial_temps[2]           = np.array([temperature_data[0,7]])

    thermo_probe_positions[0]  = np.array([1.2827,4.2799,5.207,9.8552])
    thermo_probe_positions[1]  = np.array([2.2606,3.2131])
    thermo_probe_positions[2]  = np.array([1.4605])






    flow_curves                                         = [None]*num_loops
    area_curves                                         = [None]*num_loops
    loop_lengths                                        = [None]*num_loops
    temp_curve                                          = [None]*num_loops

    for i in range(num_loops):
        flow_curves[i], area_curves[i], loop_lengths[i] = calc_geometric_curves(segment_lengths[i],
                                                                                num_nodes_in_segment[i],
                                                                                segment_areas[i])
        temp_curve[i]    = np.zeros((num_time_intervals,flow_curves[i].size))
        temp_curve[i][0] = np.interp(flow_curves[i],
                                    thermo_probe_positions[i],
                                    initial_temps[i],
                                    period=loop_lengths[i])
    




    # fig, ax = plt.subplots(3)

    # y_bounds=[0,1]

    # ax[0].vlines(np.sum(segment_lengths[0][:0]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)
    # ax[0].vlines(np.sum(segment_lengths[0][:1]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)

    # ax[0].vlines(np.sum(segment_lengths[0][:2]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)
    # ax[0].vlines(np.sum(segment_lengths[0][:3]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)

    # ax[0].vlines(np.sum(segment_lengths[0][:4]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)
    # ax[0].vlines(np.sum(segment_lengths[0][:7]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)

    # ax[0].vlines(np.sum(segment_lengths[0][:8]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)
    # ax[0].vlines(np.sum(segment_lengths[0][:9]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)

    # ax[0].vlines(thermo_probe_positions[0][0], y_bounds[0], y_bounds[1], color="black", linestyles="dotted", zorder=-1)
    # ax[0].vlines(thermo_probe_positions[0][1], y_bounds[0], y_bounds[1], color="black", linestyles="dotted", zorder=-1)
    # ax[0].vlines(thermo_probe_positions[0][2], y_bounds[0], y_bounds[1], color="black", linestyles="dotted", zorder=-1)
    # ax[0].vlines(thermo_probe_positions[0][3], y_bounds[0], y_bounds[1], color="black", linestyles="dotted", zorder=-1)

    # ax[0].fill_between([np.sum(segment_lengths[0][:0]),np.sum(segment_lengths[0][:1])], y_bounds[0], y_bounds[1], color='yellow', alpha=0.3)
    # ax[0].fill_between([np.sum(segment_lengths[0][:2]),np.sum(segment_lengths[0][:3])], y_bounds[0], y_bounds[1], color='blue', alpha=0.3)
    # ax[0].fill_between([np.sum(segment_lengths[0][:4]),np.sum(segment_lengths[0][:5])], y_bounds[0], y_bounds[1], color='pink', alpha=0.3)
    # ax[0].fill_between([np.sum(segment_lengths[0][:5]),np.sum(segment_lengths[0][:6])], y_bounds[0], y_bounds[1], color='red', alpha=0.3)
    # ax[0].fill_between([np.sum(segment_lengths[0][:6]),np.sum(segment_lengths[0][:7])], y_bounds[0], y_bounds[1], color='pink', alpha=0.3)
    # ax[0].fill_between([np.sum(segment_lengths[0][:8]),np.sum(segment_lengths[0][:9])], y_bounds[0], y_bounds[1], color='blue', alpha=0.3)

    # ax[0].set_ylim(y_bounds[0],y_bounds[1])
    # ax[0].set_xlim(np.min(flow_curves[0]),np.max(flow_curves[0]))

    # ax[1].vlines(np.sum(segment_lengths[1][:0]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)
    # ax[1].vlines(np.sum(segment_lengths[1][:1]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)

    # ax[1].vlines(np.sum(segment_lengths[1][:2]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)
    # ax[1].vlines(np.sum(segment_lengths[1][:3]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)

    # ax[1].vlines(np.sum(segment_lengths[1][:4]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)
    # ax[1].vlines(np.sum(segment_lengths[1][:7]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)

    # ax[1].vlines(thermo_probe_positions[1][0], y_bounds[0], y_bounds[1], color="black", linestyles="dotted", zorder=-1)
    # ax[1].vlines(thermo_probe_positions[1][1], y_bounds[0], y_bounds[1], color="black", linestyles="dotted", zorder=-1)

    # ax[1].fill_between([np.sum(segment_lengths[1][:0]),np.sum(segment_lengths[1][:1])], y_bounds[0], y_bounds[1], color='red', alpha=0.3)
    # ax[1].fill_between([np.sum(segment_lengths[1][:2]),np.sum(segment_lengths[1][:3])], y_bounds[0], y_bounds[1], color='cyan', alpha=0.3)
    # ax[1].fill_between([np.sum(segment_lengths[1][:4]),np.sum(segment_lengths[1][:5])], y_bounds[0], y_bounds[1], color='pink', alpha=0.3)
    # ax[1].fill_between([np.sum(segment_lengths[1][:5]),np.sum(segment_lengths[1][:6])], y_bounds[0], y_bounds[1], color='red', alpha=0.3)
    # ax[1].fill_between([np.sum(segment_lengths[1][:6]),np.sum(segment_lengths[1][:7])], y_bounds[0], y_bounds[1], color='pink', alpha=0.3)

    # ax[1].set_ylim(y_bounds[0],y_bounds[1])
    # ax[1].set_xlim(np.min(flow_curves[1]),np.max(flow_curves[1]))

    # ax[2].vlines(np.sum(segment_lengths[2][:1]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)
    # ax[2].vlines(np.sum(segment_lengths[2][:2]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)

    # ax[2].vlines(thermo_probe_positions[2][0], y_bounds[0], y_bounds[1], color="black", linestyles="dotted", zorder=-1)

    # ax[2].fill_between([np.sum(segment_lengths[2][:1]),np.sum(segment_lengths[2][:2])], y_bounds[0], y_bounds[1], color='red', alpha=0.3)

    # ax[2].set_ylim(y_bounds[0],y_bounds[1])
    # ax[2].set_xlim(np.min(flow_curves[2]),np.max(flow_curves[2]))

    # plt.show()







    simulated_probe_temps           = [None]*num_loops
    for i in range(num_loops):
        simulated_probe_temps[i]    = np.zeros((num_time_intervals, len(thermo_probe_positions[i])))
        simulated_probe_temps[i][0] = initial_temps[i]

    # fig, ax   = plt.subplots(num_loops)
    # for i in range(num_loops):
    #     ax[i].plot(flow_curves[i], temp_curve[i])
    # plt.show()

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
                interp_temps        = np.concat([[inlet_temps[step-1]],temp_curve[i][step-1][keep_indices]])
                
                sort_indices        = np.argsort(new_positions)
                temp_curve[i][step] = np.interp(flow_curves[i],
                                                new_positions[sort_indices],
                                                interp_temps[sort_indices])
            else:
                sort_indices        = np.argsort(new_positions)
                temp_curve[i][step] = np.interp(flow_curves[i],
                                                new_positions[sort_indices],
                                                temp_curve[i][step-1][sort_indices],
                                                period=loop_lengths[i])
        
        
        
        #Heat Transfer
        for i in range(num_exchangers):
            exchanger_curve                             = np.linspace(0, heat_exchanger_main_length, num_nodes_in_exchanger[i])
            heat_flux                                   = heat_exchange_coefficients[i]*(np.interp(exchanger_curve+heat_exchanger_positions[i,0],                           flow_curves[i],   temp_curve[i][step])-
                                                                                        np.interp(heat_exchanger_positions[i,1]+heat_exchanger_main_length-exchanger_curve, flow_curves[i+1], temp_curve[i+1][step]))
            
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
                    temp_curve[j+i][step][fill_indices] = temp_curve[j+i][step][fill_indices] + heat_transfer
        
        
        
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
        temp_curve[0][step][fill_indices]   = temp_curve[0][step][fill_indices] + heat_transfer
        
        
        
        #Logs temperature at probs
        for i in range(num_loops):
            simulated_probe_temps[i][step] = np.interp(thermo_probe_positions[i], flow_curves[i], temp_curve[i][step])



    simulated_data=np.zeros((time_grid.size,7))
    simulated_data[:,0]=simulated_probe_temps[0][:,3]
    simulated_data[:,1]=simulated_probe_temps[0][:,0]
    simulated_data[:,2]=simulated_probe_temps[0][:,1]
    simulated_data[:,3]=simulated_probe_temps[0][:,2]
    simulated_data[:,4]=simulated_probe_temps[1][:,0]
    simulated_data[:,5]=simulated_probe_temps[1][:,1]
    simulated_data[:,6]=simulated_probe_temps[2][:,0]

    return((simulated_data-true_temp_data).flatten())


parameters=[50,0.6,0.00191598766,0.00383197531,0,400]

print(least_squares(simulate_temps, parameters))




# fix, ax   = plt.subplots(num_loops)

# ax[0].plot(time_grid, simulated_data[:,1], linestyle="dashed", label="Simulated Temp 1", color="green")
# ax[0].plot(time_grid, simulated_data[:,2], linestyle="dashed", label="Simulated Temp 2", color="blue")
# ax[0].plot(time_grid, simulated_data[:,3], linestyle="dashed", label="Simulated Temp 3", color="orange")
# ax[0].plot(time_grid, simulated_data[:,4], linestyle="dashed", label="Simulated Temp 4", color="red")

# ax[1].plot(time_grid, simulated_data[:,5], linestyle="dashed", label="Simulated Temp 1", color="green")
# ax[1].plot(time_grid, simulated_data[:,6], linestyle="dashed", label="Simulated Temp 2", color="red")

# ax[2].plot(time_grid, simulated_data[:,8], linestyle="dashed", label="Simulated Temp 2", color="red")


# ax[0].plot(time_grid_real, temperature_data[:,0], label="True Temp  1", color="green")
# ax[0].plot(time_grid_real, temperature_data[:,1], label="True Temp  2", color="blue")
# ax[0].plot(time_grid_real, temperature_data[:,2], label="True Temp  3", color="orange")
# ax[0].plot(time_grid_real, temperature_data[:,3], label="True Temp  4", color="red")

# ax[1].plot(time_grid_real, temperature_data[:,4], label="True Temp  1", color="green")
# ax[1].plot(time_grid_real, temperature_data[:,5], label="True Temp  2", color="red")

# ax[2].plot(time_grid_real, temperature_data[:,6], label="True Temp  1", color="green")
# ax[2].plot(time_grid_real, temperature_data[:,7], label="True Temp  2", color="red")

# ax[0].legend()
# ax[1].legend()
# ax[2].legend()

# plt.show()



# fig, ax = plt.subplots(3)
# fig.set_figheight(20)
# fig.set_figwidth(7)

# def gif_Function(frame):
#     frame=int(frame*50/(time_spacing))
    
#     ax[0].cla()
#     ax[0].plot(flow_curves[0], temp_curve[0][frame], color="black")
#     y_bounds=[np.min(temp_curve[0]),np.max(temp_curve[0])]
    
#     ax[0].vlines(np.sum(segment_lengths[0][:0]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)
#     ax[0].vlines(np.sum(segment_lengths[0][:1]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)
    
#     ax[0].vlines(np.sum(segment_lengths[0][:2]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)
#     ax[0].vlines(np.sum(segment_lengths[0][:3]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)
    
#     ax[0].vlines(np.sum(segment_lengths[0][:4]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)
#     ax[0].vlines(np.sum(segment_lengths[0][:7]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)
    
#     ax[0].vlines(np.sum(segment_lengths[0][:8]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)
#     ax[0].vlines(np.sum(segment_lengths[0][:9]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)
    
#     ax[0].vlines(thermo_probe_positions[0][0], y_bounds[0], y_bounds[1], color="black", linestyles="dotted", zorder=-1)
#     ax[0].vlines(thermo_probe_positions[0][1], y_bounds[0], y_bounds[1], color="black", linestyles="dotted", zorder=-1)
#     ax[0].vlines(thermo_probe_positions[0][2], y_bounds[0], y_bounds[1], color="black", linestyles="dotted", zorder=-1)
#     ax[0].vlines(thermo_probe_positions[0][3], y_bounds[0], y_bounds[1], color="black", linestyles="dotted", zorder=-1)
    
#     ax[0].fill_between([np.sum(segment_lengths[0][:0]),np.sum(segment_lengths[0][:1])], y_bounds[0], y_bounds[1], color='yellow', alpha=0.3)
#     ax[0].fill_between([np.sum(segment_lengths[0][:2]),np.sum(segment_lengths[0][:3])], y_bounds[0], y_bounds[1], color='blue', alpha=0.3)
#     ax[0].fill_between([np.sum(segment_lengths[0][:4]),np.sum(segment_lengths[0][:5])], y_bounds[0], y_bounds[1], color='pink', alpha=0.3)
#     ax[0].fill_between([np.sum(segment_lengths[0][:5]),np.sum(segment_lengths[0][:6])], y_bounds[0], y_bounds[1], color='red', alpha=0.3)
#     ax[0].fill_between([np.sum(segment_lengths[0][:6]),np.sum(segment_lengths[0][:7])], y_bounds[0], y_bounds[1], color='pink', alpha=0.3)
#     ax[0].fill_between([np.sum(segment_lengths[0][:8]),np.sum(segment_lengths[0][:9])], y_bounds[0], y_bounds[1], color='blue', alpha=0.3)
    
#     ax[0].set_ylim(y_bounds[0],y_bounds[1])
#     ax[0].set_xlim(np.min(flow_curves[0]),np.max(flow_curves[0]))
    
    
#     ax[1].cla()
#     ax[1].plot(flow_curves[1], temp_curve[1][frame], color="black")
#     y_bounds=[np.min(temp_curve[1]),np.max(temp_curve[1])]
    
#     ax[1].vlines(np.sum(segment_lengths[1][:0]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)
#     ax[1].vlines(np.sum(segment_lengths[1][:1]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)
    
#     ax[1].vlines(np.sum(segment_lengths[1][:2]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)
#     ax[1].vlines(np.sum(segment_lengths[1][:3]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)
    
#     ax[1].vlines(np.sum(segment_lengths[1][:4]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)
#     ax[1].vlines(np.sum(segment_lengths[1][:7]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)
    
#     ax[1].fill_between([np.sum(segment_lengths[1][:0]),np.sum(segment_lengths[1][:1])], y_bounds[0], y_bounds[1], color='red', alpha=0.3)
#     ax[1].fill_between([np.sum(segment_lengths[1][:2]),np.sum(segment_lengths[1][:3])], y_bounds[0], y_bounds[1], color='cyan', alpha=0.3)
#     ax[1].fill_between([np.sum(segment_lengths[1][:4]),np.sum(segment_lengths[1][:5])], y_bounds[0], y_bounds[1], color='pink', alpha=0.3)
#     ax[1].fill_between([np.sum(segment_lengths[1][:5]),np.sum(segment_lengths[1][:6])], y_bounds[0], y_bounds[1], color='red', alpha=0.3)
#     ax[1].fill_between([np.sum(segment_lengths[1][:6]),np.sum(segment_lengths[1][:7])], y_bounds[0], y_bounds[1], color='pink', alpha=0.3)
    
#     ax[1].set_ylim(y_bounds[0],y_bounds[1])
#     ax[1].set_xlim(np.min(flow_curves[1]),np.max(flow_curves[1]))
    
    
#     ax[2].cla()
#     ax[2].plot(flow_curves[2], temp_curve[2][frame], color="black")
#     y_bounds=[np.min(temp_curve[2]),np.max(temp_curve[2])]
    
#     ax[2].vlines(np.sum(segment_lengths[2][:1]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)
#     ax[2].vlines(np.sum(segment_lengths[2][:2]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)
    
#     ax[2].fill_between([np.sum(segment_lengths[2][:1]),np.sum(segment_lengths[2][:2])], y_bounds[0], y_bounds[1], color='red', alpha=0.3)
    
#     ax[2].set_ylim(y_bounds[0],y_bounds[1])
#     ax[2].set_xlim(np.min(flow_curves[2]),np.max(flow_curves[2]))

# gif    = FuncAnimation(fig, gif_Function, frames=int((num_time_intervals)*50/(time_spacing)), interval=25)
# writer = PillowWriter(fps=50,
#                       metadata=dict(artist='Me'),
#                       bitrate=1800)
# # gif.save('Code/Images/Loop_Model.gif', writer=writer)
# plt.show()