import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.optimize import least_squares
from scipy.interpolate import RBFInterpolator, CubicSpline
import pandas as pd
from matplotlib.widgets import Button, Slider
import time
from scipy.ndimage import convolve1d
import pickle

from Modules.update_temp_curve import update_Temperature





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






filtered_data_file_path  = "Code/Data/Run 5/filtered_data.csv"






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
    mass_flow_rates[i]      = np.interp(time_grid,time_grid_real,physical_data[:,i+1]) * Gal_min_to_Kg_s

heater_flux_true            = np.max(physical_data[:,4])*np.ones(num_time_intervals)

temperature_data            = physical_data[:,5:]
temperature_data_transfer   = np.copy(temperature_data[:,:4])
temperature_data[:,0]       = temperature_data_transfer[:,3]
temperature_data[:,1]       = temperature_data_transfer[:,2]
temperature_data[:,2]       = temperature_data_transfer[:,1]
temperature_data[:,3]       = temperature_data_transfer[:,0]

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




calibrate_or_plot=False
show_geometry=False
show_gif=False
show_hist=False
show_plot=True
def simulate_temps(parameters):
    
    heater_conversion          = parameters[0]
    heater_flux                = heater_flux_true * heater_conversion
    
    heat_exchanger_positions   = np.array([[4.4831,0],
                                           [2.5146,0.2413]])
    heat_exchanger_main_length = 0.5207
    heat_exchanger_cap_length  = 0.0889
    heat_exchanger_area_ratio  = 0.6
    heat_exchanger_area_total  = parameters[1]
    heat_exchanger_areas       = np.array([heat_exchanger_area_total*heat_exchanger_area_ratio,
                                        heat_exchanger_area_total*(1-heat_exchanger_area_ratio)])
    heat_exchanger_cap_area    = parameters[2]
    num_nodes_in_exchanger     = np.array([50,50])
    heat_exchange_coefficients = np.array([parameters[3],parameters[4]])

    heater_length              = 1.143
    heater_area                = 0.004188254 

    large_tank_length          = 0.762
    large_tank_area            = parameters[8]

    small_tank_length          = 0.6223
    small_tank_area            = parameters[9]


    pipe_lengths_1             = [1.1176,1.3716,2.8448,1.27]
    pipe_lengths_2             = [0.6858,0.5969,2.7178]
    pipe_lengths_3             = [0.2413,0.762]
    pipe_area                  = 0.00053652265
    
    num_tanks=3
    tank_positions=[2.2606,
                    7.1755,
                    1.2065]
    tank_lengths=[large_tank_length,
                  large_tank_length,
                  small_tank_length]
    tank_loop=[0,0,1]
    mix_percentages=[parameters[5],
                     parameters[6],
                     parameters[7]]



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
    



    if(show_geometry):
        fig, ax = plt.subplots(3)

        y_bounds=[0,1]

        ax[0].vlines(np.sum(segment_lengths[0][:0]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)
        ax[0].vlines(np.sum(segment_lengths[0][:1]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)

        ax[0].vlines(np.sum(segment_lengths[0][:2]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)
        ax[0].vlines(np.sum(segment_lengths[0][:3]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)

        ax[0].vlines(np.sum(segment_lengths[0][:4]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)
        ax[0].vlines(np.sum(segment_lengths[0][:7]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)

        ax[0].vlines(np.sum(segment_lengths[0][:8]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)
        ax[0].vlines(np.sum(segment_lengths[0][:9]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)

        ax[0].vlines(thermo_probe_positions[0][0], y_bounds[0], y_bounds[1], color="black", linestyles="dotted", zorder=-1)
        ax[0].vlines(thermo_probe_positions[0][1], y_bounds[0], y_bounds[1], color="black", linestyles="dotted", zorder=-1)
        ax[0].vlines(thermo_probe_positions[0][2], y_bounds[0], y_bounds[1], color="black", linestyles="dotted", zorder=-1)
        ax[0].vlines(thermo_probe_positions[0][3], y_bounds[0], y_bounds[1], color="black", linestyles="dotted", zorder=-1)

        ax[0].fill_between([np.sum(segment_lengths[0][:0]),np.sum(segment_lengths[0][:1])], y_bounds[0], y_bounds[1], color='yellow', alpha=0.3)
        ax[0].fill_between([np.sum(segment_lengths[0][:2]),np.sum(segment_lengths[0][:3])], y_bounds[0], y_bounds[1], color='blue', alpha=0.3)
        ax[0].fill_between([np.sum(segment_lengths[0][:4]),np.sum(segment_lengths[0][:5])], y_bounds[0], y_bounds[1], color='pink', alpha=0.3)
        ax[0].fill_between([np.sum(segment_lengths[0][:5]),np.sum(segment_lengths[0][:6])], y_bounds[0], y_bounds[1], color='red', alpha=0.3)
        ax[0].fill_between([np.sum(segment_lengths[0][:6]),np.sum(segment_lengths[0][:7])], y_bounds[0], y_bounds[1], color='pink', alpha=0.3)
        ax[0].fill_between([np.sum(segment_lengths[0][:8]),np.sum(segment_lengths[0][:9])], y_bounds[0], y_bounds[1], color='blue', alpha=0.3)

        ax[0].set_ylim(y_bounds[0],y_bounds[1])
        ax[0].set_xlim(np.min(flow_curves[0]),np.max(flow_curves[0]))

        ax[1].vlines(np.sum(segment_lengths[1][:0]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)
        ax[1].vlines(np.sum(segment_lengths[1][:1]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)

        ax[1].vlines(np.sum(segment_lengths[1][:2]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)
        ax[1].vlines(np.sum(segment_lengths[1][:3]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)

        ax[1].vlines(np.sum(segment_lengths[1][:4]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)
        ax[1].vlines(np.sum(segment_lengths[1][:7]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)

        ax[1].vlines(thermo_probe_positions[1][0], y_bounds[0], y_bounds[1], color="black", linestyles="dotted", zorder=-1)
        ax[1].vlines(thermo_probe_positions[1][1], y_bounds[0], y_bounds[1], color="black", linestyles="dotted", zorder=-1)

        ax[1].fill_between([np.sum(segment_lengths[1][:0]),np.sum(segment_lengths[1][:1])], y_bounds[0], y_bounds[1], color='red', alpha=0.3)
        ax[1].fill_between([np.sum(segment_lengths[1][:2]),np.sum(segment_lengths[1][:3])], y_bounds[0], y_bounds[1], color='cyan', alpha=0.3)
        ax[1].fill_between([np.sum(segment_lengths[1][:4]),np.sum(segment_lengths[1][:5])], y_bounds[0], y_bounds[1], color='pink', alpha=0.3)
        ax[1].fill_between([np.sum(segment_lengths[1][:5]),np.sum(segment_lengths[1][:6])], y_bounds[0], y_bounds[1], color='red', alpha=0.3)
        ax[1].fill_between([np.sum(segment_lengths[1][:6]),np.sum(segment_lengths[1][:7])], y_bounds[0], y_bounds[1], color='pink', alpha=0.3)

        ax[1].set_ylim(y_bounds[0],y_bounds[1])
        ax[1].set_xlim(np.min(flow_curves[1]),np.max(flow_curves[1]))

        ax[2].vlines(np.sum(segment_lengths[2][:1]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)
        ax[2].vlines(np.sum(segment_lengths[2][:2]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)

        ax[2].vlines(thermo_probe_positions[2][0], y_bounds[0], y_bounds[1], color="black", linestyles="dotted", zorder=-1)

        ax[2].fill_between([np.sum(segment_lengths[2][:1]),np.sum(segment_lengths[2][:2])], y_bounds[0], y_bounds[1], color='red', alpha=0.3)

        ax[2].set_ylim(y_bounds[0],y_bounds[1])
        ax[2].set_xlim(np.min(flow_curves[2]),np.max(flow_curves[2]))

        plt.show()







    simulated_probe_temps           = [None]*num_loops
    for i in range(num_loops):
        simulated_probe_temps[i]    = np.zeros((num_time_intervals, len(thermo_probe_positions[i])))
        simulated_probe_temps[i][0] = initial_temps[i]

    # fig, ax   = plt.subplots(num_loops)
    # for i in range(num_loops):
    #     ax[i].plot(flow_curves[i], temp_curve[i])
    # plt.show()

    comp_time_initial = time.time()

    geometry_dict={"Node Areas":area_curves,
                   "Node Positions":flow_curves,
                   "Loop Lengths":loop_lengths,
                   "Number Exchangers":num_exchangers,
                   "Number Loops":num_loops,
                   "Exchanger Node Count":num_nodes_in_exchanger,
                   "Exchanger Positions":heat_exchanger_positions,
                   "Exchanger Main Length":heat_exchanger_main_length,
                   "Tank Positions":tank_positions,
                   "Tank Lengths":tank_lengths,
                   "Thermometer Positions":thermo_probe_positions,
                   "Exchanger Areas":heat_exchanger_areas,
                   "Heater Length":heater_length,
                   "Heater Areas":heater_area,
                   "Number Tanks":num_tanks,
                   "Loop of Tank":tank_loop}
    constants_dict={"Water Density":density_water,
                   "Water Specific Heat":specific_heat_water}
    parameters_dict={"Exchange Coefficients":heat_exchange_coefficients,
                    "Mix Percentages":mix_percentages,
                    "Heater Conversion":heater_conversion}
    
    with open("Code/Scripts/physics_stuff/calibrated_model_information/geometry.pkl","wb") as file:
        pickle.dump(geometry_dict,file)
    with open("Code/Scripts/physics_stuff/calibrated_model_information/constants.pkl","wb") as file:
        pickle.dump(constants_dict,file)
    with open("Code/Scripts/physics_stuff/calibrated_model_information/parameters.pkl","wb") as file:
        pickle.dump(parameters_dict,file)

    for step in range(1, num_time_intervals):
        nodal_temps=[None]*3
        for i in range(3):
            nodal_temps[i]=temp_curve[i][step-1]
        mass_flow_rates_step=[None]*3
        for i in range(3):
            mass_flow_rates_step[i]=mass_flow_rates[i][step-1]
        temp_curve_step,simulated_probes_step=update_Temperature(nodal_temps,
                                                                 geometry_dict,
                                                                 constants_dict,
                                                                 parameters_dict,
                                                                 inlet_temps[step-1],
                                                                 time_grid[step]-time_grid[step-1],
                                                                 heater_flux[step],
                                                                 mass_flow_rates_step)
        for i in range(3):
            temp_curve[i][step]=temp_curve_step[i]
        for i in range(3):
            simulated_probe_temps[i][step]=simulated_probes_step[i]



    simulated_data=np.zeros((time_grid.size,7))
    simulated_data[:,0]=simulated_probe_temps[0][:,3]
    simulated_data[:,1]=simulated_probe_temps[0][:,0]
    simulated_data[:,2]=simulated_probe_temps[0][:,1]
    simulated_data[:,3]=simulated_probe_temps[0][:,2]
    simulated_data[:,4]=simulated_probe_temps[1][:,0]
    simulated_data[:,5]=simulated_probe_temps[1][:,1]
    simulated_data[:,6]=simulated_probe_temps[2][:,0]


    if(show_gif):
        fig, ax = plt.subplots(3)
        fig.set_figheight(20)
        fig.set_figwidth(7)

        def gif_Function(frame):
            frame=int(frame*50)
            
            ax[0].cla()
            ax[0].plot(flow_curves[0], temp_curve[0][frame], color="black")
            y_bounds=[np.min(temp_curve[0]),np.max(temp_curve[0])]
            
            ax[0].vlines(np.sum(segment_lengths[0][:0]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)
            ax[0].vlines(np.sum(segment_lengths[0][:1]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)
            
            ax[0].vlines(np.sum(segment_lengths[0][:2]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)
            ax[0].vlines(np.sum(segment_lengths[0][:3]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)
            
            ax[0].vlines(np.sum(segment_lengths[0][:4]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)
            ax[0].vlines(np.sum(segment_lengths[0][:7]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)
            
            ax[0].vlines(np.sum(segment_lengths[0][:8]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)
            ax[0].vlines(np.sum(segment_lengths[0][:9]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)
            
            ax[0].vlines(thermo_probe_positions[0][0], y_bounds[0], y_bounds[1], color="black", linestyles="dotted", zorder=-1)
            ax[0].vlines(thermo_probe_positions[0][1], y_bounds[0], y_bounds[1], color="black", linestyles="dotted", zorder=-1)
            ax[0].vlines(thermo_probe_positions[0][2], y_bounds[0], y_bounds[1], color="black", linestyles="dotted", zorder=-1)
            ax[0].vlines(thermo_probe_positions[0][3], y_bounds[0], y_bounds[1], color="black", linestyles="dotted", zorder=-1)
            
            ax[0].fill_between([np.sum(segment_lengths[0][:0]),np.sum(segment_lengths[0][:1])], y_bounds[0], y_bounds[1], color='yellow', alpha=0.3)
            ax[0].fill_between([np.sum(segment_lengths[0][:2]),np.sum(segment_lengths[0][:3])], y_bounds[0], y_bounds[1], color='blue', alpha=0.3)
            ax[0].fill_between([np.sum(segment_lengths[0][:4]),np.sum(segment_lengths[0][:5])], y_bounds[0], y_bounds[1], color='pink', alpha=0.3)
            ax[0].fill_between([np.sum(segment_lengths[0][:5]),np.sum(segment_lengths[0][:6])], y_bounds[0], y_bounds[1], color='red', alpha=0.3)
            ax[0].fill_between([np.sum(segment_lengths[0][:6]),np.sum(segment_lengths[0][:7])], y_bounds[0], y_bounds[1], color='pink', alpha=0.3)
            ax[0].fill_between([np.sum(segment_lengths[0][:8]),np.sum(segment_lengths[0][:9])], y_bounds[0], y_bounds[1], color='blue', alpha=0.3)
            
            ax[0].set_ylim(y_bounds[0],y_bounds[1])
            ax[0].set_xlim(np.min(flow_curves[0]),np.max(flow_curves[0]))
            
            
            ax[1].cla()
            ax[1].plot(flow_curves[1], temp_curve[1][frame], color="black")
            y_bounds=[np.min(temp_curve[1]),np.max(temp_curve[1])]
            
            ax[1].vlines(np.sum(segment_lengths[1][:0]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)
            ax[1].vlines(np.sum(segment_lengths[1][:1]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)
            
            ax[1].vlines(np.sum(segment_lengths[1][:2]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)
            ax[1].vlines(np.sum(segment_lengths[1][:3]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)
            
            ax[1].vlines(np.sum(segment_lengths[1][:4]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)
            ax[1].vlines(np.sum(segment_lengths[1][:7]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)
            
            ax[1].fill_between([np.sum(segment_lengths[1][:0]),np.sum(segment_lengths[1][:1])], y_bounds[0], y_bounds[1], color='red', alpha=0.3)
            ax[1].fill_between([np.sum(segment_lengths[1][:2]),np.sum(segment_lengths[1][:3])], y_bounds[0], y_bounds[1], color='cyan', alpha=0.3)
            ax[1].fill_between([np.sum(segment_lengths[1][:4]),np.sum(segment_lengths[1][:5])], y_bounds[0], y_bounds[1], color='pink', alpha=0.3)
            ax[1].fill_between([np.sum(segment_lengths[1][:5]),np.sum(segment_lengths[1][:6])], y_bounds[0], y_bounds[1], color='red', alpha=0.3)
            ax[1].fill_between([np.sum(segment_lengths[1][:6]),np.sum(segment_lengths[1][:7])], y_bounds[0], y_bounds[1], color='pink', alpha=0.3)
            
            ax[1].set_ylim(y_bounds[0],y_bounds[1])
            ax[1].set_xlim(np.min(flow_curves[1]),np.max(flow_curves[1]))
            
            
            ax[2].cla()
            ax[2].plot(flow_curves[2], temp_curve[2][frame], color="black")
            y_bounds=[np.min(temp_curve[2]),np.max(temp_curve[2])]
            
            ax[2].vlines(np.sum(segment_lengths[2][:1]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)
            ax[2].vlines(np.sum(segment_lengths[2][:2]), y_bounds[0], y_bounds[1], color="grey", linestyles="dashed", zorder=-1)
            
            ax[2].fill_between([np.sum(segment_lengths[2][:1]),np.sum(segment_lengths[2][:2])], y_bounds[0], y_bounds[1], color='red', alpha=0.3)
            
            ax[2].set_ylim(y_bounds[0],y_bounds[1])
            ax[2].set_xlim(np.min(flow_curves[2]),np.max(flow_curves[2]))

        gif    = FuncAnimation(fig, gif_Function, frames=int((num_time_intervals)/50), interval=25)
        writer = PillowWriter(fps=50,
                            metadata=dict(artist='Me'),
                            bitrate=1800)
        # gif.save('Code/Images/Loop_Model.gif', writer=writer)
        plt.show()
    
    if(calibrate_or_plot):
        return(((simulated_data-true_temp_data)).flatten())
    else:
        return(simulated_data)
    


parameters=[398.72927,0.00434,0.02699,1590.10831,1023.63564,0.03767,0.13507,0.99143,0.0712,0.00358]

if(calibrate_or_plot):
    results=least_squares(simulate_temps, parameters, bounds=[[0,0,0,0,0,0,0,0,0,0],[400,0.1,0.1,2000,2000,1,1,1,1,1]])
    np.savetxt("calibration Results.txt",results.x,fmt="%10.5f")
else:
    simulated_data=simulate_temps(parameters)



if(show_hist):
    error_data=((simulated_data-true_temp_data)).flatten()
    relative_error_data=((simulated_data-true_temp_data)/true_temp_data).flatten()
    plt.hist(error_data,bins=50)
    plt.show()
    print(np.std(error_data))
    print(np.mean(np.abs(relative_error_data)))


if(show_plot):
    fix, ax   = plt.subplots(num_loops+1)

    ax[0].plot(time_grid, simulated_data[:,0], linestyle="dashed", label="Simulated Temp 1", color="green")
    ax[0].plot(time_grid, simulated_data[:,1], linestyle="dashed", label="Simulated Temp 2", color="blue")
    ax[0].plot(time_grid, simulated_data[:,2], linestyle="dashed", label="Simulated Temp 3", color="orange")
    ax[0].plot(time_grid, simulated_data[:,3], linestyle="dashed", label="Simulated Temp 4", color="red")

    ax[1].plot(time_grid, simulated_data[:,4], linestyle="dashed", label="Simulated Temp 1", color="green")
    ax[1].plot(time_grid, simulated_data[:,5], linestyle="dashed", label="Simulated Temp 2", color="red")

    ax[2].plot(time_grid, simulated_data[:,6], linestyle="dashed", label="Simulated Temp 2", color="red")


    ax[0].plot(time_grid_real, temperature_data[:,0], label="True Temp  1", color="green")
    ax[0].plot(time_grid_real, temperature_data[:,1], label="True Temp  2", color="blue")
    ax[0].plot(time_grid_real, temperature_data[:,2], label="True Temp  3", color="orange")
    ax[0].plot(time_grid_real, temperature_data[:,3], label="True Temp  4", color="red")

    ax[1].plot(time_grid_real, temperature_data[:,4], label="True Temp  1", color="green")
    ax[1].plot(time_grid_real, temperature_data[:,5], label="True Temp  2", color="red")

    ax[2].plot(time_grid_real, temperature_data[:,6], label="True Temp  1", color="green")
    ax[2].plot(time_grid_real, temperature_data[:,7], label="True Temp  2", color="red")
    
    ax[3].plot(time_grid, mass_flow_rates[0], label="Pump input 1", color="green")
    ax[3].plot(time_grid, mass_flow_rates[1], label="Pump input 2", color="blue")
    ax[3].plot(time_grid, mass_flow_rates[2], label="Pump input 3", color="red")
    ax[3].plot(time_grid, heater_flux_true[:], label="Heater input", color="orange")

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    ax[3].legend()

    plt.show()