import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.optimize import least_squares
from scipy.interpolate import RBFInterpolator, CubicSpline
import pandas as pd
from matplotlib.widgets import Button, Slider






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






filtered_data_file_path  = "/home/Aaron/Depo/DDT/Digital-Twin/Code/Images/Data Set 1/Run 1/Filtered_Data.csv"






num_loops                   = 3
num_exchangers              = 2

pump_controller_conversions = [2.8, 3.6, 0.75]
heater_conversion           = 100

physical_data               = pd.read_csv(filtered_data_file_path,index_col=False,header=None)
physical_data               = np.array(physical_data)
input_data                  = physical_data[:,3:7]

time_grid                   = physical_data[:,15]/1000
num_time_intervals          = len(time_grid)

mass_flow_rates             = [None]*num_loops
for i in range(num_loops):
    mass_flow_rates[i]      = input_data[:,i] * Gal_min_to_Kg_s * pump_controller_conversions[i]

heater_flux                 = input_data[:,3]*heater_conversion

temperature_data            = physical_data[:,7:15]



plt.plot(time_grid, input_data[:,0], label="Pump input 1", color="green")
plt.plot(time_grid, input_data[:,1], label="Pump input 2", color="blue")
plt.plot(time_grid, input_data[:,2], label="Pump input 3", color="red")
plt.plot(time_grid, input_data[:,3], label="Heater input", color="orange")

plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Input Voltage (V)")

plt.show()






heat_exchanger_positions   = np.array([[3.2512,0],
                                       [2.4257,0.3302]])
heat_exchanger_main_length = 0.5207
heat_exchanger_cap_length  = 0.0889
heat_exchanger_areas       = np.array([0.00191598766,
                                       0.00191598766])
heat_exchanger_cap_area    = 0.00383197531
num_nodes_in_exchanger     = np.array([50,50])
heat_exchange_coefficients = np.array([68,68])

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
initial_temps[2]           = np.array([temperature_data[0,6], temperature_data[0,7]])

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
    temp_curve[i] = np.interp(flow_curves[i],
                              thermo_probe_positions[i],
                              initial_temps[i],
                              period=loop_lengths[i])
    






simulated_probe_temps           = [None]*num_loops
for i in range(num_loops):
    simulated_probe_temps[i]    = np.zeros((num_time_intervals, len(thermo_probe_positions[i])))
    simulated_probe_temps[i][0] = initial_temps[i]

# fig, ax   = plt.subplots(num_loops)
# for i in range(num_loops):
#     ax[i].plot(flow_curves[i], temp_curve[i])
# plt.show()

for step in range(1, num_time_intervals):
    time_diff = time_grid[step] - time_grid[step-1]

    #Heat Shift
    for i in range(num_loops):
        #Calculate fluid flow distance
        travel_distances = mass_flow_rates[i][step-1]*time_diff/(density_water*area_curves[i])
        
        #Rotate and re-interpolate temperature data
        new_positions    = flow_curves[i]+travel_distances
        sort_indices     = np.argsort(new_positions)
        temp_curve[i]    = np.interp(flow_curves[i],
                                     new_positions[sort_indices],
                                     temp_curve[i][sort_indices],
                                     period=loop_lengths[i]) 
    
    
    
    #Heat Transfer
    for i in range(num_exchangers):
        exchanger_curve  = np.linspace(0, heat_exchanger_main_length, num_nodes_in_exchanger[i])
        heat_flux        = heat_exchange_coefficients[i]*(np.interp(exchanger_curve+heat_exchanger_positions[i,0],                           flow_curves[i],   temp_curve[i])-
                                                          np.interp(heat_exchanger_positions[i,1]+heat_exchanger_main_length-exchanger_curve, flow_curves[i+1], temp_curve[i+1]))
        
        for j in range(2):
            sub_grid_indices              = np.where(np.logical_and(flow_curves[j+i]>heat_exchanger_positions[i,j],
                                                                    flow_curves[j+i]<heat_exchanger_positions[i,j]+heat_exchanger_main_length))[0]
            next_index                    = np.max(sub_grid_indices)+1 
            
            sub_grid                      = np.ones(sub_grid_indices.size+2)
            sub_grid[0]                   = heat_exchanger_positions[i,j]
            sub_grid[-1]                  = heat_exchanger_positions[i,j]+heat_exchanger_main_length
            sub_grid[1:-1]                = flow_curves[j+i][sub_grid_indices]
            
            matched_heat_flux             = np.interp(sub_grid-heat_exchanger_positions[i,j], exchanger_curve, heat_flux)
            if(j>0):
                matched_heat_flux         = np.flip(matched_heat_flux)
            
            
                exchanger_travel_distance     = np.ones(sub_grid.size-1)*time_diff/(density_water*heat_exchanger_areas[j])
                if(mass_flow_rates[j+i][step-1]>0):
                    transfer_areas            = np.min(np.stack((exchanger_travel_distance,
                                                       (sub_grid[1:]-sub_grid[:-1])/mass_flow_rates[j+i][step-1])),
                                                       0)
                else:
                    transfer_areas            = exchanger_travel_distance
                
                heat_transfer                 = ((matched_heat_flux[1:]+matched_heat_flux[:-1])/2) * transfer_areas * (j*2-1) / specific_heat_water
                fill_indices                  = np.concat([sub_grid_indices,np.array([next_index])])
                temp_curve[j+i][fill_indices] = temp_curve[j+i][fill_indices] + heat_transfer
    
    
    
    #Heater Heat Input
    sub_grid_indices              = np.where(np.logical_and(flow_curves[0]>0,
                                                            flow_curves[0]<heater_length))[0]
    next_index                    = np.max(sub_grid_indices)+1
    fill_indices                  = np.concat([sub_grid_indices,np.array([next_index])])
    
    heater_travel_distance        = np.ones(fill_indices.size)*time_diff/(density_water*heater_area)
    if(mass_flow_rates[0][step-1]>0):
        grid_sizes                = (flow_curves[0][fill_indices]-np.concat([[0], flow_curves[0][fill_indices][:-1]]))/mass_flow_rates[0][step-1]
        heater_transfer_areas     = np.min(np.stack((heater_travel_distance,
                                           grid_sizes)),
                                           0)
    else:
        heater_transfer_areas     = heater_travel_distance
    
    heat_transfer                 = heater_flux[step-1] * transfer_areas / specific_heat_water
    temp_curve[0][fill_indices]   = temp_curve[0][fill_indices] + heat_transfer
    
    
    
    #Logs temperature at probs
    for i in range(num_loops):
        simulated_probe_temps[i][step] = np.interp(thermo_probe_positions[i], flow_curves[i], temp_curve[i])



simulated_data=np.zeros((time_grid.size,9))
simulated_data[:,0]=time_grid
simulated_data[:,1]=simulated_probe_temps[0][:,3]
simulated_data[:,2:5]=simulated_probe_temps[0][:,0:3]
simulated_data[:,5:7]=simulated_probe_temps[1]
simulated_data[:,7:9]=simulated_probe_temps[2]



fig, ax   = plt.subplots(num_loops)
for i in range(num_loops):
    ax[i].plot(flow_curves[i], temp_curve[i])
    for j in range(len(segment_lengths[i])):
        ax[i].vlines(np.sum(segment_lengths[i][:j]), ax[i].get_ylim()[0], ax[i].get_ylim()[1], color="black", zorder=-1)
plt.show()



fix, ax   = plt.subplots(num_loops)

ax[0].plot(time_grid, simulated_data[:,1], linestyle="dashed", label="Simulated Temp 1", color="green")
ax[0].plot(time_grid, simulated_data[:,2], linestyle="dashed", label="Simulated Temp 2", color="blue")
ax[0].plot(time_grid, simulated_data[:,3], linestyle="dashed", label="Simulated Temp 3", color="orange")
ax[0].plot(time_grid, simulated_data[:,4], linestyle="dashed", label="Simulated Temp 4", color="red")

ax[1].plot(time_grid, simulated_data[:,5], linestyle="dashed", label="Simulated Temp 1", color="green")
ax[1].plot(time_grid, simulated_data[:,6], linestyle="dashed", label="Simulated Temp 2", color="red")

ax[2].plot(time_grid, simulated_data[:,7], linestyle="dashed", label="Simulated Temp 1", color="green")
ax[2].plot(time_grid, simulated_data[:,8], linestyle="dashed", label="Simulated Temp 2", color="red")


ax[0].plot(time_grid, temperature_data[:,0], label="True Temp  1", color="green")
ax[0].plot(time_grid, temperature_data[:,1], label="True Temp  2", color="blue")
ax[0].plot(time_grid, temperature_data[:,2], label="True Temp  3", color="orange")
ax[0].plot(time_grid, temperature_data[:,3], label="True Temp  4", color="red")

ax[1].plot(time_grid, temperature_data[:,4], label="True Temp  1", color="green")
ax[1].plot(time_grid, temperature_data[:,5], label="True Temp  2", color="red")

ax[2].plot(time_grid, temperature_data[:,6], label="True Temp  1", color="green")
ax[2].plot(time_grid, temperature_data[:,7], label="True Temp  2", color="red")

ax[0].legend()
ax[1].legend()
ax[2].legend()

plt.show()


    
    
    
    
    
    
    
    
    
    
    
# flow_curve_2 = np.interp(temperature_data[-1],flow_data[-1],flow_data[2])

# mass_flow_rates = np.array([flow_curve_1,
#                             flow_curve_2])
# mass_flow_rates = mass_flow_rates * Gal_min_to_Kg_s

# time_grid = temperature_data[-1]/1000

# inlet_temps       = temperature_data[[4,6]]
# outlet_temps_true = temperature_data[[5,7]]



# time_grid         = time_grid[::10]
# mass_flow_rates   = mass_flow_rates[:,::10]
# inlet_temps       = inlet_temps[:,::10]
# outlet_temps_true = outlet_temps_true[:,::10]

# def model_2(time_grid,            #s
#             spacial_grids,        #m
#             mass_flow_rates,      #Kg/s
#             inlet_temps,          #C
#             initial_outlet_temps, #C
#             heat_exchange_coeff,  #J/m^2-K-s
#             specific_heat,        #J/Kg-K
#             fluid_density,        #kg/m^3
#             flow_areas,           #m^2
#             exchanger_grid):
#     heat_transfer_map           = [-1,1]
#     outlet_temps                = np.zeros(inlet_temps.shape)
#     outlet_temps[:,0]           = initial_outlet_temps
    
#     temp_distributions           = [None]*2
#     temp_distributions[0]        = np.linspace(inlet_temps[0][0],  outlet_temps[0][0], spacial_grids[0].size)              #C
#     temp_distributions[1]        = np.linspace(outlet_temps[1][0], inlet_temps[1][0],  spacial_grids[1].size)              #C
    
#     for step in range(time_grid.size-1):
#         time_diff                 = time_grid[step+1]-time_grid[step]                                                        #s
#         travel_distances          = [None]*2
#         travel_distances[0]       = mass_flow_rates[0,step]*time_diff/(fluid_density*flow_areas[0])
#         travel_distances[1]       = mass_flow_rates[1,step]*time_diff/(fluid_density*flow_areas[1])
        
        
#         #Rotates Nodes
#         outlet_temps[0,step+1]    = temp_distributions[0][-1]                                                                #C
#         outlet_temps[1,step+1]    = temp_distributions[1][0]                                                                 #C
        
#         temp_distributions[0]     = np.interp(spacial_grids[0], spacial_grids[0]+travel_distances[0], temp_distributions[0])  #C
#         temp_distributions[0][0]  = inlet_temps[0,step+1]                                                                    #C
#         temp_distributions[1]     = np.interp(spacial_grids[1], spacial_grids[1]-travel_distances[1], temp_distributions[1])  #C
#         temp_distributions[1][-1] = inlet_temps[1,step+1]                                                                    #C
        
        
#         #Transfers Heat
#         heat_fluxes               = heat_exchange_coeff*(np.interp(exchanger_grid,spacial_grids[0],temp_distributions[0])-   #J/m^2-s
#                                                         np.interp(exchanger_grid,spacial_grids[1],temp_distributions[1]))
        
#         for idx in range(2):
#             sub_grid_indices      = np.where(np.logical_and(spacial_grids[idx]>np.min(exchanger_grid[0]),
#                                                             spacial_grids[idx]<np.min(exchanger_grid[-1])))[0]
#             next_index            = np.max(sub_grid_indices)+1
            
#             sub_grid              = np.ones(sub_grid_indices.size+2)                                                         #m
#             sub_grid[0]           = exchanger_grid[0]                                                                        #m
#             sub_grid[-1]          = exchanger_grid[-1]                                                                       #m
#             sub_grid[1:-1]        = spacial_grids[idx][sub_grid_indices]                                                     #m
            
#             matched_heat_flux     = np.interp(sub_grid, exchanger_grid, heat_fluxes)                                         #J/m^2-s
            
#             fill_indices          = np.concat([sub_grid_indices,np.array([next_index])])
            
#             transfer_areas        = travel_distances[idx][fill_indices]                                                  #m^2
#             transfer_areas[0]     = np.min([transfer_areas[0],sub_grid[1]-sub_grid[0]])                                      #m^2
            
#             heat_transfer         = ((matched_heat_flux[1:]+matched_heat_flux[:-1])/2)*transfer_areas                        #J
            
#             temp_distributions[idx][fill_indices] = temp_distributions[idx][fill_indices] + heat_transfer/(mass_flow_rates[idx,step]*specific_heat)*heat_transfer_map[idx]
    
#     return(outlet_temps)















# parameters=[500,0.509,0.377,0.4325,0.565,1,
#             0.000775,0.00411,0.00110,0.000717,0.00291,0.0005,
#             2]

# fig, ax = plt.subplots()

# def update(parameters):
#     ax.cla()
    
#     heat_exchange_coeff = parameters[0]

#     geometric_parameters=np.array([[parameters[1],parameters[2]],[parameters[3],parameters[4]],[0,parameters[5]]])

#     loop_1_points=[-geometric_parameters[0,0], geometric_parameters[2,1]+geometric_parameters[0,1]]
#     loop_2_points=[-geometric_parameters[1,0], geometric_parameters[2,1]+geometric_parameters[1,1]]

#     pipe_areas_1=[parameters[6],parameters[7],parameters[8]]
#     pipe_areas_2=[parameters[9],parameters[10],parameters[11]]

#     exchanger_points=[0, geometric_parameters[2,1]]






#     num_spacial_steps   = int(10**parameters[12])

#     spacial_grid_1                       = np.linspace(loop_1_points[0], loop_1_points[1], num_spacial_steps+1)
#     spacial_grid_2                       = np.linspace(loop_2_points[0], loop_2_points[1], num_spacial_steps+1)

#     pipe_area_grid_1                     = np.ones(num_spacial_steps+1)
#     indexes                              = np.where(spacial_grid_1<exchanger_points[0])[0]
#     pipe_area_grid_1[indexes]            = pipe_area_grid_1[indexes]*pipe_areas_1[0]
#     indexes                              = np.where(np.logical_and(spacial_grid_1>=exchanger_points[0],spacial_grid_1<exchanger_points[1]))[0]
#     pipe_area_grid_1[indexes]            = pipe_area_grid_1[indexes]*pipe_areas_1[1]
#     indexes                              = np.where(spacial_grid_1>=exchanger_points[1])[0]
#     pipe_area_grid_1[indexes]            = pipe_area_grid_1[indexes]*pipe_areas_1[2]

#     pipe_area_grid_2                     = np.ones(num_spacial_steps+1)
#     indexes                              = np.where(spacial_grid_2<exchanger_points[0])[0]
#     pipe_area_grid_2[indexes]            = pipe_area_grid_2[indexes]*pipe_areas_2[0]
#     indexes                              = np.where(np.logical_and(spacial_grid_2>=exchanger_points[0],spacial_grid_2<exchanger_points[1]))[0]
#     pipe_area_grid_2[indexes]            = pipe_area_grid_2[indexes]*pipe_areas_2[1]
#     indexes                              = np.where(spacial_grid_2>=exchanger_points[1])[0]
#     pipe_area_grid_2[indexes]            = pipe_area_grid_2[indexes]*pipe_areas_2[2]

#     exchanger_grid                       = np.linspace(exchanger_points[0], exchanger_points[1],     num_spacial_steps+1)
    
#     outlet_temps = model_2(time_grid,
#                         [spacial_grid_1, spacial_grid_2],
#                         mass_flow_rates,
#                         inlet_temps,
#                         outlet_temps_true[:,0],
#                         heat_exchange_coeff,
#                         specific_heat_water,
#                         density_water,
#                         [pipe_area_grid_1, pipe_area_grid_2],
#                         exchanger_grid)

#     ax.plot(time_grid, inlet_temps[0],       color="red",                        label="True Inlet Loop 2")
#     ax.plot(time_grid, inlet_temps[1],       color="green",                      label="True Inlet Loop 3")
#     ax.plot(time_grid, outlet_temps_true[0], color="orange",                     label="True Outlet Loop 2")
#     ax.plot(time_grid, outlet_temps_true[1], color="blue",                       label="True Outlet Loop 3")
#     ax.plot(time_grid, outlet_temps[0],      color="orange", linestyle="dashed", label="New Outlet Loop 2")
#     ax.plot(time_grid, outlet_temps[1],      color="blue",   linestyle="dashed", label="New Outlet Loop 3")
#     ax.set_xlim(0,np.max(time_grid))
#     ax.set_ylim(14,22)
    
# update(parameters)






# from functools import partial

# def slider_template_update(val,el):
#     parameters[el]=val
#     update(parameters)



# slider_names=["Coeff","length 1","length 2","length 3","length 4","exchange","Area 1","Area 2","Area 3","Area 4","Area 5","Area 6","num_steps"]
# slider_bounds=[[200,1000],[0,1],[0,1],[0,1],[0,1],[0,2],[0,0.0075],[0,0.0075],[0,0.0075],[0,0.0075],[0,0.0075],[0,0.0075],[1,6]]
# sliders=[None]*13
# for i in range(13):
#     slider_adjust = fig.add_axes([0.05*(i+1), 0.25, 0.0225, 0.63])
#     sliders[i] = Slider(
#         ax=slider_adjust,
#         label=slider_names[i],
#         valmin=slider_bounds[i][0],
#         valmax=slider_bounds[i][1],
#         valinit=parameters[i],
#         orientation="vertical"
#     )
#     sliders[i].on_changed(partial(slider_template_update, el=i))

# fig.subplots_adjust(left=0.05*15)


# plt.show()
# print(parameters)





# fig, ax = plt.subplots()
# update(parameters)
# plt.show()