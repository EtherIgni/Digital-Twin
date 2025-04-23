import numpy as np

#area_curves    -> Node Areas
#flow_curves    -> Node Positions
#loop_lengths   -> Loop Lengths
#num_exchangers -> Number Exchangers

def update_temperature(node_temperatures,
                       time_diff,
                       mass_flow_rates,
                       water_density,
                       geometry_dict,
                       inlet_temp):
    #Heat Shift
    for i in range(num_loops):
        #Calculate fluid flow distance
        travel_distances         = mass_flow_rates[i]*time_diff/(density_water*geometry_dict["Node Areas"][i])
        
        #Rotate and re-interpolate temperature data
        new_positions            = geometry_dict["Node Positions"][i]+travel_distances
        if(i==2):
            keep_indices         = np.where(new_positions<=geometry_dict["Loop Lengths"][i])[0]
            new_positions        = np.concat([[0],
                                              new_positions[keep_indices]])
            interp_temps         = np.concat([[inlet_temp],
                                              temp_curve[i][keep_indices]])
            
            node_temperatures[i] = np.interp(geometry_dict["Node Positions"][i],
                                             new_positions,
                                             interp_temps)
        else:
            node_temperatures[i] = np.interp(geometry_dict["Node Positions"][i],
                                             new_positions,
                                             node_temperatures[i],
                                             period=geometry_dict["Loop Lengths"][i])
    
    
    
    #Heat Transfer
    for i in range(geometry_dict["Number Exchangers"]):
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
        
    
    
    #Tank Mixing
    for i in range(num_tanks):
        tank_indices=np.where(np.logical_and(flow_curves[tank_loop[i]]>tank_positions[i],
                                                flow_curves[tank_loop[i]]<=tank_positions[i]+tank_lengths[i]))[0]
        drawn_temps=temp_curve[tank_loop[i]][step][tank_indices]*mix_percentages[i]
        average_temps=np.mean(drawn_temps)*np.ones(drawn_temps.size)
        temp_curve[tank_loop[i]][step][tank_indices]=temp_curve[tank_loop[i]][step][tank_indices]-drawn_temps+average_temps