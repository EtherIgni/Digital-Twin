import numpy as np

#area_curves                -> Node Areas
#flow_curves                -> Node Positions
#loop_lengths               -> Loop Lengths
#num_exchangers             -> Number Exchangers
#num_loops                  -> Number Loops
#num_nodes_in_exchanger     -> Exchanger Node Count
#heat_exchanger_positions   -> Exchanger Positions
#tank_positions             -> Tank Positions
#tank_lengths               -> Tank Lengths
#thermo_probe_positions     -> Thermometer Positions
#heat_exchanger_areas       -> Exchanger Areas
#heater_length              -> Heater Length
#heater_area                -> Heater Area

#density_water              -> Water Density
#specific_heat_water        -> Water Specific Heat

#heat_exchange_coefficients -> Exchange Coefficients
#mix_percentages            -> Mix Percentages

def update_Temperature(node_temperatures,
                       geometry_dict,
                       constants_dict,
                       parameters_dict,
                       inlet_temp,
                       time_diff,
                       heater_flux,
                       mass_flow_rates):
    #Heat Shift
    for i in range(geometry_dict["Number Loops"]):
        #Calculate fluid flow distance
        travel_distances         = mass_flow_rates[i]*time_diff/(constants_dict["Water Density"]*geometry_dict["Node Areas"][i])
        
        #Rotate and re-interpolate temperature data
        new_positions            = geometry_dict["Node Positions"][i]+travel_distances
        if(i==2):
            keep_indices         = np.where(new_positions<=geometry_dict["Loop Lengths"][i])[0]
            new_positions        = np.concat([[0],
                                              new_positions[keep_indices]])
            interp_temps         = np.concat([[inlet_temp],
                                              node_temperatures[i][keep_indices]])
            
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
        exchanger_curve                             = np.linspace(0, geometry_dict["Exchanger Main Length"], geometry_dict["Exchanger Node Count"][i])
        heat_flux                                   = parameters_dict["Exchange Coefficients"][i]*(np.interp(exchanger_curve+geometry_dict["Exchanger Positions"][i,0],                           geometry_dict["Node Positions"][i],   node_temperatures[i])-
                                                                                        np.interp(geometry_dict["Exchanger Positions"][i,1]+geometry_dict["Exchanger Main Length"]-exchanger_curve, geometry_dict["Node Positions"][i+1], node_temperatures[i+1]))
        
        for j in range(2):
            sub_grid_indices                        = np.where(np.logical_and(geometry_dict["Node Positions"][j+i]>geometry_dict["Exchanger Positions"][i,j],
                                                                            geometry_dict["Node Positions"][j+i]<geometry_dict["Exchanger Positions"][i,j]+geometry_dict["Exchanger Main Length"]))[0]
            next_index                              = np.max(sub_grid_indices)+1
            
            sub_grid                                = np.ones(sub_grid_indices.size+2)
            sub_grid[0]                             = geometry_dict["Exchanger Positions"][i,j]
            sub_grid[-1]                            = geometry_dict["Exchanger Positions"][i,j]+geometry_dict["Exchanger Main Length"]
            sub_grid[1:-1]                          = geometry_dict["Node Positions"][j+i][sub_grid_indices]
            
            matched_heat_flux                       = np.interp(sub_grid-geometry_dict["Exchanger Positions"][i,j], exchanger_curve, heat_flux)
            if(j>0):
                matched_heat_flux                   = np.flip(matched_heat_flux)
            
            
            exchanger_travel_distance            = np.ones(sub_grid.size-1)*time_diff/(constants_dict["Water Density"]*geometry_dict["Exchanger Areas"][j])
            if(mass_flow_rates[j+i]>0):
                transfer_areas                   = np.min(np.stack((exchanger_travel_distance,
                                                        (sub_grid[1:]-sub_grid[:-1])/mass_flow_rates[j+i])),
                                                        0)
            else:
                transfer_areas                   = exchanger_travel_distance
            
            heat_transfer                        = ((matched_heat_flux[1:]+matched_heat_flux[:-1])/2) * transfer_areas * (j*2-1) / constants_dict["Water Specific Heat"]
            fill_indices                         = np.concat([sub_grid_indices,np.array([next_index])])
            node_temperatures[j+i][fill_indices] = node_temperatures[j+i][fill_indices] + heat_transfer
    
    
    
    #Heater Heat Input
    sub_grid_indices                    = np.where(np.logical_and(geometry_dict["Node Positions"][0]>0,
                                                            geometry_dict["Node Positions"][0]<geometry_dict["Heater Length"]))[0]
    next_index                          = np.max(sub_grid_indices)+1
    fill_indices                        = np.concat([sub_grid_indices,np.array([next_index])])
    
    heater_travel_distance              = np.ones(fill_indices.size)*time_diff/(constants_dict["Water Density"]*geometry_dict["Heater Areas"])
    if(mass_flow_rates[0]>0):
        grid_sizes                      = (geometry_dict["Node Positions"][0][fill_indices]-np.concat([[0], geometry_dict["Node Positions"][0][fill_indices][:-1]]))/mass_flow_rates[0]
        heater_transfer_areas           = np.min(np.stack((heater_travel_distance,
                                                grid_sizes)),
                                                0)
    else:
        heater_transfer_areas           = heater_travel_distance
    
    heat_transfer                       = heater_flux * heater_transfer_areas / constants_dict["Water Specific Heat"]
    node_temperatures[0][fill_indices]   = node_temperatures[0][fill_indices] + heat_transfer
        
    
    
    #Tank Mixing
    for i in range(geometry_dict["Number Tanks"]):
        tank_indices                                  = np.where(np.logical_and(geometry_dict["Node Positions"][geometry_dict["Loop of Tank"][i]]>geometry_dict["Tank Positions"][i],
                                                                                geometry_dict["Node Positions"][geometry_dict["Loop of Tank"][i]]<=geometry_dict["Tank Positions"][i]+geometry_dict["Tank Lengths"][i]))[0]
        drawn_temps                                   = node_temperatures[geometry_dict["Loop of Tank"][i]][tank_indices]*parameters_dict["Mix Percentages"][i]
        average_temps                                 = np.mean(drawn_temps)*np.ones(drawn_temps.size)
        node_temperatures[geometry_dict["Loop of Tank"][i]][tank_indices] = node_temperatures[geometry_dict["Loop of Tank"][i]][tank_indices]-drawn_temps+average_temps
        
    #Logs temperature at probs
    simulated_probe_temps        = [None]*geometry_dict["Number Loops"]
    for i in range(geometry_dict["Number Loops"]):
        simulated_probe_temps[i] = np.interp(geometry_dict["Thermometer Positions"][i], geometry_dict["Node Positions"][i], node_temperatures[i])
    
    return(node_temperatures,
           simulated_probe_temps)