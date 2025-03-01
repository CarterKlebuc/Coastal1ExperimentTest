import BFSBasedCostMap as bf
import numpy as np
import csv
import pandas as pd
import map_and_path_create as map
import matplotlib.pyplot as plt

use_pretested_maps = False
is_file_csv = False

def excel_to_matrix(file_path, sheet_name=0):
    # Read the Excel file
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Convert DataFrame to a 2D matrix (list of lists)
    matrix = df.values.tolist()

    return matrix


# Example usage:
file_path = 'UHMap.xlsx'  # Replace with your file path
sheet_name = 0  # Replace with your sheet name or index if needed
input_grid = excel_to_matrix(file_path, sheet_name)
world_size = [len(input_grid), len(input_grid[0])]
denied_map = np.zeros([world_size[0], world_size[1]])
occupancy_grid = np.zeros([world_size[0], world_size[1]])
start = (25, 0)
end = (2, 30)

for x in range(1, len(input_grid)):
    for y in range(1, len(input_grid[0])):
        if input_grid[x][y] == 1:
            occupancy_grid[x][y] = 1
        else:
            occupancy_grid[x][y] = 0


# Starting Point
#start = [12, 1]

# End
#end = [6, 38]


#print(str(occupancy_grid))

# Change so each point made has its own separate k-ratio

k_ratio = 1
travel_weight = k_ratio
info_weight = (1 - k_ratio)

# Safeguards
if (travel_weight == 0):
    travel_weight = 0.001
if (info_weight == 0):
    info_weight = 0.001

#k_value_array = [1, 0.5, 0.7, 0.001, 0.3, 0.2, 0.1, 0.8]
k_value_array = [0.999999, 0.5, 0.0000001]

#bf.create_experiment_uniform_cost(start, end, world_size, occupancy_grid, travel_weight, info_weight, denied_map,"Test Excel Map: Travel Weight = 1, Info Weight = 1")


#returned_map = map.return_info_map(start, world_size, occupancy_grid, travel_weight, info_weight)

#print(str(returned_map))

#f, (ax3) = plt.subplots(1, 1)
    #print(str(total_map))
#ax3.imshow(returned_map)
    #plt.colorbar(total_map, ax=ax3)
#plt.show()

#map.create_experiment_interface(start, end, world_size, occupancy_grid, travel_weight, info_weight, denied_map, "Interface")

# For UH Map, Start Points should be at [25, 0] and the end point should be at [2, 30] and k_values should be [0.999999, 0.5, 0.0000001]

bf.create_experiment_new_multi_exp(start, end, world_size, occupancy_grid, k_value_array, denied_map, "Test")

#travel_weight = 5
#info_weight = 1
#bf.create_experiment(start, end, world_size, occupancy_grid, travel_weight, info_weight, denied_map, "Test Excel Map: Travel Weight = 5, Info Weight = 1")

#travel_weight = 1
#info_weight = 5
#bf.create_experiment(start, end, world_size, occupancy_grid, travel_weight, info_weight,denied_map,"Test Excel Map: Travel Weight = 1, Info Weight = 5")