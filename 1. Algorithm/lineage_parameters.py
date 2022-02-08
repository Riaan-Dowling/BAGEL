def parameters(pseudo_data):
    
    #Window parameters                                     ________
    step_size = 200 #How much cells per step (Step size)  |_______/
    window_size = 150 #How many cells in window  (Widow size) |_____|

    #Gibbs sampler parameters
    itterations =  200  #Total itterations
    burn_period = 100   #Total itterations to burn "burn period"

    '''
    -----------------------------------------------------------------
    Future
    -----------------------------------------------------------------
    '''
    #window parameters
    window_step = 0 #initialize_cells_offset zero step
    #Shift cell perspective
    # pseudo_data = pseudo_data.head(3000) #Cap total cells
    initialize_cells_offset =0# Start offset

    return pseudo_data, initialize_cells_offset, step_size,window_step, window_size, itterations, burn_period

def total_windows_estimate(pseudo_data, initialize_cells_offset, window_size, step_size):
    #Estimate total windows
    cell_total = len(pseudo_data['Pseudo_Time_normal'])

    total_windows = (cell_total - initialize_cells_offset - step_size )/window_size
    total_windows_trunc = (total_windows)%1
    half_window = None
    if(total_windows_trunc>0):
        total_windows = int(total_windows) + 1 #Truncated data and leava as is because for loop starts at 0
        half_window = True
    else:
        total_windows = int(total_windows) #Truncated data and remove one itteration because for loop starts a 0
        half_window = False

    # total_windows = (cell_total - initialize_cells_offset - window_size)/step_size
    # total_windows_trunc = (total_windows)%1
    # half_window = None
    # if(total_windows_trunc>0):
    #     total_windows = int(total_windows) + 1 #Truncated data and leava as is because for loop starts at 0
    #     half_window = True
    # else:
    #     total_windows = int(total_windows) #Truncated data and remove one itteration because for loop starts a 0
    #     half_window = False
    
    return total_windows, half_window, cell_total