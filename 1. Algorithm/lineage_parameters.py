def parameters(pseudo_data):

    # Window parameters                                     ________
    pseudo_time_interval = 200  # How much cells per step (Step size)  |_______/
    window_interval = 150  # How many cells in window  (Widow size) |_____|

    # Gibbs sampler parameters
    itterations = 2000  # Total itterations
    burn_period = 500  # Total itterations to burn "burn period"

    """
    -----------------------------------------------------------------
    Future
    -----------------------------------------------------------------
    """
    # window parameters
    window_step = 0  # initialize_cells_offset zero step
    # Shift cell perspective
    # pseudo_data = pseudo_data.head(3000) #Cap total cells
    initialize_cells_offset = 0  # Start offset

    return (
        pseudo_data,
        initialize_cells_offset,
        pseudo_time_interval,
        window_step,
        window_interval,
        itterations,
        burn_period,
    )


def total_windows_estimate(
    pseudo_data, initialize_cells_offset, window_interval, pseudo_time_interval
):
    # Estimate total windows
    cell_total = len(pseudo_data["Pseudo_Time_normal"])

    total_windows = (
        cell_total - initialize_cells_offset - pseudo_time_interval
    ) / window_interval
    total_windows_trunc = (total_windows) % 1
    half_window = None
    if total_windows_trunc > 0:
        total_windows = (
            int(total_windows) + 1
        )  # Truncated data and leava as is because for loop starts at 0
        half_window = True
    else:
        total_windows = int(
            total_windows
        )  # Truncated data and remove one itteration because for loop starts a 0
        half_window = False

    # total_windows = (cell_total - initialize_cells_offset - window_interval)/pseudo_time_interval
    # total_windows_trunc = (total_windows)%1
    # half_window = None
    # if(total_windows_trunc>0):
    #     total_windows = int(total_windows) + 1 #Truncated data and leava as is because for loop starts at 0
    #     half_window = True
    # else:
    #     total_windows = int(total_windows) #Truncated data and remove one itteration because for loop starts a 0
    #     half_window = False

    return total_windows, half_window, cell_total
