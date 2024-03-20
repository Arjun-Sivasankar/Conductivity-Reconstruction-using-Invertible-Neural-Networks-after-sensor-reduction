import matplotlib.pyplot as plt
import numpy as np

#def draw_grid(highlighted_numbers):
#    # Create a 10x10 grid
#    grid = np.arange(1, 101).reshape(10, 10)
#
#    # Create the plot
#    fig, ax = plt.subplots()
#    ax.matshow(np.isin(grid, highlighted_numbers), cmap='tab20c')
#
#    # Label with numbers
#    for (i, j), val in np.ndenumerate(grid):
#        ax.text(j, i, val, ha='center', va='center', color='black' if val not in highlighted_numbers else 'black')
#
#    # Set up the axes
#    ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
#    ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
#    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
#    ax.tick_params(axis='both', which='both', length=0)
#    plt.xticks(range(10))
#    plt.yticks(range(10))
#
#    # Show the plot
##    plt.savefig("/beegfs/.global1/ws/arsi805e-Test/New/Clustering_result.png")
#    plt.savefig("/beegfs/.global1/ws/arsi805e-Test/New/Active_Sensors_AllActive.png")
#    plt.show()
    
def draw_grid(highlighted_numbers, direc="/beegfs/.global1/ws/arsi805e-Test/New/Images/SA/15Sens_SA.png"):
    # Create a 10x10 grid
    grid = np.arange(1, 101).reshape(10, 10)

    # Create the plot
    fig, ax = plt.subplots()

    # Define colors for highlighted and non-highlighted numbers
    highlight_color = 0  # Use a numerical value for the highlight color
    non_highlight_color = 1  # Use a numerical value for the non-highlight color

    # Create a mask for highlighted numbers
    mask = np.isin(grid, highlighted_numbers)

    # Use the mask to set colors for the entire grid
    colors = np.where(mask, highlight_color, non_highlight_color)

    ax.matshow(np.ones_like(grid), cmap='gray')  # Create a gray background
    ax.matshow(colors, cmap='tab20c', vmin=0, vmax=1)  # Overlay with the desired colors

    # Label with numbers
    for (i, j), val in np.ndenumerate(grid):
        ax.text(j, i, val, ha='center', va='center', color='black')

    # Set up the axes
    ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
    ax.tick_params(axis='both', which='both', length=0)
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.savefig(direc)

# Example usage
#lst = [34, 35, 36, 37, 44, 45, 46, 47, 54, 55, 56, 57, 64, 65, 66, 67]
#lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]

#lst = [41, 31, 50, 61, 60, 70, 51, 40, 79, 29, 72, 69, 62, 39, 71, 89, 32, 21, 63, 22, 19, 73, 33, 82, 38, 99, 68, 9, 80, 23, 12, 30, 45, 81, 92, 28, 48, 35, 78, 83, 42, 11, 67, 43, 47, 58, 49, 52, 65, 37, 53, 34, 59, 55, 2, 46, 44, 57, 64, 13, 91, 36, 56, 66, 54, 90, 1, 93, 20, 18, 88, 77, 3, 74, 24, 25, 75, 27, 100, 10, 8, 98, 76, 26, 84, 87, 14, 17, 85, 15, 94, 97, 4, 86, 7, 16, 95, 5, 96, 6]
#
#for i in range(0,100,10):
#    draw_grid(lst[:i], direc=f"/beegfs/.global1/ws/arsi805e-Test/New/Images/PCA/PCA_{i}.png")

lst = [1,2,3,4,5,6,7,8,9,10,15,16,18,20,21]

draw_grid(lst)
