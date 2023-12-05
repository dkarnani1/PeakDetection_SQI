import matplotlib.pyplot as plt

def graphannot(charvec, locs, amplitude):
    """
    Add text annotations to a graph at specified locations with given characters and amplitude.

    Parameters:
    - charvec: characters that represent labels to be added to the graph
    - locs: locations of annotations to add to the graph
    - amplitude: y-coordinate to mark labels
    """

    # Convert input to list of strings
    if isinstance(charvec, bool):
        charvec = str(int(charvec))
        chars = list(charvec)
    elif isinstance(charvec, str):
        chars = list(charvec)
    elif isinstance(charvec, (int, float)):
        chars = list(str(charvec))
    elif isinstance(charvec, list):
        chars = list(map(str, charvec))
    else:
        raise ValueError("Unsupported input type for charvec")

    # Plot each character at specified location
    for i, char in enumerate(chars):
        plt.text(locs[i], amplitude, char, ha='center', va='bottom')
