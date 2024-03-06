import numpy as np


def numpy_mesh_region(x: int, y: int, pm: int) -> np.ndarray:
    """
    Create a numpy mesh region
    """
    # Region indices
    xmesh = np.arange(start=x - pm, stop=x + pm + 1, dtype=int)
    ymesh = np.arange(start=y - pm, stop=y + pm + 1, dtype=int)
    # Create a meshgrid of all combinations
    xx, yy = np.meshgrid(xmesh, ymesh)
    region = np.vstack((xx.ravel(), yy.ravel())).T

    return region


def setdiff2d_idx(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """
    2d version of np.setdiff1d
    """
    delta = set(map(tuple, arr2))
    idx = [tuple(x) not in delta for x in arr1]
    return arr1[idx]


def create_square_region(
    image: np.ndarray, centerpoint: list[int], width: int
) -> np.ndarray:
    """
    Create a square boundary region given centerpoint and width

    Parameters:
    -----------
    centerpoint: list[int]
        x,y index coordinates of the center point

    width: int
        Odd integer denoting the width of the square

    Returns:
    --------
    sq_region: np.ndarray
        Square region

    sq_boundary: np.ndarray
        Boundary of the square region

    image_region: np.ndarray
        Image values in region
    """
    #Deprecated
    assert width % 2 == 1, f"Width is not odd: {width}"
    # Value to add subtract:
    pm = (width - 1) / 2

    # xy coordinations
    x, y = centerpoint

    # Boundary values
    x1 = int(x - pm)
    x2 = int(x + pm)
    y1 = int(y - pm)
    y2 = int(y + pm)

    # Region and boundary indices
    region = numpy_mesh_region(x=x, y=y, pm=pm)
    region_boundary = numpy_mesh_region(x=x, y=y, pm=pm + 1)
    # Remove overlap between boundary and region
    region_boundary = setdiff2d_idx(region_boundary, region)

    # Image Region
    image_region = image[x1 : x2 + 1, y1 : y2 + 1]

    return region, region_boundary, image_region


def inflection_points(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calculate inflection points

    Parameters:
    -----------
    x: np.ndarray
        x values, for deformable models this is
        scale

    y: np.ndarray
        y values, for deformable models this is
        the KL divergence or special cases of it

    Returns:
    --------
    ip: np.ndarray
        Inflection points
    """
    # Calculate the second derivative
    dydx = np.gradient(y, x)
    d2ydx2 = np.gradient(dydx, x)

    # Find the indices where the second derivative changes sign
    inflection_indices = np.where(np.diff(np.sign(d2ydx2)))[0]

    # Get the x and y values at the inflection points
    # ips = list(zip(x[inflection_indices], y[inflection_indices]))
    ips = list(x[inflection_indices])

    return ips


def delta(x: int) -> bool:
    """
    Delta function
    return 1 if x==0
    return 0 if x!=0
    """
    return int(x + 1 == 1)
