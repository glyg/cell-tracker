from sktracker.tracker.solver import ByFrameSolver
from sktracker.tracker.solver import GapCloseSolver
from sktracker.trajectories import Trajectories


def track_cells(positions, max_speed, cut_penalty=2.,
                max_gap=4, fragment=1, coords=['x', 'y', 'z']):
    '''Runs the tracking for the given positions. Returns the tracked positions

    Parameters
    ----------

    positions : a :class:`pandas.DataFrame`
        with the untracked positions

    max_speed : a float
        maximum allowed displacement between two positions

    cut_penalty : a float
        the penalty applied to leave a cut between two positions
        rather than making a link. When this value is low (typically
        1), there will be more fragments on the trajectory after
        the simple track. It should be between 1 and 10

    fragment : a float
        when this value is high (around 20), the trajectory will be
        more fragmented

    Returns
    -------

    tracked_positions : a :class:`sktracker.Trajectories`
        the tracked positions

    '''
    bf_solver = ByFrameSolver.for_brownian_motion(positions, max_speed,
                                                  penalty=cut_penalty)
    positions = bf_solver.track()

    ## Reverse once
    bf_solver.trajs = bf_solver.trajs.reverse()
    bf_solver.trajs = bf_solver.track()

    ## Back to original
    bf_solver.trajs = bf_solver.trajs.reverse()
    positions = bf_solver.track()
    link_percentile = 100 - fragment
    ## Gap closing
    gc_solver = GapCloseSolver.for_brownian_motion(positions,
                                                   max_speed,
                                                   maximum_gap=max_gap,
                                                   link_percentile=link_percentile,
                                                   coords=coords)
    positions = gc_solver.track()

    return positions
