"""
The factory to create all used speaker pickles in the networks.

Based on previous work of Gerber, Lukic and Vogt, adapted by Heusser
"""
from common.extrapolation.speaker import Speaker

# lehmacl1: defines which speakers are being set up initially
#
def create_all_speakers():
    """
    A generator that yields all Speakers that are needed for the Speaker Clustering Suite to function
    :return: yields Speakers
    """
    yield Speaker(False, 40, 'speakers_40_clustering_vs_reynolds')
    yield Speaker(False, 100, 'speakers_100_50w_50m_not_reynolds')
    yield Speaker(True, 40, 'speakers_40_clustering_vs_reynolds')
    yield Speaker(True, 60, 'speakers_60_clustering')
    yield Speaker(True, 80, 'speakers_80_clustering')
    ##yield Speaker(True, 80, 'speakers_80_stratified_evaluation')
    ##yield Speaker(True, 80, 'speakers_80_stratified_test')
    yield Speaker(False, 470, 'speakers_470_stratified')
    yield Speaker(False, 200, 'speakers_200_100m_100f_evaluation')
    yield Speaker(False, 160, 'speakers_160_for_eval_470')
    yield Speaker(True, 160, 'speakers_160_for_eval_470')
    yield Speaker(True, 200, 'speakers_200_100m_100f_evaluation')
    yield Speaker(True, 590, 'speakers_590_clustering_without_raynolds')
