from src.config import PREPROCESSING_NO


def compute_cuts(xs, prepocessing):

    if prepocessing.name == PREPROCESSING_NO:
        cuts = (xs == True).T

    return cuts


def compute_tangles(cuts):

    pass


def compute_clusters(xs, tangles):
    pass
