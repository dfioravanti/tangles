class OrientedCut:

    def __init__(self, cuts, orientations):

        self.current = -1
        self.cuts = cuts
        self.orientations = orientations

    def __iter__(self):
        return self

    def __add__(self, other):
        cuts = self.cuts + other.cuts
        orientations = self.orientations + other.orientations
        return OrientedCut(cuts, orientations)

    def __next__(self):
        self.current += 1
        if self.current < len(self.cuts):
            return self.cuts[self.current], self.orientations[self.current]
        raise StopIteration

    def add_oriented_cut(self, cut, orientation):
        self.cuts += [cut]
        self.orientations += [orientation]
