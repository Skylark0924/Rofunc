class TrunkREBA:
    # For calculating REBA score based on degrees
    def __init__(self, trunk_degrees):
        self.trunk_degrees = trunk_degrees

    def trunk_reba_score(self):

        trunk_flex_degree = self.trunk_degrees[0]
        trunk_side_bending_degree = self.trunk_degrees[1]
        trunk_torsion_degree = self.trunk_degrees[2]

        trunk_reba_score = 0
        trunk_flex_score = 0
        trunk_side_score = 0
        trunk_torsion_score = 0

        if trunk_flex_degree >= 0:
            # means flexion
            if 0 <= trunk_flex_degree < 5:
                trunk_reba_score += 1
                trunk_flex_score += 1
            elif 5 <= trunk_flex_degree < 20:
                trunk_reba_score += 2
                trunk_flex_score += 2
            elif 20 <= trunk_flex_degree < 60:
                trunk_reba_score += 3
                trunk_flex_score += 3
            elif 60 <= trunk_flex_degree:
                trunk_reba_score += 4
                trunk_flex_score += 4
        else:
            # means extension
            if 0 <= abs(trunk_flex_degree) < 5:
                trunk_reba_score += 1
                trunk_flex_score += 1
            elif 5 <= abs(trunk_flex_degree) < 20:
                trunk_reba_score += 2
                trunk_flex_score += 2
            elif 20 <= abs(trunk_flex_degree):
                trunk_reba_score += 3
                trunk_flex_score += 3

        if abs(trunk_side_bending_degree) >= 3:
            trunk_reba_score += 1
            trunk_side_score += 1
        if abs(trunk_torsion_degree) >= 1:
            trunk_reba_score += 1
            trunk_torsion_score += 1

        return [trunk_reba_score, trunk_flex_score, trunk_side_score, trunk_torsion_score]