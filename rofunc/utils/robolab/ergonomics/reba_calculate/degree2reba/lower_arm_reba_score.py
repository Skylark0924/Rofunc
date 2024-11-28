class LAREBA:
    # For calculating REBA score based on degrees
    def __init__(self, LA_degrees):
        self.LA_degrees = LA_degrees

    def lower_arm_score_overall(self):
        degree = self.LA_degrees
        right_degree = degree[0]
        left_degree = degree[1]
        lower_arm_reba_score = 0
        if right_degree >= left_degree:
            if 0 <= right_degree < 60:
                lower_arm_reba_score = lower_arm_reba_score + 2
            if 60 <= right_degree < 100:
                lower_arm_reba_score = lower_arm_reba_score + 1
            if 100 <= right_degree:
                lower_arm_reba_score = lower_arm_reba_score + 1
        if right_degree < left_degree:
            if 0 <= left_degree < 60:
                lower_arm_reba_score = lower_arm_reba_score + 2
            if 60 <= left_degree < 100:
                lower_arm_reba_score = lower_arm_reba_score + 1
            if 100 <= left_degree:
                lower_arm_reba_score = lower_arm_reba_score + 1

        return lower_arm_reba_score

    def lower_arm_score(self):
        degree = self.LA_degrees
        right_degree = degree[0]
        left_degree = degree[1]

        right_lower_arm_reba_score = 0
        if 0 <= right_degree < 60:
            right_lower_arm_reba_score = right_lower_arm_reba_score + 2
        if 60 <= right_degree < 100:
            right_lower_arm_reba_score = right_lower_arm_reba_score + 1
        if 100 <= right_degree:
            right_lower_arm_reba_score = right_lower_arm_reba_score + 1

        left_lower_arm_reba_score = 0
        if 0 <= left_degree < 60:
            left_lower_arm_reba_score = left_lower_arm_reba_score + 2
        if 60 <= left_degree < 100:
            left_lower_arm_reba_score = left_lower_arm_reba_score + 1
        if 100 <= left_degree:
            left_lower_arm_reba_score = left_lower_arm_reba_score + 1

        return right_lower_arm_reba_score, left_lower_arm_reba_score