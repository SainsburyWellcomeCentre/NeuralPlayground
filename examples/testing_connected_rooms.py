import numpy as np
from sehec.utils import check_crossing_wall

if __name__ == "__main__":
    wall_list = []
    wall_list.append(np.array([[0, 0], [0, -10]]))
    wall_list.append(np.array([[0, 0], [5, 5]]))
    wall_list.append(np.array([[-3, 0], [7, 0]]))

    points_set_1 = [np.array([[-2, 2], [2, -13]]),
                    np.array([[-3, -5], [1, -3]]),
                    np.array([[-2, 1], [2, 2]]),
                    np.array([[-2, -13], [4, -9]]),
                    np.array([[4, -2], [3, -6]])]

    set_1_results = [True, True, False, False, False]

    points_set_2 = [np.array([[1, 0], [2, 4]]),
                    np.array([[2, 7], [7, -2]]),
                    np.array([[3, 2], [7, 7]]),
                    np.array([[-2, 1], [0, -3]]),
                    np.array([[-8, 2], [-11, 4]])]

    set_2_results = [True, True, False, False, False]

    points_set_3 = [np.array([[-2, 1], [7, -3]]),
                    np.array([[4, -3], [8, 2]]),
                    np.array([[-6, 1], [1, -2]]),
                    np.array([[1, 1], [6, 3]]),
                    np.array([[9, 1], [10, -2]])]

    set_3_results = [True, True, False, False, False]

    point_sets = [points_set_1, points_set_2, points_set_3]
    result_sets = [set_1_results, set_2_results, set_3_results]

    for i in range(3):
        wall = wall_list[i]
        points = point_sets[i]
        results = result_sets[i]
        for j, res in enumerate(results):
            point = points[j]
            new_point, answer = check_crossing_wall(pre_state=point[0, :], new_state=point[1, :], wall=wall)
            if answer == res:
                print("passed")
            else:
                print("failed")
                print("points", "wall", "goal", "answer")
                print(point, wall, res, answer)