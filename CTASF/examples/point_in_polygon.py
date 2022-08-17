import numpy as np


class PolygonOperator(object):
    def __init__(self, polyX, polyY):
        # only convex polygon available
        # Note: polyCorners should be arranged counter-clockwise
        assert len(polyX) == len(
            polyY), "expected coordinates x_num == y_num, get x_num = {} !eq y_num = {}".format(
                len(polyX), len(polyY))
        self.polyCorners = len(polyX)  # int
        self.polyX = polyX  # array
        self.polyY = polyY  # array
        self.constant = []
        self.multiple = []

        # pre-calculation
        self.precalc_values()

    def precalc_values(self):
        j = self.polyCorners - 1
        for i in range(0, self.polyCorners):
            if self.polyY[j] == self.polyY[i]:
                self.constant.append(self.polyX[i])
                self.multiple.append(0)
            else:
                self.constant.append(self.polyX[i] - (self.polyY[i] * self.polyX[j]) /
                                     (self.polyY[j] - self.polyY[i]) +
                                     (self.polyY[i] * self.polyX[i]) /
                                     (self.polyY[j] - self.polyY[i]))
                self.multiple.append(
                    (self.polyX[j] - self.polyX[i]) / (self.polyY[j] - self.polyY[i]))
                j = i

    def is_in_polygon(self, x, y):
        oddNodes = False
        current = self.polyY[self.polyCorners - 1] > y
        for i in range(0, self.polyCorners):
            previous = current
            current = self.polyY[i] > y
            if current != previous:
                oddNodes ^= y * self.multiple[i] + self.constant[i] < x
        return oddNodes  # True or False


# def box_in_lanes(xyxy, lanes):
#     """
#     Args:
#         xyxy: torch tuple (x1, y1, x2, y2)
#         lanes: list of polygons with original image size
#     """
#     # if xyxy is of tuple of torch type
#     xyxy_np = [num.item() for num in xyxy]
#     # print(xyxy_np)
#     # xyxy_np = xyxy

#     # print(lanes)
#     for lane in lanes:
#         if box_in_lane(xyxy_np, lane):
#             return True
#     return False

# def box_in_lane(xyxy, lane):
#     # lane: np.array type, polygon info
#     x1, y1, x2, y2 = xyxy
#     x, y = (x1 + x2) / 2, (y1 + y2) / 2

#     poly_corners = len(lane)
#     polyX = [corner[0] for corner in lane]
#     polyY = [corner[1] for corner in lane]

#     jdg = Wall(poly_corners, polyX, polyY)
#     jdg.precalc_values()
#     return jdg.pointInPolygon(x, y)

if __name__ == '__main__':
    # test validity
    # xyxy = 1495, 1078, 1919, 1013
    # lane = np.array([[1495, 1078], [756, 491], [799, 492], [1919, 1013], [1919, 1075]])
    # if box_in_lane(xyxy, lane):
    #     print("Point-in-polygon algorithm tested successfully.")
    # else:
    #     print("Recheck point-in-polygon algorithm. Wrong.")
    x1, y1 = -50.0, 130.  # region B
    x2, y2 = -70.0, 135.  # out of region
    x3, y3 = -20., 100.  # out of region
    x4, y4 = -40., 60.  # region A
    xarray1, xarray2 = [-37.3, -37.3, -55.4, -55.4], [25.1, 22.7, -55.4, -63.9]
    yarray1, yarray2 = [9.0, 124.4, 124.4, 69.2], [124.4, 140.8, 140.8, 129.3]
    xarray1.reverse()  # counter-clockwise required
    xarray2.reverse()
    yarray1.reverse()
    yarray2.reverse()
    plg = PolygonOperator(yarray2, xarray2)
    print(plg.is_in_polygon(y1, x1))
    print(plg.is_in_polygon(y2, x2))
    print(plg.is_in_polygon(y3, x3))
    print(plg.is_in_polygon(y4, x4))
