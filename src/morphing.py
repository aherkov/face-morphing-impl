import cv2
import dlib
import numpy as np


class MorphingAlgorithm:
    def step(self, alpha):
        pass

    def morph(self, n_steps):
        output = []
        for i in range(1, n_steps + 1):
            if i == 1:
                output.append(self.step(0))
            elif i == n_steps:
                output.append(self.step(1))
            else:
                output.append(self.step(i / n_steps))
        return output


class LinearMorph(MorphingAlgorithm):
    def __init__(self, img_src, img_dst):
        self.img_src = img_src
        self.img_dst = img_dst

    def step(self, alpha):
        return cv2.addWeighted(self.img_src, 1 - alpha, self.img_dst, alpha, 0)


class AdvancedMorph(MorphingAlgorithm):
    def __init__(self, img_src, img_dst):
        self.img_src = img_src
        self.img_dst = img_dst

        width, height = img_src.shape[0], img_src.shape[1]
        rect = (0, 0, width, height)

        # landmarks
        pts_src = find_landmarks(img_src)
        pts_dst = find_landmarks(img_dst)

        # triangulation
        image_borders = [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1),
                         (0, height / 2), (width - 1, height / 2), (width / 2, 0), (width / 2, height - 1)]
        pts_src.extend(image_borders)
        pts_dst.extend(image_borders)

        subdiv_src = cv2.Subdiv2D(rect)
        for point in pts_src:
            subdiv_src.insert(point)

        self.triangles_src = subdiv_src.getTriangleList()

        self.triangles_dst = []
        for t in self.triangles_src:
            i1 = pts_src.index((t[0], t[1]))
            i2 = pts_src.index((t[2], t[3]))
            i3 = pts_src.index((t[4], t[5]))
            self.triangles_dst.append([pts_dst[i1][0], pts_dst[i1][1],
                                       pts_dst[i2][0], pts_dst[i2][1],
                                       pts_dst[i3][0], pts_dst[i3][1]])

        self.triangles_dst = np.array(self.triangles_dst)

    def step(self, alpha):
        triangles_morphed = []

        curr = np.array([(1 - alpha) * triangle_src + alpha * triangle_dst for (triangle_src, triangle_dst) in
                         zip(self.triangles_src, self.triangles_dst)])

        triangles_morphed.append(curr)
        output = np.zeros(self.img_src.shape)

        for j in range(0, len(self.triangles_src)):
            morph_triangle(self.img_src, self.img_dst, output,
                           list_to_triangle(self.triangles_src[j]), list_to_triangle(self.triangles_dst[j]),
                           list_to_triangle(curr[j].ravel()), alpha)

        return output


def list_to_triangle(t):
    return np.array([(t[0], t[1]), (t[2], t[3]), (t[4], t[5])])


def affine_transform(src, src_triangle, dst_triangle, size):
    warping_matrix = cv2.getAffineTransform(np.array(src_triangle, dtype=np.float32),
                                            np.array(dst_triangle, dtype=np.float32))
    output = cv2.warpAffine(src, warping_matrix, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return output


def find_landmarks(img_src):
    cascade_path = "data/haarcascade_frontalface_default.xml"
    predictor_path = "data/shape_predictor_68_face_landmarks.dat"

    # create the haar cascade
    face_detection_classifier = cv2.CascadeClassifier(cascade_path)

    # create the landmark predictor
    landmark_predictor = dlib.shape_predictor(predictor_path)

    # convert the image to grayscale
    img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

    # detect faces in the image
    faces = face_detection_classifier.detectMultiScale(
        img_gray,
        scaleFactor=1.05,
        minNeighbors=5,
        minSize=(100, 100),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    landmarks = None
    for (x, y, w, h) in faces:
        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        detected_landmarks = landmark_predictor(img_src, dlib_rect).parts()
        landmarks = [(p.x, p.y) for p in detected_landmarks]
        break

    return landmarks


# https://github.com/spmallick/learnopencv/blob/master/FaceMorph/faceMorph.py
def morph_triangle(img1, img2, img, triangle1, triangle2, triangle, alpha):
    bounding_rect1 = cv2.boundingRect(np.float32(triangle1))
    bounding_rect2 = cv2.boundingRect(np.float32(triangle2))
    rect = cv2.boundingRect(np.float32(triangle))

    triangle1_rect = []
    triangle2_rect = []
    triangle_rect = []

    for i in range(3):
        triangle_rect.append(((triangle[i][0] - rect[0]), (triangle[i][1] - rect[1])))
        triangle1_rect.append(((triangle1[i][0] - bounding_rect1[0]), (triangle1[i][1] - bounding_rect1[1])))
        triangle2_rect.append(((triangle2[i][0] - bounding_rect2[0]), (triangle2[i][1] - bounding_rect2[1])))

    mask = np.zeros((rect[3], rect[2], 3))
    cv2.fillConvexPoly(mask, np.int32(triangle_rect), (1.0, 1.0, 1.0), 16, 0)

    img1_rect = img1[bounding_rect1[1]:bounding_rect1[1] + bounding_rect1[3],
                bounding_rect1[0]:bounding_rect1[0] + bounding_rect1[2]]
    img2_rect = img2[bounding_rect2[1]:bounding_rect2[1] + bounding_rect2[3],
                bounding_rect2[0]:bounding_rect2[0] + bounding_rect2[2]]

    warp_image1 = affine_transform(img1_rect, triangle1_rect, triangle_rect, (rect[2], rect[3]))
    warp_image2 = affine_transform(img2_rect, triangle2_rect, triangle_rect, (rect[2], rect[3]))

    img_rect = (1.0 - alpha) * warp_image1 + alpha * warp_image2
    img_region = img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
    img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = img_region * (1 - mask) + img_rect * mask
