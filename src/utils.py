import math
import cv2
import logging
import pandas
from pathlib import Path
import numpy as np
import csv


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def get_cells_from_images(image_filename):
    # Global variables
    global image
    global points

    image = None
    points = []

    # Initializing the filenames
    corners_filename = image_filename[:-4] + "_corners.csv"
    annotation_filename = image_filename[:-4] + ".csv"
    # _, filename = os.path.split(image_filename)
    # destination_dir = out_folder + filename[:-4] + "/"

    # Load the image
    image = cv2.imread(image_filename)
    if image is None:
        logging.error(f"Image {image_filename} not found, please check the input path!")
        return

    clone = image.copy()

    # Reading the annotations
    df_annotation = pandas.read_csv(
        annotation_filename,
        header=None,
        sep=";|,",
        engine="python",
    )

    # Look if annotation of corners exists, otherwise create them
    if Path(corners_filename).is_file():
        corners = load_corners(corners_filename)
    else:
        corners = []
        while len(corners) < 4:
            corners = select_corners(image, points, corners_filename, df_annotation)
            points = []

    # Undistort the image and bring it to a default size
    image_unwarp = four_point_transform(clone, corners)
    image_grid = image_unwarp.copy()
    image_grid = draw_image_grid(image_grid, df_annotation)

    # Splitting the image in its part and associating the annotations
    sub_images, sub_labels = crop_image_parts(image_unwarp, df_annotation)

    return sub_images, sub_labels, image_unwarp, image_grid


def click_and_crop(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

        cv2.circle(image, (x, y), 1, (0, 255, 0), 2)
        cv2.imshow("image", image)


def load_corners(filename):
    corners = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            row = [int(row[i]) for i in range(len(row))]
            corners.append(row)

    return np.array(corners)


def select_corners(image, points, filename, annotation):
    return_points = []

    # Create a new corner annotation
    cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback("image", click_and_crop)

    # Cloning the image for resetting, if needed
    clone = image.copy()

    while True:
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF

        if len(points) == 4:
            return_points = np.array(points).copy()

            # Check selected points
            tmp_image = four_point_transform(clone, return_points)
            image_grid = draw_image_grid(tmp_image, annotation)

            # image_grid = draw_image_grid(
            #     tmp_image,
            #     annotation.shape[1],
            #     annotation.shape[0],
            # )

            cv2.imshow("Validate corners", image_grid)
            key = cv2.waitKey(0)

            cv2.destroyAllWindows()
            if key == ord("y"):
                # Saving file
                with open(filename, mode="w") as csv_file:
                    csv_writer = csv.writer(csv_file, delimiter=",")
                    for idx in range(len(points)):
                        csv_writer.writerow(points[idx])
            else:
                # Discarding everything
                return_points = np.array([])

            break

    return return_points


def order_points(corners):
    # Output will be: tl, tr, br, bl
    ordered_corners = np.zeros((4, 2), dtype="float32")

    # tl will have the smallest sum, br will have the largest sum
    c_sum = corners.sum(axis=1)
    ordered_corners[0] = corners[np.argmin(c_sum)]
    ordered_corners[2] = corners[np.argmax(c_sum)]

    # tr will have the smallest difference, bl will have the largest difference
    diff = np.diff(corners, axis=1)
    ordered_corners[1] = corners[np.argmin(diff)]
    ordered_corners[3] = corners[np.argmax(diff)]

    return ordered_corners


def four_point_transform(image, corners, cell_size=28, cell_num=15):
    rect = order_points(corners)

    new_width = cell_size * cell_num
    new_height = cell_size * cell_num

    new_coords = np.array(
        [
            [0, 0],
            [new_width - 1, 0],
            [new_width - 1, new_height - 1],
            [0, new_height - 1],
        ],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(rect, new_coords)

    warped = cv2.warpPerspective(image, M, (new_width, new_height))

    return warped

colors = [
    (11, 10, 24),
    (11, 21, 17),
    (21, 22, 14),
    (7, 7, 7),
]

def draw_image_grid(image, annotation):

    # colors = [
    #     (112, 106, 241),
    #     (119, 216, 177),
    #     (218, 220, 140),
    #     (77, 77, 77),
    # ]
    img_cols = annotation.shape[1]
    img_rows = annotation.shape[0]

    max_x = image.shape[0] - 1
    max_y = image.shape[1] - 1

    col_step = float(image.shape[0]) / img_cols
    row_step = float(image.shape[1]) / img_rows

    col_step_num = math.floor(image.shape[0] / col_step)
    row_step_num = math.floor(image.shape[1] / row_step)

    for c in range(col_step_num):
        x_coord = int(c * col_step)

        cv2.line(image, (x_coord, 0), (x_coord, max_y), (0, 0, 255), 1)

    for r in range(row_step_num):
        y_coord = int(r * row_step)
        cv2.line(image, (0, y_coord), (max_x, y_coord), (0, 0, 255), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.3
    lineType = 1
    for c in range(col_step_num):
        for r in range(row_step_num):
            x_coord = int(c * col_step) + 5
            y_coord = int(r * row_step) + int(row_step - 5)
            label = annotation[c][r]

            if label:
                text_position = (x_coord, y_coord)
                color = colors[label]

                cv2.putText(
                    image,
                    f"{label}",
                    text_position,
                    font,
                    fontScale,
                    color,
                    lineType,
                )

    return image


def add_label_pred(img, label, pred):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.35
    lineType = 1

    x_coord = 0
    y_coord = int(img.shape[1] - 5)

    text_position = (x_coord, y_coord)

    if label == pred:
        color =  (0, 150, 0)
    else:
        color =  (0, 0, 200)

    cv2.putText(
        img,
        f"{pred}({label})",
        text_position,
        font,
        fontScale,
        color,
        lineType,
    )

def crop_image_parts(image, annotation):
    sub_images = []
    sub_labels = []

    cols = image.shape[0]
    rows = image.shape[1]

    img_cols = annotation.shape[1]
    img_rows = annotation.shape[0]

    col_step = float(cols) / img_cols
    row_step = float(rows) / img_rows

    col_step_num = math.floor(cols / col_step)
    row_step_num = math.floor(rows / row_step)

    for c in range(col_step_num):
        min_val_c = int(c * col_step)
        max_val_c = min(cols, int((c + 1) * col_step))

        for r in range(row_step_num):
            min_val_r = int(r * row_step)
            max_val_r = min(rows, int((r + 1) * row_step))

            sub_img = image[min_val_c:max_val_c, min_val_r:max_val_r, :]
            sub_images.append(sub_img)
            sub_labels.append(annotation[r][c])

    return sub_images, sub_labels


def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [
        cv2.resize(
            im,
            (int(im.shape[1] * h_min / im.shape[0]), h_min),
            interpolation=interpolation,
        )
        for im in im_list
    ]
    return cv2.hconcat(im_list_resize)



def cm_analysis(y_true, y_pred, filename, labels, ymap=None, figsize=(40,40)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            s = cm_sum[i]
            if i == j:
                annot[i, j] = '%.1f%%\n(%d/%d)' % (p, c, s)
            elif c == 0:
                annot[i, j] = '%.1f%%\n(%d/%d)' % (0.0, c, s)
            else:
                annot[i, j] = '%.1f%%\n(%d/%d)' % (p, c, s)
    cm = pd.DataFrame(cm_perc, index=labels, columns=labels)
    cm.index.name = 'True class'
    cm.columns.name = 'Predicted class'
    fig, ax = plt.subplots(figsize=figsize)
    pp = sns.heatmap(cm, vmax=100.1,annot=annot, fmt='', ax=ax, cmap=plt.get_cmap('Blues'),annot_kws={"size":14})
    pp.set_yticklabels([0,1,2,3], fontweight='bold', fontsize=14, rotation=0)
    pp.set_xticklabels([0,1,2,3], fontweight='bold', fontsize=14)
    pp.set_ylabel('True class', fontweight='bold', fontsize=14)
    pp.set_xlabel('Predicted class', fontweight='bold', fontsize=14)
    pp.axes.xaxis.set_ticks_position("top")
    pp.axes.xaxis.set_label_position("top")
    plt.savefig(filename)