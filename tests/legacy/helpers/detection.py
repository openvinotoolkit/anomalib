"""Helpers for detection tests."""
import os
import xml.etree.ElementTree as ET  # nosec
from glob import glob
from typing import List, Tuple

import cv2
import numpy as np


class BBFromMasks:
    """Creates temporary XML files from masks for testing. Intended to be used
    as a context so that the XML files are automatically deleted when the
    execution goes out of scope.

    Example:

        >>> with BBFromMasks(root="/tmp/datasets/MVTec", datast_name="MVTec"):
        >>>     tests_case()

    Args:
        root (str, optional): Path to the dataset location. Defaults to "datasets/MVTec".
        dataset_name (str, optional): Name of the dataset to write to the XML file. Defaults to "MVTec".
    """

    def __init__(self, root: str = "datasets/MVTec", dataset_name: str = "MVTec") -> None:
        self.root = root
        self.dataset_name = dataset_name
        self.generated_xml_files: List[str] = []

    def __enter__(self):
        """Generate XML files."""
        for mask_path in glob(os.path.join(self.root, "*/ground_truth/*/*_mask.png")):
            path_tree = mask_path.split("/")
            image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            image = np.array(image, dtype=np.uint8)
            im_size = image.shape
            contours, _ = cv2.findContours(image, 1, 1)

            boxes = []
            for contour in contours:
                p1 = [np.min(contour[..., 0]), np.min(contour[..., 1])]
                p2 = [np.max(contour[..., 0]), np.max(contour[..., 1])]
                boxes.append([p1, p2])

            contents = self._create_xml_contents(boxes, path_tree, im_size)
            tree = ET.ElementTree(contents)
            output_loc = "/".join(path_tree[:-1]) + f"/{path_tree[-1].rstrip('_mask.png')}.xml"
            tree.write(output_loc)
            # write the xml
            self.generated_xml_files.append(output_loc)

    def __exit__(self, _exc_type, _exc_value, _exc_traceback):
        """Cleans up generated XML files."""
        for file in self.generated_xml_files:
            os.remove(file)

    def _create_xml_contents(
        self, boxes: List[List[List[np.int]]], path_tree: List[str], image_size: Tuple[int, int]
    ) -> ET.Element:
        """Create the contents of the annotation file in Pascal VOC format.

        Args:
            boxes (List[List[List[np.int]]]): The calculated pox corners from the masks
            path_tree (List[str]): The entire filepath of the mask.png image split into a list
            image_size (Tuple[int, int]): Tuple of image size for writing into annotation

        Returns:
            ET.Element: annotation root element
        """
        annotation = ET.Element("annotation")
        ET.SubElement(annotation, "folder").text = path_tree[-2]
        ET.SubElement(annotation, "filename").text = path_tree[-1]

        source = ET.SubElement(annotation, "source")
        ET.SubElement(source, "database").text = self.dataset_name
        ET.SubElement(source, "annotation").text = "PASCAL VOC"

        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = str(image_size[0])
        ET.SubElement(size, "height").text = str(image_size[1])
        ET.SubElement(size, "depth").text = "1"
        for box in boxes:
            object = ET.SubElement(annotation, "object")
            ET.SubElement(object, "name").text = "anomaly"
            ET.SubElement(object, "difficult").text = "1"
            bndbox = ET.SubElement(object, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(box[0][0])
            ET.SubElement(bndbox, "ymin").text = str(box[0][1])
            ET.SubElement(bndbox, "xmax").text = str(box[1][0])
            ET.SubElement(bndbox, "ymax").text = str(box[1][1])

        return annotation
