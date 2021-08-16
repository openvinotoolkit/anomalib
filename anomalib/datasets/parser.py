"""
This script contains parsers for different annotations for object detection task.
    Parsers include pascal-voc,.
"""
import logging
from typing import Any, Dict, List, Optional, Union
from xml.etree import ElementTree

from lxml import etree

XML_EXT = ".xml"
ENCODE_METHOD = "utf-8"

logger = logging.getLogger(name="Dataset: Anomaly")


class PascalVocReader:
    """
    Data parser for Pascal-VOC labels
    """

    def __init__(self, file_path: str):
        # shapes type:
        self.labels: List[str] = list()
        self.boxes: List[List[int]] = list()
        self.file_path = file_path
        self.verified: bool = False
        self.xml_tree: Optional[ElementTree.Element] = None
        try:
            self.parse_xml()
        except RuntimeError:
            logger.warning(f"Incorrect format: Unable to parse xml from {file_path}")

    def get_shapes(self) -> Dict[str, Union[List, Any]]:
        """
        Returns:
            annotated bounding boxes and corresponding labels
        """

        return {"boxes": self.boxes, "labels": self.labels}

    def add_shape(self, label: str, bnd_box: ElementTree.Element):
        """
        Args:
            label: label for target object
            bnd_box: bounding box coordinates
        """

        x_min = int(float(bnd_box.find("xmin").text))
        y_min = int(float(bnd_box.find("ymin").text))
        x_max = int(float(bnd_box.find("xmax").text))
        y_max = int(float(bnd_box.find("ymax").text))
        points = [x_min, y_min, x_max - x_min, y_max - y_min]
        self.boxes.append(points)
        self.labels.append(label)

    def parse_xml(self):
        """
        Function to read xml file and parse annotations
        """

        assert self.file_path.endswith(XML_EXT), "Unsupported file format"
        parser = etree.XMLParser(encoding=ENCODE_METHOD)
        self.xml_tree = ElementTree.parse(self.file_path, parser=parser).getroot()
        if "verified" in self.xml_tree.attrib and self.xml_tree.attrib["verified"] == "yes":
            self.verified = True
        else:
            self.verified = False

        for object_iter in self.xml_tree.findall("object"):
            bnd_box = object_iter.find("bndbox")
            label = object_iter.find("name").text
            self.add_shape(label, bnd_box)
