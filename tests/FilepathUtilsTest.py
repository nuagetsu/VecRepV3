import os
import unittest
import src.data_processing.FilepathUtils as utils
import os


class TestGetFilePath(unittest.TestCase):
    def setUp(self):
        self.project_root = utils.get_project_root()

    def test_get_filepath_imageType(self):
        imageType = "testImage"
        expected = os.path.join(self.project_root, "data", imageType)
        self.assertEqual(utils.get_filepath(imageType=imageType), expected)

    def test_get_filepath_filters(self):
        imageType = "testImage"
        filters = ["filter1", "filter2"]
        expected = os.path.join(self.project_root, "data", imageType + "-filter1--filter2")
        self.assertEqual(utils.get_filepath(imageType=imageType, filters=filters), expected)

    def test_get_filepath_imageProductType(self):
        imageType = "testImage"
        filters = ["filter1", "filter2"]
        imageProductType = "product1"
        expected = os.path.join(self.project_root, "data", imageType + "-filter1--filter2", imageProductType)
        self.assertEqual(utils.get_filepath(imageType=imageType, filters=filters, imageProductType=imageProductType),
                         expected)

    def test_get_filepath_approximatorType(self):
        imageType = "testImage"
        filters = ["filter1", "filter2"]
        imageProductType = "product1"
        embeddingType = "approx1"
        expected = os.path.join(self.project_root, "data", imageType + "-filter1--filter2", imageProductType,
                                embeddingType)
        self.assertEqual(utils.get_filepath(imageType=imageType, filters=filters, imageProductType=imageProductType,
                                            embeddingType=embeddingType), expected)


if __name__ == '__main__':
    unittest.main()
