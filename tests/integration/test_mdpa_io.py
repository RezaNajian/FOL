import pytest
import unittest
import os
import numpy as np
from fol.IO.mdpa_io import MdpaIO
from fol.tools.usefull_functions import *

class TestMdpaIO(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def _request_debug_mode(self,request):
        self.debug_mode = request.config.getoption('--debug-mode')

    def setUp(self):
        # problem setup
        test_name = 'test_mdpa_io'
        self.test_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), test_name)
        create_clean_directory(self.test_directory)
        self.mdpa_io = MdpaIO("mdpa_io",file_name="coarse_sphere.mdpa",
                              case_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),"meshes"))

    def test_io(self):
        self.mdpa_io.Initialize()
        self.assertEqual(self.mdpa_io.GetNumberOfNodes(),85)
        self.assertEqual(self.mdpa_io.GetNumberOfElements("tetra"),249)
        self.assertEqual(len(self.mdpa_io.GetNodeSet("Skin_Part")),66)
        self.assertEqual(len(self.mdpa_io.GetNodeSet("Partial_Skin_Part")),13)

        try:
            field_to_export = np.zeros(self.mdpa_io.GetNumberOfNodes())
            field_to_export[self.mdpa_io.GetNodeSet("Skin_Part")] = np.ones((len(self.mdpa_io.GetNodeSet("Skin_Part"))))
            field_to_export[self.mdpa_io.GetNodeSet("Partial_Skin_Part")] = 2*np.ones((len(self.mdpa_io.GetNodeSet("Partial_Skin_Part"))))
            self.mdpa_io['T'] = field_to_export
            self.mdpa_io.Finalize(export_dir=self.test_directory)
        except Exception as e:
            pytest.fail("value assignment & export failed")

        if self.debug_mode=="false":
            shutil.rmtree(self.test_directory)

if __name__ == '__main__':
    unittest.main()