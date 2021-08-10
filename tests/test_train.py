import pytest
import os
from subprocess import call

@pytest.mark.skip(reason="This is to be addressed later")
class Test_Train_Results():
    def test_train(self):
# run unit test to verify train.py runs with stfpm model and create checkpoint file size = 104.9 MB
        call(['python3', 'train.py', '--model', 'stfpm'])
        model_file_size = os.stat('./results/stfpm/mvtec/leather/weights/model-v1.ckpt').st_size
        assert model_file_size == 104855073
# run unit test to verify train.py runs with dfkde model and create checkpoint file size = 119.4 MB
        call(['python3', 'train.py', '--model', 'dfkde'])
        model_file_size = os.stat('./results/dfkde/mvtec/leather/weights/model.ckpt').st_size
        assert model_file_size == 119379237
# run unit test to verify train.py runs with padim model and create checkpoint file size = 212.3 MB
        call(['python3', 'train.py', '--model', 'padim'])
        model_file_size = os.stat('./results/padim/mvtec/leather/weights/model.ckpt').st_size
        assert model_file_size == 212337483
# run unit test to verify train.py runs with patchcore model and create checkpoint file size = 277.3 MB
        call(['python3', 'train.py', '--model', 'patchcore'])
        model_file_size = os.stat('./results/patchcore/mvtec/carpet/weights/model.ckpt').st_size
        assert model_file_size == 277302264
