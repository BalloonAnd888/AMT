from models.endtoend.endtoend import ETE
from models.onsetsandframes.of import OnsetsAndFrames
from models.onsetsandvelocities.inference import BATCH_NORM, CONV1X1_HEAD, DROPOUT, IN_CHANS, LEAKY_RELU_SLOPE
from models.onsetsandvelocities.ov import OnsetsAndVelocities
from preprocessing.constants import DEVICE, N_KEYS, N_MELS

class ModelManager:
    def __init__(self):
        self.model = None
        self.config = {}

    def load_model(self, model_name):
        if model_name == "End to End":
            return self._load_end_to_end()
        elif model_name == "Onsets and Velocities":
            return self._load_onsets_and_velocities()
        elif model_name == "Onsets and Frames":
            return self._load_onsets_and_frames()

    def _load_end_to_end(self):
        self.model = ETE(input_shape=1, 
                         output_shape=N_KEYS).to(DEVICE)
        # print(f"Loading End to End model with config: {self.model}")
        print(f"Loading End to End model")
        return self.model

    def _load_onsets_and_velocities(self):
        self.model = OnsetsAndVelocities(in_chans=IN_CHANS,
                                         in_height=N_MELS,
                                         out_height=N_KEYS,
                                         conv1x1head=CONV1X1_HEAD,
                                         bn_momentum=BATCH_NORM,
                                         leaky_relu_slope=LEAKY_RELU_SLOPE,
                                         dropout_drop_p=DROPOUT).to(DEVICE)
        # print(f"Loading Onsets and Velocities model with config: {self.model}")
        print(f"Loading Onsets and Velocities model")
        return self.model

    def _load_onsets_and_frames(self):
        self.model = OnsetsAndFrames(input_features=N_MELS, 
                                     output_features=N_KEYS, 
                                     model_complexity=48).to(DEVICE)
        # print(f"Loading Onsets and Frames model with config: {self.model}")
        print(f"Loading Onsets and Frames model")
        return self.model 
    