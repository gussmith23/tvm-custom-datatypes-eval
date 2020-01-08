import tvm
from tvm import relay
from tvm.relay.frontend import from_keras
import keras

mobilenet = keras.applications.mobilenet.MobileNet(include_top=True,
                                                   weights=None,
                                                   input_shape=(224, 224, 3),
                                                   classes=1000)
shape_dict = {'input_1': (1, 3, 224, 224)}
module, params = from_keras(mobilenet, shape_dict)
print(params[list(params.keys())[0]])

