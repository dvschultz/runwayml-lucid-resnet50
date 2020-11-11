# Import the Runway SDK. Please install it first with
# `pip install runway-python`.
import runway
from runway.data_types import category, image, number, boolean
import numpy as np
import tensorflow as tf
from PIL import Image
# from example_model import ExampleModel

import lucid.modelzoo.vision_models as models
from lucid.misc.io import show
import lucid.optvis.param as param
import lucid.optvis.render as render
import lucid.optvis.transform as transform

setup_options = {
    "network": category(choices=["ResnetV2_50_slim"], default="ResnetV2_50_slim")
}
@runway.setup(options=setup_options)
def setup(opts):
    msg = '[SETUP] Ran with options: network = {}'
    print(msg.format(opts['network']))
    
    model = models.ResnetV2_50_slim()
    model.load_graphdef()
    return model

input_options = {
  'layer': category(choices=["block1/unit_1/bottleneck_v2/preact/Relu (max:64)",
        "block1/unit_1/bottleneck_v2/add (max:256)",
        "block1/unit_2/bottleneck_v2/add (max:256)",
        "block1/unit_3/bottleneck_v2/add (max:256)",
        "block2/unit_1/bottleneck_v2/preact/Relu (max:64)",
        "block2/unit_1/bottleneck_v2/add (max:512)",
        "block2/unit_2/bottleneck_v2/add (max:512)",
        "block2/unit_3/bottleneck_v2/add (max:512)",
        "block2/unit_4/bottleneck_v2/add (max:512)",
        "block3/unit_1/bottleneck_v2/preact/Relu (max:512)",
        "block3/unit_1/bottleneck_v2/add (max:1024)",
        "block3/unit_2/bottleneck_v2/add (max:1024)",
        "block3/unit_3/bottleneck_v2/add (max:1024)",
        "block3/unit_4/bottleneck_v2/add (max:1024)",
        "block3/unit_5/bottleneck_v2/add (max:1024)",
        "block3/unit_6/bottleneck_v2/add (max:1024)",
        "block4/unit_1/bottleneck_v2/preact/Relu (max:1024)",
        "block4/unit_1/bottleneck_v2/add (max:2048)",
        "block4/unit_2/bottleneck_v2/add (max:2048)",
        "block4/unit_3/bottleneck_v2/add (max:2048)",
        "postnorm/Relu (max:2048)"
        ], default="block3/unit_1/bottleneck_v2/preact/Relu (max:512)", description='choose layer of network to visualize'),
  'neuron': number(default=0, min=0, max=2047, step=1, description='Neuron ID'),
  'size': number(default=128, min=128, max=1024, step=128, description='Image Size'),
  'transforms': boolean(default=False, description='Vary size of visualization'),
  'transform_min': number(default=0.3, min=0.0, max=1.0, step=.1, description='Minimum scaling amount'),
  'transform_max': number(default=0.5, min=0.0, max=1.0, step=.1, description='Maximum scaling amount')
}

@runway.command(name='generate',
                inputs=input_options,
                outputs={ 'image': image() },
                description='Use Lucid to visualize the layers and neurons of a specific ML network.')
def generate(model, args):
    print('[GENERATE] Ran with layer {} and neuron {}'.format(args['layer'], args['neuron']))

    layer_id = args['layer'].split(' ')[0]
    layer_neuron = 'resnet_v2_50/{}:{}'.format(layer_id, args['neuron'])
    print(layer_neuron)

    s = int(args['size'])
    min_scale = args['transform_min']
    max_scale = args['transform_max']
    scale_offset = (max_scale - min_scale) * 10

    # https://github.com/tensorflow/lucid/issues/148
    with tf.Graph().as_default() as graph, tf.Session() as sess:
  
        t_img = param.image(s)
        crop_W = int(s/2)
        t_offset = tf.random.uniform((2,), 0, s - crop_W, dtype="int32")
        t_img_crop = t_img[:, t_offset[0]:t_offset[0]+crop_W, t_offset[1]:t_offset[1]+crop_W]
        
        
        if(args['transforms']):
            transforms=[transform.jitter(2), 
                transform.random_scale([min_scale + n/10. for n in range(20)]),
                transform.random_rotate(range(-10,11)),
                transform.jitter(2)]
      
            T = render.make_vis_T(model, layer_neuron, t_img_crop, transforms=transforms)
        else:
            T = render.make_vis_T(model, layer_neuron, t_img_crop)

        tf.initialize_all_variables().run()

        for i in range(1024):
            T("vis_op").run()
          
        img = t_img.eval()[0]

    # https://github.com/tensorflow/lucid/issues/108
    # img = render.render_vis(model, layer_neuron)[-1][0]
    img = Image.fromarray(np.uint8(img*255))
    return {
        'image': img
    }

if __name__ == '__main__':
    # run the model server using the default network interface and ports,
    # displayed here for convenience
    runway.run(host='0.0.0.0', port=8000)
