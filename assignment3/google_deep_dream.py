# imports and basic notebook setup

from io import StringIO
import numpy as np
import scipy.ndimage as nd
import PIL.Image
from IPython.display import clear_output, Image, display
from google.protobuf import text_format

import caffe

# If your GPU supports CUDA and Caffe was built with CUDA support,
# uncomment the following to run Caffe operations on the GPU.
# caffe.set_mode_gpu()
# caffe.set_device(0) # select GPU device if multiple devices exist

'''
Speichert Array in ein Bild
'''
def showarray(a, fmt='jpeg', fileName='deep'):
    a = np.uint8(np.clip(a, 0, 255))
    f = StringIO()
    arr = PIL.Image.fromarray(a)
    arr.save(fileName + ".jpg", fmt)
    display(Image(data=f.getvalue()))
    
# NN laden
model_path = 'models/' # substitute your path here
net_fn   = model_path + 'deploy.prototxt'
param_fn = model_path + 'bvlc_googlenet.caffemodel'

# Patching model to be able to compute gradients.
# Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
model = caffe.io.caffe_pb2.NetParameter()
text_format.Merge(open(net_fn).read(), model)
model.force_backward = True
open('tmp.prototxt', 'w').write(str(model))

net = caffe.Classifier('tmp.prototxt', param_fn,
                       mean = np.float32([104.0, 116.0, 122.0]), # ImageNet mean, training set dependent
                       channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB

# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']

def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])

#euclidean distance
def objective_L2(dst):
    # überschreibe kompletten Array
    dst.diff[:] = dst.data 


def make_step(net, step_size=1.5, end_layer='inception_4c/output', 
              jitter=32, clip=True, objective=objective_L2):
    '''Basic gradient ascent step.'''

    # input image is stored in Net's 'data' blob
    # vorbereitetes Bild (mit Detail von der vorherigen Octave angewendet)
    src = net.blobs['data'] 
    
    # Die Werte die am end_layer gespeichert sind
    dst = net.blobs[end_layer]
    
    # verschieben das  Bild in ox und oy Richtung
    ox, oy = np.random.randint(-jitter, jitter+1, 2)
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift
            
    # Process data normally to the layer end_layer
    # aktuell ist dst.data leer, hiermit wird das gefüllt
    net.forward(end=end_layer)
    
    # dst.diff wird auf dst.data gesetzt, also das was durch forward() errechnet wurde
    objective(dst)  # specify the optimization objective
    
    # Process data from the back to the beginnen, start at layer end_layer
    # hier wird src.diff verändert (das sind die Gradients, 
    # also wie die weights angepasst werden sollten, aber NICHT werden)
    net.backward(start=end_layer)
    
    # die Gradients haben die selbe Shape wie das Input Image, da die Gradients von
    # src.diff[0] die Gradients des ersten, also den Input Layers sind
    # die Gradient der Layer die dazwichen liegen werden zwar auch berechnet, aber
    # für Deep Dream nicht verwendet
    gradients = src.diff[0]
    
    # apply normalized ascent step to the input image
    # Passe Bild mit der errechneten Differenz an
    # "Abschwächen" der Änderung
    src.data[0] += step_size / np.abs(gradients).mean() * gradients
    
    # unshift image
    src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) 
            
    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255-bias)


def deepdream(net, base_img, iter_n=10, octave_n=4, octave_scale=1.4, 
              end='inception_4c/output', clip=True, **step_params):
    # prepare base images for all octaves
    # octaves sind verschiedene Skalierungen des orignalen Bildes
    octaves = [preprocess(net, base_img)]
    for i in range(octave_n-1):
        octaves.append(nd.zoom(octaves[-1], (1, 1.0/octave_scale,1.0/octave_scale), order=1))
    
    # src ist Anfangs ein leeres Bild, dass dann iterativ gefüllt wird
    src = net.blobs['data']
    
    # erstelle Array mit nullen der die selbe Größe hat wie die größte Skalierung des Bildes
    detail = np.zeros_like(octaves[-1]) # allocate image for network-produced details
    
    for octave_idx, octave_base_img in enumerate(octaves[::-1]):
        # Ausmaße des Bildes herausfinden
        h, w = octave_base_img.shape[-2:]
        
        if octave_idx > 0:
            # upscale details from the previous octave_idx
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0*h/h1,1.0*w/w1), order=1)

        # resize the network's input image size to the currect octavesize
        # num (?), channels, h, w
        src.reshape(1,3,h,w) 
    
        # apply the detail (modification) of the previous iteration to the current
        src.data[0] = octave_base_img+detail
        
        for i in range(iter_n):
            make_step(net, end_layer=end, clip=clip, **step_params)
            
            # visualization
            vis = deprocess(net, src.data[0])
            if not clip: # adjust image contrast if clipping is disabled
                vis = vis*(255.0/np.percentile(vis, 99.98))
            showarray(vis, fileName= end.replace("/","_") + "_step_" + str(i))
            clear_output(wait=True)
            
        # extract details produced on the current octave_idx
        detail = src.data[0]-octave_base_img
    # returning the resulting image
    return deprocess(net, src.data[0])

img = np.float32(PIL.Image.open('images/forest.jpg'))

_=deepdream(net, img)

_=deepdream(net, img, end='inception_3b/5x5_reduce')

net.blobs.keys()

frame = img
frame_i = 0


h, w = frame.shape[:2]
s = 0.05 # scale coefficient
for i in range(100):
    frame = deepdream(net, frame)
    PIL.Image.fromarray(np.uint8(frame)).save("frames/%04d.jpg"%frame_i)
    frame = nd.affine_transform(frame, [1-s,1-s,1], [h*s/2,w*s/2,0], order=1)
    frame_i += 1



guide = np.float32(PIL.Image.open('images/black.jpg'))

end = 'inception_3b/output'
h, w = guide.shape[:2]
src, dst = net.blobs['data'], net.blobs[end]
src.reshape(1,3,h,w)

# anstatt leerem Bild mit Guide anfangen
src.data[0] = preprocess(net, guide)
net.forward(end=end)
guide_features = dst.data[0].copy()

def objective_guide(dst):
    x = dst.data[0].copy()
    y = guide_features
    ch = x.shape[0]
    x = x.reshape(ch,-1)
    y = y.reshape(ch,-1)
    A = x.T.dot(y) # compute the matrix of dot-products with guide features
    # select ones that match best (neuron der am meisten aktiviert wurde)
    dst.diff[0].reshape(ch,-1)[:] = y[:,A.argmax(1)] 
    

_=deepdream(net, img, end=end, objective=objective_guide)


 
