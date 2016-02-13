import numpy as np
import sys
import caffe
import os

class caffe_classifier(object):
    def __init__(self):
        print "Make sure you run the following command with correct caffe-install/ path before this!"
        print "setenv LD_LIBRARY_PATH $HOME/caffe-install/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib"
        print "setenv PYTHONPATH $HOME/caffe-install/lib:$HOME/caffe-install/python"
        # Make sure that caffe is on the python path:
        #caffe_root = '../caffe-install/'  # this file is expected to be in {caffe_root}/examples
        #sys.path.insert(0, caffe_root + 'python')
        
        caffe.set_device(0)
        caffe.set_mode_gpu()
        caffe_root=os.environ['HOME']+'/caffe-install'  # change to correct one
		
        self.net = caffe.Net(caffe_root+'/caffe/models/bvlc_reference_caffenet/deploy.prototxt',
                        caffe_root+'/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                        caffe.TEST)

        # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2,0,1))
        transformer.set_mean('data', np.load(caffe_root + '/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
        transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
        transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB    
        self.transformer=transformer
        
        # load labels
        imagenet_labels_filename = caffe_root + '/caffe/data/ilsvrc12/synset_words.txt'
        self.labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
        
    def get_class_by_file(self,img_path):
        img=caffe.io.load_image(img_path)
        scores,numbers,labels=self.get_class(img)
        return scores,numbers,labels
    
    def get_class(self,img):
        img = self.transformer.preprocess('data', img)
        self.net.blobs['data'].data[...] = img
        out = self.net.forward()
        n=10
        scores=np.sort(out['prob'][0])[-1:-(n+1):-1]
        numbers=out['prob'][0].argsort()[-1:-(n+1):-1]        
        # sort top k predictions from softmax output
        top_k = self.net.blobs['prob'].data[0].flatten().argsort()[-1:-(n+1):-1]        
        return scores,numbers,self.labels[top_k]