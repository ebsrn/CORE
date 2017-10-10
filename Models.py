from chainer.functions.array import concat
from chainer.functions.noise import dropout
from chainer.functions.pooling import average_pooling_2d as A
from chainer.functions.pooling import max_pooling_2d as M
from chainer import link
import chainer, cupy
import numpy as np
from chainercv.evaluations import eval_semantic_segmentation
import chainer.functions as F
from chainer.links.connection import convolution_2d as C
from chainer.links.connection import linear
from chainer.links.normalization import batch_normalization as B
from chainer.links.connection import dilated_convolution_2d as D
from chainer import Variable, cuda, initializers
import h5py, os

        
class DilatedConvBN(link.Chain):
    def __init__(self, ich, och, ksize, stride, pad, dilate, init_weights, pool=None):
        super(DilatedConvBN, self).__init__(
            conv = D.DilatedConvolution2D(ich, och, ksize, stride, pad, dilate, nobias=True),
            bn = B.BatchNormalization(och),
            )
        self.pool = pool
        if init_weights:
            f = h5py.File('%s/data/dump/%s.h5' % (os.getcwd(),init_weights),'r')
            self.conv.W.data  = np.array(f['weights']).transpose([3, 2, 0, 1])
            self.bn.beta.data = np.array(f['beta'])
            self.bn.gamma.data = np.array(f['gamma'])
            self.bn.avg_mean = np.array(f['mean'])
            self.bn.avg_var = np.array(f['var'])
    def __call__(self, x):
        if self.pool:
            x = self.pool(x)
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x

class ConvBN(link.Chain):
    def __init__(self, ich, och, ksize, stride, pad, init_weights, pool=None):
        super(ConvBN, self).__init__(
            conv = C.Convolution2D(ich, och, ksize, stride, pad, nobias=True),
            bn = B.BatchNormalization(och),
            )
        self.pool = pool
        if init_weights:
            f = h5py.File('%s/data/dump/%s.h5' % (os.getcwd(),init_weights),'r')
            self.conv.W.data  = np.array(f['weights']).transpose([3, 2, 0, 1])
            self.bn.beta.data = np.array(f['beta'])
            self.bn.gamma.data = np.array(f['gamma'])
            self.bn.avg_mean = np.array(f['mean'])
            self.bn.avg_var = np.array(f['var'])
    def __call__(self, x):
        if self.pool:
            x = self.pool(x)
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x


class Conv(link.Chain):
    def __init__(self, ich, och, ksize, stride, pad, init_weights, pool=None, nobias=False):
        super(Conv, self).__init__(
            conv = C.Convolution2D(ich, och, ksize, stride, pad, nobias),            
            )
        self.pool = pool
    def __call__(self, x):
        if self.pool:
            x = self.pool(x)
        x = self.conv(x)                
        return x


class Sequential(link.ChainList):
    def __call__(self, x, *args, **kwargs):
        for l in self:
            x = l(x, *args, **kwargs)
        return x


class Inception(link.ChainList):
    def __init__(self, *links, **kw):
        super(Inception, self).__init__(*links)
        self.pool = kw.get('pool', None)
    def __call__(self, x):
        xs = [l(x) for l in self]
        if self.pool:
            xs.append(self.pool(x))
        return concat.concat(xs)


class InceptionV3(link.Chain):
    def __init__(self, args):

        convolution = link.ChainList(ConvBN(3, 32, 3, 2, 0, 'conv'),
            ConvBN(32, 32, 3, 1, 0, 'conv_1'),
            ConvBN(32, 64, 3, 1, 1, 'conv_2'),
            ConvBN(64, 80, 1, 1, 0, 'conv_3'),
            ConvBN(80, 192, 3, 1, 0, 'conv_4'))


        def inception_35(ich, pch, name):
            # 1x1
            s1 = ConvBN(ich, 64, 1, 1, 0, name['1x1'][0])
            # 5x5
            s21 = ConvBN(ich, 48, 1, 1, 0, name['5x5'][0])
            s22 = ConvBN(48, 64, 5, 1, 2, name['5x5'][1])
            s2 = Sequential(s21, s22)
            # double 3x3
            s31 = ConvBN(ich, 64, 1, 1, 0, name['3x3'][0])
            s32 = ConvBN(64, 96, 3, 1, 1, name['3x3'][1])
            s33 = ConvBN(96, 96, 3, 1, 1, name['3x3'][2])
            s3 = Sequential(s31, s32, s33)
            # pool
            s4 = ConvBN(ich, pch, 1, 1, 0, name['pool'][1],
             pool=A.AveragePooling2D(3, 1, 1))
            return Inception(s1, s2, s3, s4)

        inception35_names = ({
        '1x1':['mixed_conv'], 
        '5x5':['mixed_tower_conv','mixed_tower_conv_1'],
        '3x3':['mixed_tower_1_conv','mixed_tower_1_conv_1','mixed_tower_1_conv_2'],
        'pool':['mixed_tower_2_pool','mixed_tower_2_conv']
        },
        {
        '1x1':['mixed_1_conv'], 
        '5x5':['mixed_1_tower_conv','mixed_1_tower_conv_1'],
        '3x3':['mixed_1_tower_1_conv','mixed_1_tower_1_conv_1','mixed_1_tower_1_conv_2'],
        'pool':['mixed_1_tower_2_pool','mixed_1_tower_2_conv']
        },
        {
        '1x1':['mixed_2_conv'], 
        '5x5':['mixed_2_tower_conv','mixed_2_tower_conv_1'],
        '3x3':['mixed_2_tower_1_conv','mixed_2_tower_1_conv_1','mixed_2_tower_1_conv_2'],
        'pool':['mixed_2_tower_2_pool','mixed_2_tower_2_conv']
        })               

        inception35 = Sequential(*[inception_35(ich, pch, name)
                                  for ich, pch, name
                                  in zip([192, 256, 288], [32, 64, 64], inception35_names)])
        
        reduction35 = Inception(
            # strided 3x3
            ConvBN(288, 384, 3, 2, 0, 'mixed_3_conv'), # originally stride-pad: 2-0
            # double 3x3
            Sequential(
                ConvBN(288, 64, 1, 1, 0, 'mixed_3_tower_conv'),
                ConvBN(64, 96, 3, 1, 1, 'mixed_3_tower_conv_1'),
                ConvBN(96, 96, 3, 2, 0, 'mixed_3_tower_conv_2') # originally stride-pad: 2-0
                ),
            # pool
            pool=M.MaxPooling2D(3, 2, 0, cover_all=False)) # originally stride-pad: 2-0


        def inception_17(hidden_channel, name):
            # 1x1
            s1 = ConvBN(768, 192, 1, 1, 0, name['1x1'][0])
            # 7x7
            s21 = ConvBN(768, hidden_channel, 1, 1, 0, name['7x7'][0])
            s22 = ConvBN(hidden_channel, hidden_channel, (1,7), (1,1), (0,3), name['7x7'][1])
            s23 = ConvBN(hidden_channel, 192, (7,1), (1,1), (3,0), name['7x7'][2])
            s2 = Sequential(s21, s22, s23)
            # double 7x7
            s31 = ConvBN(768, hidden_channel, 1, 1, 0, name['double7x7'][0])
            s32 = ConvBN(hidden_channel, hidden_channel, (7,1), (1,1), (3,0), name['double7x7'][1])
            s33 = ConvBN(hidden_channel, hidden_channel, (1,7), (1,1), (0,3), name['double7x7'][2])
            s34 = ConvBN(hidden_channel, hidden_channel, (7,1), (1,1), (3,0), name['double7x7'][3])
            s35 = ConvBN(hidden_channel, 192, (1,7), (1,1), (0,3), name['double7x7'][4])
            s3 = Sequential(s31, s32, s33, s34, s35)
            # pool
            s4 = ConvBN(768, 192, 1, 1, 0, name['pool'][1],
                pool=A.AveragePooling2D(3, 1, 1))
            return Inception(s1, s2, s3, s4)

        inception17_names = ({
        '1x1':['mixed_4_conv'], 
        '7x7':['mixed_4_tower_conv','mixed_4_tower_conv_1','mixed_4_tower_conv_2'],
        'double7x7':['mixed_4_tower_1_conv','mixed_4_tower_1_conv_1',
        'mixed_4_tower_1_conv_2','mixed_4_tower_1_conv_3','mixed_4_tower_1_conv_4'],
        'pool':['mixed_4_tower_2_pool','mixed_4_tower_2_conv']
        },
        {
        '1x1':['mixed_5_conv'], 
        '7x7':['mixed_5_tower_conv','mixed_5_tower_conv_1','mixed_5_tower_conv_2'],
        'double7x7':['mixed_5_tower_1_conv','mixed_5_tower_1_conv_1',
        'mixed_5_tower_1_conv_2','mixed_5_tower_1_conv_3','mixed_5_tower_1_conv_4'],
        'pool':['mixed_5_tower_2_pool','mixed_5_tower_2_conv']
        },        
        {
        '1x1':['mixed_6_conv'], 
        '7x7':['mixed_6_tower_conv','mixed_6_tower_conv_1','mixed_6_tower_conv_2'],
        'double7x7':['mixed_6_tower_1_conv','mixed_6_tower_1_conv_1',
        'mixed_6_tower_1_conv_2','mixed_6_tower_1_conv_3','mixed_6_tower_1_conv_4'],
        'pool':['mixed_6_tower_2_pool','mixed_6_tower_2_conv']
        },                
        {
        '1x1':['mixed_7_conv'], 
        '7x7':['mixed_7_tower_conv','mixed_7_tower_conv_1','mixed_7_tower_conv_2'],
        'double7x7':['mixed_7_tower_1_conv','mixed_7_tower_1_conv_1',
        'mixed_7_tower_1_conv_2','mixed_7_tower_1_conv_3','mixed_7_tower_1_conv_4'],
        'pool':['mixed_7_tower_2_pool','mixed_7_tower_2_conv']
        })               

        inception17 = Sequential(*[inception_17(c, name)
                                  for c, name in zip([128, 160, 160, 192], inception17_names)])

        # Reduction 17 to 8
        reduction17 = Inception(
            # strided 3x3
            Sequential(
                ConvBN(768, 192, 1, 1, 0, 'mixed_8_tower_conv'),
                ConvBN(192, 320, 3, 1, 1, 'mixed_8_tower_conv_1')), # originally stride-pad: 2-0
            # 7x7 and 3x3
            Sequential(
                ConvBN(768, 192, 1, 1, 0, 'mixed_8_tower_1_conv'),
                ConvBN(192, 192, (1,7), (1,1), (0,3), 'mixed_8_tower_1_conv_1'),
                ConvBN(192, 192, (7,1), (1,1), (3,0), 'mixed_8_tower_1_conv_2'),
                ConvBN(192, 192, 3, 1, 1, 'mixed_8_tower_1_conv_3')), # originally stride-pad: 2-0
            # pool
            pool=M.MaxPooling2D(3, 1, 1, cover_all=False)) # originally stride-pad: 2-0

        def inception_8(input_channel, name):
            # 1x1
            s1 = ConvBN(input_channel, 320, 1,1,0, name['1x1'][0])
            # 3x3
            s21 = ConvBN(input_channel, 384, 1,1,0, name['3x3'][0])
            s22 = Inception(ConvBN(384, 384, (1, 3),(1,1),(0, 1), name['3x3'][1]),
                            ConvBN(384, 384, (3, 1),(1,1),(1, 0), name['3x3'][2]))
            s2 = Sequential(s21, s22)
            # double 3x3
            s31 = ConvBN(input_channel, 448, 1,1,0, name['double3x3'][0])
            s32 = ConvBN(448, 384, 3, 1,1, name['double3x3'][1])
            s331 = ConvBN(384, 384, (1, 3),(1,1),(0, 1), name['double3x3'][2])
            s332 = ConvBN(384, 384, (3, 1), (1,1),(1, 0), name['double3x3'][3])
            s33 = Inception(s331, s332)
            s3 = Sequential(s31, s32, s33)
            # pool
            s4 = ConvBN(input_channel, 192, 1,1,0, name['pool'][1],
                         pool=A.AveragePooling2D(3, 1, 1))
            return Inception(s1, s2, s3, s4)

        inception8_names = ({
        '1x1':['mixed_9_conv'], 
        '3x3':['mixed_9_tower_conv','mixed_9_tower_mixed_conv','mixed_9_tower_mixed_conv_1'],
        'double3x3':['mixed_9_tower_1_conv','mixed_9_tower_1_conv_1',
        'mixed_9_tower_1_mixed_conv','mixed_9_tower_1_mixed_conv_1'],
        'pool':['mixed_9_tower_2_pool','mixed_9_tower_2_conv']
        },        
        {
        '1x1':['mixed_10_conv'], 
        '3x3':['mixed_10_tower_conv','mixed_10_tower_mixed_conv','mixed_10_tower_mixed_conv_1'],
        'double3x3':['mixed_10_tower_1_conv','mixed_10_tower_1_conv_1',
        'mixed_10_tower_1_mixed_conv','mixed_10_tower_1_mixed_conv_1'],
        'pool':['mixed_10_tower_2_pool','mixed_10_tower_2_conv']
        })               

        inception8 = Sequential(*[inception_8(input_channel, name)
                                  for input_channel, name in zip([1280, 2048],inception8_names)])        


        super(InceptionV3, self).__init__(
            convolution=convolution,            
            inception=link.ChainList(inception35, inception17, inception8),
            grid_reduction=link.ChainList(reduction35, reduction17),
            )        

    def __call__(self, x):

        def convolution(x):
            x = self.convolution[0](x)
            x = self.convolution[1](x)
            x = self.convolution[2](x)
            x = M.max_pooling_2d(x, 3, 2)
            x = self.convolution[3](x)
            x = self.convolution[4](x)
            x = M.max_pooling_2d(x, 3, 2)
            return x

        x = convolution(x)
        x = self.inception[0](x)
        x = self.grid_reduction[0](x)
        x = self.inception[1](x)
        x = self.grid_reduction[1](x)
        x = self.inception[2](x)
        return x

class Classifier(link.Chain):
    def __init__(self, label_dim):
        super(Classifier, self).__init__(
            ASPP=link.ChainList(
                ConvBN(2048, 512, 1, 1, 0, init_weights=None, pool=None),
                ConvBN(2048, 384, 1, 1, 0, init_weights=None, pool=None),
                DilatedConvBN(2048, 384, 3, 1, 4, 4, init_weights=None, pool=None),
                DilatedConvBN(2048, 384, 3, 1, 8, 8, init_weights=None, pool=None),
                DilatedConvBN(2048, 384, 3, 1, 16, 16, init_weights=None, pool=None),                
                ConvBN(2048, 2048, 1, 1, 0, init_weights=None, pool=None)
                ),
            classifier = link.ChainList(
                Conv(2048, label_dim, 1, 1, 0, init_weights=None, pool=None, nobias=False)
                ),
            )        
    def __call__(self, x, train=True):
        
        # Forward
        with chainer.using_config('train', train):
            with chainer.using_config('enable_backprop', train):
                # ASPP
                def ASPP(x):
                    y = [F.tile(self.ASPP[0](F.average_pooling_2d(x, ksize=x.shape[-2:])), x.shape[-2:])]
                    y.extend([self.ASPP[i](x) for i in range(1,len(self.ASPP)-1)])
                    y = F.concat(y, axis=1)
                    y = dropout.dropout(y, ratio=0.5)
                    y = self.ASPP[-1](y)
                    return y
                x = ASPP(x)                
                # Classifier
                y = self.classifier[0](x)
                return y

class InceptionV3Classifier(link.Chain):
    def __init__(self, predictor, classifiers, args):
        super(InceptionV3Classifier, self).__init__(
            predictor=predictor,
            classifiers=link.ChainList(*classifiers)
            )
        self.args = args        
    def __call__(self, x, t, dataset, train=True):
        
        # Create variables
        x = Variable(x)
        x.to_gpu(self.gpu_id)
        t = Variable(t)
        t.to_gpu(self.gpu_id)

        # Config mode
        if len(t.shape) == 3:
            config_mode = 'segmentation'
        elif len(t.shape) == 2:
            config_mode = 'recognition'
        else:
            raise ValueError('label format is not supported')

        # Forward
        with chainer.using_config('train', train):
            with chainer.using_config('enable_backprop', train):
                # InceptionV3 backbone
                x = self.predictor(x)
                # Classifiers
                classifier_indx = self.args.dataset.split('+').index(dataset)                
                y = self.classifiers[classifier_indx](x, train)
                # Loss
                if config_mode == 'segmentation':                    
                    self.y = F.resize_images(y, t.shape[-2:]) # Upsampling logits
                    self.loss = F.softmax_cross_entropy(self.y, t)
                elif config_mode == 'recognition':
                    self.y = F.squeeze(F.average_pooling_2d(y, ksize=y.shape[-2:]),axis=(2,3)) # Global Average Pooling
                    self.loss = F.sigmoid_cross_entropy(self.y, t)
        # Backward
        if train:
            # Clear grads for uninitialized params
            self.cleargrads()
            # Backwards
            self.loss.backward()
        
        # Reporter
        if config_mode == 'segmentation':            
            self.y = F.argmax(self.y, axis=1)
            self.y.to_cpu()
            t.to_cpu()                        
            result = eval_semantic_segmentation(list(self.y.data), list(t.data))
            del result['iou'], result['class_accuracy']
            result.update({'loss':self.loss.data.tolist()})
            self.reporter.update({dataset: result})
        elif config_mode == 'recognition':
            self.reporter.update({dataset:{
                'loss': self.loss.data.tolist(),
                'prediction': F.sigmoid(self.y).data.tolist(),
                'groundtruth': t.data.tolist()}})