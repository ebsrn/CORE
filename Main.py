print '---- THIS CODE REQUIRES CHAINER V2 ----'

import warnings
warnings.simplefilter('ignore', UserWarning)
warnings.simplefilter('ignore', RuntimeWarning)
warnings.simplefilter('ignore', FutureWarning)

import numpy as np
import time, os, copy, random, h5py
from argparse import ArgumentParser

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import optimizers, Variable, cuda, serializers
from chainer.iterators import MultiprocessIterator
from chainer.optimizer import WeightDecay, GradientClipping
import DataChef 
import Models


def parse_args():

  def_minibatch = 8
  def_scales_tr = '512,512'
  def_scales_ts = '512,512'
  def_optimizer = 'lr:0.1--lr_pretrained:0.01--momentum:0.9--weightdecay:0.0005--lr_decay_power:0.9--gradientclipping:2.0'
  def_GPUs = '0'
  def_suffix = ''  
  def_checkpoint = 0
  def_dataset = 'WIDER+LIP'  
  def_label_dim = '14+20' #WIDER:14, LIP:20, BAPD:9
  def_eval_split = 'test+validation' #WIDER:test, LIP:validation, BAPD:test
  def_nb_processes = 4
  def_max_iter = 50000
  def_report_interval = 500
  def_save_interval = 2500
  def_eval_interval = 5000
  def_project_folder = '/data/PAMI/CORE'
  def_dataset_folder = '/data2/Datasets'  
  p = ArgumentParser()
  p.add_argument('--minibatch', default=def_minibatch, type=int)
  p.add_argument('--suffix', default=def_suffix, type=str)  
  p.add_argument('--scales_tr', default=def_scales_tr, type=str)
  p.add_argument('--scales_ts', default=def_scales_ts, type=str)
  p.add_argument('--optimizer', default=def_optimizer, type=str)
  p.add_argument('--GPUs', default=def_GPUs, type=str)
  p.add_argument('--dataset', default=def_dataset, type=str)
  p.add_argument('--checkpoint', default=def_checkpoint, type=int)
  p.add_argument('--label_dim', default=def_label_dim, type=str)
  p.add_argument('--eval_split', default=def_eval_split, type=str)
  p.add_argument('--nb_processes', default=def_nb_processes, type=int)
  p.add_argument('--max_iter', default=def_max_iter, type=int)  
  p.add_argument('--report_interval', default=def_report_interval, type=int)
  p.add_argument('--save_interval', default=def_save_interval, type=int)
  p.add_argument('--eval_interval', default=def_eval_interval, type=int)
  p.add_argument('--project_folder', default=def_project_folder, type=str)
  p.add_argument('--dataset_folder', default=def_dataset_folder, type=str)
  args = p.parse_args()
  return args


def Evaluation(splits='validation'):
  # FOR SEMANTIC SEGMENTATION TASK, SET args.minibatch TO 1
  # Creat data generator
  for dataset, split in zip(args.dataset.split('+'), splits.split('+')):
    for image_size in args.scales_ts:
      batch_tuple = MultiprocessIterator(
        DataChef.GetExample(datasets[dataset][split], False, dataset, image_size),
        args.minibatch, n_prefetch=2, n_processes = args.nb_processes, shared_mem=20000000, repeat=False, shuffle=False)
      # Keep the log in history
      if dataset in ['LIP','MSCOCO','PASCAL_SBD']:
        history = {dataset:{'loss':[], 'miou':[], 'pixel_accuracy':[], 'mean_class_accuracy':[], 'image_size': image_size}}
      elif dataset in ['WIDER', 'BAPD']:
        history = {dataset:{'loss':[], 'prediction':[], 'groundtruth':[], 'image_size': image_size}}
      # Evaluate  
      for dataBatch in batch_tuple:
        dataBatch = zip(*dataBatch)    
        # Prepare batch data
        IMG = np.array_split(np.array(dataBatch[0]), len(Model), axis=0)
        LBL = np.array_split(np.array(dataBatch[1]), len(Model), axis=0)        
        # Forward
        for device_id, img, lbl in zip(range(len(Model)), IMG, LBL):
          Model[device_id](img, lbl, dataset, train=False)        
        # Aggregate reporters from all GPUs        
        reporters = []      
        for i in range(len(Model)):        
          reporters.append(Model[i].reporter)
          Model[i].reporter = {} # clear reporter
        # History      
        for reporter in reporters:        
            for k in reporter[dataset].keys():
              history[dataset][k].append(reporter[dataset][k])    
      # Report    
      DataChef.Report(history, args.report_interval * len(args.GPUs), split=split)


def Train():
  # Creat data generator 
  batch_tuples, history = {}, {}
  for dataset in args.dataset.split('+'):
    batch_tuples.update({dataset:[]})
    for image_size in args.scales_tr:
      iterator = MultiprocessIterator(
        DataChef.GetExample(datasets[dataset]['train'], True, dataset, image_size),
        args.minibatch, n_prefetch=2, n_processes = args.nb_processes, shared_mem=20000000, repeat=True, shuffle=True)
      batch_tuples[dataset].append(iterator)
    # Keep the log in history
    if dataset in ['LIP','MSCOCO','PASCAL_SBD']:
      history.update({dataset:{'loss':[], 'miou':[], 'pixel_accuracy':[], 'mean_class_accuracy':[]}})
    elif dataset in ['WIDER', 'BAPD']:
      history.update({dataset:{'loss':[], 'prediction':[], 'groundtruth':[]}})
  # Random input image size (change it after every x minibatch)
  batch_tuple_indx = np.random.choice(range(len(args.scales_tr)), args.max_iter/10)
  batch_tuple_indx = list(np.repeat(batch_tuple_indx, 10))
  # Train
  start_time = time.time()
  for iterk in range(args.checkpoint, len(batch_tuple_indx)):    
    # Get a minibatch while sequentially rotating between datasets
    for dataset in args.dataset.split('+'):     
      dataBatch = batch_tuples[dataset][batch_tuple_indx[iterk]].next()    
      dataBatch = zip(*dataBatch)    
      # Prepare batch data
      IMG = np.array_split(np.array(dataBatch[0]), len(Model), axis=0)
      LBL = np.array_split(np.array(dataBatch[1]), len(Model), axis=0)    
      # Forward
      for device_id, img, lbl in zip(range(len(Model)), IMG, LBL):
        Model[device_id](img, lbl, dataset, train=True)    
      # Aggregate reporters from all GPUs
      reporters = []
      for i in range(len(Model)):
        reporters.append(Model[i].reporter)
        Model[i].reporter = {} # clear reporter
      # History
      for reporter in reporters:
        for k in reporter[dataset].keys():
          history[dataset][k].append(reporter[dataset][k])    
      # Accumulate grads
      for i in range(1,len(Model)):
        Model[0].addgrads(Model[i])
      # Update
      opt.update()
      # Update params of other models
      for i in range(1,len(Model)):
        Model[i].copyparams(Model[0])    
    # Report
    if (iterk+1) % args.report_interval == 0:
      DataChef.Report(
        history, args.report_interval * len(args.GPUs), (iterk+1), time.time()-start_time, split='train')    
    # Saving the model
    if (iterk+1) % args.save_interval == 0 or (iterk+1) == len(batch_tuple_indx):
      serializers.save_hdf5('%s/checkpoints/%s_iter_%d_%s.chainermodel' %
        (args.project_folder, args.dataset, iterk+1, args.suffix), Model[0])
      serializers.save_npz('%s/checkpoints/%s_iter_%d_%s.chaineropt' %
        (args.project_folder, args.dataset, iterk+1, args.suffix), opt)    
    # Evaluation
    if (iterk+1) % args.eval_interval == 0:
      Evaluation(splits=args.eval_split)    
    # Decrease learning rate (poly in 10 steps)
    if (iterk + 1) % int(args.max_iter/10) == 0:
      decay_rate = (1.0 - float(iterk)/args.max_iter) ** args.optimizer['lr_decay_power']
      # Learning rate of fresh layers
      opt.lr *= decay_rate
      # Learning rate of pretrained layers
      for name, param in opt.target.namedparams():
        if name.startswith('/predictor/'):
          param.update_rule.hyperparam.lr *= decay_rate

def SetupOptimizer(model): 
  opt = optimizers.NesterovAG(
    lr=args.optimizer['lr'], momentum=args.optimizer['momentum'])  
  opt.setup(model)  
  return opt


def toGPU():
  # main model is always first
  Model = [opt.target]
  for i in range(1,len(args.GPUs)):
    _model = copy.deepcopy(opt.target)
    _model.to_gpu(args.GPUs[i])
    _model.gpu_id = args.GPUs[i]
    _model.reporter = {}
    Model.append(_model)
  # First GPU device is by default the main one
  opt.target.to_gpu(args.GPUs[0])
  opt.target.gpu_id = args.GPUs[0]
  opt.target.reporter = {}
  return Model


def ResumeFromCheckpoint(path_to_checkpoint, model):
  init_weights = h5py.File(path_to_checkpoint,'r')
  for name, link in model.namedlinks():
    # load pretrained weights
    if name.endswith('/conv') or name.endswith('/bn'):
      path_to_link = ['init_weights']
      for i in name.split('/')[1:]:
        path_to_link.append('["%s"]' % i)
      f = eval(''.join(path_to_link))
      if name.endswith('/conv'):
        link.W.data  = np.array(f['W'])
      elif name.endswith('/bn'):
        link.beta.data  = np.array(f['beta'])
        link.gamma.data  = np.array(f['gamma'])
        link.avg_mean  = np.array(f['avg_mean'])
        link.avg_var  = np.array(f['avg_var'])


# MAIN BODY
args = parse_args()
args.optimizer = dict(zip(['lr','lr_pretrained','momentum','weightdecay','lr_decay_power','gradientclipping'] ,
 [float(x.split(':')[-1]) for x in args.optimizer.split('--')]))

args.label_dim = map(int, args.label_dim.split('+'))
args.scales_tr = [map(int,x.split(',')) for x in args.scales_tr.split('--')]
args.scales_ts = [map(int,x.split(',')) for x in args.scales_ts.split('--')]


# Adjust params w.r.t number of GPUs
args.GPUs = map(int,args.GPUs.split('/'))
args.minibatch *= len(args.GPUs)
args.optimizer['lr'] /= len(args.GPUs)
args.optimizer['lr_pretrained'] /= len(args.GPUs)
args.report_interval /= len(args.GPUs)
args.save_interval /= len(args.GPUs)
args.eval_interval /= len(args.GPUs)
print vars(args)

print 'Prepare Dataset'
datasets = DataChef.PrepData(args)


print 'Initialize Model'
predictor = Models.InceptionV3(args)
classifiers = [Models.Classifier(label_dim) for label_dim in args.label_dim]
model = Models.InceptionV3Classifier(predictor, classifiers, args)


print 'Setup optimizer'
opt = SetupOptimizer(model)

# Use lower learning rate for pretrained parts
for name, param in opt.target.namedparams():
  if name.startswith('/predictor/'):
    param.update_rule.hyperparam.lr = args.optimizer['lr_pretrained']
opt.add_hook(WeightDecay(args.optimizer['weightdecay']))
opt.add_hook(GradientClipping(args.optimizer['gradientclipping']))


# Resume training from a checkpoint
if args.checkpoint > 0:
  print 'Resume training from checkpoint' 
  # Load model weights
  ResumeFromCheckpoint('%s/checkpoints/%s_iter_%d_%s.chainermodel' %
   (args.project_folder, args.dataset, args.checkpoint, args.suffix), model)
  # Load optimizer status
  serializers.load_npz('%s/checkpoints/%s_iter_%d_%s.chaineropt' %
   (args.project_folder, args.dataset, args.checkpoint, args.suffix), opt)
  # Adjust the learning rate
  decay_rate = 1.0
  for iterk in range(args.checkpoint):
    if (iterk + 1) % int(args.max_iter/10) == 0:
      decay_rate *= (1.0 - float(iterk)/args.max_iter) ** args.optimizer['lr_decay_power']  
  # Learning rate of fresh layers
  opt.lr *= decay_rate
  # Learning rate of pretrained layers
  for name, param in opt.target.namedparams():
    if name.startswith('/predictor/'):
      param.update_rule.hyperparam.lr *= decay_rate

print 'Push Model to GPU'
Model = toGPU()

print 'Main Begins'
Train()  