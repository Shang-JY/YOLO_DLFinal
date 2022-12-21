from __future__ import print_function
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torch.autograd import Variable

from resnet import *
from yoloLoss import YoloLoss
from dataset import Yolodata
import numpy as np
#from visualize import Visualizer
import numpy as np
import pandas as pd
import hiddenlayer as hl
from tqdm import tqdm
import time
from collections import defaultdict
from eval_voc import Evaluation

RESUME = True
ROOT = './data/VOCdevkit'
BATCH_SIZE = 32
LEARN_RATE = 0.01
EPOCH_NUM = 150
NAME = 'resnet50'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CLASS_NUM = 20
BBOX_NUM = 2
GRID_NUM = 7
best_loss = 1000000  
# best test accuracy


VOC_CLASSES = (    # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
'sheep', 'sofa', 'train', 'tvmonitor')

def categorize(pred):
    '''
    pred (tensor) 1x7x7x30
    return (tensor) box[[x1,y1,x2,y2]] label[...]
    '''
    # print(pred.shape)
    grid_num = GRID_NUM
    boxes=[]
    cls_indexs=[]
    probs = []
    cell_size = 1./grid_num
    pred = pred.data
    # pred = pred.squeeze(0) #7x7x30
    contain1 = pred[:,:,4].unsqueeze(2)
    contain2 = pred[:,:,9].unsqueeze(2)
    contain = torch.cat((contain1,contain2),2)
    mask1 = contain > 0.1 #大于阈值
    mask2 = (contain==contain.max()) #we always select the best contain_prob what ever it>0.9
    mask = (mask1+mask2).gt(0)
    # min_score,min_index = torch.min(contain,2) #每个cell只选最大概率的那个预测框
    for i in range(grid_num):
        for j in range(grid_num):
            for b in range(2):
                # index = min_index[i,j]
                # mask[i,j,index] = 0
                if mask[i,j,b] == 1:
                    #print(i,j,b)
                    box = pred[i,j,b*5:b*5+4]
                    contain_prob = torch.FloatTensor([pred[i,j,b*5+4]])
                    xy = torch.FloatTensor([j,i])*cell_size #cell左上角  up left of cell
                    box[:2] = box[:2]*cell_size + xy # return cxcy relative to image
                    box_xy = torch.FloatTensor(box.size())#转换成xy形式    convert[cx,cy,w,h] to [x1,xy1,x2,y2]
                    box_xy[:2] = box[:2] - 0.5*box[2:]
                    box_xy[2:] = box[:2] + 0.5*box[2:]
                    max_prob,cls_index = torch.max(pred[i,j,10:],0)
                    if float((contain_prob*max_prob)[0]) > 0.1:
                        boxes.append(box_xy.view(1,4))
                        cls_indexs.append(cls_index)
                        probs.append(contain_prob*max_prob)
    if len(boxes) ==0:
        boxes = torch.zeros((1,4))
        probs = torch.zeros(1)
        cls_indexs = torch.zeros(1)
    else:
        boxes = torch.cat(boxes,0) #(n,4)
        # print(probs)
        probs = torch.cat(probs,0) #(n,)
        # print(probs)
        # print(cls_indexs)
        cls_indexs = torch.stack(cls_indexs,0) #(n,)
        # print(cls_indexs)
    keep = nms(boxes,probs)
    return boxes[keep],cls_indexs[keep],probs[keep]

def nms(bboxes,scores,threshold=0.5):
    '''
    bboxes(tensor) [N,4]
    scores(tensor) [N,]
    '''
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    areas = (x2-x1) * (y2-y1)

    _,order = scores.sort(0,descending=True)
    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            i = order.item()
        else:
            i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2-xx1).clamp(min=0)
        h = (yy2-yy1).clamp(min=0)
        inter = w*h
        union = areas[i] + areas[order[1:]] - inter

        ovr = inter / union
        ids = (ovr<=threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids+1]
    return torch.LongTensor(keep)

def summ(pred,image_name,w,h):
    result = []
    pred = pred.cpu()
    boxes,cls_indexs,probs = categorize(pred)

    for i,box in enumerate(boxes):
        x1 = int(box[0]*w)
        x2 = int(box[2]*w)
        y1 = int(box[1]*h)
        y2 = int(box[3]*h)
        cls_index = cls_indexs[i]
        cls_index = int(cls_index) # convert LongTensor to int
        prob = probs[i]
        prob = float(prob)
        result.append([(x1,y1),(x2,y2),VOC_CLASSES[cls_index],image_name,prob])
    return result




def train(epoch,net,Datasetinstance,train_loader,optimizer,criterion,Evaluation):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    start_time=time.time()
    origin = Datasetinstance.train_info
    preds = defaultdict(list)
    
    for batch_idx, (inputs, targets, w, h, fname) in enumerate(tqdm(train_loader)):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        pred = net(inputs)
        # print(pred.shape)
        # print(targets.shape)
        loss = criterion(pred,targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        for i in range(pred.shape[0]):
            result = summ(pred[i,:,:,:],fname[i],w[i],h[i])
            for (x1,y1),(x2,y2),class_name,image_id,prob in result:
                preds[class_name].append([image_id,prob,x1,y1,x2,y2])

        # print('batch %s of total batch %s' % (batch_idx, len(train_loader)), 'Loss: %.3f ' % (train_loss/(batch_idx+1)))

    
    
    end_time=time.time()
    epoch_time=end_time-start_time
    aps = Evaluation(preds, origin, threshold=0.6).evaluate()
    mAP = np.mean(aps)
    data=[epoch,train_loss/(batch_idx+1),epoch_time/(batch_idx+1),mAP]
    print('trainloss:{},time_per:{},train_mAP:{}'.format(train_loss/(batch_idx+1),epoch_time/(batch_idx+1),mAP))
    return data

def test(epoch,net,Datasetinstance,test_loader,criterion,Evaluation):
    print('\nEpoch: %d' % epoch)
    net.eval()
    test_loss = 0

    start_time=time.time()
    origin = Datasetinstance.test_info
    preds = defaultdict(list)

    for batch_idx, (inputs, targets, w, h, fname) in enumerate(tqdm(test_loader)):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        pred = net(inputs)
        loss = criterion(pred,targets)

        test_loss += loss.item()
        for i in range(pred.shape[0]):
            result = summ(pred[i,:,:,:],fname[i],w[i],h[i])
            for (x1,y1),(x2,y2),class_name,image_id,prob in result:
                preds[class_name].append([image_id,prob,x1,y1,x2,y2])

        # print('batch %s of total batch %s' % (batch_idx, len(test_loader)), 'Loss: %.3f ' % (test_loss/(batch_idx+1)))

    end_time=time.time()
    epoch_time=end_time-start_time
    aps = Evaluation(preds, origin, threshold=0.6).evaluate()
    mAP = np.mean(aps)
    data=[epoch,test_loss/(batch_idx+1),epoch_time/(batch_idx+1),mAP]
    print('testloss:{},time_per:{},test_mAP:{}'.format(test_loss/(batch_idx+1),epoch_time/(batch_idx+1),mAP))

    return data





def run():
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    print('loading dataset ...')
    Datasetinstance = Yolodata(train_file_root = './data/train/VOCdevkit/VOC2007/JPEGImages/', train_listano = './data/voc2007train.txt', test_file_root = './data/test/VOCdevkit/VOC2007/JPEGImages/', test_listano = './data/voc2007test.txt' ,batchsize=BATCH_SIZE, snumber = GRID_NUM, bnumber = BBOX_NUM, cnumber = CLASS_NUM)
    train_loader, test_loader = Datasetinstance.getdata()

    print('the dataset has %d images for train' % (len(train_loader)))
    print('the batch_size is %d' % (BATCH_SIZE))

    print('loading network structure ...')
    net = resnet50()
    net = net.to(DEVICE)
    # print(net) 

    print('load pre-trined model')
    refer_net = models.resnet50(pretrained=True)
    refer_net_state_dict = refer_net.state_dict()
    net_state = net.state_dict()
    for k in refer_net_state_dict.keys():
        if k in net_state.keys() and not k.startswith('fc'):
            net_state[k] = refer_net_state_dict[k]
    net.load_state_dict(net_state)


    history=hl.History()
    canvas=hl.Canvas()
    savepath='./records/'+ NAME +'/checkpoints/'
    if RESUME:
        print('Loading snapshot..')
        assert os.path.isdir('records'), 'Error: no snapshot directory found!'
        snapshot = torch.load(savepath + 'best_check.plk')

        net.load_state_dict(snapshot['net'])
        best_loss = snapshot['loss']
        start_epoch = snapshot['epoch'] + 1
        print('Loading history..')
        assert os.path.isdir('history'), 'Error: no history directory found!'
        history.load('./history/'+ NAME +"_history.plk")
        history.summary()

    criterion = YoloLoss(GRID_NUM,BBOX_NUM, 5, 0.5)

    net.train()
    # different learning rate
    params=[]
    params_dict = dict(net.named_parameters())
    for key,value in params_dict.items():
        if key.startswith('features'):
            params += [{'params':[value],'lr':LEARN_RATE*1}]
        else:
            params += [{'params':[value],'lr':LEARN_RATE}]
    optimizer = torch.optim.SGD(params, lr=LEARN_RATE, momentum=0.9, weight_decay=5e-4)

    for epoch in range(start_epoch, start_epoch+EPOCH_NUM):
        learning_rate = LEARN_RATE
        if epoch == 30:
            learning_rate=0.001
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
        if epoch == 45:
            learning_rate=0.0001
        #optimizer = torch.optim.SGD(net.parameters(),lr=learning_rate,momentum=0.9,weight_decay=1e-4)
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
                
        print('\n\nStarting epoch %d / %d' % (epoch + 1, EPOCH_NUM))
        print('Learning Rate for this epoch: {}'.format(learning_rate))
        nd = train(epoch,net,Datasetinstance,train_loader,optimizer,criterion,Evaluation)
        # print(nd)
        ed = test(epoch,net,Datasetinstance,test_loader,criterion,Evaluation)
        # print(ed)
        history.log(epoch, train_loss=nd[1], train_time=nd[2], train_mAP=nd[3], test_loss=ed[1], test_time=ed[2], test_mAP=ed[3])
        history.progress()
        test_loss = ed[1]
        with canvas:
            canvas.draw_plot([history['train_loss'],history['test_loss']])
            canvas.draw_plot([history['train_mAP'],history['test_mAP']])

        if test_loss < best_loss:
            print('Updating best validation loss')
            best_loss = test_loss
            print('Saving..best_record')
            state = {
                'net': net.state_dict(),
                'loss': best_loss,
                'epoch': epoch,
            }
            if not os.path.isdir('./records'):
                os.mkdir('./records')
            if not os.path.isdir('./figure'):
                os.mkdir('./figure')
            if not os.path.isdir('./history'):
                os.mkdir('./history')

            filename = savepath + 'best_check.plk'
            if not os.path.isdir(savepath):
                os.makedirs(savepath)
            torch.save(state, filename)
            canvas.save('./figure/'+ NAME +"_progress.png")
            history.save('./history/'+ NAME +"_history.plk")


if __name__ == '__main__':
    run()

    