import os
import torch
import torch.nn.parallel
from torch.utils.data import DataLoader
import torch.optim
import numpy as np
import models
import dataset as ds
import configs.configs as config
from utils.factory import CosineAnnealingLR
import torch.nn.functional as F

def load_my_state_dict(model, state_dict):
    own_state = model.state_dict()
    ckpt_name = []
    cnt = 0
    for name, param in state_dict.items():
        if name not in list(own_state.keys()) or 'output_conv' in name:
                ckpt_name.append(name)
                continue
        own_state[name].copy_(param)
        cnt += 1
    print('#reused param: {}'.format(cnt))

    return model

def get_miou(pred, label, num_classes=config.DATA['num_class'] + 1):
    pred = pred.cpu().data.numpy().flatten()
    label = label.cpu().data.numpy().flatten()

    hist = np.bincount(num_classes * label.astype(int) + pred, minlength=num_classes ** 2)
    hist = hist.reshape(num_classes, num_classes)

    miou = np.diag(hist) / (np.sum(hist, axis=1) + np.sum(hist, axis=0) - np.diag(hist))
    miou = np.nanmean(miou)

    return miou

def valid(net, data_loader, optimizer):
    net.eval()

    avg_loss_total = 0
    avg_loss_att = 0
    avg_loss_seg = 0

    avg_iou = 0

    weights = [1.0 for _ in range(5)]
    weights[0] = 0.4
    class_weights = torch.FloatTensor(weights).cuda()

    CEL = torch.nn.CrossEntropyLoss(weight=class_weights)
    MSEL = torch.nn.MSELoss()


    for b_idx, (_, input, seg_target_2D, seg_target_3D, _) in enumerate(data_loader):
        img, seg_label_2D, seg_label_3D = input.cuda(), seg_target_2D.cuda(), seg_target_3D.cuda()
        seg_out, att_map, _ = net(img)

        seg_out_3D = (F.softmax(seg_out, dim=1))[:, 1:, :, :]
        att_out = seg_out_3D * att_map
        att_target = seg_label_3D * att_map

        loss_seg = CEL(seg_out, seg_label_2D)
        loss_att = MSEL(att_out, att_target) + torch.abs(torch.mean(att_target) - 0.8 * torch.mean(seg_label_3D))
        loss_att = loss_att * 50

        seg_out_2D = torch.argmax(seg_out, dim=1)

        iou = get_miou(seg_out_2D, seg_label_2D)

        loss = loss_seg + avg_loss_att

        avg_loss_total += loss.item()
        avg_loss_att += loss_att.item()
        avg_loss_seg += loss_seg.item()

        avg_iou += iou

    avg_loss_total = avg_loss_total / (b_idx + 1)
    avg_loss_att = avg_loss_att / (b_idx + 1)
    avg_loss_seg = avg_loss_seg / (b_idx + 1)

    avg_iou = avg_iou / (b_idx + 1)
    
    print(('Validation: lr: {lr:.5f}\t'  'Loss total {loss_t:.4f}\t' 'Loss att {loss_att:.4f}\t' 'Loss seg {loss_seg:.4f}\t'\
        'mIOU {avg_iou:.4f}\t'\
                .format(loss_t=avg_loss_total, loss_att=avg_loss_att, loss_seg=avg_loss_seg, \
        avg_iou=avg_iou, lr=optimizer.param_groups[-1]['lr'])))

    return avg_iou

def train(net, data_loader, optimizer, scheduler, epoch):
    net.train()

    avg_loss_total = 0
    avg_loss_att = 0
    avg_loss_seg = 0

    avg_iou = 0

    weights = [1.0 for _ in range(5)]
    weights[0] = 0.4
    class_weights = torch.FloatTensor(weights).cuda()

    CEL = torch.nn.CrossEntropyLoss(weight=class_weights)
    BCEL = torch.nn.BCELoss()
    MSEL = torch.nn.MSELoss()

    print_freq = config.TRAIN['print_freq']

    for b_idx, (input, seg_target_2D, seg_target_3D, target_exist) in enumerate(data_loader):
        global_step = epoch * len(data_loader) + b_idx

        img, seg_label_2D, seg_label_3D, lane_label = input.cuda(), seg_target_2D.cuda(), seg_target_3D.cuda(), target_exist.float().cuda()
        seg_out, att_map, lane_out = net(img)

        seg_out_3D = (F.softmax(seg_out, dim=1))[:, 1:, :, :]
        att_out = seg_out_3D * att_map
        att_target = seg_label_3D * att_map

        loss_seg = CEL(seg_out, seg_label_2D)
        loss_att = MSEL(att_out, att_target) + torch.abs(torch.mean(att_target) - 0.8 * torch.mean(seg_label_3D))
        loss_att = loss_att * 50
        loss_exist = BCEL(lane_out, lane_label)

        seg_out_2D = torch.argmax(seg_out, dim=1)

        iou = get_miou(seg_out_2D, seg_label_2D)

        loss = loss_seg + loss_att + loss_exist * 0.1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(global_step)

        avg_loss_total += loss.item()
        avg_loss_att += loss_att.item()
        avg_loss_seg += loss_seg.item()

        avg_iou += iou

        if (b_idx + 1) % print_freq == 0:
            avg_loss_total = avg_loss_total / print_freq
            avg_loss_att = avg_loss_att / print_freq
            avg_loss_seg = avg_loss_seg / print_freq

            avg_iou = avg_iou / print_freq
            
            print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t' 'Loss total {loss_t:.4f}\t' 'Loss att {loss_att:.4f}\t' 'Loss seg {loss_seg:.4f}\t'\
                'mIOU {avg_iou:.4f}\t'\
                .format(epoch, b_idx, len(train_loader), loss_t=avg_loss_total, loss_att=avg_loss_att, loss_seg=avg_loss_seg, \
                avg_iou=avg_iou, lr=optimizer.param_groups[-1]['lr'])))

            avg_loss_total = 0
            avg_loss_att = 0
            avg_loss_seg = 0
            avg_iou = 0
        
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = config.TRAIN['device']
    torch.backends.cudnn.benchmark = True

    train_dataset = ds.MY_VOCAugDataSet(dataset_path=config.DATA['data_root'], data_list='train_gt', transform=True, mode='train')
    val_dataset = ds.MY_VOCAugDataSet(dataset_path=config.DATA['data_root'], data_list='test_gt', transform=True, mode='test')

    train_loader = DataLoader(train_dataset, batch_size=config.TRAIN['train_batch_size'], shuffle=True, num_workers=config.TRAIN['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config.TRAIN['valid_batch_size'], shuffle=True, num_workers=config.TRAIN['num_workers'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = models.mymodel(num_classes=config.DATA['num_class'] + 1)

    checkpoint = torch.load('./pretrained/ERFNet_pretrained.tar')
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    net.to(device)

    net = load_my_state_dict(net, checkpoint['state_dict'])

    training_params = filter(lambda p: p.requires_grad, net.parameters())
    optimizer = torch.optim.SGD(training_params, lr=0.1, momentum=0.9, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=150000 , eta_min=0.001, warmup=None, warmup_iters=None)

    best = 0

    for epoch in range(config.TRAIN['epoch']):
        train(net, train_loader, optimizer, scheduler, epoch)
        miou = valid(net, val_loader, optimizer)

        if miou > best:
            model_state_dict = net.state_dict()

            if not os.path.exists('./trained'):
                os.makedirs('./trained')

            state = {'model': model_state_dict}
            model_path = os.path.join('./trained', 'ep%03d.pth' % epoch)
            torch.save(state, model_path)

            best = miou