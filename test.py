import os
import time
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import cv2
import numpy as np
from models.mymodel import mymodel
import dataset as ds
import torch.nn.functional as F
from torch.utils.data import DataLoader
import configs.configs as config

best_mIoU = 0
print_freq = 1

name = 'ERFnet'

def save_result(val_loader, model):
    batch_time = AverageMeter()
    mIoU = 0

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (ori, input, target, img_name) in enumerate(val_loader):
        input_var = torch.autograd.Variable(input, volatile=True)

        # compute output
        output, _, _ = model(input_var.cuda())
        output = F.upsample(output, size=(288, 800), mode='bilinear')
        output = F.softmax(output, dim=1)

        pred = output.data.cpu().numpy() # BxCxHxW

        for cnt in range(len(img_name)):
            for num in range(config.DATA['num_class']):
                prob_map = (pred[cnt][num+1] * 255).astype(int)
                black = np.zeros((prob_map.shape[0], prob_map.shape[1]), np.uint8)

                if num == 0:
                    sum = np.dstack((prob_map, black))
                    blue_lane = np.dstack((sum, black)).astype('uint8')
                
                elif num == 1:
                    sum = np.dstack((black, prob_map))
                    green_lane = np.dstack((sum, black)).astype('uint8')

                elif num == 2:
                    sum = np.dstack((black, prob_map))
                    red_lane = np.dstack((black, sum)).astype('uint8')
                
                elif num == 3:
                    sum = np.dstack((black, prob_map))
                    yellow_lane = np.dstack((prob_map, sum)).astype('uint8')

            directory = 'result/' + name + img_name[cnt][:-10]
            if not os.path.exists(directory):
                os.makedirs(directory)

            re_input = ori[cnt]
            input_img = ((re_input).data.cpu().numpy())*255
            input_img = np.transpose(input_img, (1, 2, 0)).astype('uint8')
            input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)

            lane = cv2.add(green_lane, blue_lane)
            lane = cv2.add(lane, red_lane)
            lane = cv2.add(lane, yellow_lane)
            result_img = cv2.add(lane, input_img)

            cv2.imwrite('result/' + name + img_name[cnt], result_img)

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print(('Test: [{0}/{1}]\t' 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time)))

    print('finished, #test:{}'.format(i))

    return mIoU


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def update(self, val, n=1):
        if self.val is None:
            self.val = val
            self.sum = val * n
            self.count = n
            self.avg = self.sum / self.count
        else:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = config.TEST['device']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = mymodel(num_classes=5)

    model = torch.nn.DataParallel(model)
    model.to(device)

    checkpoint = torch.load(config.TEST['checkpoint'])
    torch.nn.Module.load_state_dict(model, checkpoint['model'])

    cudnn.benchmark = True
    cudnn.fastest = True

    # Data loading code
    batch_size = 16
    num_workers = 4

    test_dataset = ds.VOCAugDataSet(dataset_path=config.DATA['data_root'], data_list='test_gt', transform=True, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    ### evaluate ###
    save_result(test_loader, model)

if __name__ == '__main__':
    main()