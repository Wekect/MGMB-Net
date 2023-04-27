import os
from os.path import realpath
import torch
from skimage import io
import numpy as np
from util.config import cfg as test_cfg
from data.test_dataset import TestDataset
from util import util
from models.networks import msmbnet
from models.normalize import RAIN
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from tqdm import tqdm
import cv2

from options.train_options import TrainOptions

def sebeltrans(img):
    result = img[:,0,:,:].unsqueeze(1).clone()
    for tempath in range(len(img[:,0,0,0])):
        imgtemp = img[tempath,:,:,:].detach().cpu().float().numpy()
        imgtemp = (np.transpose(imgtemp,(1,2,0))+1)/2*255.0
        x = cv2.sebel(imgtemp, cv2.CV_16S,1,0)
        y = cv2.sebel(imgtemp, cv2.CV_16S,0,1)
        Scale_absX = cv2.convertScaleAbs(x)
        Scale_absY = cv2.convertScaleAbs(y)
        imgsebel = cv2.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)[:,:,0]
        imgsebel = torch.from_numpy(imgsebel).unsqueeze(0).unsqueeze(0).cuda()
        result[tempath,:,:,] = imgsebel

    return result
def load_network(cfg):
    net = msmbnet(input_nc=cfg.input_nc, 
                output_nc=cfg.output_nc, 
                ngf=cfg.ngf, 
                norm_layer=RAIN, 
                use_dropout=not cfg.no_dropout)
    
    load_path = os.path.join(cfg.checkpoints_dir, cfg.name, 'net_G_last.pth')
    print(f'loading the model from {load_path}')
    state_dict = torch.load(load_path, map_location='cpu')
    util.copy_state_dict(net.state_dict(), state_dict)
    return net

def save_img(path, img):
    fold, name = os.path.split(path)
    if not os.path.exists(fold):
        os.makedirs(fold)
    io.imsave(path, img)
def evalresult(model, data, device):
    all_mse, all_psnr, sebel_mse, sebel_psnr, diff_mse, diff_psnr = [], [], [], [], [], []
    eval_path = os.path.join(opt.checkpoints_dir, opt.name, 'Eval.csv')
    eval_results_fstr = open(eval_path, 'w')
    for i in tqdm(range(len(data))):
        sample = data[i]
        comp = sample['comp'].unsqueeze().to(device)
        mask = sample['mask'].unsqueeze()[:,0:1,:,:].to(device)
        real = sample['real'].unsqueeze().to(device)
        real_mask = sample['real_mask'].unsqueeze().to(device)

        mask_diff = mask -real_mask
        mask_diff[mask_diff>0.25] = 1
        mask_diff[mask_diff<-0.25] =1

        pred = model.processImage(comp, mask)
        pre_sebel = sebeltrans(pred*mask_diff)[0,0,:].cpu().float().detach().numpy()
        real_sebel = sebeltrans(real*mask_diff)[0,0,:].cpu().float().detach().numpy()

        pred_rgb = util.tensor2im(pred)
        real_rgb = util.tensor2im(real)
        mse_single = mean_squared_error(pred_rgb, real_rgb)
        psnr_single = peak_signal_noise_ratio(pred_rgb, real_rgb, data_range=255)

        mse_sebel = mean_squared_error(pre_sebel, real_sebel)
        psnr_sebel = peak_signal_noise_ratio(pre_sebel, real_sebel,data_range=255)
        mse_diff = mean_squared_error(util.tensor2im(pred*mask_diff), util.tensor2im(real*mask_diff))
        psnr_diff = peak_signal_noise_ratio(util.tensor2im(pred*mask_diff), util.tensor2im(real*mask_diff),data_range=255)
        eval_results_fstr.writelines('%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n' % (sample['img_path'], mse_single, psnr_single, mse_sebel, psnr_sebel, mse_diff, psnr_diff))
        eval_results_fstr.flush()
        all_mse.append(mse_single)
        all_psnr.append(psnr_single)
        sebel_mse.append(mse_sebel)
        sebel_psnr.append(psnr_sebel)
        diff_mse.append(mse_diff)
        diff_psnr.append(psnr_diff)
    
    ave_mse = sum(all_mse)/len(all_mse)
    ave_psnr = sum(all_psnr)/len(all_psnr)
    ave_mse_sebel = sum(sebel_mse)/len(sebel_mse)
    ave_psnr_sbel = sum(sebel_psnr)/len(sebel_psnr)
    ave_mse_diff = sum(diff_mse)/len(diff_mse)
    ave_psnr_diff = sum(diff_psnr)/len(diff_psnr)

    eval_results_fstr.writelines('%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n' % (ave_mse, ave_psnr, ave_mse_sebel, ave_psnr_sbel, ave_mse_diff, ave_psnr_diff))

    eval_results_fstr.flush()
    eval_results_fstr.close()

    return ave_mse, ave_psnr, ave_mse_sebel, ave_psnr_sbel, ave_mse_diff, ave_psnr_diff



if __name__ == '__main__':
    image_paths, mask_paths, gt_paths, real_masks = [], [], [], []
    opt = TrainOptions().parse()
    testfile = os.path.join(opt.dataset_root, 'IHD_train.txt')
    with open(testfile,'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            name_parts = line.split('_')
            mask_path = line.replace('composite_images', 'masks')
            mask_path = mask_path.replace('.jpg','.png')
            real_maskpath = mask_path.replace('masks','real_masks')
            gt_path = line.replace('composite_images', 'real_images')
            gt_path = gt_path.replace('_'+name_parts[-2]+'_'+name_parts[-1], '.jpg')
            image_paths.append(os.path.join(opt.dataset_root, line))
            mask_paths.append(os.path.join(opt.dataset_root, mask_path))
            gt_paths.append(os.path.join(opt.dataset_root, gt_path))
            real_masks.append(os.path.join(opt.dataset_root, real_maskpath))    


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    testdata = TestDataset(foreground_paths=image_paths, mask_paths=mask_paths, background_paths=gt_paths, real_masks = real_masks, load_size=512)
    msmbnet = load_network(opt).to(device)
    msmbnet.eval()

    print('compute----------------------------------')
    ave_mse, ave_psnr, ave_mse_sebel, ave_psnr_sbel, ave_mse_diff, ave_psnr_diff = evalresult(msmbnet, testdata, device)
    print('MSE:%.3f, PSNR:%.3f, MSE_sebel:%.3f, PSNR_sebel:%.3f, MSE_bound:%.3f, PSNR_bound:%.3f' %( ave_mse, ave_psnr, ave_mse_sebel, ave_psnr_sbel, ave_mse_diff, ave_psnr_diff))
    
    for i in range(len(testdata)):
        sample = testdata[i]
        # inference
        comp = sample['comp'].unsqueeze(0).to(device)
        mask = sample['mask'].unsqueeze(0)[:,0:1,:,:].to(device)
        real = sample['real'].unsqueeze(0).to(device)
        img_path = sample['img_path']
        pred = msmbnet.processImage(comp, mask)
        # save
        pred_rgb = util.tensor2im(pred[0:1])
        comp_rgb = util.tensor2im(comp[:1])
        mask_rgb = util.tensor2im(mask[:1])
        real_rgb = util.tensor2im(real[:1])
        save_img(opt.checkpoints_dir + '/'+ img_path.split('.')[0] + '-results.png', np.hstack([comp_rgb, mask_rgb, pred_rgb, real_rgb]))

