import time
from options.train_options import TrainOptions
from data import CustomDataset
from models import create_model
from torch.utils.tensorboard import SummaryWriter
import os
from util import util
import numpy as np
import torch
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from tqdm import tqdm
import cv2

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
def calculateMean(vars):
    return sum(vars) / len(vars)

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

def evaluateModel(model, opt, test_dataset, epoch, iters=None):
    model.netG.eval()
    if iters is not None:
        eval_path = os.path.join(opt.checkpoints_dir, opt.name, 'Eval_%s_iter%d.csv' % (epoch, iters))  # define the website directory
    else:
        eval_path = os.path.join(opt.checkpoints_dir, opt.name, 'Eval_%s.csv' % (epoch))  # define the website directory
    eval_results_fstr = open(eval_path, 'w')
    eval_results = {'mask': [], 'mse': [], 'mse_sebel': [], 'psnr_sebel': [], 'mse_bound': [], 'psnr_bound': []}

    for i, data in tqdm(enumerate(test_dataset)):
        model.set_input(data)  # unpack data from data loader
        model.test()  # inference
        visuals = model.get_current_visuals()  # get image results

        output = visuals['attentioned']
        real = visuals['real']
        mask = visuals['mask']
        real_mask = visuals['real_mask0']

        mask_diff = mask -real_mask
        mask_diff[mask_diff>0.25] = 1
        mask_diff[mask_diff<-0.25] =1

        outputsebel = sebeltrans(output*mask_diff)
        realsebel = sebeltrans(real*mask_diff)


        for i_img in range(real.size(0)):
            gt, pred, ma, sebelout, sebelreal = real[i_img:i_img+1], output[i_img:i_img+1], mask_diff[i_img:i_img+1], outputsebel[i_img:i_img+1], realsebel[i_img:i_img+1]
            mse_score_op = mean_squared_error(util.tensor2im(pred), util.tensor2im(gt))
            psnr_score_op = peak_signal_noise_ratio(util.tensor2im(gt), util.tensor2im(pred), data_range=255)
            mse_sebel = mean_squared_error(sebelout[0,0,:].cpu().float().detach().numpy(), sebelreal[0,0,:].cpu().float().detach().numpy())
            psnr_sebel = peak_signal_noise_ratio(sebelout[0,0,:].cpu().float().detach().numpy(), sebelreal[0,0,:].cpu().float().detach().numpy(),data_range=255)
            mse_boundary = mean_squared_error(util.tensor2im(pred*ma), util.tensor2im(gt*ma))
            psnr_boundary = peak_signal_noise_ratio(util.tensor2im(pred*ma), util.tensor2im(gt*ma), data_range=255)
            # update calculator
            eval_results['mse'].append(mse_score_op)
            eval_results['psnr'].append(psnr_score_op)
            eval_results['mask'].append(data['mask'][i_img].mean().item())
            eval_results['mse_sebel'].append(mse_sebel)
            eval_results['psnr_sebel'].append(psnr_sebel)
            eval_results['mse_bound'].append(mse_boundary)
            eval_results['psnr_bound'].append(psnr_boundary)            
            eval_results_fstr.writelines('%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n' % (data['img_path'][i_img], eval_results['mask'][-1],mse_score_op, psnr_score_op,mse_sebel,psnr_sebel, mse_boundary,psnr_boundary))
        if i + 1 % 100 == 0:
            # print('%d images have been processed' % (i + 1))
            eval_results_fstr.flush()


    all_mse, all_psnr,all_mse_sebel, all_psnr_sebel, all_mse_bound, all_psnr_bound = calculateMean(eval_results['mse']), calculateMean(eval_results['psnr']), calculateMean(eval_results['mse_sebel']), calculateMean(eval_results['psnr_sebel']), calculateMean(eval_results['mse_bound']), calculateMean(eval_results['psnr_bound'])
    print('MSE:%.3f, PSNR:%.3f, MSE_sebel:%.3f, PSNR_sebel:%.3f, MSE_bound:%.3f, PSNR_bound:%.3f' % (all_mse, all_psnr,all_mse_sebel, all_psnr_sebel, all_mse_bound, all_psnr_bound))
    eval_results_fstr.flush()
    eval_results_fstr.close()
    model.netG.train()
    return all_mse, all_psnr,all_mse_sebel, all_psnr_sebel, all_mse_bound, all_psnr_bound, resolveResults(eval_results)

def resolveResults(results):
    interval_metrics = {}
    mask, mse, psnr = np.array(results['mask']), np.array(results['mse']), np.array(results['psnr'])
    interval_metrics['0.00-0.05'] = [np.mean(mse[np.logical_and(mask <= 0.05, mask > 0.0)]),
                                     np.mean(psnr[np.logical_and(mask <= 0.05, mask > 0.0)])]
    interval_metrics['0.05-0.15'] = [np.mean(mse[np.logical_and(mask <= 0.15, mask > 0.05)]),
                                     np.mean(psnr[np.logical_and(mask <= 0.15, mask > 0.05)])]
    interval_metrics['0.15-0.25'] = [np.mean(mse[np.logical_and(mask <= 0.25, mask > 0.15)]),
                                     np.mean(psnr[np.logical_and(mask <= 0.25, mask > 0.15)])]
    interval_metrics['0.25-0.50'] = [np.mean(mse[np.logical_and(mask <= 0.5, mask > 0.25)]),
                                     np.mean(psnr[np.logical_and(mask <= 0.5, mask > 0.25)])]
    interval_metrics['0.50-1.00'] = [np.mean(mse[mask > 0.5]), np.mean(psnr[mask > 0.5])]
    return interval_metrics

def updateWriterInterval(writer, metrics, epoch):
    for k, v in metrics.items():
        writer.add_scalar('interval/{}-MSE'.format(k), v[0], epoch)
        writer.add_scalar('interval/{}-PSNR'.format(k), v[1], epoch)

if __name__ == '__main__':
    # setup_seed(6)
    opt = TrainOptions().parse()   # get training 
    train_dataset = CustomDataset(opt, is_for_train=True)
    test_dataset = CustomDataset(opt, is_for_train=False)
    train_dataset_size = len(train_dataset)    # get the number of images in the dataset.
    test_dataset_size = len(test_dataset)
    print('The number of training images = %d' % train_dataset_size)
    print('The number of testing images = %d' % test_dataset_size)
    
    train_dataloader = train_dataset.load_data()
    test_dataloader = test_dataset.load_data()
    print('The total batches of training images = %d' % len(train_dataset.dataloader))
    print(len(train_dataloader))

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    total_iters = 0                # the total number of training iterations
    writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name))

    loss_path = os.path.join(opt.checkpoints_dir, opt.name, 'loss.csv')
    loss_res = open(loss_path, 'w')
    mess_path = os.path.join(opt.checkpoints_dir, opt.name, 'mes_psnr.csv')
    mess_res = open(mess_path, 'w')

    for epoch in range(opt.load_iter+1, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in tqdm(enumerate(train_dataloader)):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_iters += 1
            epoch_iter += 1
            model.set_input(data)         # unpack data from dataset
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                
            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        # evaluate for every epoch
        losses = model.get_current_losses()
        loss_res.writelines('%s, %s\n'%(epoch, losses))
        loss_res.flush()
        epoch_mse, epoch_psnr,  epoch_mse_sebel, epoch_psnr_sebel, epoch_mse_boundary, epoch_psnr_boundary,epoch_interval_metrics = evaluateModel(model, opt, test_dataloader, epoch)
        writer.add_scalar('overall/MSE', epoch_mse, epoch)
        writer.add_scalar('overall/PSNR', epoch_psnr, epoch)
        writer.add_scalar('overall/MSE_sebel', epoch_mse_sebel, epoch)
        writer.add_scalar('overall/PSNR_sebel', epoch_psnr_sebel, epoch)
        writer.add_scalar('overall/MSE_boundary', epoch_mse_boundary, epoch)
        writer.add_scalar('overall/PSNR_boundary', epoch_psnr_boundary, epoch)
        updateWriterInterval(writer, epoch_interval_metrics, epoch)

        mess_res.writelines('%s, MSE:%.3f, PSNR:%.3f, MSE_sebel:%.3f, OSNR_sebel:%.3f, MSE_bound:%.3f, PSNR_bound:%.3f\n'%(epoch_mse, epoch_psnr,  epoch_mse_sebel, epoch_psnr_sebel, epoch_mse_boundary, epoch_psnr_boundary))
        mess_res.flush()
        torch.cuda.empty_cache()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks('%d' % epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
        print('Current learning rate: {}, {}'.format(model.schedulers[0].get_lr(), model.schedulers[1].get_lr()))

    writer.close()
    loss_res.close()
    mess_res.close()
