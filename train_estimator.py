from mymodel import UNet_Att_Cascade6_Cube128_Regression
from myutils import file_exist, join_path, ls, gn, load_pkl, save_pkl
from torch.utils import data

dataloader_numworkers = 12
model_in_channels, model_out_channels = 1, 1
initial_lr = 0.0002
num_epochs = 1000
only_run_inference = False
generated_data_paths = [
    "/data7/chenghaoliu/Codes/Tumor_WMH_Analysis/digicare/experiments/022_ki67_real_tumor/generated_data/",
]
model_output_path = './trained_models/v7_att_negsamp/'
data_samples = {}

import argparse
parser = argparse.ArgumentParser(
    description='Run ablation study for tumor diagnosis.')
parser.add_argument(
    '--run-on-which-gpu', '-g', type=int, 
    help='GPU ID (starts with 0).', 
    required=False, default=0)
args = vars(parser.parse_args())

# step 1 : collect data samples from directory
def collect_data_samples(root_dir):
    all_cases = set()
    for folder in ls(root_dir, full_path=True):
        case_name = gn(folder)
        if case_name not in all_cases:
            all_cases.add(case_name)
    print('found %d unique cases.' % len(all_cases))
    all_cases = sorted(all_cases)
    for case in all_cases:
        expand_ratio = case[-4:]
        case_name = case[:-5]
        print(case_name, expand_ratio)
        # if expand_ratio == '1.00':
        #     print('skip identity')
        #     continue
        input_file = join_path(root_dir, case, case_name + '.nii.gz')
        label_file = join_path(root_dir, case, case_name + '_label.nii.gz')
        output_file = join_path(root_dir, case, case_name + '_displacement.nii.gz')
        if file_exist(input_file) and file_exist(label_file) and file_exist(output_file):
            data_samples[case] = {
                'input': input_file,
                'label': label_file,
                'output': output_file,
            }
    return data_samples
data_samples = {}
for generated_data_path in generated_data_paths:
    data_samples.update(collect_data_samples(generated_data_path))
print('collected data:')
print(data_samples)

import random
def random_split_cases(data_samples, train_ratio = 0.5):
    all_cases = [case for case in data_samples]
    num_train_cases = int(len(all_cases) * train_ratio)
    random.seed(6957)
    random.shuffle(all_cases)
    train_cases = all_cases[:num_train_cases]
    val_cases = all_cases[num_train_cases:]
    return train_cases, val_cases
def split_data_sample(data_samples, train_cases, val_cases):
    train_data_samples, val_data_samples = {}, {}
    for case in train_cases:
        train_data_samples[case] = data_samples[case]
    for case in val_cases:
        val_data_samples[case] = data_samples[case]
    return train_data_samples, val_data_samples

if not file_exist('train_cases.pkl') or not file_exist('val_cases.pkl'):
    train_cases, val_cases = random_split_cases(data_samples)
    save_pkl(train_cases, 'train_cases.pkl')
    save_pkl(val_cases, 'val_cases.pkl')
else:
    train_cases = load_pkl('train_cases.pkl')
    val_cases = load_pkl('val_cases.pkl')

train_data_samples, val_data_samples = split_data_sample(data_samples, train_cases, val_cases)
print('train:', [case for case in train_data_samples])
print('val:', [case for case in val_data_samples])

from myutils import get_nifti_pixdim, load_nifti, load_nifti_simple, save_nifti, save_nifti_simple, try_load_nifti, barycentric_coordinate, center_crop, z_score, masked_mean, masked_std
import numpy as np
from scipy.ndimage import zoom as zoom_image

# step 2 : build dataloader
def rolldice(prob):
    return np.random.rand() < prob
def process_data_sample_to_model_inputs(data_sample, outres='1.5mm'):
    case_name, input_image, input_label, output_displacement = data_sample
    input_data = load_nifti_simple(input_image)
    input_label = load_nifti_simple(input_label)
    output_disp_data = load_nifti_simple(output_displacement)
    if outres == '1.5mm':
        cx, cy, cz = [int(value) for value in barycentric_coordinate(input_data)]
        input_data = center_crop(input_data, [cx, cy, cz], [192, 192, 192], default_fill=np.min(input_data))
        input_label = center_crop(input_label, [cx, cy, cz], [192, 192, 192], default_fill=np.min(input_label))
        output_disp_data = center_crop(output_disp_data, [cx, cy, cz], [192, 192, 192], default_fill=0.0)
        input_data = zoom_image(input_data, 2/3, order=1)
        input_label = zoom_image(input_label, 2/3, order=1)
        output_disp_data = zoom_image(output_disp_data, 2/3, order=1)
    elif outres == '2mm':
        input_data = zoom_image(input_data, 0.5, order=1)
        input_label = zoom_image(input_label, 0.5, order=1)
        output_disp_data = zoom_image(output_disp_data, 0.5, order=1)
    else:
        raise RuntimeError('invalid out res setting.')
    assert input_data.shape[0] == 128 and input_data.shape == output_disp_data.shape == input_label.shape
    # apply data augmentation
    # mirroring
    r_mirror = np.random.rand()
    if r_mirror < 0.2:
        input_data = input_data[::,::,::-1]
        input_label = input_label[::,::,::-1]
        output_disp_data = output_disp_data[::,::,::-1]
    elif r_mirror < 0.4:
        input_data = input_data[::,::-1,::]
        input_label = input_label[::,::-1,::]
        output_disp_data = output_disp_data[::,::-1,::]
    elif r_mirror < 0.6:
        input_data = input_data[::-1,::,::]
        input_label = input_label[::-1,::,::]
        output_disp_data = output_disp_data[::-1,::,::]
    elif r_mirror < 0.8:
        input_data = input_data[::-1,::-1,::-1]
        input_label = input_label[::-1,::-1,::-1]
        output_disp_data = output_disp_data[::-1,::-1,::-1]
    input_data = input_data.copy()
    input_label = input_label.copy()
    output_disp_data = output_disp_data.copy()
    # gamma transform
    input_data = (input_data - np.min(input_data))/(np.max(input_data) - np.min(input_data))
    assert np.min(input_data) >= 0.0 and np.max(input_data) <= 1.0
    gamma_value = np.random.rand() * 1.5 + 0.5 # 0.5~2.0
    input_data = np.power(input_data, gamma_value)
    # finally, z-score
    input_data = z_score(input_data, input_label > 0.5)
    assert -0.05 < masked_mean(input_data, input_label > 0.5) < +0.05, masked_mean(input_data, input_label > 0.5) 
    assert 0.95 < masked_std(input_data, input_label > 0.5) < 1.05, masked_std(input_data, input_label > 0.5)
    return case_name, input_data[None], input_label[None], output_disp_data[None]

def paste_image(image_src, image_dst, src_center_pos, dst_mask):
    assert image_src.shape == image_dst.shape
    image_src_translated = center_crop( image_src, src_center_pos, image_src.shape )
    new_image = np.where(dst_mask > 0.5, image_src_translated, image_dst)
    return new_image

class DataLoader(data.Dataset):
    def __init__(self, data_samples, outres='1.5mm'):
        super().__init__()
        self.data_samples = data_samples
        self.outres = outres
        self.index_to_case_name = {}
        for case_name in self.data_samples:
            case_index = len(self.index_to_case_name)
            self.index_to_case_name[case_index] = case_name
        print(self.index_to_case_name)

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, index):
        case_name = self.index_to_case_name[index]
        data_sample = self.data_samples[self.index_to_case_name[index]]

        case_name, input_data, input_label, output_disp_data = process_data_sample_to_model_inputs(
            (case_name, data_sample['input'], data_sample['label'], data_sample['output']), outres=self.outres)
                
        return case_name, input_data, output_disp_data

train_loader = data.DataLoader(
    DataLoader(train_data_samples, outres='2mm'), batch_size=1, shuffle=True, num_workers=dataloader_numworkers
)
val_loader = data.DataLoader(
    DataLoader(val_data_samples, outres='2mm'), batch_size=1, shuffle=True, num_workers=dataloader_numworkers
)

# step 3: train model
from torch.optim import Adam as AdamOptimizer
gpu = int(args['run_on_which_gpu'])
model = UNet_Att_Cascade6_Cube128_Regression(model_in_channels, model_out_channels).to('cuda:%d' % gpu)
optim = AdamOptimizer(model.parameters(), lr=initial_lr, betas=(0.9, 0.999))

from model_trainer import ModelTrainer_PyTorch
from myutils import Timer, minibar, mkdir, gd
import torch
import torch.nn as nn
class Trainer(ModelTrainer_PyTorch):
    def __init__(self, 
        output_folder: str, 
        gpu_index: int, 
        model, optim, 
        train_loader, val_loader):
        super().__init__(output_folder=output_folder, 
            gpu_index=gpu_index, model=model, optim=optim, lr_scheduler=None, 
            train_loader=train_loader, val_loader=val_loader, test_loader=None)        
    def _on_epoch(self, epoch: int, msg: str, data_loader, phase):
        self.model.train() if phase=='train' else self.model.eval()
        timer = Timer()
        losses = []
        for batch_idx, (sample) in enumerate(data_loader):
            case_name, input_data, output_disp_data = sample
            input_data, output_disp_data = input_data.to('cuda:%d' % gpu).float(), output_disp_data.to('cuda:%d' % gpu).float()
            loss_function = nn.MSELoss()
            if phase=='train':
                output_estimation = self.model(input_data)
                assert output_estimation.shape == output_disp_data.shape
                loss = loss_function(output_estimation, output_disp_data)
                optim.zero_grad()
                loss.backward()
                optim.step()
            else:
                with torch.no_grad():
                    output_estimation = self.model(input_data)
                    assert output_estimation.shape == output_disp_data.shape
                    loss = loss_function(output_estimation, output_disp_data)
                if epoch == 1 or epoch % 10 == 0:
                    # save outputs
                    out_dir = mkdir('./model_train_out_v7/')
                    save_nifti_simple(input_data[0,0].detach().cpu().numpy(), join_path(out_dir, '%s_in.nii.gz' % case_name)) 
                    save_nifti_simple(output_estimation[0,0].detach().cpu().numpy(), join_path(out_dir, '%s_out.nii.gz' % case_name)) 
                    save_nifti_simple(output_disp_data[0,0].detach().cpu().numpy(), join_path(out_dir, '%s_gt.nii.gz' % case_name)) 
            losses.append(loss.item())
            minibar(msg, batch_idx+1, len(data_loader), timer.elapsed(), last='%.4f' % loss.item())
        return np.mean(losses)

trainer = Trainer(model_output_path, gpu, model, optim, train_loader, val_loader)
trainer.train(num_epochs=(num_epochs if not only_run_inference else 0))
trainer.load_model(trainer.model_best)
