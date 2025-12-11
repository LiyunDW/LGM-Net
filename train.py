import torch, os, glob, cv2, random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser
from LGMNet import *
from utils import *
from skimage.metrics import structural_similarity as ssim
from time import time
from tqdm import tqdm

parser = ArgumentParser(description='LGM-net')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--end_epoch', type=int, default=200)
parser.add_argument('--Iter_D', type=int, default=4)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--block_size', type=int, default=32)
parser.add_argument('--model_dir', type=str, default='model')
parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--log_dir', type=str, default='log')
parser.add_argument('--save_interval', type=int, default=10)
parser.add_argument('--testset_name', type=str, default=["Set11"])
parser.add_argument('--gpu_list', type=str, default='0')
parser.add_argument('--dim', type=int, default=8)
parser.add_argument('--dimf', type=int, default=8)
parser.add_argument('--cs_ratio', type=float, default=0.1)
args = parser.parse_args()

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_list
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

start_epoch, end_epoch = args.start_epoch, args.end_epoch
learning_rate = args.learning_rate
Iter_D = args.Iter_D
Block = args.block_size
dim = args.dim
dimf = args.dimf
cs_ratio = args.cs_ratio

if cs_ratio in [0.01, 0.04, 0.1]:
    learning_rate, end_epoch, m = 1e-4, 200, [150, 180, 190]
else:
    learning_rate, end_epoch, m = 8e-5, 250, [200, 230, 240]

seed = 2023
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

batch_size, patch_size, iter_num = 64, 128, 2000
N = Block * Block
q = int(np.ceil(N * cs_ratio))

print('reading files...')
start_time = time()
training_image_paths = glob.glob(os.path.join(args.data_dir, "CoCo2017") + '/*')
training_image_num = len(training_image_paths)
print('training_image_num', training_image_num, 'read time', time() - start_time)

Init_Phi = torch.nn.init.xavier_normal_(torch.Tensor(q, N))
model = LGMNet(Iter_D, Block, Init_Phi, dim, dimf)
net_params = sum([p.numel() for p in model.parameters()]) - model.Phi_weight.numel()
print("total para num: %d" % net_params)
model = torch.nn.DataParallel(model).to(device)


class MyDataset(Dataset):
    def __init__(self):
        self.len = iter_num * batch_size
        self.real_len_1 = training_image_num - 1

    def __getitem__(self, index):
        while True:
            index = random.randint(0, self.real_len_1)
            path = training_image_paths[index]
            training_image_ycrcb = cv2.imread(path, 1)
            training_image_ycrcb = cv2.cvtColor(training_image_ycrcb, cv2.COLOR_BGR2YCrCb)
            training_image_y = training_image_ycrcb[:, :, 0]
            training_image_y_tensor = torch.Tensor(training_image_y) / 255.0
            h, w = training_image_y.shape
            max_h, max_w = h - patch_size, w - patch_size
            if max_h < 0 or max_w < 0:
                continue
            start_h = random.randint(0, max_h)
            start_w = random.randint(0, max_w)

            return training_image_y_tensor[start_h:start_h + patch_size, start_w:start_w + patch_size]

    def __len__(self):
        return self.len


my_loader = DataLoader(dataset=MyDataset(), batch_size=batch_size, num_workers=4, pin_memory=True)
optimizer = torch.optim.Adam([{'params': model.parameters(), 'initial_lr': learning_rate}], lr=learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=m, gamma=0.1, last_epoch=start_epoch - 1)
model_dir = r"model/ratio_%.2f_D_%d_dim_%d" % (cs_ratio, Iter_D, dim)
os.makedirs(model_dir, exist_ok=True)
log_path = './%s/ratio_%.2f_D_%d_dim_%d.txt' % (model_dir, cs_ratio, Iter_D, dim)


def test():
    with torch.no_grad():
        for ipath in args.testset_name:
            test_image_paths = glob.glob(os.path.join(args.data_dir, ipath) + '/*')
            test_image_num = len(test_image_paths)
            PSNR_list, SSIM_list = [], []
            for i in tqdm(range(test_image_num)):
                test_image = cv2.imread(test_image_paths[i], 1)
                test_image_ycrcb = cv2.cvtColor(test_image, cv2.COLOR_BGR2YCrCb)
                img, old_h, old_w, img_pad, new_h, new_w = my_zero_pad(test_image_ycrcb[:, :, 0])
                img_pad = img_pad.reshape(1, 1, new_h, new_w) / 255.0
                x_input = torch.from_numpy(img_pad)
                x_input = x_input.type(torch.FloatTensor).to(device)
                x_output = model(x_input)
                x_output = x_output.cpu().data.numpy().squeeze()
                x_output = np.clip(x_output[:old_h, :old_w], 0, 1).astype(np.float64) * 255.0
                PSNR = psnr(x_output, img)
                SSIM = ssim(x_output, img, data_range=255)
                PSNR_list.append(PSNR)
                SSIM_list.append(SSIM)
            log_data = 'CS Ratio is %.2f, %s: PSNR is %.2f, SSIM is %.4f.' % (
                cs_ratio, ipath, float(np.mean(PSNR_list)), float(np.mean(SSIM_list)))
            print(log_data)
            with open(log_path, 'a') as log_file:
                log_file.write(log_data + '\n')


if start_epoch > 0:
    model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (model_dir, start_epoch)))

print('start training...')
for epoch_i in range(start_epoch + 1, end_epoch + 1):
    start_time = time()
    lr = scheduler.optimizer.param_groups[0]['lr']
    loss_avg = 0.0
    for data in tqdm(my_loader, ncols=100, colour="cyan"):
        x = data.unsqueeze(1).to(device)
        x = H(x, random.randint(0, 7))
        loss = (model(x) - x).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_avg += loss.item()
    scheduler.step()
    loss_avg /= iter_num
    log_data = '[%d/%d] Average loss: %.f, lr: %.7f, time cost: %.2fs.' % (
        epoch_i, end_epoch, loss_avg, lr, time() - start_time)
    print(log_data)
    with open(log_path, 'a') as log_file:
        log_file.write(log_data + '\n')
    if epoch_i % args.save_interval == 0:
        torch.save(model.state_dict(), './%s/net_params_%d.pkl' % (model_dir, epoch_i))
    if epoch_i % 5 == 0:
        test()
