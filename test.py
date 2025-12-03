import os, glob, cv2, argparse, random, warnings
from time import time
from skimage.metrics import structural_similarity as ssim
from utils import *
from LGMNet import *

warnings.filterwarnings("ignore")


def main():
    global args
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    seed = 2023
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    Block = 32
    Iter_D = args.Iter_D
    dim = args.dim
    cs_ratio = args.cs_ratio
    N = Block * Block
    flag = args.save_flag
    Init_Phi = torch.nn.init.xavier_normal_(torch.zeros(int(np.ceil(cs_ratio * N)), N))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.enabled = True
    model = LGMNet(Iter_D, Block, Init_Phi, dim, dim)
    num_params = sum([p.numel() for p in model.parameters()]) - model.Phi_weight.numel()
    print("total para num: %d" % num_params)

    model = nn.DataParallel(model).to(device)
    model_dir = r"model/ratio_%.2f_D_%d_dim_%d" % (cs_ratio, Iter_D, dim)
    pth = torch.load("%s/net_params_best.pkl" % (model_dir), map_location=device)
    model.load_state_dict(pth)
    with torch.no_grad():
        for ipath in args.test_name:
            test_image_paths = glob.glob(os.path.join(args.data, ipath) + '/*')
            test_image_num = len(test_image_paths)
            PSNR_list, SSIM_list, TIME_list = [], [], []
            for i in range(test_image_num):
                test_image = cv2.imread(test_image_paths[i], 1)
                test_image_ycrcb = cv2.cvtColor(test_image, cv2.COLOR_BGR2YCrCb)
                img, old_h, old_w, img_pad, new_h, new_w = my_zero_pad(test_image_ycrcb[:, :, 0])
                img_pad = img_pad.reshape(1, 1, new_h, new_w) / 255.0
                x_input = torch.from_numpy(img_pad)
                x_input = x_input.type(torch.FloatTensor).to(device)
                start = time()
                x_output = model(x_input)
                end = time() - start
                x_output = x_output.cpu().data.numpy().squeeze()
                x_output = np.clip(x_output[:old_h, :old_w], 0, 1).astype(np.float64) * 255.0
                PSNR = psnr(x_output, img)
                SSIM = ssim(x_output, img, data_range=255)
                PSNR_list.append(PSNR)
                SSIM_list.append(SSIM)
                TIME_list.append(end)
                name = os.path.split(test_image_paths[i])[-1].split(".")[0]
                print(f"[{i + 1:02d}/{test_image_num:02d}] "
                      f"Run time for {name}: {end:.4f}, "
                      f"PSNR: {PSNR:.2f}, SSIM: {SSIM:.4f}")
                if flag:
                    test_image_ycrcb[:, :, 0] = x_output
                    im_rec_rgb = cv2.cvtColor(test_image_ycrcb, cv2.COLOR_YCrCb2BGR)
                    im_rec_rgb = np.clip(im_rec_rgb, 0, 255).astype(np.uint8)
                    save_path = os.path.join(args.result_dir, ipath, str(args.cs_ratio))
                    os.makedirs(save_path, exist_ok=True)
                    cv2.imwrite(
                        "%s/%s/%s/%s_PSNR_%.2f_SSIM_%.4f.png" % (
                            args.result_dir, ipath, str(args.cs_ratio), name, PSNR, SSIM), im_rec_rgb)
                    del x_output
            log_data = 'CS Ratio: %.2f, %s: PSNR: %.2f, SSIM: %.4f,  TIME: %.3f.\n' % (
                cs_ratio, ipath, float(np.mean(PSNR_list)), float(np.mean(SSIM_list)), float(np.mean(TIME_list)))
            print(log_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='LGM-Net', help="model name")
    parser.add_argument('--data', type=str, default='data', help="data path")
    parser.add_argument('--cs_ratio', type=float, default=0.1, help="cs ratio: {0.01, 0.04, 0.1, 0.25, 0.5}")
    parser.add_argument('--Iter_D', type=int, default=4, help="Iter number")
    parser.add_argument('--dim', type=int, default=8, help="dimension")
    parser.add_argument('--test_name', type=str, default=["Set11"], help="testset list: [ , , ,]")
    parser.add_argument('--result_dir', type=str, default='result', help="result path for test")
    parser.add_argument('--save_flag', type=bool, default=True, help="save flag when test")
    main()
