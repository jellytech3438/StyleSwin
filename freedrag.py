import torch
import math
import numpy as np
from models.generator import Generator
import torch.nn.functional as functional
from torchvision.transforms import functional as ttfunc
import matplotlib.pyplot as plt
import cv2
import lpips
from einops import rearrange
import torch.nn.functional as FUNC
from dift_sd import SDFeaturizer


def linear(feature, p0, p1, d, axis=0):
    f0 = feature[..., p0[0], p0[1]]
    f1 = feature[..., p1[0], p1[1]]
    weight = abs(d[axis])
    f = (1 - weight) * f0 + weight * f1
    return f


def bilinear(feature, qi, d):
    y0, x0 = qi
    dy, dx = d
    d = (dx, dy)
    dx = 1 if dx >= 0 else -1
    dy = 1 if dy >= 0 else -1
    x1 = x0 + dx
    y1 = y0 + dy
    fx1 = linear(feature, (x0, y0), (x1, y0), d, axis=0)
    fx2 = linear(feature, (x0, y1), (x1, y1), d, axis=0)
    weight = abs(d[1])
    fx = (1 - weight) * fx1 + weight * fx2
    return fx


def motion_supervision(F0, F, pi, ti, r1=3, M=None, lambda_mask=20):
    # print("M F size:", F.transpose(-1, -2).size())
    # print("M F0 size:", F0.transpose(-1, -2).size())

    # layer1: [1,1024,512] -> [1,512,1024] -> [1,512,32,32] -> [1,512,256,256]
    # F = F.transpose(-1,-2).reshape(1,512,32,32)
    # F0 = F0.transpose(-1,-2).reshape(1,512,32,32)

    # layer2: [1,16384,256] -> [1,256,16384] -> [1,256,128,128] -> [1,256,256,256]
    # F = F.transpose(-1,-2).reshape(1,256,128,128)
    # F0 = F0.transpose(-1,-2).reshape(1,256,128,128)
    #
    # F = functional.interpolate(F, [256, 256], mode="bilinear")
    # F0 = functional.interpolate(F0, [256, 256], mode="bilinear")

    # layer3: [1,65536,128] -> [1,128,65536] -> [1,128,256,256]
    F = F.transpose(-1, -2).reshape(1, 128, 256, 256)
    F0 = F0.transpose(-1, -2).reshape(1, 128, 256, 256)

    dw, dh = ti[0] - pi[0], ti[1] - pi[1]
    norm = math.sqrt(dw**2 + dh**2)
    w = (max(0, pi[0] - r1), min(256, pi[0] + r1))
    h = (max(0, pi[1] - r1), min(256, pi[1] + r1))
    # d = di
    d = torch.tensor(
        (dw / norm, dh / norm),
        dtype=F.dtype, device=F.device,
    ).reshape(1, 1, 1, 2)
    grid_h, grid_w = torch.meshgrid(
        torch.tensor(range(h[0], h[1]), device=F.device),
        torch.tensor(range(w[0], w[1]), device=F.device),
        indexing='xy',
    )
    grid = torch.stack([grid_w, grid_h], dim=-1).unsqueeze(0)
    grid = (grid / 255 - 0.5) * 2
    grid_d = grid + 2 * d / 255

    sample = functional.grid_sample(
        F, grid, mode='bilinear', padding_mode='border',
        align_corners=True,
    )
    sample_d = functional.grid_sample(
        F, grid_d, mode='bilinear', padding_mode='border',
        align_corners=True,
    )

    loss = (sample_d - sample.detach()).abs().mean(1).sum()
    og_loss = loss.data.cpu().numpy()
    mask_loss = 0

    if M is not None:
        # check if there's mask( min = 0 and max = 1 => have mask / min = 0 and max = 0 => no mask )
        if M.min() == 0 and M.max() == 1:
            mask = torch.from_numpy(M).float().to(
                F.device).unsqueeze(0).unsqueeze(0)

            mask_loss = ((F - F0) * (1 - mask)).abs().mean() * lambda_mask
            loss += mask_loss

    return loss, og_loss, mask_loss


def check_handle_reach_target(handle_points,
                              target_points):
    # dist = (torch.cat(handle_points,dim=0) - torch.cat(target_points,dim=0)).norm(dim=-1)
    all_dist = list(map(lambda p, q: math.sqrt(
        (p[0]-q[0])**2 + (p[1]-q[1])**2), handle_points, target_points))
    dist = torch.tensor(all_dist)
    print("tensor distance: ", dist)
    return (dist < 2.0).all()


@torch.no_grad()
def point_tracking(F0, F, pi, p0, r2=12):
    # print("P F size:", F.transpose(-1, -2).size())
    # print("P F0 size:", F0.transpose(-1, -2).size())

    # layer1: [1,1024,512] -> [1,512,1024] -> [1,512,32,32] -> [1,512,256,256]
    # F = F.transpose(-1,-2).reshape(1,512,32,32)
    # F0 = F0.transpose(-1,-2).reshape(1,512,32,32)

    # layer2: [1,16384,256] -> [1,256,16384] -> [1,256,128,128] -> [1,256,256,256]
    # F = F.transpose(-1,-2).reshape(1,256,128,128)
    # F0 = F0.transpose(-1,-2).reshape(1,256,128,128)
    #
    # F = functional.interpolate(F, [256, 256], mode="bilinear")
    # F0 = functional.interpolate(F0, [256, 256], mode="bilinear")

    # layer3: [1,65536,128] -> [1,128,65536] -> [1,128,256,256]
    F = F.transpose(-1, -2).reshape(1, 128, 256, 256)
    F0 = F0.transpose(-1, -2).reshape(1, 128, 256, 256)

    x = (max(0, pi[0] - r2), min(256, pi[0] + r2))
    y = (max(0, pi[1] - r2), min(256, pi[1] + r2))
    base = F0[..., p0[1], p0[0]].reshape(1, -1, 1, 1)
    diff = (F[..., y[0]:y[1], x[0]:x[1]] - base).abs().mean(1)
    idx = diff.argmin()
    dy = int(idx / (x[1] - x[0]))
    dx = int(idx % (x[1] - x[0]))
    npi = (x[0] + dx, y[0] + dy)
    return npi


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


# Free Drag Functions

def get_position_for_feature(win_r, handle_size, full_size):
    k = torch.linspace(-(win_r*(handle_size/full_size)),
                       win_r*(handle_size/full_size), steps=win_r)
    # k = torch.linspace(-(win_r//2),win_r//2,steps= win_r)
    k1 = k.repeat(win_r, 1).transpose(1, 0).flatten(0).unsqueeze(0)
    k2 = k.repeat(1, win_r)
    return torch.cat((k1, k2), dim=0).transpose(1, 0)


def get_features_plus(feature, position):
    # feature: (1,C,H,W)
    # position: (N,2)
    # return: (N,C)
    print("feature: ", torch.tensor(feature).shape)
    d1 = feature.size(dim=1)
    d2 = feature.size(dim=2)
    if d2 != 128:
        d3 = d2 / 128
        feature = feature.reshape(1, int(d1 * d3), 128)
    # feature = torch.tensor([item.numpy() for item in feature])
    # feature = torch.tensor(feature, device='cpu').transpose(-1, -2)
    feature = feature.transpose(-1, -2)
    d1 = feature.size(dim=1)
    d2 = feature.size(dim=2)
    feature = feature.reshape(1, d1, int(
        math.sqrt(d2)), int(math.sqrt(d2)))
    feature = functional.interpolate(feature, [256, 256], mode="bilinear")
    device = feature.device

    y = position[:, 0]
    x = position[:, 1]

    x0 = x.long()
    x1 = x0+1
    y0 = y.long()
    y1 = y0+1

    wa = ((x1.float() - x) * (y1.float() - y)).to(device).unsqueeze(1).detach()
    wb = ((x1.float() - x) * (y - y0.float())).to(device).unsqueeze(1).detach()
    wc = ((x - x0.float()) * (y1.float() - y)).to(device).unsqueeze(1).detach()
    wd = ((x - x0.float()) * (y - y0.float())).to(device).unsqueeze(1).detach()

    Ia = feature[:, :, y0, x0].squeeze(0).transpose(1, 0)
    Ib = feature[:, :, y1, x0].squeeze(0).transpose(1, 0)
    Ic = feature[:, :, y0, x1].squeeze(0).transpose(1, 0)
    Id = feature[:, :, y1, x1].squeeze(0).transpose(1, 0)

    output = Ia * wa + Ib * wb + Ic * wc + Id * wd
    return output


def get_each_point(current, target_final, L, feature_map, max_distance, template_feature,
                   loss_initial, loss_end, position_local, threshold_l):
    d_max = max_distance
    d_remain = (current-target_final).pow(2).sum().pow(0.5)
    interval_number = 10  # for point localization
    intervals = torch.arange(
        0, 1+1/interval_number, 1/interval_number, device=current.device)[1:].unsqueeze(1)

    if loss_end < threshold_l:
        target_max = current + \
            min(d_max/(d_remain+1e-8), 1)*(target_final-current)
        candidate_points = (1-intervals)*current.unsqueeze(0) + \
            intervals*target_max.unsqueeze(0)
        candidate_points_repeat = candidate_points.repeat_interleave(
            position_local.shape[0], dim=0)
        position_local_repeat = position_local.repeat(intervals.shape[0], 1)

        candidate_points_local = candidate_points_repeat + position_local_repeat
        features_all = get_features_plus(feature_map, candidate_points_local)

        features_all = features_all.reshape((intervals.shape[0], -1))
        dif_location = abs(
            features_all-template_feature.flatten(0).unsqueeze(0)).mean(1)
        min_idx = torch.argmin(abs(dif_location-L))
        current_best = candidate_points[min_idx, :]
        return current_best

    elif loss_end < loss_initial:
        return current

    else:
        current = current - min(d_max/(d_remain+1e-8), 1) * \
            (target_final-current)  # rollback
        d_remain = (current-target_final).pow(2).sum().pow(0.5)
        # double the localization range
        target_max = current + \
            min(2*d_max/(d_remain+1e-8), 1)*(target_final-current)

        candidate_points = (1-intervals)*current.unsqueeze(0) + \
            intervals*target_max.unsqueeze(0)
        candidate_points_repeat = candidate_points.repeat_interleave(
            position_local.shape[0], dim=0)
        position_local_repeat = position_local.repeat(intervals.shape[0], 1)
        candidate_points_local = candidate_points_repeat + position_local_repeat
        features_all = get_features_plus(feature_map, candidate_points_local)
        features_all = features_all.reshape((intervals.shape[0], -1))
        dif_location = abs(
            features_all-template_feature.flatten(0).unsqueeze(0)).mean(1)
        min_idx = torch.argmin(dif_location)   # l=0 in this case
        current_best = candidate_points[min_idx, :]
        return current_best


def get_current_target(sign_points, current_target, target_point, L, feature_map, max_distance, template_feature,
                       loss_initial, loss_end, position_local, threshold_l):
    for k in range(target_point.shape[0]):
        # sign_points ==0 means the remains distance to target point is larger than the preset threshold
        if sign_points[k] == 0:
            current_target[k, :] = get_each_point(current_target[k, :], target_point[k, :],
                                                  L, feature_map, max_distance, template_feature[k], loss_initial[k], loss_end[k], position_local, threshold_l)
    return current_target


def update_signs(sign_point_pairs, current_point, target_point, loss_supervised, threshold_d, threshold_l):

    distance = (current_point-target_point).pow(2).sum(dim=1).pow(0.5)
    sign_point_pairs[distance < threshold_d] = 1
    sign_point_pairs[distance >= threshold_d] = 0
    sign_point_pairs[loss_supervised > threshold_l] = 0


def get_xishu(loss_k, a, b):
    xishu = xishu = 1/(1+(a*(loss_k-b)).exp())
    return xishu


class DragGAN():
    def __init__(self, device, layer_index=3):
        self.generator = Generator(256, 512, 8).to(device)
        requires_grad(self.generator, False)
        self._device = device
        self.layer_index = layer_index
        self.latent = None
        self.F0 = None
        self.optimizer = None
        self.p0 = None
        self.lambda_mask = 20
        self.input_image_size = 256
        self.og_loss_recod = []
        self.mask_loss_record = []

    def load_ckpt(self, path):
        print(f'loading checkpoint from {path}')
        temp = 0
        for chr in path:
            if chr.isdigit():
                temp = temp * 10 + int(chr)
        print("input size: ", temp)

        # set the image_width and image_height to 256 or 1024
        # the condition change if still unfinish
        if temp == 1024:
            print("change generator to 1024 weights")
            self.generator = Generator(
                1024, 512, 8, channel_multiplier=1).to(self._device)
            self.input_image_size = 1024
        elif temp == 512:
            print("change generator to 512 weights")
            self.generator = Generator(
                512, 512, 8, channel_multiplier=1).to(self._device)
            self.input_image_size = 512
        elif temp == 256:
            print("change generator to 256 weights")
            self.generator = Generator(
                256, 512, 8).to(self._device)
            self.input_image_size = 256

        ckpt = torch.load(path, map_location=self._device)
        self.generator.load_state_dict(ckpt["g_ema"], strict=False)
        print('loading checkpoint successed!')

    def to(self, device):
        if self._device != device:
            self.generator = self.generator.to(device)
            self._device = device

    def plot_loss(self):
        plt.plot(np.array(self.og_loss_recod), label='og loss')
        plt.plot(np.array(self.mask_loss_record), label='mask loss')

        plt.title("loss record")
        plt.xlabel("step")
        plt.legend(loc='upper right')

        plt.savefig("200lambda_mask_seed512.png")

    @torch.no_grad()
    def generate_image(self, seed):
        # z = torch.from_numpy(
        #     np.random.RandomState(seed).randn(1, 512).astype(np.float32)
        # ).to(self._device)
        gen = torch.Generator()
        gen = gen.manual_seed(seed)
        z = torch.randn(1, 512, generator=gen).to(self._device)
        print("z: ", z.size())
        image, self.latent, self.F0 = self.generator(
            z, return_latents=True, return_features=True
        )
        # v2
        # imaeg = image[0]
        # v1
        image, self.F0 = image[0], self.F0[self.layer_index*2+1].detach()
        # OG
        # image, self.F0 = image[0], self.F0[self.layer_index+1].detach()
        image = (1/(2*2.8)) * \
            image.detach().cpu().permute(1, 2, 0).numpy() + 0.5
        image = image.clip(0, 1).reshape(-1)
        self.optimizer = None
        return image

    @property
    def device(self):
        return self._device

    def __call__(self, *args, **kwargs):
        return self.generator(*args, **kwargs)

    def store_feature(self, steps, image, features):
        plt.figure(figsize=(20, 10))

        for i in range(3):
            for j in range(1, 4):
                d = features[2*i+j].size(dim=1)
                d2 = features[2*i+j].size(dim=2)
                feature = features[2*i+j].transpose(-1, -2).reshape(
                    1, d2, int(math.sqrt(d)), int(math.sqrt(d)))

                img = feature[0]
                img = torch.sum(img, 0)
                img = img / feature[0].shape[0]
                plt.subplot(2, 4, 2*i+j)
                plt.imshow(img.cpu().detach().numpy())

        plt.subplot(2, 4, 8)
        # already change in 'step' method
        # img = (1/(2*2.8)) * image[0].permute(1, 2, 0).cpu().numpy() + 0.5
        img = ttfunc.to_pil_image(image, mode="RGB")
        plt.savefig("./features/{}.png".format(steps))

    # overlay = [0, 1]
    # select_layer = [6, 7]
    # feature_channel = range(128)
    def step(self, points, mask, overlay=None, select_layer=6, feature_channel=4):
        if self.optimizer is None:
            len_pts = (len(points) // 2) * 2
            if len_pts == 0:
                print('Select at least one pair of points')
                return False, None
            self.trainable = self.latent[:, :self.layer_index *
                                         2, :].detach().requires_grad_(True)
            self.fixed = self.latent[:, self.layer_index *
                                     2:, :].detach().requires_grad_(False)
            # self.trainable = self.latent[:, :self.layer_index, :].detach().requires_grad_(True)
            # self.fixed = self.latent[:, self.layer_index:, :].detach().requires_grad_(False)

            self.optimizer = torch.optim.Adam(
                [self.trainable], lr=2e-3, eps=1e-08, weight_decay=0, amsgrad=False)
            points = points[:len_pts]
            self.p0 = points[::2]
            print("p0: ", self.p0)

        trainable_fixed = torch.cat([self.trainable, self.fixed], dim=1)
        image, _, store_features = self.generator(
            trainable_fixed, input_is_latent=True,
            return_features=True
        )

        # v2
        # fetures = store_features
        # v1
        features = store_features[self.layer_index*2+1]
        # OG
        # features = features[self.layer_index+1]

        store_image = (1/(2*2.8)) * \
            image[0].detach().cpu().permute(1, 2, 0).numpy() + 0.5
        image = store_image.clip(0, 1).reshape(-1)

        # l_expected: expected loss at the beginning of each sub-motion
        # d_max: max distance for each sub-motion (in the feature map)
        l_expected = 1
        d_max = 2
        win_r = 3
        sample_interval = 10

        # handle_size: the size of handled feature map
        # set to 256 for temporary
        full_size = 256
        handle_size = 256

        threshold_l = 0.5*l_expected
        threshold_d = handle_size/full_size

        aa = torch.log(torch.tensor(9.0, device=self.device))/(0.6*l_expected)
        bb = 0.2*l_expected

        handle_point = []
        target_point = []
        for i in range(len(self.p0)):
            handle_point.append(torch.tensor(
                points[2*i], device=self.device).float())
            target_point.append(torch.tensor(
                points[2*i + 1], device=self.device).float())

        handle_point = torch.stack(handle_point)
        target_point = torch.stack(target_point)
        handle_point = handle_point * (handle_size/full_size)
        target_point = target_point * (handle_size/full_size)

        position_local = get_position_for_feature(
            win_r, handle_size, full_size).to(self.device)

        point_pairs_number = target_point.shape[0]
        template_feature = []
        for k in range(point_pairs_number):
            template_feature.append(get_features_plus(
                features, handle_point[k, :]+position_local))

        Loss_l1 = torch.nn.L1Loss()

        if np.any(mask == 0):
            mask = torch.tensor(mask, dtype=torch.float32,
                                device=self.device).unsqueeze(0).unsqueeze(0)
            mask_resized = FUNC.interpolate(mask, size=(
                handle_size, handle_size), mode='bilinear')
            mask_resized = mask_resized.repeat(1, features.shape[1], 1, 1) > 0

        max_steps = 2000
        step_number = 0
        current_target = handle_point.clone().to(self.device)
        current_feature_map = features.detach()

        # determiner if the localization point is closest to target point
        sign_points = torch.zeros(point_pairs_number).to(self.device)
        loss_ini = torch.zeros(point_pairs_number).to(self.device)
        loss_end = torch.zeros(point_pairs_number).to(self.device)
        step_threshold = max_steps

        while step_number < max_steps:
            if torch.all(sign_points == 1):
                img_show, _, _ = self.generator(
                    trainable_fixed, input_is_latent=True)
                yield img_show, current_target*(full_size/handle_size), step_number, full_size, trainable_fixed
                break

            current_target = get_current_target(sign_points, current_target, target_point, l_expected, current_feature_map,
                                                d_max, template_feature, loss_ini, loss_end, position_local, threshold_l)
            d_remain = (current_target-target_point).pow(2).sum(dim=1).pow(0.5)

            for step in range(5):
                step_number += 1

                # size=input_image_size
                img_mid, _, feature_mid = self.generator(
                    trainable_fixed, input_is_latent=True, return_features=True)

                loss_supervised = torch.zeros(
                    point_pairs_number).to(self.device)
                current_feature = []
                print("ppn: ", point_pairs_number)
                for k in range(point_pairs_number):
                    # for feat in feature_mid[1::]:
                    temp = get_features_plus(
                        feature_mid[-1], current_target[k, :]+position_local)
                    current_feature.append(temp)
                    loss_supervised[k] = Loss_l1(
                        current_feature[k], template_feature[k].detach())

                loss_feature = loss_supervised.sum()

                if np.any(mask == 0):
                    loss_mask = Loss_l1(
                        feature_mid[~mask_resized], features[~mask_resized].detach())
                    loss = loss_feature + 10*loss_mask
                else:
                    loss = loss_feature
                print("loss: ", loss)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if step_number % sample_interval == 0:
                    img_show, _, _ = self.generator(
                        trainable_fixed, input_is_latent=True)
                    yield img_show, current_target*(full_size/handle_size), step_number, full_size, trainable_fixed

                if step == 0:
                    loss_ini = loss_supervised

                if loss_supervised.max() < 0.5*threshold_l:
                    break

                if step_number == max_steps or step_number > step_threshold+10:
                    img_show, _, _ = self.generator(
                        trainable_fixed, input_is_latent=True)
                    yield img_show, current_target*(full_size/handle_size), step_number, full_size, trainable_fixed
                    break

            with torch.no_grad():
                img_mid, _, feature_mid = self.generator(
                    trainable_fixed, input_is_latent=True, return_features=True)

                current_feature = []
                for k in range(point_pairs_number):
                    current_feature.append(get_features_plus(
                        feature_mid[-1], current_target[k, :]+position_local))
                    loss_end[k] = Loss_l1(
                        current_feature[k], template_feature[k].detach())
            if d_remain.max() < threshold_d:
                step_threshold = step_number
            update_signs(sign_points, current_target, target_point,
                         loss_end, threshold_d, 0.5*threshold_l)
            for k in range(point_pairs_number):
                if sign_points[k] == 1:
                    xishu = 1
                else:
                    xishu = get_xishu(loss_end[k].detach(), aa, bb)
                template_feature[k] = xishu*current_feature[k].detach() + \
                    (1-xishu)*template_feature[k]

            current_feature_map = feature_mid[-1].detach()

        status = True

        # if check_handle_reach_target(pi, ti):
        #     status = False

        # return bool means keep iterating or not
        return status, (points, image), store_features, store_image

    def preprocess_image(self, image):
        image = image * 255
        image = image.astype(np.uint8)
        for i in range(len(image), -1, -1):
            if i % 4 == 3:
                image = np.delete(image, i)
        print(image.shape)
        image = image.reshape((256, 256, 3))
        image = torch.from_numpy(image).float() / 127.5 - 1  # [-1, 1]
        image = rearrange(image, "h w c -> 1 c h w")
        image = image.to(self._device)
        return image

    def lpip_score(self, origin_image, raw_data):
        all_lpips = []

        source_image = self.preprocess_image(np.array(origin_image))
        dragged_image = self.preprocess_image(np.array(raw_data))

        # compute LPIP
        loss_fn_alex = lpips.LPIPS(net='alex').to(self._device)
        with torch.no_grad():
            source_image_224x224 = FUNC.interpolate(
                source_image, (224, 224), mode='bilinear')
            dragged_image_224x224 = FUNC.interpolate(
                dragged_image, (224, 224), mode='bilinear')
            cur_lpips = loss_fn_alex(
                source_image_224x224, dragged_image_224x224)
            all_lpips.append(cur_lpips.item())
        return np.mean(all_lpips)

    def mean_distance_score(self, origin_image, raw_data, points):
        all_dist = []
        handle_points = []
        target_points = []
        for idx, point in enumerate(points):
            # from now on, the point is in row,col coordinate
            cur_point = torch.tensor([point[1], point[0]])
            if idx % 2 == 0:
                handle_points.append(cur_point)
            else:
                target_points.append(cur_point)

        dift = SDFeaturizer('stabilityai/stable-diffusion-2-1')

        source_image_tensor = self.preprocess_image(np.array(origin_image))
        dragged_image_tensor = self.preprocess_image(np.array(raw_data))
        print(source_image_tensor.shape, dragged_image_tensor.shape)

        _, C, H, W = source_image_tensor.shape
        ft_source = dift.forward(source_image_tensor,
                                 prompt='',
                                 t=261,
                                 up_ft_index=1,
                                 ensemble_size=8)
        ft_source = FUNC.interpolate(ft_source, (H, W), mode='bilinear')

        ft_dragged = dift.forward(dragged_image_tensor,
                                  prompt='',
                                  t=261,
                                  up_ft_index=1,
                                  ensemble_size=8)
        ft_dragged = FUNC.interpolate(ft_dragged, (H, W), mode='bilinear')

        cos = torch.nn.CosineSimilarity(dim=1)
        for pt_idx in range(len(handle_points)):
            hp = handle_points[pt_idx]
            tp = target_points[pt_idx]

            num_channel = ft_source.size(1)
            src_vec = ft_source[0, :, hp[0], hp[1]].view(1, num_channel, 1, 1)
            cos_map = cos(src_vec, ft_dragged).cpu().numpy()[0]  # H, W
            max_rc = np.unravel_index(
                cos_map.argmax(), cos_map.shape)  # the matched row,col

            # calculate distance
            dist = (tp - torch.tensor(max_rc)).float().norm()
            all_dist.append(dist)

        return torch.tensor(all_dist).mean().item()
