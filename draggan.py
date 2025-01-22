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
from PIL import Image
from torchvision.transforms import PILToTensor


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


def motion_supervision(F0, F, pi, ti, layer_index, r1=3, M=None, lambda_mask=20):
    # print("M F size:", F.transpose(-1, -2).size())
    # print("M F0 size:", F0.transpose(-1, -2).size())

    if layer_index == 3:
        # layer1: [1,1024,512] -> [1,512,1024] -> [1,512,32,32] -> [1,512,256,256]
        F = F.transpose(-1,-2).reshape(1,512,32,32)
        F0 = F0.transpose(-1,-2).reshape(1,512,32,32)
        F = functional.interpolate(F, [256, 256], mode="bilinear")
        F0 = functional.interpolate(F0, [256, 256], mode="bilinear")
    elif layer_index == 4:
        # layer 4: [1,4096,512] -> [1,512,4096] -> [1,512,64,64] -> [1,512,256,256]
        F = F.transpose(-1,-2).reshape(1,512,64,64)
        F0 = F0.transpose(-1,-2).reshape(1,512,64,64)
        
        F = functional.interpolate(F, [256, 256], mode="bilinear")
        F0 = functional.interpolate(F0, [256, 256], mode="bilinear")
    elif layer_index == 5:
        # layer2: [1,16384,256] -> [1,256,16384] -> [1,256,128,128] -> [1,256,256,256]
        F = F.transpose(-1,-2).reshape(1,256,128,128)
        F0 = F0.transpose(-1,-2).reshape(1,256,128,128)

        F = functional.interpolate(F, [256, 256], mode="bilinear")
        F0 = functional.interpolate(F0, [256, 256], mode="bilinear")
    elif layer_index >= 6:
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
def point_tracking(F0, F, pi, p0, layer_index, r2=12):
    # print("P F size:", F.transpose(-1, -2).size())
    # print("P F0 size:", F0.transpose(-1, -2).size())

    if layer_index == 3:
        # layer 3: [1,1024,512] -> [1,512,1024] -> [1,512,32,32] -> [1,512,256,256]
        F = F.transpose(-1, -2).reshape(1, 512, 32, 32)
        F0 = F0.transpose(-1, -2).reshape(1, 512, 32, 32)
        
        F = functional.interpolate(F, [256, 256], mode="bilinear")
        F0 = functional.interpolate(F0, [256, 256], mode="bilinear")
    elif layer_index == 4:
        # layer 4: [1, 4096, 512] -> [1, 512, 4096] -> [1, 512, 64, 64] -> [1, 512, 256, 256]
        F = F.transpose(-1, -2).reshape(1, 512, 64, 64)
        F0 = F0.transpose(-1, -2).reshape(1, 512, 64, 64)
        
        F = functional.interpolate(F, [256, 256], mode="bilinear")
        F0 = functional.interpolate(F0, [256, 256], mode="bilinear")
    elif layer_index == 5:
        # layer 5: [1, 16384, 256] -> [1, 256, 16384] -> [1, 256, 128, 128] -> [1, 256, 256, 256]
        F = F.transpose(-1, -2).reshape(1, 256, 128, 128)
        F0 = F0.transpose(-1, -2).reshape(1, 256, 128, 128)

        F = functional.interpolate(F, [256, 256], mode="bilinear")
        F0 = functional.interpolate(F0, [256, 256], mode="bilinear")
    elif layer_index >= 6:
        # layer 6 7: [1, 65536, 128] -> [1, 128, 65536] -> [1, 128, 256, 256]
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


class DragGAN():
    def __init__(self, device, train_fixed_layer=6, layer_index=7):
        # layer index = 3~7 becuase of pt and ms interpolation
        # train fixed index = 0~7
        self.generator = Generator(256, 512, 8).to(device)
        requires_grad(self.generator, False)
        self._device = device
        self.layer_index = layer_index
        self.train_fixed_index = train_fixed_layer
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
        plt.plot(np.array(self.mask_loss_record.detach().cpu().numpy()),
                 label='mask loss')

        plt.title("loss record")
        plt.xlabel("step")
        plt.legend(loc='upper right')

        plt.savefig("200lambda_mask_seed512.png")

    @torch.no_grad()
    def generate_image(self, seed):
        # np random
        # z = torch.from_numpy(
        #     np.random.RandomState(seed).randn(1, 512).astype(np.float32)
        # ).to(self._device)

        # torch random
        gen = torch.Generator()
        gen = gen.manual_seed(seed)
        z = torch.randn(1, 512, generator=gen).to(self._device)

        image, self.latent, self.F0, _ = self.generator(
            z, return_latents=True, return_features=True
        )
        image, self.F0 = image[0], self.F0[self.layer_index].detach()
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
        plt.imshow(img)
        plt.savefig("./features/{}.png".format(steps))

    # select_layer = [6, 7]
    # feature_channel = range(128)
    def step(self, points, mask, step, feature_channel=4, visiualize_attention=False):
        if self.optimizer is None:
            len_pts = (len(points) // 2) * 2
            if len_pts == 0:
                print('Select at least one pair of points')
                return False, None
            self.trainable = self.latent[:, :self.train_fixed_index, :].detach().requires_grad_(True)
            self.fixed = self.latent[:, self.train_fixed_index:, :].detach().requires_grad_(False)

            self.optimizer = torch.optim.Adam([self.trainable], lr=2e-3)
            points = points[:len_pts]
            self.p0 = points[::2]
            print("p0: ", self.p0)
            
        self.optimizer.zero_grad()
        trainable_fixed = torch.cat([self.trainable, self.fixed], dim=1)
        

        # init features and image for point tracking
        image, _, store_features, attentions = self.generator(
            trainable_fixed,  # this is why noise = [tensor[]]
            input_is_latent=True,
            return_features=True,
            return_attention=visiualize_attention,
            mask1=torch.tensor(1-mask, dtype=torch.float).to("cuda")
        )

        # v1 and OG
        features = store_features[self.layer_index]
        # v2
        # features = store_features
        
        new_points = points

        store_image = (1/(2*2.8)) * \
            image[0].detach().cpu().permute(1, 2, 0).numpy() + 0.5
        image = store_image.clip(0, 1).reshape(-1)

        # pt start
        for i in range(len(self.p0)):
            new_points[2*i] = point_tracking(self.F0,
                                             features, points[2*i], self.p0[i], self.layer_index)

        loss = 0
        # ms start
        for i in range(len(self.p0)):
            msloss, og_loss, mask_loss = motion_supervision((self.F0),
                                                            features, points[2*i], points[2*i+1], self.layer_index ,M=mask, lambda_mask=self.lambda_mask)
            loss += msloss
            self.og_loss_recod.append(og_loss)
            self.mask_loss_record.append(mask_loss)
            print("loss ratio: origin => ", og_loss,
                  ", mask => ", mask_loss)

        # visiualize attention
        # if true then this will save attention map under "attention" dir
        # else it will skip

        if visiualize_attention:
            nh = attentions[-1][0].shape[1]  # number of head
            wh = attentions[-1][0].shape[0]
            correction = (1, 2, 0)
            plt.figure(figsize=(10, 15))

            for i in range(4):
                # i = 0, 1 => no mask
                # i = 2, 3 => with mask
                # each i has 2 head
                # 0,1 was originally one latent that seperated
                # 2,3 was originally one latent that seperated

                if i < 2:
                    latent_num = 1
                else:
                    latent_num = 2

                if i % 2 == 0:
                    window = "w-msa"
                else:
                    window = "sw-msa"

                attention_pre = attentions[-1][i][:,
                                                  :1, 0, 0].reshape(nh-1, wh)
                attention_post = attentions[-1][i][:,
                                                   1:, 0, 0].reshape(nh-1, wh)
                attention_pre = attention_pre.reshape(nh-1, 32, 32)
                attention_post = attention_post.reshape(nh-1, 32, 32)
                attention_pre = np.transpose(
                    attention_pre.detach().cpu().numpy(), correction)
                attention_post = np.transpose(
                    attention_post.detach().cpu().numpy(), correction)

                plt.subplot(4, 2, 2*i+1)
                plt.title(f"head 1 / latent {latent_num} / {window}")
                plt.imshow(attention_pre.squeeze())
                plt.subplot(4, 2, 2*i+2)
                plt.title(f"head 2 / latent {latent_num} / {window}")
                plt.imshow(attention_post.squeeze())
                plt.savefig(f"attention\\{step}.png")
            plt.close()


        print(loss)
        loss.backward()
        self.optimizer.step()

        points = new_points

        status = True
        pi = []
        ti = []
        for i in range(len(self.p0)):
            pi.append(points[2*i])
            ti.append(points[2*i + 1])

        if check_handle_reach_target(pi, ti):
            status = False

        # status means keep iterating or not
        return status, (points, image), store_features, store_image

    def preprocess_image(self, image):
        image = image * 255
        image = image.astype(np.uint8)
        image = Image.fromarray(image.reshape((256, 256, 4)))
        image = image.convert('RGB')
        image = torch.from_numpy(
            np.array(image)).float() / 127.5 - 1  # [-1, 1]
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

        # source_image_tensor = self.preprocess_image(np.array(origin_image))
        # dragged_image_tensor = self.preprocess_image(np.array(raw_data))
        # print(source_image_tensor.shape, dragged_image_tensor.shape)

        source_image = np.array(origin_image) * 255
        source_image = source_image.astype(np.uint8)
        source_image = Image.fromarray(source_image.reshape((256, 256, 4)))
        source_image = source_image.convert('RGB')
        source_image_tensor = (PILToTensor()(
            source_image) / 255.0 - 0.5) * 2

        dragged_image = np.array(raw_data) * 255
        dragged_image = dragged_image.astype(np.uint8)
        dragged_image = Image.fromarray(dragged_image.reshape((256, 256, 4)))
        dragged_image = dragged_image.convert('RGB')
        dragged_image_tensor = (PILToTensor()(
            dragged_image) / 255.0 - 0.5) * 2

        _, H, W = source_image_tensor.shape
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
