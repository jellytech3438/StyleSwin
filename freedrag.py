import torch
import math
import numpy as np
from models.generator import Generator
import torch.nn.functional as functional
from torchvision.transforms import functional as ttfunc
import matplotlib.pyplot as plt


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def check_handle_reach_target(handle_points,
                              target_points):
    # dist = (torch.cat(handle_points,dim=0) - torch.cat(target_points,dim=0)).norm(dim=-1)
    all_dist = list(map(lambda p, q: math.sqrt(
        (p[0]-q[0])**2 + (p[1]-q[1])**2), handle_points, target_points))
    dist = torch.tensor(all_dist)
    print("tensor distance: ", dist)
    return (dist < 2.0).all()


class FreeDrag():
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
        image, self.F0 = image[0], self.F0[self.layer_index*2+1].detach()
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
                # print("b4:", features[0].shape)
                d = features[2*i+j].size(dim=1)
                d2 = features[2*i+j].size(dim=2)
                feature = features[2*i+j].transpose(-1, -2).reshape(
                    1, d2, int(math.sqrt(d)), int(math.sqrt(d)))
                # print("af:", feature.shape)

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

    def step(self, points, mask):
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

            self.optimizer = torch.optim.Adam([self.trainable], lr=2e-3)
            points = points[:len_pts]
            self.p0 = points[::2]
            print("p0: ", self.p0)
        self.optimizer.zero_grad()
        trainable_fixed = torch.cat([self.trainable, self.fixed], dim=1)
        image, _, store_features = self.generator(
            trainable_fixed,  # this is why noise = [tensor[]]
            input_is_latent=True,
            return_features=True
        )
        features = store_features[self.layer_index*2+1]
        # features = features[self.layer_index+1]

        loss = 0
        for i in range(len(self.p0)):
            loss += motion_supervision((self.F0),
                                       features, points[2*i], points[2*i+1], M=mask, lambda_mask=self.lambda_mask)

        # calculate with selection layer mask if user choose

        print(loss)
        loss.backward()
        self.optimizer.step()

        # pt start
        # init features and image for point tracking
        image, _, features = self.generator(
            trainable_fixed,
            input_is_latent=True,
            return_features=True
        )
        features = features[self.layer_index*2+1]
        # features = features[self.layer_index+1]

        store_image = (1/(2*2.8)) * \
            image[0].detach().cpu().permute(1, 2, 0).numpy() + 0.5
        image = store_image.clip(0, 1).reshape(-1)
        for i in range(len(self.p0)):
            points[2*i] = point_tracking(self.F0,
                                         features, points[2*i], self.p0[i])

        status = True
        pi = []
        ti = []
        for i in range(len(self.p0)):
            pi.append(points[2*i])
            ti.append(points[2*i + 1])

        if check_handle_reach_target(pi, ti):
            status = False

        # return bool means keep iterating or not
        return status, (points, image), store_features, store_image
