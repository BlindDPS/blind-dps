import torch
import numpy as np
import torch.nn.functional as F

from pathlib import Path
from tqdm import tqdm
from utils import Blurkernel, clear


# params
device = 'cuda:0'
H, W = 256, 256
M = 500
N = 32
ksize = 7
sig = 3.0
S = np.random.uniform(0.1, 0.4)
B = np.random.uniform(0.1, 1.0)

save_root = Path(f"./turbulence/tilt_256x256")
save_root.mkdir(exist_ok=True, parents=True)


for idx in tqdm(range(50000)):
    u = torch.zeros([H, W], device=device)
    v = torch.zeros([H, W], device=device)

    conv = Blurkernel(blur_type="gaussian",
                      kernel_size=ksize, std=1.0,
                      device=device).to(device)
    kernel = conv.get_kernel()
    kernel = kernel.type(torch.float32)
    kernel = kernel.to(device).view(1, 1, ksize, ksize)

    # generate tilt map
    for i in range(M):
        x = np.random.randint(H - 2 * N) + N
        y = np.random.randint(W - 2 * N) + N

        S = np.random.uniform(0.1, 0.4)

        N_u_tmp = torch.randn([2 * N, 2 * N], device=device).view(1, 1, 2 * N, 2 * N)
        N_u = F.conv2d(N_u_tmp, kernel, padding="same")[0, 0, ...]
        N_v_tmp = torch.randn([2 * N, 2 * N], device=device).view(1, 1, 2 * N, 2 * N)
        N_v = F.conv2d(N_v_tmp, kernel, padding="same")[0, 0, ...]
        u[x - N:x + N, y - N:y + N] += N_u * S
        v[x - N:x + N, y - N:y + N] += N_v * S

    # save
    tilt_map = torch.stack((u, v), dim=2)
    tilt_map = clear(tilt_map, normalize=False)
    np.save(str(save_root / f"{str(idx).zfill(5)}.npy"), tilt_map)