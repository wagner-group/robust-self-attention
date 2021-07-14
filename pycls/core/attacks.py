import torch
import torch.nn as nn
import torch.nn.functional as F

import pycls.core.logging as logging

logger = logging.get_logger(__name__)



def get_random_mask(B, img_size, patch_size, grid_aligned):
    # get patch indices
    b_idx = torch.arange(B)[:, None]
    c_idx = torch.zeros((B, 1), dtype=torch.long)
    if grid_aligned:
        h_idx = torch.randint(0, (img_size // patch_size) - 1, (B, 1)) * patch_size
        w_idx = torch.randint(0, (img_size // patch_size) - 1, (B, 1)) * patch_size
    else:
        h_idx = torch.randint(0 , img_size - patch_size, (B, 1))
        w_idx = torch.randint(0 , img_size - patch_size, (B, 1))

    idx = torch.cat([b_idx, c_idx, h_idx, w_idx], dim=1)
    idx_list = [idx + torch.tensor([0, 0, h, w]) for h in range(patch_size)\
                                                    for w in range(patch_size)]
    idx_list = torch.cat(idx_list, dim=0)

    # create mask
    mask = torch.zeros([B, 1, img_size, img_size], dtype=torch.bool).cuda()
    mask[idx_list[:, 0], idx_list[:, 1], idx_list[:, 2], idx_list[:, 3]] = True

    return mask


def get_grid_masks(img_size, patch_size, patch_stride):
    for h in torch.arange(0, img_size + 1 - patch_size, patch_stride):
        for w in torch.arange(0, img_size + 1 - patch_size, patch_stride):
            idxs = [torch.tensor([h + i, w + j]) for i in range(patch_size) for j in range(patch_size)]
            idxs = torch.stack(idxs, dim=1)

            mask = torch.zeros([img_size, img_size], dtype=torch.bool)
            mask[idxs[0], idxs[1]] = True
            
            yield mask


def DLRUntargetedLoss(x, y):
    x_s, idx_s = x.sort(dim=1)
    ind = (idx_s[:, -1] == y).float()
    
    return - (x[torch.arange(x.size(0)), y] - ind * x_s[:, -2] - (1. - ind) * x_s[:, -1]) / (x_s[:, -1] - x_s[:, -3] + 1e-12)


class PatchPGDTrain(nn.Module):
    def __init__(self, model, patch_size, img_size, num_steps, step_size, grid_aligned):
        super().__init__()
        self.model = model
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_steps = num_steps
        self.step_size = step_size
        self.grid_aligned = grid_aligned

    def forward(self, x, y):
        patch = torch.rand_like(x)
        if len(self.patch_size) > 1:
            size = torch.randint(self.patch_size[0], self.patch_size[1] + 1, (1,))[0]
        else:
            size = self.patch_size[0]
        mask = get_random_mask(x.size(0), self.img_size, size, self.grid_aligned)

        for i in range(self.num_steps):
            patch.requires_grad_()
            with torch.enable_grad():
                logits = self.model(torch.where(mask, patch, x))
                loss = F.cross_entropy(logits, y, reduction='sum')
            grad = torch.autograd.grad(loss, patch, only_inputs=True)[0]

            patch = patch.detach() + self.step_size * torch.sign(grad.detach())
            patch = patch.clamp(0, 1)

        return torch.where(mask, patch, x)


class PatchPGD(nn.Module):
    def __init__(self, model, patch_size, img_size, num_steps, step_size, num_restarts, grid_aligned, verbose=False):
        super().__init__()
        self.model = model
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_steps = num_steps
        self.step_size = step_size
        self.num_restarts = num_restarts
        self.grid_aligned = grid_aligned
        self.verbose = verbose

    def perturb(self, x, y):
        patch = torch.rand_like(x)
        mask = get_random_mask(x.size(0), self.img_size, self.patch_size, self.grid_aligned)

        for i in range(self.num_steps):
            patch.requires_grad_()
            with torch.enable_grad():
                logits = self.model(torch.where(mask, patch, x))
                loss = F.cross_entropy(logits, y, reduction='sum')
                
                if self.verbose:
                    logger.info("Average loss: {}".format(loss.item() / x.size(0)))

            grad = torch.autograd.grad(loss, patch, only_inputs=True)[0]

            patch = patch.detach() + self.step_size * torch.sign(grad.detach())
            patch = patch.clamp(0, 1)

        with torch.no_grad():
            logits = self.model(torch.where(mask, patch, x))
            fooled = logits.argmax(1) != y

        return patch, fooled, mask

    def forward(self, x, y):
        B = x.size(0)
        
        with torch.no_grad():
            logits = self.model(x)

        correct = logits.argmax(1) == y
        # probably should init patch to zeros
        patch = torch.rand_like(x)
        mask = torch.zeros(B, 1, self.img_size, self.img_size, dtype=torch.bool).cuda()

        if self.verbose:
            logger.info("Starting new batch.")

        for i in range(self.num_restarts):
            if torch.sum(correct) == 0:
                if self.verbose:
                    logger.info("All images fooled, skipping remaining restarts.")
                break

            ind = torch.nonzero(correct).squeeze(1)
            x_i, y_i = x[ind], y[ind]
            patch_i, fooled_i, mask_i = self.perturb(x_i, y_i)
            correct[ind[fooled_i]] = False
            # dont' think we need this
            if i == 0:
                patch[ind] = patch_i
                mask[ind] = mask_i
            else:
                patch[ind[fooled_i]] = patch_i[fooled_i]
                mask[ind[fooled_i]] = mask_i[fooled_i]

            if self.verbose:
                logger.info("Correct after restart {} : {} / {}".format(i, torch.sum(correct), B))
        
        return torch.where(mask, patch, x)


class PatchPGDGrid(nn.Module):
    def __init__(self, model, patch_size, patch_stride, img_size, num_steps, step_size, verbose=False):
        super().__init__()
        self.model = model
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.img_size = img_size
        self.num_steps = num_steps
        self.step_size = step_size
        self.verbose = verbose

    def perturb(self, x, y, mask):
        mask_b = mask[None, None, :, :]
        patch = torch.rand_like(x)

        for i in range(self.num_steps):
            patch.requires_grad_()
            with torch.enable_grad():
                logits = self.model(torch.where(mask_b, patch, x))
                loss = F.cross_entropy(logits, y, reduction='sum')
                
                if self.verbose:
                    logger.info("Average loss: {}".format(loss.item() / x.size(0)))

            grad = torch.autograd.grad(loss, patch, only_inputs=True)[0]

            patch = patch.detach() + self.step_size * torch.sign(grad.detach())
            patch = patch.clamp(0, 1)

        with torch.no_grad():
            logits = self.model(torch.where(mask_b, patch, x))
            fooled = logits.argmax(1) != y

        return patch, fooled

    def forward(self, x, y):
        B = x.size(0)
        
        with torch.no_grad():
            logits = self.model(x)

        correct = logits.argmax(1) == y
        # probably should init patch to zeros
        patch = torch.rand_like(x)
        mask = torch.zeros(B, 1, self.img_size, self.img_size, dtype=torch.bool).cuda()

        if self.verbose:
            logger.info("Starting new batch.")

        for i, mask_i in enumerate(get_grid_masks(self.img_size, self.patch_size, self.patch_stride)):
            if torch.sum(correct) == 0:
                if self.verbose:
                    logger.info("All images fooled, skipping remaining restarts.")
                break

            mask_i = mask_i.cuda()
            ind = torch.nonzero(correct).squeeze(1)
            x_i, y_i = x[ind], y[ind]
            patch_i, fooled_i = self.perturb(x_i, y_i, mask_i)
            correct[ind[fooled_i]] = False
            patch[ind[fooled_i]] = patch_i[fooled_i]
            mask[ind[fooled_i]] = mask_i

            if self.verbose:
                logger.info("Correct after mask {} : {} / {}".format(i, torch.sum(correct), B))
        
        return torch.where(mask, patch, x)



class PatchAutoPGD(nn.Module):
    def __init__(self, model, patch_size, img_size, num_steps, num_restarts, grid_aligned,
                 loss='dlr', num_eot_steps=1, rho=0.75, verbose=False):
        super().__init__()
        self.model = model
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_steps = num_steps
        self.num_restarts = num_restarts
        self.grid_aligned = grid_aligned
        self.loss = loss
        self.num_eot_steps = num_eot_steps
        self.rho = rho
        self.verbose = verbose

        p_step = max(int(0.22 * self.num_steps), 1)
        p_decr = max(int(0.03 * self.num_steps), 1)
        p_min = max(int(0.06 * self.num_steps), 1)
        checkpoints = [0]
        while checkpoints[-1] + p_step < self.num_steps:
            checkpoints.append(checkpoints[-1] + p_step)
            p_step = max(p_step - p_decr, p_min)

        self.checkpoints = checkpoints
    
    def check_condition1(self, loss_log, loss, i):
        k = self.checkpoints[self.checkpoints.index(i) - 1]
        t = torch.zeros(loss_log.size(1)).cuda()
        for j in range(k, i):
            t += loss_log[j + 1] > loss_log[j]
        return t <= (i - k) * self.rho

    def check_condition2(self, loss_checkpoint, plateaued_checkpoint, loss):
        return torch.logical_and(~plateaued_checkpoint, loss_checkpoint >= loss)

    def perturb(self, x, y):
        B = x.size(0)
        if self.loss == 'ce':
            criterion = nn.CrossEntropyLoss(reduce=False, reduction='none')
        elif self.loss == 'dlr':
            criterion = DLRUntargetedLoss
        else:
            raise ValueError('Unknown loss')

        loss_log = torch.zeros(self.num_steps, B).cuda()
        mask = get_random_mask(B, self.img_size, self.patch_size, self.grid_aligned).cuda()
        patch = torch.rand_like(x)
        patch_old = patch.detach()
        patch_best = patch.detach()
        
        step_size = 2 * torch.ones(B).cuda()
        loss_best = float('-inf') * torch.ones(B).cuda()
        loss_checkpoint = loss_best.detach()
        plateaued_checkpoint = torch.ones(B, dtype=torch.bool).cuda()

        patch.requires_grad_()
        grad = torch.zeros_like(patch)
        for _ in range(self.num_eot_steps):
            with torch.enable_grad():
                logits = self.model(torch.where(mask, patch, x))
                loss = criterion(logits, y)
                loss_sum = loss.sum()

            grad += torch.autograd.grad(loss_sum, patch, only_inputs=True)[0]

        grad /= self.num_eot_steps
        grad_best = grad.detach()

        for i in range(self.num_steps):
            with torch.no_grad():
                patch = patch.detach()
                momentum = patch - patch_old
                patch_old = patch.detach()

                if i == 0:
                    patch = torch.clamp(patch + step_size[:, None, None, None] * torch.sign(grad.detach()), 0, 1)
                else:
                    patch_z = torch.clamp(patch + step_size[:, None, None, None] * torch.sign(grad.detach()), 0, 1)
                    patch = torch.clamp(patch + 0.75 * (patch_z - patch) + 0.25 * momentum, 0, 1)

            patch.requires_grad_()
            grad = torch.zeros_like(patch)
            for _ in range(self.num_eot_steps):
                with torch.enable_grad():
                    logits = self.model(torch.where(mask, patch, x))
                    loss = criterion(logits, y)
                    loss_sum = loss.sum()

                grad += torch.autograd.grad(loss_sum, patch, only_inputs=True)[0]

            grad /= self.num_eot_steps

            if torch.all(logits.argmax(1) != y):
                patch_best = patch
                break

            with torch.no_grad():
                loss_log[i] = loss.detach()
                improved = loss > loss_best
                patch_best[improved] = patch[improved].detach()
                loss_best[improved] = loss[improved].detach()
                grad_best[improved] = grad[improved].detach()
              
                if i in self.checkpoints:
                    condition1 = self.check_condition1(loss_log, loss, i)
                    condition2 = self.check_condition2(loss_checkpoint, plateaued_checkpoint, loss)

                    plateaued = torch.logical_or(condition1, condition2)
                    loss_checkpoint = loss_best.detach()
                    plateaued_checkpoint = plateaued

                    step_size[plateaued] /= 2
                    patch[plateaued] = patch_best[plateaued].detach()
                    grad[plateaued] = grad_best[plateaued].detach()

                    if self.verbose:
                        logger.info("Average loss: {}".format(loss_sum.item() / x.size(0)))

        with torch.no_grad():
            logits = self.model(torch.where(mask, patch_best, x))
            fooled = logits.argmax(1) != y

        return patch_best, fooled, mask
    
    def forward(self, x, y):
        B = x.size(0)
        
        with torch.no_grad():
            logits = self.model(x)

        correct = logits.argmax(1) == y
        patch = torch.rand_like(x)
        mask = torch.zeros(B, 1, self.img_size, self.img_size, dtype=torch.bool).cuda()

        if self.verbose:
            logger.info("Starting new batch.")

        for i in range(self.num_restarts):
            if torch.sum(correct) == 0:
                if self.verbose:
                    logger.info("All images fooled, skipping remaining restarts.")
                break
            
            ind = torch.nonzero(correct).squeeze(1)
            x_i, y_i = x[ind], y[ind]
            patch_i, fooled_i, mask_i = self.perturb(x_i, y_i)
            correct[ind[fooled_i]] = False
            if i == 0:
                patch[ind] = patch_i
                mask[ind] = mask_i
            else:
                patch[ind[fooled_i]] = patch_i[fooled_i]
                mask[ind[fooled_i]] = mask_i[fooled_i]

            if self.verbose:
                logger.info("Correct after restart {} : {} / {}".format(i, torch.sum(correct), B))
        
        return torch.where(mask, patch, x)


class PatchAutoPGDGrid(nn.Module):
    def __init__(self, model, patch_size, patch_stride, img_size, num_steps,
                 loss='dlr', num_eot_steps=1, rho=0.75, verbose=False):
        super().__init__()
        self.model = model
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.img_size = img_size
        self.num_steps = num_steps
        self.loss = loss
        self.num_eot_steps = num_eot_steps
        self.rho = rho
        self.verbose = verbose

        p_step = max(int(0.22 * self.num_steps), 1)
        p_decr = max(int(0.03 * self.num_steps), 1)
        p_min = max(int(0.06 * self.num_steps), 1)
        checkpoints = [0]
        while checkpoints[-1] + p_step < self.num_steps:
            checkpoints.append(checkpoints[-1] + p_step)
            p_step = max(p_step - p_decr, p_min)

        self.checkpoints = checkpoints
    
    def check_condition1(self, loss_log, loss, i):
        k = self.checkpoints[self.checkpoints.index(i) - 1]
        t = torch.zeros(loss_log.size(1)).cuda()
        for j in range(k, i):
            t += loss_log[j + 1] > loss_log[j]
        return t <= (i - k) * self.rho

    def check_condition2(self, loss_checkpoint, plateaued_checkpoint, loss):
        return torch.logical_and(~plateaued_checkpoint, loss_checkpoint >= loss)

    def perturb(self, x, y, mask):
        B = x.size(0)
        
        if self.loss == 'ce':
            criterion = nn.CrossEntropyLoss(reduce=False, reduction='none')
        elif self.loss == 'dlr':
            criterion = DLRUntargetedLoss
        else:
            raise ValueError('Unknown loss')

        loss_log = torch.zeros(self.num_steps, B).cuda()
        mask_b = mask[None, None, :, :]
        patch = torch.rand_like(x)
        patch_old = patch.detach()
        patch_best = patch.detach()
        
        step_size = 2 * torch.ones(B).cuda()
        loss_best = float('-inf') * torch.ones(B).cuda()
        loss_checkpoint = loss_best.detach()
        plateaued_checkpoint = torch.ones(B, dtype=torch.bool).cuda()

        patch.requires_grad_()
        grad = torch.zeros_like(patch)
        for _ in range(self.num_eot_steps):
            with torch.enable_grad():
                logits = self.model(torch.where(mask_b, patch, x))
                loss = criterion(logits, y)
                loss_sum = loss.sum()

            grad += torch.autograd.grad(loss_sum, patch, only_inputs=True)[0]

        grad /= self.num_eot_steps
        grad_best = grad.detach()

        for i in range(self.num_steps):
            with torch.no_grad():
                patch = patch.detach()
                momentum = patch - patch_old
                patch_old = patch.detach()

                if i == 0:
                    patch = torch.clamp(patch + step_size[:, None, None, None] * torch.sign(grad.detach()), 0, 1)
                else:
                    patch_z = torch.clamp(patch + step_size[:, None, None, None] * torch.sign(grad.detach()), 0, 1)
                    patch = torch.clamp(patch + 0.75 * (patch_z - patch) + 0.25 * momentum, 0, 1)

            patch.requires_grad_()
            grad = torch.zeros_like(patch)
            for _ in range(self.num_eot_steps):
                with torch.enable_grad():
                    logits = self.model(torch.where(mask_b, patch, x))
                    loss = criterion(logits, y)
                    loss_sum = loss.sum()

                grad += torch.autograd.grad(loss_sum, patch, only_inputs=True)[0]

            grad /= self.num_eot_steps

            if torch.all(logits.argmax(1) != y):
                patch_best = patch
                break

            with torch.no_grad():
                loss_log[i] = loss.detach()
                improved = loss > loss_best
                patch_best[improved] = patch[improved].detach()
                loss_best[improved] = loss[improved].detach()
                grad_best[improved] = grad[improved].detach()
              
                if i in self.checkpoints:
                    condition1 = self.check_condition1(loss_log, loss, i)
                    condition2 = self.check_condition2(loss_checkpoint, plateaued_checkpoint, loss)

                    plateaued = torch.logical_or(condition1, condition2)
                    loss_checkpoint = loss_best.detach()
                    plateaued_checkpoint = plateaued

                    step_size[plateaued] /= 2
                    patch[plateaued] = patch_best[plateaued].detach()
                    grad[plateaued] = grad_best[plateaued].detach()

                    if self.verbose:
                        logger.info("Average loss: {}".format(loss_sum.item() / x.size(0)))

        with torch.no_grad():
            logits = self.model(torch.where(mask_b, patch_best, x))
            fooled = logits.argmax(1) != y

        return patch_best, fooled
    
    def forward(self, x, y):
        B = x.size(0)
        
        with torch.no_grad():
            logits = self.model(x)

        correct = logits.argmax(1) == y
        patch = torch.rand_like(x)
        mask = torch.zeros(B, 1, self.img_size, self.img_size, dtype=torch.bool).cuda()

        if self.verbose:
            logger.info("Starting new batch.")

        for i, mask_i in enumerate(get_grid_masks(self.img_size, self.patch_size, self.patch_stride)):
            if torch.sum(correct) == 0:
                if self.verbose:
                    logger.info("All images fooled, skipping remaining restarts.")
                break
            
            mask_i = mask_i.cuda()
            ind = torch.nonzero(correct).squeeze(1)
            x_i, y_i = x[ind], y[ind]
            patch_i, fooled_i = self.perturb(x_i, y_i, mask_i)
            correct[ind[fooled_i]] = False
            patch[ind[fooled_i]] = patch_i[fooled_i]
            mask[ind[fooled_i]] = mask_i

            if self.verbose:
                logger.info("Correct after mask {} : {} / {}".format(i, torch.sum(correct), B))
        
        return torch.where(mask, patch, x)
