import os
import numpy as np
import torch
import torch.utils.data.dataloader
from tqdm import tqdm
import torch.multiprocessing as mp
import sys
sys.path.append("../")
import warnings
# Ignore specific FutureWarning about functorch.vmap deprecation (PyTorch 2.x migration)
warnings.filterwarnings(
    "ignore",
    message=r".*functorch.vmap.*deprecated.*",
    category=FutureWarning,
)
# Broader filter to catch other FutureWarnings originating from functorch integration
warnings.filterwarnings(
    "ignore",
    message=r".*functorch.*",
    category=FutureWarning,
)

def mean_next_batch(m, n, mean_batch, mean_old):
    mean_new = mean_old + m*(mean_batch - mean_old)/n
    return mean_new

def var_next_batch(m, n, mean_batch, var_batch, mean_old, var_old):
    var_new = (n-m)*var_old/n + m*(n-m)*(mean_batch-mean_old)**2/n**2 + m*var_batch/n
    return var_new

def get_gradient_norms_var(model, inputs, targets, device="cuda:0",
                           criterion_vector=torch.nn.MSELoss(reduction='none'), bs=100):
    """Get gradient norms via vectorization operations.

    Parameters:
    -----------
        model: the neural network model under training.
        inputs: torch.tensor, training data.
        targets: torch.tensor, labels of training data.
        device: string, cuda device.
        criterion_vector: loss function with no reduction.
        bs: int, batch size for vectorization operations.

    Returns:
    --------
        grad_norms: torch.tensor, L2-norms of gradient of training data.
        grad_mean: torch.tensor, mean gradient of training data.
        grad_var: torch.tensor, trace of gradient variance.
    """
    m = bs
    n = 0
    mean, var = 0, 0
    n_inputs = len(inputs)
    grad_outputs = torch.eye(bs).view(bs,bs,1).to(device)
    grad_norms_list = []
    for i in tqdm(range(0, n_inputs, bs)):
        if i == bs * (n_inputs // bs):
            bs = n_inputs % bs
            grad_outputs = torch.eye(bs).view(bs,bs,1).to(device)
        input = inputs[i:i+bs]
        target = targets[i:i+bs]
        output = model(input)
        losses = criterion_vector(output, target)
        grads = torch.autograd.grad(losses, model.parameters(), 
                                    grad_outputs=grad_outputs,
                                    retain_graph=True, 
                                    is_grads_batched=True)
        grad_ = torch.cat([grad.view(bs,-1) for grad in grads], dim=1)
        grad_norm = torch.norm(grad_, dim=1)
        grad_norms_list.append(grad_norm)
        n += m
        mean_batch = torch.mean(grad_, dim=0)
        var_batch = torch.var(grad_, dim=0)
        var = var_next_batch(m, n, mean_batch, var_batch, mean, var)
        mean = mean_next_batch(m, n, mean_batch, mean)
    grad_norms = torch.cat(grad_norms_list).cpu()
    grad_mean = mean.cpu()
    grad_var = var.cpu()
    return grad_norms, grad_mean, grad_var

def get_gradient_norms_var_batched(model, inputs, targets, device="cuda:0",
                           criterion_vector=torch.nn.MSELoss(), bs=500):
    """Get batched gradient norms.

    Parameters:
    -----------
        model: the neural network model under training.
        inputs: torch.tensor, training data.
        targets: torch.tensor, labels of training data.
        device: string, cuda device.
        criterion_vector: loss function with no reduction.
        bs: int, batch size for vectorization operations.

    Returns:
    --------
        grad_norms: torch.tensor, L2-norms of gradient of training data.
    """
    n_inputs = len(inputs)
    grad_norms_list = []
    for i in tqdm(range(0, n_inputs, bs)):
        if i == bs * (n_inputs // bs):
            bs = n_inputs % bs
        input = inputs[i:i+bs]
        target = targets[i:i+bs]
        output = model(input)
        loss = criterion_vector(output, target)
        grads = torch.autograd.grad(loss, model.parameters())
        grad_ = torch.cat([grad.view(-1) for grad in grads], dim=0)
        grad_norm = torch.norm(grad_, dim=0)
        grad_norms_list.append(grad_norm.view(-1))
    grad_norms = torch.cat(grad_norms_list).cpu()
    return grad_norms

def get_gradient_norms_var_cls_parallel(
        model,
        data_loader: torch.utils.data.dataloader.DataLoader, 
        device="cuda:0",
        criterion=torch.nn.CrossEntropyLoss().to("cuda:0")
    ):
    """Get batched gradient norms for image classification.

    Parameters:
    -----------
        model: the neural network model under training.
        data_loader: torch.utils.data.dataloader.DataLoader, the data loader.
        device: string, cuda device.
        criterion_vector: loss function with no reduction.

    Returns:
    --------
        grad_norms: torch.tensor, L2-norms of gradient of training data.
        grad_mean: torch.tensor, mean gradient of training data.
        grad_var: torch.tensor, trace of gradient variance.
    """
    from functorch import make_functional_with_buffers, vmap, grad
    grad_norms_list = []
    func_model, params, buffers = make_functional_with_buffers(model)

    def compute_loss(params, buffers, image, label):
        images = image.unsqueeze(0)
        labels = label.unsqueeze(0)
        outputs = func_model(params, buffers, images)
        loss = criterion(outputs, labels)
        return loss
    
    for images, labels in tqdm(data_loader):
        m = len(images)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        inputs = (params, buffers, images, labels)
        grads = vmap(grad(compute_loss), (None, None, 0, 0))(*inputs)
        grad_ = torch.cat([grad.detach().view(m,-1) for grad in grads], dim=1)
        del grads
        grad_norm = torch.norm(grad_, dim=1)
        grad_norm[torch.isnan(grad_norm)] = 0
        grad_norms_list.append(grad_norm)
    torch.cuda.empty_cache()
    grad_norms = torch.cat(grad_norms_list).cpu().float()
    return grad_norms

def get_gradient_norms_var_cls_parallel_full(
        model,
        data_loader: torch.utils.data.dataloader.DataLoader, 
        device="cuda:0",
        criterion=torch.nn.CrossEntropyLoss().to("cuda:0")
    ):
    """Get gradient norms for image classification with `vmap`.

    Parameters:
    -----------
        model: the neural network model under training.
        data_loader: torch.utils.data.dataloader.DataLoader, the data loader.
        device: string, cuda device.
        criterion: loss function.

    Returns:
    --------
        grad_norms: torch.tensor, L2-norms of gradient of training data.
        grad_mean: torch.tensor, mean gradient of training data.
        grad_var: torch.tensor, trace of gradient variance.
    """
    from functorch import make_functional_with_buffers, vmap, grad
    # model.half()
    func_model, params, buffers = make_functional_with_buffers(model)

    def compute_loss(params, buffers, image, label):
        images = image.unsqueeze(0)
        labels = label.unsqueeze(0)
        outputs = func_model(params, buffers, images)
        loss = criterion(outputs, labels)
        return loss
    
    n = 0
    mean, var = 0, 0
    grad_norms_list = []
    for images, labels in tqdm(data_loader):
        m = len(images)
        # images = images.half().to(device, non_blocking=True)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        inputs = (params, buffers, images, labels)
        grads = vmap(grad(compute_loss), (None, None, 0, 0))(*inputs)
        grad_ = torch.cat([grad.detach().view(m,-1) for grad in grads], dim=1)
        # grad_[torch.isnan(grad_)] = 0
        # grad_ = grad_.float()
        del grads
        grad_norm = torch.norm(grad_, dim=1)
        # grad_norm[torch.isnan(grad_norm)] = 0
        grad_norms_list.append(grad_norm)
        n += m
        mean_batch = torch.mean(grad_, dim=0)
        # mean_batch[torch.isnan(mean_batch)] = 0
        # mean_batch = mean_batch.float()
        var_batch = torch.var(grad_, dim=0)
        # var_batch[torch.isnan(var_batch)] = 0
        # var_batch = var_batch.float()
        var = var_next_batch(m, n, mean_batch, var_batch, mean, var)
        mean = mean_next_batch(m, n, mean_batch, mean)
    grad_norms = torch.cat(grad_norms_list).cpu()
    grad_mean = mean.cpu()
    grad_var = var.cpu()
    # model.float()
    return grad_norms, grad_mean, grad_var

def under_sampling(
        data_loader: torch.utils.data.dataloader.DataLoader,
        ratio = 5,
    ):
    grad_norms_list = []
    for images, labels in tqdm(data_loader):
        grad_norm = []
        for i in range(len(images)):
            if labels[i] != 0:
                grad_norm.append(1)
            elif torch.rand(1) < (1 / ratio):
                grad_norm.append(1)
            else:
                grad_norm.append(0)
        grad_norm = torch.tensor(grad_norm)
        grad_norms_list.append(grad_norm)
    grad_norms = torch.cat(grad_norms_list).cpu()
    return grad_norms

def get_gradient_norms_var_cls_batch(
        model, 
        data_loader: torch.utils.data.dataloader.DataLoader, 
        device="cuda:0",
        criterion=torch.nn.CrossEntropyLoss(),
    ):
    """Get gradient norms via vectorization operations for image classification.

    Parameters:
    -----------
        model: the neural network model under training.
        data_loader: torch.utils.data.dataloader.DataLoader, the data loader.
        device: string, cuda device.
        criterion_vector: loss function with no reduction.

    Returns:
    --------
        grad_norms: torch.tensor, L2-norms of gradient of training data.
        grad_mean: torch.tensor, mean gradient of training data.
        grad_var: torch.tensor, trace of gradient variance.
    """
    model.half()
    grad_norms_list = []
    criterion.to(device)
    for i, (images, labels) in enumerate(tqdm(data_loader)):
        images = images.half().to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        grads = torch.autograd.grad(loss, model.parameters())
        grad_ = torch.cat([grad.view(-1) for grad in grads], dim=0)
        grad_norm = torch.norm(grad_)
        grad_norms_list.append(grad_norm)
        if i % 100 == 0:
            torch.cuda.empty_cache()
    grad_norms = torch.tensor(grad_norms_list).cpu().float()
    model.float()
    return grad_norms

def get_gradient_norms_var_cls_batch_parallel(
        model, 
        data_loader, 
        processes_per_gpu,
        device="cuda:0",
        criterion=torch.nn.CrossEntropyLoss(),
    ):
    """Get gradient norms via vectorization operations for image classification.

    Parameters:
    -----------
        model: the neural network model under training.
        data_loader: torch.utils.data.dataloader.DataLoader, the data loader.
        device: string, cuda device.
        criterion_vector: loss function with no reduction.

    Returns:
    --------
        grad_norms: torch.tensor, L2-norms of gradient of training data.
        grad_mean: torch.tensor, mean gradient of training data.
        grad_var: torch.tensor, trace of gradient variance.
    """
    model.half()
    mp.set_start_method('spawn')
    grad_norms_list = mp.Manager().list([0] * len(data_loader))
    criterion.to(device)
    workers = [mp.Process(target=worker_fn, args=(model, data_loader, device, criterion, i, processes_per_gpu, grad_norms_list)) for i in range(processes_per_gpu)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    grad_norms = torch.tensor(list(grad_norms_list)).cpu().float()
    model.float()
    return grad_norms

def worker_fn(model, data_loader, device, criterion, worker_id, processes_per_gpu, grad_norms_list):
    for i, (images, labels) in enumerate(data_loader):
        if i % processes_per_gpu == worker_id:
            images = images.half().to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            grads = torch.autograd.grad(loss, model.parameters())
            grad_ = torch.cat([grad.view(-1) for grad in grads], dim=0)
            grad_norm = torch.norm(grad_)
            grad_norms_list[i] = grad_norm.cpu().clone().item()
    torch.cuda.empty_cache()
    return

def get_gradient_norms_var_cls_approx(
        model, 
        data_loader: torch.utils.data.dataloader.DataLoader, 
        device="cuda:0",
        criterion=torch.nn.CrossEntropyLoss(),
    ):
    """Get gradient norms via vectorization operations for image classification.

    Parameters:
    -----------
        model: the neural network model under training.
        data_loader: torch.utils.data.dataloader.DataLoader, the data loader.
        device: string, cuda device.
        criterion_vector: loss function with no reduction.

    Returns:
    --------
        grad_norms: torch.tensor, L2-norms of gradient of training data.
        grad_mean: torch.tensor, mean gradient of training data.
        grad_var: torch.tensor, trace of gradient variance.
    """
    model.half()
    grad_norms_list = []
    criterion.to(device)
    for i, (images, labels) in enumerate(tqdm(data_loader)):
        images = images.half().to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        one_hot_labels = torch.zeros_like(outputs).scatter_(1, labels.view(-1, 1), 1)
        outputs = torch.nn.functional.softmax(outputs, dim=1)
        loss = torch.sum(-one_hot_labels * torch.log(outputs), dim=1).detach().cpu().numpy().tolist()
        grad_norms_list += loss
        if (i-1) % 100000 == 0:
            torch.cuda.empty_cache()
    grad_norms = torch.tensor(grad_norms_list).cpu().float()
    model.float()
    return grad_norms