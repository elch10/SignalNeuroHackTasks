import torch
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import colors

def to_var(x, requires_grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return torch.autograd.Variable(x, requires_grad=requires_grad)

def weight_prune_mask(model, pruning_perc):
    """
    Compute masks for model parameters so as pruning_perc smallest parameters
    would be reset after multiplying masks by corresponding parameters
    """
    all_weights = []
    for p in model.parameters():
        if len(p.data.size()) != 1:
            all_weights += list(p.cpu().data.abs().numpy().flatten())

    threshold = np.percentile(np.array(all_weights), pruning_perc)

    masks = []
    for p in model.parameters():
        if len(p.data.size()) != 1:
            pruned_inds = p.data.abs() >= threshold
            masks.append(pruned_inds.float())

    return masks

def diff_init_weight_prune_mask(model, pruning_perc):
    """
    Compute masks for model parameters so as pruning_perc smallest
    changed parameters as compared to initial state of parameter
    would be reset after multiplying masks by theirs parameters
    """
    all_weights = []
    all_init_weights = []
    for p in model.parameters():
        if len(p.data.size()) != 1:
            all_weights += list(p.cpu().data.abs().numpy().flatten())

    for p in model.init_parameters():
        if len(p.data.size()) != 1:
            all_init_weights += list(p.cpu().data.abs().numpy().flatten())

    threshold = np.percentile(abs(np.array(all_init_weights)) - abs(np.array(all_weights)), pruning_perc)

    # generate mask
    masks = []
    for p, p_init in zip(model.parameters(), model.init_parameters()):
        if len(p.data.size()) != 1 and len(p_init.data.size()) != 1:
            pruned_inds = p_init.data.abs().cuda() - p.data.abs() > threshold
            masks.append(pruned_inds.float())

    return masks

def prune_rate(model, verbose=True):
    """
    Compute percent of prunning parameters of the model. This function used for
    check prunning and for printing information about prunning percent of each layer
    """
    total_nb_param = 0
    nb_zero_param = 0

    layer_id = 0

    for parameter in model.parameters():
        param_this_layer = np.prod(parameter.data.size())
        total_nb_param += param_this_layer

        # only pruning linear and conv layers
        if len(parameter.data.size()) != 1:
            layer_id += 1
            zero_param_this_layer = \
                np.count_nonzero(parameter.cpu().data.numpy()==0)
            nb_zero_param += zero_param_this_layer

            if verbose:
                print("Layer {} | {} layer | {:.2f}% parameters pruned" \
                    .format(
                        layer_id,
                        'Conv' if len(parameter.data.size()) == 4 \
                            else 'Linear',
                        100.*zero_param_this_layer/param_this_layer,
                        ))

    pruning_perc = 100. * nb_zero_param/total_nb_param

    if verbose:
        print("Current pruning rate: {:.2f}%".format(pruning_perc))

    return pruning_perc

def compute_accuracy(model, loader):
    model.eval()

    num_correct = 0
    for x, y in loader:
        x_var = to_var(x)
        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()

    return num_correct.item() / len(loader.dataset)

def train(model, loss_fn, optimizer, epochs, train_loader, val_loader=None):
    """
    Train a model and save loss and accuracy on validation set on each epoch
    """
    loss_list = []
    accuracy_list = []

    for epoch in range(epochs):
        print('Training epoch', epoch)
        loss_sum = 0
        for t, (x, y) in enumerate(train_loader):
            x_var, y_var = to_var(x), to_var(y.long())

            scores = model(x_var)
            loss = loss_fn(scores, y_var)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            loss_sum += loss

        loss_list.append((loss_sum / t).item())
        if val_loader:
            accuracy_list.append(compute_accuracy(model, val_loader))

    return {'train_loss': loss_list, 'validation_accuracy': accuracy_list}

def train_one_shot(model, loss_fn, optimizer,
                   prepare_epochs, prune_epochs, prune_percent,
                   train_loader, val_loader=None):
    """
    Implements one shot prunning described in the original paper.
    """
    history = {}
    print('Preparing: Train before prune')
    history['train'] = train(model, loss_fn, optimizer,
                             prepare_epochs, train_loader, val_loader)

    print('Pruning...')
    masks = weight_prune_mask(model, prune_percent)
    model.reset_parameters()
    model.set_masks(masks)
    prune_rate(model)

    print('Train after prune')
    history['after_prune'] = train(model, loss_fn, optimizer,
                                   prune_epochs, train_loader, val_loader)

    return history


def visualize(weights, title, figsize=None):
    """
    Visualizes weights
    """

    data = weights.data.cpu().detach().abs().numpy().flatten()

    # make weights array as square matrix
    half = int(np.sqrt(data.size)) + 1
    data = np.concatenate((data, np.zeros(half * half - data.size)), axis=0)
    data = data.reshape(half, half)

    mx = np.max(data)

    cmap = ListedColormap(['k', 'w', 'r'])
    bounds=[0,0.000000000000001,mx/2,mx]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=figsize)
    plt.imshow(data, cmap=cmap, norm=norm)
    plt.colorbar()
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
