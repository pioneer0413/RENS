# %%
import tonic
from tqdm.notebook import tqdm

import torch
import torchvision

sensor_size = tonic.datasets.NMNIST.sensor_size


transform_original = tonic.transforms.Compose(
    [
        tonic.transforms.Denoise(filter_time=10000),
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=30),
        torch.from_numpy
    ]
)

transform_noised = tonic.transforms.Compose(
    [
        tonic.transforms.Denoise(filter_time=10000),
        tonic.transforms.UniformNoise(sensor_size=sensor_size, n=30000),
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=30),
        torch.from_numpy
    ]
)

situation = 'normal'

if situation == 'normal':
    train_dataset = tonic.datasets.NMNIST(save_to="/DATA/hwkang", train=True, transform=transform_original)
elif situation == 'noised':
    train_dataset = tonic.datasets.NMNIST(save_to="/DATA/hwkang", train=True, transform=transform_noised)

test_dataset = tonic.datasets.NMNIST(save_to="/DATA/hwkang", train=False, transform=transform_noised)

cached_train_dataset = tonic.cached_dataset.MemoryCachedDataset(train_dataset)
cached_test_dataset = tonic.cached_dataset.MemoryCachedDataset(test_dataset)

# %%
import matplotlib.pyplot as plt
from IPython.display import HTML

frames, _ = cached_test_dataset[0]
ani = tonic.utils.plot_animation(frames)
HTML(ani.to_jshtml())

# %%
from torch.utils.data import DataLoader
import multiprocessing

train_loader = DataLoader(cached_train_dataset, batch_size=100, num_workers=multiprocessing.cpu_count() // 2, shuffle=True, collate_fn=tonic.collation.PadTensors(batch_first=False))
test_loader = DataLoader(cached_test_dataset, batch_size=100, num_workers=multiprocessing.cpu_count() // 2, shuffle=False, collate_fn=tonic.collation.PadTensors(batch_first=False))

# %%
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import spikeplot as splt
from snntorch import utils
import torch.nn as nn

# %%
torch.manual_seed(0)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# neuron and simulation parameters
spike_grad = surrogate.atan()
beta = 0.5

#model = SpikingCNN(beta, spike_grad)
model = CNN()
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-2, betas=(0.9, 0.999))
#criterion = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
criterion = nn.CrossEntropyLoss()

# %%
num_epochs = 3

skip_train = True
if skip_train is True:
    model.load_state_dict(torch.load('experiment/experiment-09/weights/nmnist_snntorch_cnn_no_train.pt'))
if skip_train is False:

    epoch_loss_rec = []
    batch_loss_rec = []

    epoch_loss = 0.0
    for epoch in range(num_epochs):
        batch_loss = 0.0
        current_step = 0
        current_size = 0
        for inputs, targets in train_loader: # >> [t, b, c, x, y] [b]
            optimizer.zero_grad()
            #outputs, _ = model(inputs.to(device)) # >> [t, b, num_neurons] 
            outputs = model(inputs.to(device))
            loss = criterion(outputs, targets.to(device))
            loss.backward()
            optimizer.step()

            batch_loss += loss.item()

            current_step += 1
            current_size += inputs.size(1)

            if current_step % 100 == 0:
                print(f'Epoch: {epoch}/{num_epochs} | Inputs: {current_size}/{len(train_loader.dataset)} | Batch Loss: {batch_loss / current_size:.6f}')

            batch_loss_rec.append(batch_loss / current_size)

        epoch_loss = batch_loss / len(train_loader.dataset)
        print(f'Epoch: {epoch+1}/{num_epochs} | Epoch Loss: {epoch_loss:.6f}\n')
        epoch_loss_rec.append(epoch_loss)

# %%
PATH = 'experiment/experiment-09/weights'
filename = 'nmnist_snntorch_cnn_no_train.pt'
if skip_train is False:
    import os
    torch.save(model.state_dict(), os.path.join(PATH,filename))

# %%
import torchmetrics
acc_metrics = torchmetrics.Accuracy(task='multiclass', num_classes=10).to(device)

with torch.no_grad():
    current_step = 0
    all_predictions = []
    all_targets = []
    for inputs, targets in test_loader: # >> [b]
        #outputs, _ = model(inputs.to(device)) # >> [t, b, num_neurons]
        #batch_acc = SF.accuracy_rate(outputs, targets.to(device))
        outputs = model(inputs.to(device))
        predictions = torch.argmax(outputs, dim=1) # >> [b]
        batch_acc = acc_metrics(predictions, targets.to(device))

        current_step += 1
        if current_step % 5 == 0:
            print(f'Step: {current_step}/{len(test_loader)} | Batch Accuracy: {batch_acc * 100:.3f}%')

        #all_predictions.append(outputs.detach().cpu())
        all_predictions.append(predictions.detach().cpu())
        all_targets.append(targets.detach().cpu())

    #all_predictions = torch.cat(all_predictions, dim=1)
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

# %%
#total_accuracy = SF.accuracy_rate(all_predictions, all_targets)
total_accuracy = acc_metrics(all_predictions, all_targets)

print(f'Total Accuracy: {total_accuracy * 100:.4f}%')

# %%
# Loss trend
epoch_indices = [599, 1199, 1799]
epoch_loss_plot = [None] * len(batch_loss_rec)
for idx, epoch_idx in enumerate(epoch_indices):
    epoch_loss_plot[epoch_idx] = epoch_loss_rec[idx]

plt.figure(figsize=(12,5))
plt.plot(batch_loss_rec)
plt.plot(epoch_loss_plot, marker='x')
plt.grid(True)
plt.show()

# %%
inputs, targets = next(iter(test_loader))
spk_out, mem_out = model(inputs.to(device)) # >> [t, b, num_neurons] [t, b, num_neurons]

example_spk_train = spk_out[:, 0, :].detach().cpu() # << [t, num_neurons]
example_mem_trend = mem_out[:, 0, :].detach().cpu() # << [t, num_neurons]

# %%
# Spike count
fig, ax = plt.subplots(facecolor='w', figsize=(12, 7))
labels=['0', '1', '2', '3', '4', '5', '6', '7', '8','9']
print(f"The target label is: {targets[0]}")

anim = splt.spike_count(example_spk_train, fig, ax, labels=labels, animate=True, interpolate=1)

HTML(anim.to_html5_video())

# %%
splt.traces(example_mem_trend, spk=example_spk_train, dim=(10,1), spk_height=100)


