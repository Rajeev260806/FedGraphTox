import torch
from torch import nn, optim
from opacus import PrivacyEngine
# Define model, optimizer, and data loader
# Create a Dataset directly from in-memory Tensors
x = torch.randn(100, 10)  # 100 samples, 10 features
y = torch.randn(100, 1)   # 100 targets
dataset = torch.utils.data.TensorDataset(x, y)

model = nn.Linear(10, 1)
optimizer = optim.SGD(model.parameters(), lr=0.1)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32)
# Initialize PrivacyEngine
privacy_engine = PrivacyEngine()
# Make the model, optimizer, and data loader private
model, optimizer, data_loader = privacy_engine.make_private(
   module=model,
   optimizer=optimizer,
   data_loader=data_loader,
   noise_multiplier=1.1, # Controls the level of noise added
   max_grad_norm=1.0 # Clips gradients to this norm
)
# Train the model as usual
for epoch in range(5):
   for data, target in data_loader:
       optimizer.zero_grad()
       output = model(data)
       loss = nn.MSELoss()(output, target)
       loss.backward()
       optimizer.step()