import os
import torch
from torchvision import datasets
from facenet_pytorch import MTCNN, InceptionResnetV1, training, fixed_image_standardization, utils

# Path to the directory containing your dataset
data_dir = 'your_dataset_directory'

# Define device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Define MTCNN module
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

# Define Inception Resnet V1 module
resnet = InceptionResnetV1(
    classify=True,
    pretrained='vggface2',
    num_classes=len(os.listdir(data_dir))
).to(device)

# Define optimizer, scheduler, loss function
optimizer = torch.optim.Adam(resnet.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [5, 10])
loss_fn = torch.nn.CrossEntropyLoss()

# Define data loader
dataset = datasets.ImageFolder(data_dir, transform=utils.get_transform(mtcnn, normalize=True))
loader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=4, shuffle=True)

# Train the model
training.train(
    mtcnn=mtcnn, 
    resnet=resnet, 
    dataloader=loader, 
    optimizer=optimizer, 
    loss_fn=loss_fn, 
    scheduler=scheduler, 
    num_epochs=15, 
    device=device
)

# Save the model
torch.save(resnet.state_dict(), 'fine_tuned_model.pt')
