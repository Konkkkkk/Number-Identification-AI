import torch
from PIL import Image
import torchvision.transforms as transforms

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
class DigitNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28*28, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load model
model = DigitNet().to(device)
model.load_state_dict(torch.load("digit_model.pth", map_location=device))
model.eval()

# Load image
img_path = "./test/my_digit.png"
img = Image.open(img_path).convert("L")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
img = transform(img).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    output = model(img)
    pred = output.argmax(dim=1).item()
print(f"The number is: {pred}")