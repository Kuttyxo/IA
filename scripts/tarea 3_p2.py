import matplotlib.pyplot as plt  
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os

# ==========================================
# 1. Configuración y Hiperparámetros
# ==========================================

BATCH_SIZE = 32
IMG_SIZE = 64
LEARNING_RATE = 0.001
SEED = 42  # Semilla para reproducibilidad 

# Fijamos la semilla en CPU y GPU
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


# ==========================================
# 2. Preparación de Datos
# ==========================================

print("--- Preparando Datos ---")

# Transformaciones: Convertir a Tensor y Normalizar
data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Lógica para encontrar la carpeta
if os.path.exists('../Imagenes'):
    data_dir = '../Imagenes'
elif os.path.exists('IA/Imagenes'):
    data_dir = 'IA/Imagenes'
else:
    data_dir = './Imagenes' 

try:
    full_dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)
    print(f"✅ Dataset encontrado. Clases: {full_dataset.classes}")
except Exception as e:
    print(f"❌ Error: No se encontró el dataset en '{data_dir}'. Verifica la ruta.")
    exit()

# División del Dataset (70% Train, 15% Val, 15% Test)
total_size = len(full_dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(SEED)
)

print(f"📊 Distribución: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ==========================================
# 3. Arquitecturas CNN
# ==========================================

# --- MODELO 1: BASE (El que ya probaste) ---
class SatelliteCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(SatelliteCNN, self).__init__()
        # Bloques Convolucionales
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Clasificador
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 8 * 8, 512) 
        self.relu_fc = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu_fc(self.fc1(x))
        x = self.fc2(x)
        return x

# --- MODELO 2: CON DROPOUT (NUEVO PARA PARTE 2) ---
class SatelliteCNNDropout(nn.Module):
    def __init__(self, num_classes=5):
        super(SatelliteCNNDropout, self).__init__()
        
        # Usamos los mismos bloques convolucionales para comparar justamente
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Clasificador
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.relu_fc = nn.ReLU()
        
        # -------------------------------------------------------
        # AQUÍ ESTÁ EL CAMBIO IMPORTANTE: DROPOUT
        # -------------------------------------------------------
        # Apagamos aleatoriamente el 50% de las neuronas en cada paso de entrenamiento
        self.dropout = nn.Dropout(p=0.5) 
        
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        
        x = self.flatten(x)
        x = self.relu_fc(self.fc1(x))
        
        # Aplicamos Dropout antes de la capa final
        x = self.dropout(x)
        
        x = self.fc2(x)
        return x


# ==========================================
# 4. Funciones de Entrenamiento
# ==========================================

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Iniciando entrenamiento en: {device}")
    model.to(device)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    start_time = time.time()

    for epoch in range(num_epochs):
        # FASE DE ENTRENAMIENTO
        model.train() # Habilita el Dropout
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        # FASE DE VALIDACIÓN
        model.eval() # Deshabilita Dropout para evaluar
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)
                
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc.item())
        
        print(f'Época {epoch+1}/{num_epochs} | Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

    time_elapsed = time.time() - start_time
    print(f'\n🏁 Entrenamiento completado en {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    return model, history

def plot_training_curves(history, filename='curvas_entrenamiento.png', title_suffix=''):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title(f'Pérdida (Loss) {title_suffix}')
    plt.xlabel('Época'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs, history['val_acc'], label='Val Accuracy')
    plt.title(f'Precisión (Accuracy) {title_suffix}')
    plt.xlabel('Época'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"📈 Gráfico guardado como '{filename}'")
    plt.show()

# ==========================================
# 5. Ejecución Principal
# ==========================================

if __name__ == "__main__":
    # NOTA: He comentado el entrenamiento del modelo base para ahorrar tiempo
    # y pasar directo a la Parte 2 (Dropout).
    
    # ---------------------------------------------------------
    # EJECUCIÓN PARTE 2: MODELO CON DROPOUT
    # ---------------------------------------------------------
    print("\n--- 🏗️ Entrenando Modelo CON Dropout (Parte 2) ---")
    
    # Instanciamos la NUEVA clase
    model_dropout = SatelliteCNNDropout(num_classes=5)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_dropout.parameters(), lr=LEARNING_RATE)
    
    # Entrenar
    trained_model, history_dropout = train_model(
        model_dropout, train_loader, val_loader, criterion, optimizer, num_epochs=10
    )
    
    # Graficar y guardar con nombre diferente
    plot_training_curves(history_dropout, filename='curvas_dropout.png', title_suffix='(con Dropout)')