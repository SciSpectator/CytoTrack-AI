"""
Cell Tracker - AI Cell Classification Training Module
======================================================
Train custom models to classify different cell types.

Uses Transfer Learning with Vision Transformer (ViT) or 
simple CNN fallback if transformers not available.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Try importing deep learning libraries
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms, models
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠️ PyTorch not installed. Run: pip install torch torchvision")

try:
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("⚠️ Transformers not installed. Using CNN fallback.")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

import cv2


class CellDataset(Dataset):
    """Dataset for cell classification."""

    def __init__(self, data_dir: str, transform=None, processor=None,
                 prior_classes: Optional[List[str]] = None):
        """
        Args:
            data_dir: Root with one subfolder per class.
            prior_classes: If given, these class names keep their original
                indices. New class folders get indices appended after them
                in sorted order. Used for incremental training so old
                classifier-head rows remain valid.
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.processor = processor
        self.samples: List[Tuple[str, int]] = []
        self.classes: List[str] = []
        self.class_to_idx: Dict[str, int] = {}
        self.prior_classes: List[str] = list(prior_classes or [])

        self._scan_directory()

    def _scan_directory(self):
        """Find all class folders and images."""
        class_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])

        if not class_dirs:
            raise ValueError(f"No class folders found in {self.data_dir}")

        # Preserve prior class ordering when resuming, then append new
        # folders in sorted order. That keeps old head weights pointing
        # at the right class index.
        found = [d.name for d in class_dirs]
        ordered: List[str] = [c for c in self.prior_classes if c in found]
        for name in found:
            if name not in ordered:
                ordered.append(name)
        self.classes = ordered
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        class_dirs = [self.data_dir / name for name in self.classes]
        
        image_exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
        
        for class_dir in class_dirs:
            class_idx = self.class_to_idx[class_dir.name]
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in image_exts:
                    self.samples.append((str(img_path), class_idx))
        
        print(f"Found {len(self.samples)} images in {len(self.classes)} classes:")
        for cls in self.classes:
            count = sum(1 for s in self.samples if s[1] == self.class_to_idx[cls])
            print(f"  - {cls}: {count} images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        if HAS_PIL:
            image = Image.open(img_path).convert('RGB')
        else:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(img)
        
        # Apply transforms
        if self.processor:
            inputs = self.processor(images=image, return_tensors="pt")
            return {'pixel_values': inputs['pixel_values'].squeeze(0), 'labels': label}
        elif self.transform:
            image = self.transform(image)
            return {'pixel_values': image, 'labels': label}
        
        return {'image': image, 'labels': label}


class SimpleCNN(nn.Module):
    """Simple CNN for cell classification (fallback if no transformers)."""
    
    def __init__(self, num_classes: int):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class CellClassifierTrainer:
    """
    Train a cell classifier using transfer learning.
    
    Uses ViT (Vision Transformer) if available, otherwise CNN fallback.
    """
    
    def __init__(self, output_dir: str = "cell_classifier", use_gpu: bool = True,
                 resume_from: Optional[str] = None):
        """
        Args:
            output_dir: Where to save the trained model (``<output_dir>/model/``).
            use_gpu:    Use CUDA if available.
            resume_from: Path to an existing ``model/`` folder to continue
                training from. If provided, the prior class list is loaded
                and the classifier head is grown to accommodate any new
                classes found in the new training folder — all old
                encoder + head weights for shared classes are preserved
                (transfer learning / incremental training).
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required. Run: pip install torch torchvision")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = None
        self.processor = None
        self.classes: List[str] = []
        self.use_vit = HAS_TRANSFORMERS
        self.resume_from: Optional[Path] = Path(resume_from) if resume_from else None
        self._prior_classes: List[str] = []
        if self.resume_from is not None:
            map_path = self.resume_from / "class_map.json"
            if not map_path.exists():
                raise FileNotFoundError(
                    f"resume_from does not contain class_map.json: {self.resume_from}")
            with open(map_path, "r") as f:
                prior = json.load(f)
            self._prior_classes = list(prior.get("classes", []))
            self.use_vit = prior.get("model_type", "vit") == "vit" and HAS_TRANSFORMERS
            print(f"Resuming from {self.resume_from} "
                  f"({len(self._prior_classes)} prior classes: {self._prior_classes})")
    
    def class_to_idx_map(self) -> Dict[str, int]:
        return {cls: idx for idx, cls in enumerate(self.classes)}

    def prepare_data(self, data_dir: str, val_split: float = 0.2) -> Tuple[DataLoader, DataLoader]:
        """Prepare training and validation data loaders."""
        
        print(f"\nPreparing data from: {data_dir}")
        
        if self.use_vit:
            # Prefer the resumed model's own processor if available,
            # otherwise download the base ViT processor.
            proc_source = str(self.resume_from) if self.resume_from else \
                "google/vit-base-patch16-224"
            self.processor = AutoImageProcessor.from_pretrained(proc_source)
            dataset = CellDataset(data_dir, processor=self.processor,
                                  prior_classes=self._prior_classes)
        else:
            # Use torchvision transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            dataset = CellDataset(data_dir, transform=transform,
                                  prior_classes=self._prior_classes)

        self.classes = dataset.classes

        # Training needs at least one sample. A single-class dataset is
        # allowed — users can train on one phenotype, save, then come
        # back later and add more phenotypes via ``resume_from``.
        n_samples = len(dataset)
        if n_samples < 2:
            raise ValueError(
                f"Not enough images to train ({n_samples}). Add more "
                "images to your class folder(s).")

        # Use at least one validation sample, but don't reserve more than
        # val_split fraction. For tiny datasets (e.g. 3 images) this keeps
        # the training loop alive.
        n_val = max(1, int(n_samples * val_split))
        n_val = min(n_val, n_samples - 1)
        n_train = n_samples - n_val

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [n_train, n_val]
        )

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

        print(f"Training samples: {n_train}")
        print(f"Validation samples: {n_val}")

        return train_loader, val_loader
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 10, learning_rate: float = 2e-5,
              progress_callback=None) -> Dict:
        """Train the classifier."""
        
        print(f"\n{'='*50}")
        print("Starting Training")
        print(f"{'='*50}")
        print(f"Classes: {self.classes}")
        print(f"Epochs: {epochs}")
        print(f"Model: {'ViT' if self.use_vit else 'CNN'}")
        print()
        
        num_classes = len(self.classes)

        # HuggingFace ViT with num_labels==1 drops into a regression loss
        # (MSE on float targets), which breaks our integer-label training
        # loop. Always keep at least 2 head slots; the extra slot is
        # unused until a future incremental-training run claims it.
        head_slots = max(2, num_classes)

        # Create model — either from the ViT base, or resume from an
        # existing checkpoint and grow its head to cover any new classes.
        if self.use_vit:
            base_source = str(self.resume_from) if self.resume_from else \
                "google/vit-base-patch16-224"
            self.model = AutoModelForImageClassification.from_pretrained(
                base_source,
                num_labels=head_slots,
                ignore_mismatched_sizes=True,
            )
            if self.resume_from is not None and self._prior_classes:
                # ``ignore_mismatched_sizes`` re-initialises the head on
                # shape mismatch. We manually copy rows that correspond to
                # prior classes back in so prior knowledge survives.
                try:
                    prior_model = AutoModelForImageClassification.from_pretrained(
                        str(self.resume_from))
                    old_w = prior_model.classifier.weight.data
                    old_b = prior_model.classifier.bias.data
                    new_w = self.model.classifier.weight.data
                    new_b = self.model.classifier.bias.data
                    for old_idx, cls in enumerate(self._prior_classes):
                        if cls in self.class_to_idx_map():
                            new_idx = self.class_to_idx_map()[cls]
                            if (old_idx < old_w.shape[0]
                                    and new_idx < new_w.shape[0]):
                                new_w[new_idx] = old_w[old_idx]
                                new_b[new_idx] = old_b[old_idx]
                    print(f"  preserved {len(self._prior_classes)} "
                          f"prior class head rows")
                except Exception as exc:
                    print(f"  warning: could not preserve prior head rows: {exc}")
        else:
            self.model = SimpleCNN(head_slots)
            if self.resume_from is not None:
                ckpt_path = self.resume_from / "model.pth"
                if ckpt_path.exists():
                    try:
                        old_state = torch.load(ckpt_path,
                                               map_location=self.device)
                        # Remove the final linear layer so we can grow it.
                        own_state = self.model.state_dict()
                        for k, v in old_state.items():
                            if k in own_state and own_state[k].shape == v.shape:
                                own_state[k] = v
                        self.model.load_state_dict(own_state)
                        print("  resumed CNN feature weights")
                    except Exception as exc:
                        print(f"  warning: could not resume CNN: {exc}")

        self.model.to(self.device)
        
        # Optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training history
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        best_val_acc = 0
        
        total_steps = epochs * len(train_loader)
        current_step = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                pixel_values = batch['pixel_values'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                
                if self.use_vit:
                    outputs = self.model(pixel_values=pixel_values, labels=labels)
                    loss = outputs.loss
                    logits = outputs.logits
                else:
                    logits = self.model(pixel_values)
                    loss = criterion(logits, labels)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                preds = logits.argmax(dim=-1)
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)
                
                current_step += 1
                if progress_callback:
                    progress_callback(current_step, total_steps, 
                                    f"Epoch {epoch+1}/{epochs} - Training...")
            
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    pixel_values = batch['pixel_values'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    if self.use_vit:
                        outputs = self.model(pixel_values=pixel_values, labels=labels)
                        loss = outputs.loss
                        logits = outputs.logits
                    else:
                        logits = self.model(pixel_values)
                        loss = criterion(logits, labels)
                    
                    val_loss += loss.item()
                    preds = logits.argmax(dim=-1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
            
            val_loss /= len(val_loader)
            val_acc = val_correct / max(1, val_total)
            
            # Record history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
            
            # Save best model (strictly greater so the FIRST epoch is
            # always saved, not just the ones that improve).
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                self._save_model()

        # Always make sure the final model is on disk, even if no epoch
        # beat the initial best-acc of 0 (single-class run or tiny dataset).
        if not (self.output_dir / "model" / "class_map.json").exists():
            self._save_model()

        print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")

        return history
    
    def _save_model(self):
        """Save the trained model."""
        model_dir = self.output_dir / "model"
        model_dir.mkdir(exist_ok=True)
        
        if self.use_vit:
            self.model.save_pretrained(str(model_dir))
            self.processor.save_pretrained(str(model_dir))
        else:
            torch.save(self.model.state_dict(), model_dir / "model.pth")
        
        # Save class mapping
        class_map = {
            "classes": self.classes,
            "class_to_idx": {cls: idx for idx, cls in enumerate(self.classes)},
            "idx_to_class": {idx: cls for idx, cls in enumerate(self.classes)},
            "model_type": "vit" if self.use_vit else "cnn"
        }
        
        with open(model_dir / "class_map.json", 'w') as f:
            json.dump(class_map, f, indent=2)
        
        print(f"Model saved to: {model_dir}")
    
    @classmethod
    def load_model(cls, model_dir: str) -> 'CellClassifierTrainer':
        """Load a trained model for inference."""
        model_dir = Path(model_dir)
        
        # Load class mapping
        with open(model_dir / "class_map.json", 'r') as f:
            class_map = json.load(f)
        
        trainer = cls.__new__(cls)
        trainer.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trainer.classes = class_map["classes"]
        trainer.use_vit = class_map.get("model_type", "vit") == "vit"
        trainer.output_dir = model_dir.parent
        
        if trainer.use_vit and HAS_TRANSFORMERS:
            trainer.processor = AutoImageProcessor.from_pretrained(str(model_dir))
            trainer.model = AutoModelForImageClassification.from_pretrained(str(model_dir))
        else:
            trainer.model = SimpleCNN(len(trainer.classes))
            trainer.model.load_state_dict(torch.load(model_dir / "model.pth", map_location=trainer.device))
            trainer.processor = None
            trainer.use_vit = False
        
        trainer.model.to(trainer.device)
        trainer.model.eval()
        
        print(f"Model loaded: {len(trainer.classes)} classes")
        return trainer
    
    def predict(self, image) -> Tuple[str, float]:
        """Predict cell type for a single image."""
        if self.model is None:
            raise RuntimeError("No model loaded")
        
        # Handle different input types
        if isinstance(image, str):
            if HAS_PIL:
                image = Image.open(image).convert('RGB')
            else:
                img = cv2.imread(image)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(img)
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        # Process image
        if self.use_vit and self.processor:
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(self.device)
        else:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            pixel_values = transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            if self.use_vit:
                outputs = self.model(pixel_values=pixel_values)
                logits = outputs.logits
            else:
                logits = self.model(pixel_values)
            
            probs = torch.softmax(logits, dim=-1)
            pred_idx = probs.argmax().item()
            confidence = probs[0, pred_idx].item()
        
        return self.classes[pred_idx], confidence
    
    def predict_batch(self, images: List, batch_size: int = 16) -> List[Tuple[str, float]]:
        """Predict cell types for multiple images."""
        results = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            
            for img in batch_images:
                try:
                    cell_type, confidence = self.predict(img)
                    results.append((cell_type, confidence))
                except Exception as e:
                    results.append(("Unknown", 0.0))
        
        return results


def check_training_requirements() -> Tuple[bool, str]:
    """Check if training requirements are met."""
    if not HAS_TORCH:
        return False, "PyTorch not installed. Run: pip install torch torchvision"

    if not HAS_PIL:
        return False, "Pillow not installed. Run: pip install Pillow"

    return True, "All requirements met"


# ======================================================================
# FeatureClassifier — MLP trained on Cellpose-SAM per-cell features.
# ======================================================================
#
# Replaces the raw-pixel ViT classifier when Cellpose-SAM is present.
# Input: 256-dim style vector from `CellDetector.extract_cell_features`.
# Output: phenotype label + confidence.
#
# Architecture: Linear(256->128) + ReLU + Dropout + Linear(128->N_classes).
# Tiny (~35k params) and trains on a handful of cells per phenotype because
# the heavy lifting already happened inside Cellpose-SAM's ViT encoder.
#
# Supports incremental learning: `resume_from=<dir>` preserves the existing
# output rows for prior phenotypes and appends fresh rows for new ones,
# exactly like `CellClassifierTrainer`.

class _FeatureMLP(nn.Module if HAS_TORCH else object):
    def __init__(self, in_dim: int, num_classes: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class FeatureClassifier:
    """Phenotype classifier over Cellpose-SAM per-cell feature vectors."""

    FEATURE_DIM = 256

    def __init__(self, output_dir: str = "feature_classifier",
                 use_gpu: bool = True,
                 resume_from: Optional[str] = None):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for FeatureClassifier")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.device = torch.device(
            "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
        self.classes: List[str] = []
        self.model: Optional[_FeatureMLP] = None
        self.resume_from: Optional[Path] = Path(resume_from) if resume_from else None
        self._prior_classes: List[str] = []
        if self.resume_from is not None:
            cmap = self.resume_from / "class_map.json"
            if not cmap.exists():
                raise FileNotFoundError(
                    f"resume_from missing class_map.json: {self.resume_from}")
            with open(cmap, "r") as f:
                self._prior_classes = list(json.load(f).get("classes", []))

    def class_to_idx_map(self) -> Dict[str, int]:
        return {c: i for i, c in enumerate(self.classes)}

    def fit(
        self,
        features: List[np.ndarray],
        labels: List[str],
        epochs: int = 50,
        batch_size: int = 16,
        learning_rate: float = 1e-3,
        val_split: float = 0.15,
    ) -> Dict:
        """
        Train on a list of 256-dim feature vectors + string labels.
        When `resume_from` was set, prior-class output rows are carried
        forward and new-class rows are appended.
        """
        if len(features) != len(labels):
            raise ValueError("features and labels length mismatch")
        if len(features) == 0:
            raise ValueError("no training samples")

        # Preserve prior class ordering so old rows stay at their indices.
        ordered: List[str] = list(self._prior_classes)
        for lbl in labels:
            if lbl not in ordered:
                ordered.append(lbl)
        self.classes = ordered
        idx_map = self.class_to_idx_map()
        num_classes = max(2, len(self.classes))  # ≥2 for cross-entropy

        self.model = _FeatureMLP(self.FEATURE_DIM, num_classes).to(self.device)

        # Restore prior weights.
        if self.resume_from is not None:
            ckpt = self.resume_from / "model.pth"
            if ckpt.exists():
                prior_state = torch.load(ckpt, map_location="cpu")
                prior_num = prior_state.get("_num_classes", len(self._prior_classes))
                try:
                    prior_model = _FeatureMLP(self.FEATURE_DIM, max(2, prior_num))
                    prior_model.load_state_dict(prior_state["state_dict"])
                    old_w = prior_model.net[-1].weight.data
                    old_b = prior_model.net[-1].bias.data
                    new_w = self.model.net[-1].weight.data
                    new_b = self.model.net[-1].bias.data
                    # Copy everything up to the final layer.
                    for i in range(len(self.model.net) - 1):
                        if hasattr(self.model.net[i], "weight"):
                            self.model.net[i].weight.data.copy_(
                                prior_model.net[i].weight.data)
                            self.model.net[i].bias.data.copy_(
                                prior_model.net[i].bias.data)
                    # Map old class rows to their new indices.
                    for old_idx, cls in enumerate(self._prior_classes):
                        new_idx = idx_map.get(cls)
                        if new_idx is None:
                            continue
                        if old_idx < old_w.shape[0] and new_idx < new_w.shape[0]:
                            new_w[new_idx].copy_(old_w[old_idx])
                            new_b[new_idx].copy_(old_b[old_idx])
                    print(f"  preserved {len(self._prior_classes)} prior "
                          f"phenotype rows from {self.resume_from}")
                except Exception as e:
                    print(f"  warning: could not load prior state: {e}")

        # Build train/val split.
        n = len(features)
        perm = np.random.permutation(n)
        n_val = max(1, int(n * val_split))
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        X = torch.tensor(np.stack(features), dtype=torch.float32)
        y = torch.tensor([idx_map[l] for l in labels], dtype=torch.long)

        X_tr, y_tr = X[train_idx].to(self.device), y[train_idx].to(self.device)
        X_va, y_va = X[val_idx].to(self.device), y[val_idx].to(self.device)

        optim = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        best_val = -1.0
        best_state = None
        for epoch in range(epochs):
            self.model.train()
            # Mini-batch shuffle.
            order = torch.randperm(len(X_tr))
            total_loss = 0.0
            for i in range(0, len(order), batch_size):
                sel = order[i:i + batch_size]
                xb, yb = X_tr[sel], y_tr[sel]
                optim.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optim.step()
                total_loss += float(loss.item()) * len(sel)

            self.model.eval()
            with torch.no_grad():
                va_logits = self.model(X_va)
                va_pred = va_logits.argmax(dim=1)
                va_acc = float((va_pred == y_va).float().mean().item())
            print(f"  epoch {epoch+1}/{epochs}  "
                  f"loss={total_loss/len(X_tr):.4f}  val_acc={va_acc:.3f}")
            if va_acc >= best_val:
                best_val = va_acc
                best_state = {k: v.detach().clone()
                              for k, v in self.model.state_dict().items()}

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.save_model()
        return {"val_accuracy": best_val, "classes": self.classes}

    def save_model(self, path: Optional[str] = None) -> Path:
        out = Path(path) if path else self.output_dir
        out.mkdir(exist_ok=True, parents=True)
        if self.model is None:
            raise RuntimeError("no model to save — call fit() first")
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "_num_classes": max(2, len(self.classes)),
                "feature_dim": self.FEATURE_DIM,
            },
            out / "model.pth",
        )
        with open(out / "class_map.json", "w") as f:
            json.dump({"classes": self.classes}, f, indent=2)
        with open(out / "model_info.json", "w") as f:
            json.dump({
                "type": "feature_mlp",
                "feature_source": "cellpose-sam styles",
                "feature_dim": self.FEATURE_DIM,
                "num_classes": len(self.classes),
            }, f, indent=2)
        print(f"FeatureClassifier saved to {out}")
        return out

    @classmethod
    def load_model(cls, model_dir: str) -> "FeatureClassifier":
        p = Path(model_dir)
        with open(p / "class_map.json", "r") as f:
            classes = list(json.load(f).get("classes", []))
        inst = cls(output_dir=str(p), use_gpu=torch.cuda.is_available())
        inst.classes = classes
        ckpt = torch.load(p / "model.pth", map_location="cpu")
        num_classes = max(2, ckpt.get("_num_classes", len(classes)))
        inst.model = _FeatureMLP(cls.FEATURE_DIM, num_classes).to(inst.device)
        inst.model.load_state_dict(ckpt["state_dict"])
        inst.model.eval()
        return inst

    def predict(self, feature: np.ndarray) -> Tuple[str, float]:
        if self.model is None:
            raise RuntimeError("no model loaded")
        if feature is None or feature.size == 0:
            return ("Unknown", 0.0)
        v = np.asarray(feature, dtype=np.float32).reshape(-1)
        if v.shape[0] != self.FEATURE_DIM:
            return ("Unknown", 0.0)
        x = torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]
            k = int(probs.argmax().item())
            conf = float(probs[k].item())
        if k >= len(self.classes):
            return ("Unknown", conf)
        return (self.classes[k], conf)
