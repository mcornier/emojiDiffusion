#!/usr/bin/env python3
"""
Script de diagnostic pour identifier pourquoi le modèle génère du bruit pur
"""

import torch
import torch.nn as nn
from pytorch_diffusion.model import DiffusionModel
from pytorch_diffusion.dataset import ImageDataset
from pytorch_diffusion.utils import get_device, forward_noise, TIMESTEPS, linear_beta_schedule
from torch.utils.data import DataLoader
import numpy as np

def test_model_gradients():
    """Test si les gradients se propagent correctement"""
    print("=== TEST DES GRADIENTS ===")
    
    device = get_device()
    print(f"Device: {device}")
    
    # Créer un petit modèle
    model = DiffusionModel(
        img_size=64,
        patch_size=8,
        img_channels=3,
        latent_dim=256,
        time_dim=64,
        num_transformer_blocks=2,  # Plus petit pour le test
        num_heads=8
    ).to(device)
    
    # Données de test
    batch_size = 2
    x_t = torch.randn(batch_size, 3, 64, 64, device=device, requires_grad=True)
    t = torch.randint(0, TIMESTEPS, (batch_size,), device=device)
    context = torch.randn(batch_size, 3, 8, 8, device=device)
    target_noise = torch.randn_like(x_t)
    
    print(f"Input shapes: x_t={x_t.shape}, t={t.shape}, context={context.shape}")
    print(f"Input ranges: x_t=[{x_t.min():.3f}, {x_t.max():.3f}]")
    print(f"Context ranges: [{context.min():.3f}, {context.max():.3f}]")
    print(f"Timesteps: {t}")
    
    # Forward pass
    model.train()
    predicted_noise = model(x_t, t, context)
    
    print(f"Predicted noise shape: {predicted_noise.shape}")
    print(f"Predicted noise range: [{predicted_noise.min():.3f}, {predicted_noise.max():.3f}]")
    print(f"Target noise range: [{target_noise.min():.3f}, {target_noise.max():.3f}]")
    
    # Loss
    loss_fn = nn.MSELoss()
    loss = loss_fn(predicted_noise, target_noise)
    print(f"Loss: {loss.item():.6f}")
    
    # Backward pass
    loss.backward()
    
    # Vérifier les gradients
    total_grad_norm = 0
    param_count = 0
    zero_grad_count = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm
            param_count += 1
            if grad_norm < 1e-8:
                zero_grad_count += 1
                
            if grad_norm > 1e-4 or grad_norm < 1e-8:  # Log significative gradients
                print(f"  {name}: grad_norm = {grad_norm:.6f}")
        else:
            print(f"  {name}: NO GRADIENT!")
    
    avg_grad_norm = total_grad_norm / param_count if param_count > 0 else 0
    print(f"\nGradient summary:")
    print(f"  Total parameters: {param_count}")
    print(f"  Zero gradients: {zero_grad_count}")
    print(f"  Average gradient norm: {avg_grad_norm:.6f}")
    
    if zero_grad_count > param_count * 0.5:
        print("❌ WARNING: Plus de 50% des gradients sont nuls!")
    elif avg_grad_norm < 1e-6:
        print("❌ WARNING: Gradients très petits, possible problème de vanishing gradients")
    elif avg_grad_norm > 10:
        print("❌ WARNING: Gradients très grands, possible problème d'exploding gradients")
    else:
        print("✅ Gradients semblent normaux")
    
    return model, loss.item()

def test_diffusion_parameters():
    """Test si les paramètres de diffusion sont cohérents"""
    print("\n=== TEST DES PARAMETRES DE DIFFUSION ===")
    
    # Paramètres du train.py
    timesteps = 200
    beta_schedule_train = linear_beta_schedule(timesteps)
    alphas_train = 1.0 - beta_schedule_train
    alphas_cumprod_train = torch.cumprod(alphas_train, dim=0)
    
    # Paramètres d'utils.py (importés)
    from pytorch_diffusion.utils import BETAS, ALPHAS, ALPHAS_CUMPROD
    
    print(f"Train timesteps: {timesteps}")
    print(f"Utils timesteps: {TIMESTEPS}")
    
    # Comparer les paramètres
    beta_diff = torch.abs(beta_schedule_train - BETAS).max()
    alphas_diff = torch.abs(alphas_train - ALPHAS).max()
    alphas_cumprod_diff = torch.abs(alphas_cumprod_train - ALPHAS_CUMPROD).max()
    
    print(f"Beta schedule diff: {beta_diff:.10f}")
    print(f"Alphas diff: {alphas_diff:.10f}")
    print(f"Alphas cumprod diff: {alphas_cumprod_diff:.10f}")
    
    if beta_diff > 1e-6:
        print("❌ WARNING: Différence dans beta_schedule!")
    else:
        print("✅ Beta schedules identiques")
    
    # Test de forward_noise
    device = get_device()
    x_0 = torch.randn(2, 3, 64, 64, device=device)
    t = torch.tensor([50, 100], device=device)
    
    x_t, noise = forward_noise(x_0, t, device)
    
    print(f"\nTest forward_noise:")
    print(f"  x_0 range: [{x_0.min():.3f}, {x_0.max():.3f}]")
    print(f"  x_t range: [{x_t.min():.3f}, {x_t.max():.3f}]")
    print(f"  noise range: [{noise.min():.3f}, {noise.max():.3f}]")
    
    # Vérifier que le bruit est gaussien
    noise_std = noise.std().item()
    noise_mean = noise.mean().item()
    print(f"  noise mean: {noise_mean:.6f} (devrait être ~0)")
    print(f"  noise std: {noise_std:.6f} (devrait être ~1)")
    
    if abs(noise_mean) > 0.1:
        print("❌ WARNING: Bruit pas centré!")
    if abs(noise_std - 1.0) > 0.1:
        print("❌ WARNING: Bruit pas normalisé!")

def test_dataset_normalization():
    """Test si le dataset est correctement normalisé"""
    print("\n=== TEST DE NORMALISATION DU DATASET ===")
    
    dataset_path = "H:/Repositories/ImgDatasetGen_256"
    
    try:
        dataset = ImageDataset(dataset_path, img_size=64, context_size=8)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        # Analyser quelques batches
        batch_count = 0
        total_main_mean = 0
        total_main_std = 0
        total_context_mean = 0
        total_context_std = 0
        
        for main_images, context_images, _ in dataloader:
            total_main_mean += main_images.mean().item()
            total_main_std += main_images.std().item()
            total_context_mean += context_images.mean().item()
            total_context_std += context_images.std().item()
            
            batch_count += 1
            if batch_count >= 10:  # Analyser 10 batches
                break
        
        avg_main_mean = total_main_mean / batch_count
        avg_main_std = total_main_std / batch_count
        avg_context_mean = total_context_mean / batch_count
        avg_context_std = total_context_std / batch_count
        
        print(f"Moyenne sur {batch_count} batches:")
        print(f"  Main images - mean: {avg_main_mean:.6f}, std: {avg_main_std:.6f}")
        print(f"  Context images - mean: {avg_context_mean:.6f}, std: {avg_context_std:.6f}")
        
        # Vérifier la normalisation [-1, 1]
        if abs(avg_main_mean) > 0.1:
            print("❌ WARNING: Images principales pas centrées!")
        if abs(avg_main_std - 1.0) > 0.3:
            print("❌ WARNING: Images principales mal normalisées!")
        if abs(avg_context_mean) > 0.1:
            print("❌ WARNING: Images de contexte pas centrées!")
        if abs(avg_context_std - 1.0) > 0.3:
            print("❌ WARNING: Images de contexte mal normalisées!")
        
        print("✅ Normalisation semble correcte")
        
    except Exception as e:
        print(f"❌ Erreur lors du test du dataset: {e}")

def test_training_step():
    """Test d'un step de training complet"""
    print("\n=== TEST D'UN STEP DE TRAINING ===")
    
    device = get_device()
    
    # Charger un petit batch
    dataset_path = "H:/Repositories/ImgDatasetGen_256"
    dataset = ImageDataset(dataset_path, img_size=64, context_size=8)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Modèle simple
    model = DiffusionModel(
        img_size=64,
        patch_size=8,
        img_channels=3,
        latent_dim=256,
        time_dim=64,
        num_transformer_blocks=2,
        num_heads=8
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()
    
    # Un step de training
    model.train()
    optimizer.zero_grad()
    
    for main_images, context_images, _ in dataloader:
        main_images = main_images.to(device)
        context_images = context_images.to(device)
        
        # Timesteps aléatoires
        t = torch.randint(0, TIMESTEPS, (main_images.shape[0],), device=device)
        
        # Add noise
        x_t, noise = forward_noise(main_images, t, device)
        
        # Prédiction
        predicted_noise = model(x_t, t, context_images)
        
        # Loss
        loss = loss_fn(predicted_noise, noise)
        
        print(f"Training step:")
        print(f"  Batch size: {main_images.shape[0]}")
        print(f"  Main images range: [{main_images.min():.3f}, {main_images.max():.3f}]")
        print(f"  Context range: [{context_images.min():.3f}, {context_images.max():.3f}]")
        print(f"  Timesteps: {t}")
        print(f"  x_t range: [{x_t.min():.3f}, {x_t.max():.3f}]")
        print(f"  True noise range: [{noise.min():.3f}, {noise.max():.3f}]")
        print(f"  Predicted noise range: [{predicted_noise.min():.3f}, {predicted_noise.max():.3f}]")
        print(f"  Loss: {loss.item():.6f}")
        
        # Backward
        loss.backward()
        
        # Vérifier les gradients avant optimisation
        total_grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        print(f"  Total gradient norm: {total_grad_norm:.6f}")
        
        optimizer.step()
        
        break  # Un seul batch pour le test
    
    return loss.item()

if __name__ == '__main__':
    print("🔍 DIAGNOSTIC COMPLET DU MODELE DE DIFFUSION")
    print("=" * 60)
    
    try:
        # Test 1: Gradients
        model, test_loss = test_model_gradients()
        
        # Test 2: Paramètres de diffusion
        test_diffusion_parameters()
        
        # Test 3: Dataset
        test_dataset_normalization()
        
        # Test 4: Training step
        training_loss = test_training_step()
        
        print("\n" + "=" * 60)
        print("📊 RÉSUMÉ DU DIAGNOSTIC:")
        print(f"  Test loss (modèle aléatoire): {test_loss:.6f}")
        print(f"  Training loss (un step): {training_loss:.6f}")
        
        if test_loss > 0.5 and training_loss > 0.5:
            print("✅ Les losses semblent dans la plage normale pour un modèle non-entraîné")
        
        print("\n🔍 Si le problème persiste, vérifiez:")
        print("  1. Le learning rate (peut-être trop faible)")
        print("  2. L'architecture du modèle")
        print("  3. La qualité du dataset")
        print("  4. Les hyperparamètres d'optimisation")
        
    except Exception as e:
        print(f"❌ Erreur lors du diagnostic: {e}")
        import traceback
        traceback.print_exc()
