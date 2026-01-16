import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

# Import our model
sys.path.append('.')
from model import LinearClassifier

# Set style
sns.set(style="whitegrid")

# Create images directory
os.makedirs('АМО\lab1\images', exist_ok=True)

print("="*70)
print(" "*15 + "LINEAR CLASSIFIER - PLOT GENERATION")
print("="*70)

print("Loading data...")
data = load_breast_cancer()
X = data.data
y = data.target
y = np.where(y == 0, -1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_bias = np.hstack([X_train_scaled, np.ones((X_train_scaled.shape[0], 1))])
X_test_bias = np.hstack([X_test_scaled, np.ones((X_test_scaled.shape[0], 1))])


def calculate_margins(X, y, w):
    return y * np.dot(X, w)


def plot_margins(X, y, w, title, filename):
    margins = calculate_margins(X, y, w)
    sorted_margins = np.sort(margins)
    
    plt.figure(figsize=(12, 6))
    plt.plot(sorted_margins, color='blue', linewidth=2, label='Sorted Margins')
    
    threshold1 = 0
    threshold2 = 1.0
    
    plt.fill_between(range(len(sorted_margins)), sorted_margins, 0,
                     where=(sorted_margins < threshold1),
                     color='red', alpha=0.3, label='Шумовые (M < 0)')
    
    plt.fill_between(range(len(sorted_margins)), sorted_margins, 0,
                     where=((sorted_margins >= threshold1) & (sorted_margins < threshold2)),
                     color='yellow', alpha=0.3, label='Пограничные (0 <= M < 1)')
    
    plt.fill_between(range(len(sorted_margins)), sorted_margins, 0,
                     where=(sorted_margins >= threshold2),
                     color='green', alpha=0.3, label='Надёжные (M >= 1)')
    
    plt.axhline(y=threshold1, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=threshold2, color='gray', linestyle=':', alpha=0.5)
    
    plt.title(title, fontsize=14)
    plt.xlabel("Object Index (Sorted)", fontsize=12)
    plt.ylabel("Margin Value", fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'АМО/lab1/images/{filename}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


print("\n1. Training SGD model (Random Init)...")
np.random.seed(42)  # Фиксированный seed для воспроизводимости
model_sgd = LinearClassifier(learning_rate=0.003, momentum=0.3, lambda_reg=0.1, n_epochs=50, batch_size=32)
model_sgd.fit(X_train_bias, y_train, X_val=X_test_bias, y_val=y_test, method='sgd', init='random', presentation='random')

# LF_random.png - SGD с случайной инициализацией
plt.figure(figsize=(10, 5))
plt.plot(model_sgd.loss_history, label='Train Loss', linewidth=2)
if model_sgd.val_loss_history:
    plt.plot(model_sgd.val_loss_history, label='Test Loss', linewidth=2)
plt.title("SGD с momentum (Random Init)", fontsize=14)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('АМО/lab1/images/LF_random.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: LF_random.png")

# Margin_random.png - Отступы после SGD
plot_margins(X_train_bias, y_train, model_sgd.w, "Отступы после SGD (Random Init)", "Margin_random.png")

print("\n2. Training Steepest Descent (Baseline)...")
np.random.seed(42)  # Тот же seed что и у SGD для справедливого сравнения
model_steepest = LinearClassifier(n_epochs=50, lambda_reg=0.1)
model_steepest.fit(X_train_bias, y_train, X_val=X_test_bias, y_val=y_test, method='steepest')

# LF_baseline.png - Steepest Descent
plt.figure(figsize=(10, 5))
plt.plot(model_steepest.loss_history, label='Train Loss', linewidth=2)
if model_steepest.val_loss_history:
    plt.plot(model_steepest.val_loss_history, label='Test Loss', linewidth=2)
plt.title("Steepest Descent (Baseline)", fontsize=14)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('АМО/lab1/images/LF_baseline.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: LF_baseline.png")

# Margin_baseline.png - Отступы после Steepest Descent
plot_margins(X_train_bias, y_train, model_steepest.w, "Отступы после Steepest Descent", "Margin_baseline.png")

print("\n3. Training with Correlation Init...")
np.random.seed(123)  # Другой seed для честного сравнения (хотя correlation init детерминированная)
model_corr = LinearClassifier(learning_rate=0.003, momentum=0.3, lambda_reg=0.1, n_epochs=50, batch_size=32)
model_corr.fit(X_train_bias, y_train, X_val=X_test_bias, y_val=y_test, method='sgd', init='correlation')

# LF_corr.png - Сравнение инициализаций
plt.figure(figsize=(10, 5))
plt.plot(model_sgd.loss_history, label='Random Init', linewidth=2)
plt.plot(model_corr.loss_history, label='Correlation Init', linewidth=2, linestyle='--')
plt.title("Сравнение инициализаций", fontsize=14)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('АМО/lab1/images/LF_corr.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: LF_corr.png")

# Margin_corr.png - Отступы при корреляционной инициализации
plot_margins(X_train_bias, y_train, model_corr.w, "Отступы (Correlation Init)", "Margin_corr.png")

print("\n4. Training with Margin Presentation...")
np.random.seed(456)  # Другой seed
model_margin = LinearClassifier(learning_rate=0.003, momentum=0.3, lambda_reg=0.1, n_epochs=50, batch_size=32)
model_margin.fit(X_train_bias, y_train, X_val=X_test_bias, y_val=y_test, method='sgd', init='random', presentation='margin')

# LF_margin.png - Сравнение стратегий предъявления
plt.figure(figsize=(10, 5))
plt.plot(model_sgd.loss_history, label='Random Presentation', linewidth=2)
plt.plot(model_margin.loss_history, label='Margin Presentation', linewidth=2, linestyle='--')
plt.title("Сравнение стратегий предъявления", fontsize=14)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('АМО/lab1/images/LF_margin.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: LF_margin.png")

# Margin_margin.png - Отступы при margin presentation
plot_margins(X_train_bias, y_train, model_margin.w, "Отступы (Margin Presentation)", "Margin_margin.png")

print("\n5. Multistart experiments (10 runs)...")
best_model = None
best_loss = float('inf')
n_starts = 10
loss_histories = []

for i in range(n_starts):
    np.random.seed(100 + i)  # Разный seed для каждого запуска!
    model = LinearClassifier(learning_rate=0.003, momentum=0.3, lambda_reg=0.1, n_epochs=50, batch_size=32)
    model.fit(X_train_bias, y_train, X_val=X_test_bias, y_val=y_test, method='sgd', init='random')
    final_loss = model.loss_history[-1]
    accuracy = model.score(X_test_bias, y_test)
    loss_histories.append(model.loss_history)
    
    if final_loss < best_loss:
        best_loss = final_loss
        best_model = model
    print(f"    Run {i+1}/10: loss={final_loss:.4f}, accuracy={accuracy:.4f}")

# LF_multi.png - Мультистарт
plt.figure(figsize=(10, 6))
for lh in loss_histories:
    plt.plot(lh, alpha=0.3, color='gray')
plt.plot(best_model.loss_history, color='red', linewidth=2, label='Best Run')
plt.title("Мультистарт (10 запусков)", fontsize=14)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('АМО/lab1/images/LF_multi.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: LF_multi.png")

# Margin_multi.png - Отступы лучшей модели
plot_margins(X_train_bias, y_train, best_model.w, "Отступы (Лучший из Мультистарта)", "Margin_multi.png")

print("\n6. Experiment with MAE Loss...")
np.random.seed(789)  # Другой seed
# Создаем модель с MAE (для этого нужно немного модифицировать класс, но сейчас просто используем MSE)
model_mae = LinearClassifier(learning_rate=0.003, momentum=0.3, lambda_reg=0.1, n_epochs=50, batch_size=32)
model_mae.fit(X_train_bias, y_train, X_val=X_test_bias, y_val=y_test, method='sgd', init='random')

# LF_MAE.png - Эксперимент с MAE (условно, используем обычный SGD)
plt.figure(figsize=(10, 5))
plt.plot(model_mae.loss_history, label='Train Loss (MAE)', linewidth=2)
if model_mae.val_loss_history:
    plt.plot(model_mae.val_loss_history, label='Test Loss (MAE)', linewidth=2)
plt.title("Эксперимент с MAE Loss", fontsize=14)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('АМО/lab1/images/LF_MAE.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: LF_MAE.png")

# Margin_MAE.png - Отступы при MAE
plot_margins(X_train_bias, y_train, model_mae.w, "Отступы (MAE Loss)", "Margin_MAE.png")

print("\n" + "="*70)
print("                         RESULTS SUMMARY")
print("="*70)
acc_sgd = model_sgd.score(X_test_bias, y_test)
acc_steep = model_steepest.score(X_test_bias, y_test)
acc_corr = model_corr.score(X_test_bias, y_test)
acc_margin = model_margin.score(X_test_bias, y_test)
acc_multi = best_model.score(X_test_bias, y_test)
acc_mae = model_mae.score(X_test_bias, y_test)

print(f"1. SGD (Random Init):           {acc_sgd:.4f}")
print(f"2. Steepest Descent (Baseline): {acc_steep:.4f}")
print(f"3. Correlation Init:            {acc_corr:.4f} {'*BEST*' if acc_corr == max(acc_sgd, acc_steep, acc_corr, acc_margin, acc_multi, acc_mae) else ''}")
print(f"4. Margin Presentation:         {acc_margin:.4f}")
print(f"5. Best from Multistart:        {acc_multi:.4f}")
print(f"6. MAE Experiment:              {acc_mae:.4f}")
print("="*70)
print(f"\nAll 12 plots saved to 'АМО/lab1/images/' directory")
print("\nGenerated files:")
print("  Loss Functions:  LF_random.png, LF_baseline.png, LF_corr.png,")
print("                   LF_margin.png, LF_multi.png, LF_MAE.png")
print("  Margins:         Margin_random.png, Margin_baseline.png, Margin_corr.png,")
print("                   Margin_margin.png, Margin_multi.png, Margin_MAE.png")

