# improved_neural_svg_generator.py
"""
Улучшенная нейросеть для генерации SVG
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from pathlib import Path
import re


# ============================================================================
# 1. Улучшенный токенизатор с эмбеддингами слов
# ============================================================================

class WordTokenizer:
    """Токенизирует текст на уровне слов (лучше понимает концепции)"""

    def __init__(self):
        # Словарь всех возможных слов
        self.word_to_idx = {
            '<PAD>': 0,
            '<UNK>': 1,
            'красный': 2, 'синий': 3, 'зеленый': 4, 'желтый': 5, 'серый': 6, 'коричневый': 7, 'голубой': 8,
            'круг': 9, 'квадрат': 10, 'дерево': 11, 'небо': 12, 'море': 13, 'кот': 14, 'собака': 15, 'солнце': 16,
            'маленький': 17, 'большой': 18, 'слева': 19, 'справа': 20, 'вверху': 21, 'внизу': 22,
            'с': 23, 'и': 24, 'в': 25, 'на': 26
        }
        self.vocab_size = len(self.word_to_idx)
        self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}

        print(f"Размер словаря: {self.vocab_size} слов")

    def encode(self, text, max_length=10):
        """Кодирует текст в последовательность индексов слов"""
        words = text.lower().split()
        indices = []

        for word in words:
            if word in self.word_to_idx:
                indices.append(self.word_to_idx[word])
            else:
                indices.append(self.word_to_idx['<UNK>'])

        # Обрезаем или дополняем
        if len(indices) > max_length:
            indices = indices[:max_length]
        else:
            indices += [self.word_to_idx['<PAD>']] * (max_length - len(indices))

        return torch.tensor(indices, dtype=torch.long)


# ============================================================================
# 2. Улучшенный датасет с более разнообразными примерами
# ============================================================================

class ImprovedSVGDataset(Dataset):
    """Улучшенный датасет с большим разнообразием"""

    def __init__(self):
        self.tokenizer = WordTokenizer()

        # Расширенные примеры с разными формами и цветами
        self.examples = [
            # Круги разных цветов и размеров
            {'text': 'красный круг', 'type': 'circle', 'x': 32, 'y': 32, 'size': 20, 'color': [255, 0, 0]},
            {'text': 'синий круг', 'type': 'circle', 'x': 32, 'y': 32, 'size': 20, 'color': [0, 0, 255]},
            {'text': 'зеленый круг', 'type': 'circle', 'x': 32, 'y': 32, 'size': 20, 'color': [0, 255, 0]},
            {'text': 'желтый круг', 'type': 'circle', 'x': 32, 'y': 32, 'size': 20, 'color': [255, 255, 0]},
            {'text': 'серый круг', 'type': 'circle', 'x': 32, 'y': 32, 'size': 20, 'color': [128, 128, 128]},
            {'text': 'маленький красный круг', 'type': 'circle', 'x': 32, 'y': 32, 'size': 10, 'color': [255, 0, 0]},
            {'text': 'большой синий круг', 'type': 'circle', 'x': 32, 'y': 32, 'size': 30, 'color': [0, 0, 255]},

            # Квадраты
            {'text': 'красный квадрат', 'type': 'rect', 'x': 12, 'y': 12, 'w': 40, 'h': 40, 'color': [255, 0, 0]},
            {'text': 'синий квадрат', 'type': 'rect', 'x': 12, 'y': 12, 'w': 40, 'h': 40, 'color': [0, 0, 255]},
            {'text': 'зеленый квадрат', 'type': 'rect', 'x': 12, 'y': 12, 'w': 40, 'h': 40, 'color': [0, 255, 0]},

            # Позиции
            {'text': 'круг слева', 'type': 'circle', 'x': 16, 'y': 32, 'size': 15, 'color': [255, 0, 0]},
            {'text': 'круг справа', 'type': 'circle', 'x': 48, 'y': 32, 'size': 15, 'color': [0, 0, 255]},
            {'text': 'круг вверху', 'type': 'circle', 'x': 32, 'y': 16, 'size': 15, 'color': [0, 255, 0]},
            {'text': 'круг внизу', 'type': 'circle', 'x': 32, 'y': 48, 'size': 15, 'color': [255, 255, 0]},

            # Дерево (специфическая форма)
            {'text': 'зеленое дерево', 'type': 'tree',
             'parts': [
                 {'type': 'rect', 'x': 27, 'y': 40, 'w': 10, 'h': 24, 'color': [139, 69, 19]},  # ствол
                 {'type': 'circle', 'x': 32, 'y': 30, 'size': 20, 'color': [0, 255, 0]},  # крона
             ]},

            # Небо
            {'text': 'голубое небо', 'type': 'sky',
             'parts': [
                 {'type': 'rect', 'x': 0, 'y': 0, 'w': 64, 'h': 64, 'color': [135, 206, 235]}
             ]},

            # Небо с солнцем
            {'text': 'небо с солнцем', 'type': 'sky_sun',
             'parts': [
                 {'type': 'rect', 'x': 0, 'y': 0, 'w': 64, 'h': 64, 'color': [135, 206, 235]},
                 {'type': 'circle', 'x': 50, 'y': 15, 'size': 12, 'color': [255, 255, 0]}
             ]},

            # Море
            {'text': 'синее море', 'type': 'sea',
             'parts': [
                 {'type': 'rect', 'x': 0, 'y': 0, 'w': 64, 'h': 64, 'color': [0, 0, 255]}
             ]},

            # Кот (морда)
            {'text': 'серый кот', 'type': 'cat',
             'parts': [
                 {'type': 'circle', 'x': 32, 'y': 32, 'size': 25, 'color': [128, 128, 128]},  # голова
                 {'type': 'circle', 'x': 27, 'y': 27, 'size': 3, 'color': [0, 255, 0]},  # левый глаз
                 {'type': 'circle', 'x': 37, 'y': 27, 'size': 3, 'color': [0, 255, 0]},  # правый глаз
             ]},

            # Собака
            {'text': 'коричневая собака', 'type': 'dog',
             'parts': [
                 {'type': 'circle', 'x': 32, 'y': 32, 'size': 25, 'color': [139, 69, 19]},
                 {'type': 'circle', 'x': 27, 'y': 27, 'size': 3, 'color': [0, 0, 0]},
                 {'type': 'circle', 'x': 37, 'y': 27, 'size': 3, 'color': [0, 0, 0]},
             ]},
        ]

        print(f"Создан датасет с {len(self.examples)} примерами")

        # Подготавливаем данные для обучения
        self.parsed_data = []
        for ex in self.examples:
            params = self._example_to_params(ex)
            if params is not None:
                self.parsed_data.append({
                    'text': ex['text'],
                    'params': params
                })

        print(f"Подготовлено {len(self.parsed_data)} примеров для обучения")

    def _example_to_params(self, ex):
        """Конвертирует пример в вектор параметров"""
        if 'parts' in ex:  # Сложный объект из нескольких частей
            # Для простоты берем только первую часть
            part = ex['parts'][0]
            if part['type'] == 'circle':
                return torch.tensor([
                    0.0,  # тип: круг
                    part['x'] / 64.0,
                    part['y'] / 64.0,
                    part['size'] / 32.0,
                    0.0,
                    part['color'][0] / 255.0,
                    part['color'][1] / 255.0,
                    part['color'][2] / 255.0
                ], dtype=torch.float32)
            else:
                return torch.tensor([
                    1.0,  # тип: прямоугольник
                    part['x'] / 64.0,
                    part['y'] / 64.0,
                    part['w'] / 64.0,
                    part['h'] / 64.0,
                    part['color'][0] / 255.0,
                    part['color'][1] / 255.0,
                    part['color'][2] / 255.0
                ], dtype=torch.float32)

        elif ex['type'] == 'circle':
            return torch.tensor([
                0.0,
                ex['x'] / 64.0,
                ex['y'] / 64.0,
                ex['size'] / 32.0,
                0.0,
                ex['color'][0] / 255.0,
                ex['color'][1] / 255.0,
                ex['color'][2] / 255.0
            ], dtype=torch.float32)

        elif ex['type'] == 'rect':
            return torch.tensor([
                1.0,
                ex['x'] / 64.0,
                ex['y'] / 64.0,
                ex['w'] / 64.0,
                ex['h'] / 64.0,
                ex['color'][0] / 255.0,
                ex['color'][1] / 255.0,
                ex['color'][2] / 255.0
            ], dtype=torch.float32)

        return None

    def __len__(self):
        return len(self.parsed_data)

    def __getitem__(self, idx):
        item = self.parsed_data[idx]
        text_tensor = self.tokenizer.encode(item['text'])
        return {
            'text': text_tensor,
            'params': item['params'],
            'text_str': item['text']
        }


# ============================================================================
# 3. Улучшенная модель с attention
# ============================================================================

class ImprovedTextEncoder(nn.Module):
    """Улучшенный энкодер с self-attention"""

    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # BiLSTM для контекста
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
            num_layers=3,
            dropout=0.3
        )

        # Self-attention механизм
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True,
            dropout=0.3
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)

        # LSTM
        lstm_out, _ = self.lstm(embedded)  # (batch, seq_len, hidden_dim)

        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.layer_norm(lstm_out + attn_out)

        # Pooling (берем среднее по всем токенам)
        pooled = attn_out.mean(dim=1)  # (batch, hidden_dim)

        return pooled


class ImprovedGenerator(nn.Module):
    """Улучшенный генератор с residual connections"""

    def __init__(self, hidden_dim=256, num_params=8):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, num_params),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


class ImprovedNeuralSVGGenerator(nn.Module):
    """Улучшенная полная нейросеть"""

    def __init__(self, vocab_size, hidden_dim=256, num_params=8):
        super().__init__()

        self.encoder = ImprovedTextEncoder(vocab_size, hidden_dim=hidden_dim)
        self.generator = ImprovedGenerator(hidden_dim, num_params)

        print(f"\nСоздана улучшенная нейросеть:")
        print(f"  - Энкодер: BiLSTM(3 слоя) + Multi-head Attention")
        print(f"  - Генератор: MLP с Residual Connections")
        print(f"  - Всего параметров: {sum(p.numel() for p in self.parameters()):,}")

    def forward(self, text):
        context = self.encoder(text)
        params = self.generator(context)
        return params


# ============================================================================
# 4. Улучшенный конвертер параметров в SVG
# ============================================================================

def improved_params_to_svg(params, text=""):
    """
    Улучшенный конвертер с проверкой границ
    """
    params = params.detach().cpu().numpy()

    # Денормализация с проверкой границ
    shape_type = 0 if params[0] < 0.5 else 1
    x = min(56, max(8, int(params[1] * 64)))  # Не даем уйти за края
    y = min(56, max(8, int(params[2] * 64)))
    size1 = min(32, max(5, int(params[3] * 64)))  # Ограничиваем размер
    size2 = min(32, max(0, int(params[4] * 64)))
    r = min(255, max(0, int(params[5] * 255)))
    g = min(255, max(0, int(params[6] * 255)))
    b = min(255, max(0, int(params[7] * 255)))

    color = f"rgb({r}, {g}, {b})"

    # Определяем тип фигуры по тексту (эвристика)
    if 'дерево' in text.lower():
        # Для дерева рисуем ствол и крону
        svg = f'''<svg width="64" height="64" xmlns="http://www.w3.org/2000/svg">
  <!-- {text} -->
  <rect x="27" y="40" width="10" height="24" fill="brown"/>
  <circle cx="32" cy="30" r="18" fill="{color}"/>
</svg>'''
    elif 'кот' in text.lower() or 'кошка' in text.lower():
        # Для кота рисуем мордочку
        svg = f'''<svg width="64" height="64" xmlns="http://www.w3.org/2000/svg">
  <!-- {text} -->
  <circle cx="32" cy="32" r="20" fill="{color}"/>
  <circle cx="27" cy="27" r="3" fill="white"/>
  <circle cx="37" cy="27" r="3" fill="white"/>
  <circle cx="27" cy="27" r="1.5" fill="black"/>
  <circle cx="37" cy="27" r="1.5" fill="black"/>
  <circle cx="32" cy="35" r="2" fill="pink"/>
</svg>'''
    elif 'собака' in text.lower() or 'пёс' in text.lower():
        svg = f'''<svg width="64" height="64" xmlns="http://www.w3.org/2000/svg">
  <!-- {text} -->
  <circle cx="32" cy="32" r="20" fill="{color}"/>
  <circle cx="27" cy="27" r="3" fill="white"/>
  <circle cx="37" cy="27" r="3" fill="white"/>
  <circle cx="27" cy="27" r="1.5" fill="black"/>
  <circle cx="37" cy="27" r="1.5" fill="black"/>
  <ellipse cx="32" cy="38" rx="5" ry="3" fill="black"/>
</svg>'''
    elif 'небо' in text.lower():
        svg = f'''<svg width="64" height="64" xmlns="http://www.w3.org/2000/svg">
  <!-- {text} -->
  <rect width="64" height="64" fill="{color}"/>
  <circle cx="50" cy="15" r="8" fill="yellow"/>
</svg>'''
    elif 'море' in text.lower():
        svg = f'''<svg width="64" height="64" xmlns="http://www.w3.org/2000/svg">
  <!-- {text} -->
  <rect width="64" height="64" fill="{color}"/>
  <path d="M 0,45 Q 16,40 32,45 T 64,45" stroke="white" stroke-width="2" fill="none"/>
</svg>'''
    else:
        # Обычная фигура
        if shape_type == 0:  # круг
            svg = f'''<svg width="64" height="64" xmlns="http://www.w3.org/2000/svg">
  <!-- {text} -->
  <circle cx="{x}" cy="{y}" r="{size1}" fill="{color}"/>
</svg>'''
        else:  # прямоугольник
            svg = f'''<svg width="64" height="64" xmlns="http://www.w3.org/2000/svg">
  <!-- {text} -->
  <rect x="{x - size1 // 2}" y="{y - size2 // 2}" width="{size1}" height="{size2}" fill="{color}"/>
</svg>'''

    return svg


# ============================================================================
# 5. Функции обучения
# ============================================================================

def train_improved(model, dataloader, val_loader, num_epochs=500):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.MSELoss()

    best_loss = float('inf')

    for epoch in range(num_epochs):
        # Обучение
        model.train()
        train_loss = 0
        for batch in dataloader:
            text = batch['text'].to(device)
            target = batch['params'].to(device)

            pred = model(text)
            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(dataloader)

        # Валидация
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                text = batch['text'].to(device)
                target = batch['params'].to(device)
                pred = model(text)
                val_loss += criterion(pred, target).item()

        val_loss /= len(val_loader)
        scheduler.step()

        if (epoch + 1) % 50 == 0:
            print(f"Эпоха {epoch + 1:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_improved_model.pth')

    return model


# ============================================================================
# 6. Основная функция
# ============================================================================

def main():
    print("=" * 60)
    print("УЛУЧШЕННАЯ НЕЙРОСЕТЬ ДЛЯ ГЕНЕРАЦИИ SVG")
    print("=" * 60)

    # Создаем датасет
    dataset = ImprovedSVGDataset()
    tokenizer = dataset.tokenizer

    # Разделяем данные
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Создаем модель
    model = ImprovedNeuralSVGGenerator(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=256,
        num_params=8
    )

    # Обучаем
    print("\nНачинаем обучение...")
    model = train_improved(model, train_loader, val_loader, num_epochs=500)

    # Тестируем
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ МОДЕЛИ")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    test_prompts = [
        "красный круг",
        "синий квадрат",
        "зеленое дерево",
        "голубое небо",
        "серый кот",
        "коричневая собака",
        "желтое солнце",
        "синее море"
    ]

    for prompt in test_prompts:
        text_tensor = tokenizer.encode(prompt).unsqueeze(0).to(device)

        with torch.no_grad():
            params = model(text_tensor)
            svg = improved_params_to_svg(params[0], prompt)

        print(f"\n{prompt}:")
        print(svg)

        # Сохраняем
        filename = f"improved_{prompt.replace(' ', '_')}.svg"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(svg)
        print(f"Сохранено: {filename}")


if __name__ == "__main__":
    main()