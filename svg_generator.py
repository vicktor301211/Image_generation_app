# train_svg_model.py
"""
Полный код обучения нейросети для генерации SVG
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import random
from pathlib import Path
from typing import List, Tuple
import math


# ============================================================================
# 1. Создание датасета (если его нет)
# ============================================================================

def create_training_dataset():
    """Создает тренировочный датасет"""

    dataset = {
        "name": "SVG Training Dataset",
        "samples": [
            # Деревья
            {
                "id": "tree_01",
                "category": "tree",
                "prompt": "зеленое дерево с коричневым стволом",
                "svg": '''<svg width="64" height="64">
  <rect x="27" y="40" width="10" height="24" fill="#8B4513"/>
  <circle cx="32" cy="30" r="20" fill="#228B22"/>
</svg>'''
            },
            {
                "id": "tree_02",
                "category": "tree",
                "prompt": "дерево с пышной кроной",
                "svg": '''<svg width="64" height="64">
  <rect x="29" y="35" width="6" height="29" fill="#654321"/>
  <circle cx="32" cy="25" r="15" fill="#2E8B57"/>
  <circle cx="20" cy="30" r="12" fill="#388E3C"/>
  <circle cx="44" cy="30" r="12" fill="#388E3C"/>
  <circle cx="32" cy="15" r="10" fill="#4CAF50"/>
</svg>'''
            },
            # Небо
            {
                "id": "sky_01",
                "category": "sky",
                "prompt": "голубое небо с облаками",
                "svg": '''<svg width="64" height="64">
  <rect width="64" height="64" fill="#87CEEB"/>
  <circle cx="20" cy="30" r="8" fill="#FFFFFF"/>
  <circle cx="30" cy="25" r="10" fill="#FFFFFF"/>
  <circle cx="40" cy="30" r="8" fill="#FFFFFF"/>
  <circle cx="50" cy="40" r="6" fill="#FFFFFF"/>
</svg>'''
            },
            {
                "id": "sky_02",
                "category": "sky",
                "prompt": "небо с солнцем",
                "svg": '''<svg width="64" height="64">
  <rect width="64" height="64" fill="#1E90FF"/>
  <circle cx="50" cy="15" r="12" fill="#FFD700"/>
  <circle cx="20" cy="45" r="5" fill="#FFFFFF"/>
  <circle cx="40" cy="50" r="7" fill="#FFFFFF"/>
</svg>'''
            },
            # Море
            {
                "id": "sea_01",
                "category": "sea",
                "prompt": "синее море с волной",
                "svg": '''<svg width="64" height="64">
  <rect width="64" height="64" fill="#1E90FF"/>
  <path d="M 0,40 Q 16,30 32,40 T 64,40" stroke="#FFFFFF" stroke-width="3" fill="none"/>
  <path d="M 0,50 Q 20,45 40,50 T 64,50" stroke="#FFFFFF" stroke-width="2" fill="none"/>
</svg>'''
            },
            # Коты
            {
                "id": "cat_01",
                "category": "cat",
                "prompt": "серый кот",
                "svg": '''<svg width="64" height="64">
  <ellipse cx="32" cy="40" rx="18" ry="15" fill="#808080"/>
  <circle cx="32" cy="25" r="12" fill="#808080"/>
  <circle cx="27" cy="23" r="3" fill="#32CD32"/>
  <circle cx="37" cy="23" r="3" fill="#32CD32"/>
  <circle cx="32" cy="28" r="2" fill="#FF69B4"/>
</svg>'''
            },
            {
                "id": "cat_02",
                "category": "cat",
                "prompt": "рыжий кот с ушами",
                "svg": '''<svg width="64" height="64">
  <ellipse cx="32" cy="40" rx="20" ry="16" fill="#D2691E"/>
  <circle cx="32" cy="25" r="14" fill="#D2691E"/>
  <polygon points="25,15 20,25 30,20" fill="#D2691E"/>
  <polygon points="39,15 44,25 34,20" fill="#D2691E"/>
  <circle cx="28" cy="24" r="3" fill="#00FF00"/>
  <circle cx="36" cy="24" r="3" fill="#00FF00"/>
</svg>'''
            },
            # Собаки
            {
                "id": "dog_01",
                "category": "dog",
                "prompt": "коричневая собака",
                "svg": '''<svg width="64" height="64">
  <rect x="22" y="35" width="20" height="15" rx="5" fill="#8B4513"/>
  <circle cx="32" cy="28" r="10" fill="#8B4513"/>
  <circle cx="28" cy="26" r="2" fill="#000000"/>
  <circle cx="36" cy="26" r="2" fill="#000000"/>
  <circle cx="32" cy="30" r="1.5" fill="#000000"/>
</svg>'''
            },
            {
                "id": "dog_02",
                "category": "dog",
                "prompt": "собака с виляющим хвостом",
                "svg": '''<svg width="64" height="64">
  <rect x="20" y="35" width="24" height="18" rx="6" fill="#A0522D"/>
  <circle cx="32" cy="28" r="12" fill="#A0522D"/>
  <circle cx="27" cy="26" r="2.5" fill="#000000"/>
  <circle cx="37" cy="26" r="2.5" fill="#000000"/>
  <path d="M 44,40 Q 54,30 58,45" stroke="#A0522D" stroke-width="4" fill="none"/>
</svg>'''
            }
        ]
    }

    # Сохраняем датасет
    with open("training_dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"Создан тренировочный датасет: {len(dataset['samples'])} примеров")
    return dataset


# ============================================================================
# 2. Парсинг SVG и подготовка данных
# ============================================================================

class SVGTokenizer:
    """Токенизатор для SVG команд"""

    def __init__(self):
        # Словарь команд
        self.commands = {
            'rect': 0,
            'circle': 1,
            'ellipse': 2,
            'polygon': 3,
            'path': 4,
            'start_svg': 5,
            'end_svg': 6
        }

        # Цвета (упрощенно)
        self.colors = {
            '#8B4513': 0,  # коричневый
            '#228B22': 1,  # зеленый
            '#654321': 2,  # темно-коричневый
            '#2E8B57': 3,  # морская волна
            '#388E3C': 4,  # зеленый
            '#4CAF50': 5,  # светлый зеленый
            '#87CEEB': 6,  # голубой
            '#FFFFFF': 7,  # белый
            '#1E90FF': 8,  # синий
            '#FFD700': 9,  # золотой
            '#00008B': 10,  # темно-синий
            '#808080': 11,  # серый
            '#32CD32': 12,  # лаймовый
            '#FF69B4': 13,  # розовый
            '#D2691E': 14,  # шоколадный
            '#00FF00': 15,  # зеленый
            '#A0522D': 16,  # сиена
            '#000000': 17  # черный
        }

        self.reverse_commands = {v: k for k, v in self.commands.items()}
        self.reverse_colors = {v: k for k, v in self.colors.items()}

    def parse_svg(self, svg_text: str) -> List[List[int]]:
        """Парсит SVG в последовательность токенов"""
        tokens = []

        # Добавляем начало
        tokens.append([self.commands['start_svg'], 0, 0, 0, 0, 0, 0])

        # Простой парсинг (в реальности нужен XML парсер)
        lines = svg_text.strip().split('\n')

        for line in lines:
            line = line.strip()

            if '<rect' in line:
                # Парсим rect
                x = self._extract_number(line, 'x="', '"')
                y = self._extract_number(line, 'y="', '"')
                w = self._extract_number(line, 'width="', '"')
                h = self._extract_number(line, 'height="', '"')
                fill = self._extract_color(line)

                tokens.append([
                    self.commands['rect'],
                    int(x), int(y), int(w), int(h),
                    self.colors.get(fill, 0),
                    0
                ])

            elif '<circle' in line:
                # Парсим circle
                cx = self._extract_number(line, 'cx="', '"')
                cy = self._extract_number(line, 'cy="', '"')
                r = self._extract_number(line, 'r="', '"')
                fill = self._extract_color(line)

                tokens.append([
                    self.commands['circle'],
                    int(cx), int(cy), int(r), 0,
                    self.colors.get(fill, 0),
                    0
                ])

            elif '<ellipse' in line:
                # Парсим ellipse
                cx = self._extract_number(line, 'cx="', '"')
                cy = self._extract_number(line, 'cy="', '"')
                rx = self._extract_number(line, 'rx="', '"')
                ry = self._extract_number(line, 'ry="', '"')
                fill = self._extract_color(line)

                tokens.append([
                    self.commands['ellipse'],
                    int(cx), int(cy), int(rx), int(ry),
                    self.colors.get(fill, 0),
                    0
                ])

        # Добавляем конец
        tokens.append([self.commands['end_svg'], 0, 0, 0, 0, 0, 0])

        return tokens

    def _extract_number(self, text: str, start: str, end: str) -> float:
        """Извлекает число из строки"""
        try:
            start_idx = text.find(start)
            if start_idx == -1:
                return 0
            start_idx += len(start)
            end_idx = text.find(end, start_idx)
            num_str = text[start_idx:end_idx]
            return float(num_str)
        except:
            return 0

    def _extract_color(self, text: str) -> str:
        """Извлекает цвет из строки"""
        try:
            start_idx = text.find('fill="')
            if start_idx == -1:
                return "#000000"
            start_idx += 6
            end_idx = text.find('"', start_idx)
            return text[start_idx:end_idx]
        except:
            return "#000000"

    def tokens_to_svg(self, tokens: List[List[int]]) -> str:
        """Конвертирует токены обратно в SVG"""
        svg_lines = ['<svg width="64" height="64">']

        for token in tokens:
            cmd_idx = token[0]

            if cmd_idx == self.commands['rect']:
                x, y, w, h, color_idx, _ = token[1:7]
                color = self.reverse_colors.get(color_idx, "#000000")
                svg_lines.append(f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{color}"/>')

            elif cmd_idx == self.commands['circle']:
                cx, cy, r, _, color_idx, _ = token[1:7]
                color = self.reverse_colors.get(color_idx, "#000000")
                svg_lines.append(f'  <circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}"/>')

            elif cmd_idx == self.commands['ellipse']:
                cx, cy, rx, ry, color_idx, _ = token[1:7]
                color = self.reverse_colors.get(color_idx, "#000000")
                svg_lines.append(f'  <ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="{color}"/>')

        svg_lines.append('</svg>')
        return '\n'.join(svg_lines)


# ============================================================================
# 3. Датасет для PyTorch
# ============================================================================

class SVGDataset(Dataset):
    """Датасет SVG изображений"""

    def __init__(self, dataset_file="training_dataset.json"):
        # Загружаем или создаем датасет
        if Path(dataset_file).exists():
            with open(dataset_file, 'r', encoding='utf-8') as f:
                self.dataset = json.load(f)['samples']
        else:
            print("Датасет не найден, создаем новый...")
            dataset = create_training_dataset()
            self.dataset = dataset['samples']

        self.tokenizer = SVGTokenizer()
        self.max_seq_len = 20  # Максимальная длина последовательности команд

        # Создаем кодировку промптов
        self.prompt_vocab = self._create_prompt_vocab()
        self.prompt_to_idx = {word: i for i, word in enumerate(self.prompt_vocab)}

        print(f"Загружено {len(self.dataset)} примеров")
        print(f"Размер словаря промптов: {len(self.prompt_vocab)}")

    def _create_prompt_vocab(self):
        """Создает словарь из всех слов в промптах"""
        all_words = set()
        for sample in self.dataset:
            words = sample['prompt'].lower().split()
            all_words.update(words)

        # Добавляем специальные токены
        vocab = ['<PAD>', '<SOS>', '<EOS>', '<UNK>'] + sorted(list(all_words))
        return vocab

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """Кодирует текстовый промпт в тензор"""
        words = prompt.lower().split()
        indices = [self.prompt_to_idx.get(word, self.prompt_to_idx['<UNK>']) for word in words]

        # Ограничиваем длину и добавляем специальные токены
        indices = [self.prompt_to_idx['<SOS>']] + indices[:10] + [self.prompt_to_idx['<EOS>']]

        # Дополняем до фиксированной длины
        while len(indices) < 15:
            indices.append(self.prompt_to_idx['<PAD>'])

        return torch.tensor(indices[:15], dtype=torch.long)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        # Кодируем промпт
        prompt_tensor = self.encode_prompt(sample['prompt'])

        # Парсим SVG
        tokens = self.tokenizer.parse_svg(sample['svg'])

        # Создаем входную и целевую последовательности
        input_seq = []
        target_seq = []

        for i in range(min(len(tokens), self.max_seq_len - 1)):
            input_seq.append(tokens[i])
            target_seq.append(tokens[i + 1] if i + 1 < len(tokens) else [0] * 7)

        # Дополняем последовательности
        while len(input_seq) < self.max_seq_len:
            input_seq.append([0] * 7)
            target_seq.append([0] * 7)

        # Конвертируем в тензоры
        input_tensor = torch.tensor(input_seq[:self.max_seq_len], dtype=torch.float32)
        target_tensor = torch.tensor(target_seq[:self.max_seq_len], dtype=torch.float32)

        return {
            'prompt': prompt_tensor,
            'input_seq': input_tensor,
            'target_seq': target_tensor,
            'original_svg': sample['svg'],
            'category': sample['category']
        }


# ============================================================================
# 4. Модель нейросети
# ============================================================================

class SVGGeneratorModel(nn.Module):
    """Модель для генерации SVG по текстовому описанию"""

    def __init__(self, prompt_vocab_size, hidden_size=128, num_params=7):
        super(SVGGeneratorModel, self).__init__()

        # Энкодер для текста (промпта)
        self.prompt_embedding = nn.Embedding(prompt_vocab_size, 32)
        self.prompt_encoder = nn.LSTM(
            input_size=32,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=False,  # Упрощаем: убираем bidirectional
            num_layers=1
        )

        # Энкодер для последовательности команд
        self.command_encoder = nn.Linear(num_params, hidden_size)

        # Декодер (LSTM)
        # Исправлено: input_size = hidden_size + hidden_size = 256
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size + hidden_size,  # command_emb + prompt_context
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        # Головы для предсказания
        self.command_head = nn.Linear(hidden_size, 7)  # 7 типов команд
        self.param_head = nn.Linear(hidden_size, 6)  # 6 параметров (5 чисел + цвет)

        self.hidden_size = hidden_size
        self.num_params = num_params

    def forward(self, prompt, input_seq):
        """
        Args:
            prompt: (batch_size, prompt_len) - индексы слов промпта
            input_seq: (batch_size, seq_len, num_params) - входная последовательность команд
        """
        batch_size = prompt.size(0)
        seq_len = input_seq.size(1)

        # 1. Кодируем промпт
        prompt_emb = self.prompt_embedding(prompt)  # (batch, prompt_len, 32)

        # Получаем последний hidden state из LSTM
        _, (hidden, _) = self.prompt_encoder(prompt_emb)
        prompt_context = hidden[-1]  # (batch, hidden_size)

        # Расширяем для конкатенации с каждым шагом последовательности
        prompt_context = prompt_context.unsqueeze(1)  # (batch, 1, hidden)
        prompt_context = prompt_context.expand(-1, seq_len, -1)  # (batch, seq_len, hidden)

        # 2. Кодируем входную последовательность команд
        command_emb = self.command_encoder(input_seq)  # (batch, seq_len, hidden)

        # 3. Объединяем промпт и команды
        combined = torch.cat([command_emb, prompt_context], dim=-1)  # (batch, seq_len, hidden*2)

        # Отладочный вывод
        print(f"DEBUG: command_emb shape = {command_emb.shape}")
        print(f"DEBUG: prompt_context shape = {prompt_context.shape}")
        print(f"DEBUG: combined shape = {combined.shape}")
        print(f"DEBUG: LSTM expects input_size = {self.decoder_lstm.input_size}")

        # 4. Декодируем
        lstm_out, _ = self.decoder_lstm(combined)

        # 5. Предсказываем следующую команду и параметры
        command_pred = self.command_head(lstm_out)  # (batch, seq_len, 7)
        param_pred = self.param_head(lstm_out)  # (batch, seq_len, 6)

        return command_pred, param_pred


# ============================================================================
# 5. Функции обучения
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Одна эпоха обучения"""
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(dataloader):
        # Перемещаем данные на устройство
        prompt = batch['prompt'].to(device)
        input_seq = batch['input_seq'].to(device)
        target_seq = batch['target_seq'].to(device)

        # Forward pass
        pred_commands, pred_params = model(prompt, input_seq)

        # Разделяем целевые значения
        target_commands = target_seq[:, :, 0].long()  # Индексы команд
        target_params = target_seq[:, :, 1:]  # Параметры

        # Вычисляем loss
        command_loss = criterion[0](pred_commands.reshape(-1, pred_commands.size(-1)),
                                    target_commands.reshape(-1))
        param_loss = criterion[1](pred_params, target_params)

        loss = command_loss + param_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}, Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device, tokenizer):
    """Валидация модели"""
    model.eval()
    total_loss = 0
    samples = []

    # Ограничиваем количество батчей для валидации
    max_batches = 2
    batch_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_count >= max_batches:
                break

            prompt = batch['prompt'].to(device)
            input_seq = batch['input_seq'].to(device)
            target_seq = batch['target_seq'].to(device)

            # Forward pass
            pred_commands, pred_params = model(prompt, input_seq)

            # Разделяем целевые значения
            target_commands = target_seq[:, :, 0].long()
            target_params = target_seq[:, :, 1:]

            # Вычисляем loss
            command_loss = criterion[0](pred_commands.reshape(-1, pred_commands.size(-1)),
                                        target_commands.reshape(-1))
            param_loss = criterion[1](pred_params, target_params)

            loss = command_loss + param_loss
            total_loss += loss.item()

            # Сохраняем пример генерации (только для первого батча)
            if batch_idx == 0:
                # Конвертируем предсказания в SVG
                pred_tokens = []
                for i in range(pred_commands.size(1)):
                    cmd_idx = torch.argmax(pred_commands[0, i]).item()
                    params = [int(round(p.item())) for p in pred_params[0, i]]
                    pred_tokens.append([cmd_idx] + params[:6])

                generated_svg = tokenizer.tokens_to_svg(pred_tokens)
                samples.append({
                    'prompt': batch['prompt'][0],
                    'generated': generated_svg,
                    'original': batch['original_svg'][0]
                })

            batch_count += 1

    return total_loss / batch_count, samples


def generate_svg(model, prompt_text, tokenizer, prompt_to_idx, device, max_len=15):
    """Генерация SVG по промпту"""
    model.eval()

    # Подготовка промпта
    words = prompt_text.lower().split()
    indices = [prompt_to_idx.get(word, prompt_to_idx['<UNK>']) for word in words]
    indices = [prompt_to_idx['<SOS>']] + indices[:10] + [prompt_to_idx['<EOS>']]
    while len(indices) < 15:
        indices.append(prompt_to_idx['<PAD>'])

    prompt_tensor = torch.tensor(indices[:15], dtype=torch.long).unsqueeze(0).to(device)

    # Начальная последовательность (только start token)
    start_token = [[tokenizer.commands['start_svg'], 0, 0, 0, 0, 0, 0]]
    current_seq = torch.tensor(start_token, dtype=torch.float32).unsqueeze(0).to(device)

    generated_tokens = []

    with torch.no_grad():
        for step in range(max_len):
            # Предсказываем следующую команду
            pred_commands, pred_params = model(prompt_tensor, current_seq)

            # Берем последнее предсказание
            last_command = pred_commands[0, -1:]
            last_params = pred_params[0, -1:]

            # Выбираем команду с максимальной вероятностью
            cmd_idx = torch.argmax(last_command, dim=-1).item()

            # Округляем параметры
            params = [int(round(p.item())) for p in last_params[0]]

            # Добавляем в последовательность
            generated_tokens.append([cmd_idx] + params[:6])

            # Обновляем текущую последовательность
            new_token = torch.tensor([[cmd_idx] + params[:6]], dtype=torch.float32).to(device)
            current_seq = torch.cat([current_seq, new_token.unsqueeze(0)], dim=1)

            # Если предсказали конец, останавливаемся
            if cmd_idx == tokenizer.commands['end_svg']:
                break

    # Конвертируем в SVG
    svg_code = tokenizer.tokens_to_svg(generated_tokens)
    return svg_code


# ============================================================================
# 6. Основная функция обучения
# ============================================================================

def main():
    """Основная функция обучения"""
    # Настройки
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используемое устройство: {device}")

    # Создаем датасет
    dataset = SVGDataset()

    # Разделяем на train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Создаем DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0
    )

    # Создаем модель
    model = SVGGeneratorModel(
        prompt_vocab_size=len(dataset.prompt_vocab),
        hidden_size=128,
        num_params=7
    ).to(device)

    # Оптимизатор и loss функции
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    criterion = [
        nn.CrossEntropyLoss(),  # Для команд
        nn.MSELoss()  # Для параметров
    ]

    # Обучение
    num_epochs = 50
    best_val_loss = float('inf')

    print("\nНачинаем обучение...")
    print(f"Размер тренировочного датасета: {train_size}")
    print(f"Размер валидационного датасета: {val_size}")
    print(f"Количество эпох: {num_epochs}")
    print("-" * 50)

    for epoch in range(num_epochs):
        print(f"\nЭпоха {epoch + 1}/{num_epochs}")

        # Обучение
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Train Loss: {train_loss:.4f}")

        # Валидация
        val_loss, val_samples = validate(model, val_loader, criterion, device, dataset.tokenizer)
        print(f"Val Loss: {val_loss:.4f}")

        # Сохраняем лучшую модель
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'prompt_vocab': dataset.prompt_vocab,
                'prompt_to_idx': dataset.prompt_to_idx
            }, 'best_svg_model.pth')
            print(f"  ✓ Сохранена лучшая модель (loss: {val_loss:.4f})")

        # Показываем примеры генерации каждые 10 эпох
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print("\nПримеры генерации:")

            test_prompts = [
                "зеленое дерево",
                "голубое небо",
                "серый кот",
                "коричневая собака"
            ]

            for prompt in test_prompts:
                try:
                    svg = generate_svg(
                        model,
                        prompt,
                        dataset.tokenizer,
                        dataset.prompt_to_idx,
                        device
                    )
                    print(f"\nПромпт: '{prompt}'")
                    print(f"Сгенерированный SVG (первые 3 строки):")
                    print('\n'.join(svg.split('\n')[:3]) + "...")
                except Exception as e:
                    print(f"Ошибка генерации для '{prompt}': {e}")

    print("\n" + "=" * 50)
    print("Обучение завершено!")
    print(f"Лучшая валидационная ошибка: {best_val_loss:.4f}")
    print(f"Модель сохранена в: best_svg_model.pth")

    # Финальные примеры
    print("\nФинальные примеры генерации:")
    print("-" * 30)

    # Загружаем лучшую модель для демонстрации
    checkpoint = torch.load('best_svg_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    final_prompts = [
        "дерево с листьями",
        "небо с облаками",
        "рыжий кот",
        "собака с хвостом"
    ]

    for prompt in final_prompts:
        svg = generate_svg(
            model,
            prompt,
            dataset.tokenizer,
            dataset.prompt_to_idx,
            device
        )

        # Сохраняем SVG в файл
        filename = f"generated_{prompt.replace(' ', '_')}.svg"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(svg)

        print(f"\nПромпт: '{prompt}'")
        print(f"Сохранено в: {filename}")
        print("SVG:")
        print(svg[:200] + "..." if len(svg) > 200 else svg)


# ============================================================================
# 7. Скрипт для использования обученной модели
# ============================================================================

def generate_from_model():
    """Генерация SVG с использованием обученной модели"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Загружаем чекпоинт
    checkpoint = torch.load('best_svg_model.pth', map_location=device)

    # Восстанавливаем tokenizer и словари
    tokenizer = SVGTokenizer()

    # Создаем модель
    model = SVGGeneratorModel(
        prompt_vocab_size=len(checkpoint['prompt_vocab']),
        hidden_size=128,
        num_params=7
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Интерактивная генерация
    print("SVG Generator - обученная модель")
    print("Введите промпт (или 'quit' для выхода):")
    print("-" * 40)

    while True:
        prompt = input("\nПромпт: ").strip()

        if prompt.lower() in ['quit', 'exit', 'выход']:
            break

        if not prompt:
            print("Введите промпт!")
            continue

        try:
            # Генерация
            svg = generate_svg(
                model,
                prompt,
                tokenizer,
                checkpoint['prompt_to_idx'],
                device
            )

            # Сохраняем в файл
            filename = f"generated_{prompt[:20].replace(' ', '_')}.svg"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(svg)

            print(f"✓ SVG сгенерирован и сохранен в: {filename}")
            print("\nСодержимое:")
            print(svg)

            # Показываем превью (если есть браузер)
            try:
                import webbrowser
                import tempfile

                with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
                    f.write(f"<html><body><h1>{prompt}</h1>{svg}</body></html>")
                    temp_file = f.name

                webbrowser.open(f'file://{temp_file}')
            except:
                pass

        except Exception as e:
            print(f"Ошибка генерации: {e}")


# ============================================================================
# Запуск
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        generate_from_model()
    else:
        main()