# train_svg_model_simple.py
"""
Упрощенная версия обучения нейросети для генерации SVG
Без аргументов командной строки, только с input() меню
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from pathlib import Path
from typing import List, Optional
import os
import sys


# ============================================================================
# 1. Создание датасета
# ============================================================================

def create_training_dataset():
    """Создает тренировочный датасет с 10 примерами"""

    print("\nСоздание нового датасета...")

    dataset = {
        "name": "Simple SVG Dataset",
        "samples": [
            # 1. Дерево
            {
                "id": "tree_01",
                "category": "tree",
                "prompt": "зеленое дерево",
                "svg": '''<svg width="64" height="64">
  <rect x="27" y="40" width="10" height="24" fill="#8B4513"/>
  <circle cx="32" cy="30" r="20" fill="#228B22"/>
</svg>'''
            },
            # 2. Небо с облаками
            {
                "id": "sky_01",
                "category": "sky",
                "prompt": "голубое небо с облаками",
                "svg": '''<svg width="64" height="64">
  <rect width="64" height="64" fill="#87CEEB"/>
  <circle cx="20" cy="30" r="8" fill="#FFFFFF"/>
  <circle cx="30" cy="25" r="10" fill="#FFFFFF"/>
  <circle cx="40" cy="30" r="8" fill="#FFFFFF"/>
</svg>'''
            },
            # 3. Море
            {
                "id": "sea_01",
                "category": "sea",
                "prompt": "синее море",
                "svg": '''<svg width="64" height="64">
  <rect width="64" height="64" fill="#1E90FF"/>
  <path d="M 0,40 Q 16,30 32,40 T 64,40" stroke="#FFFFFF" stroke-width="3" fill="none"/>
</svg>'''
            },
            # 4. Серый кот
            {
                "id": "cat_01",
                "category": "cat",
                "prompt": "серый кот",
                "svg": '''<svg width="64" height="64">
  <ellipse cx="32" cy="40" rx="18" ry="15" fill="#808080"/>
  <circle cx="32" cy="25" r="12" fill="#808080"/>
  <circle cx="27" cy="23" r="3" fill="#32CD32"/>
  <circle cx="37" cy="23" r="3" fill="#32CD32"/>
</svg>'''
            },
            # 5. Рыжий кот
            {
                "id": "cat_02",
                "category": "cat",
                "prompt": "рыжий кот",
                "svg": '''<svg width="64" height="64">
  <ellipse cx="32" cy="40" rx="20" ry="16" fill="#D2691E"/>
  <circle cx="32" cy="25" r="14" fill="#D2691E"/>
  <polygon points="25,15 20,25 30,20" fill="#D2691E"/>
  <polygon points="39,15 44,25 34,20" fill="#D2691E"/>
</svg>'''
            },
            # 6. Коричневая собака
            {
                "id": "dog_01",
                "category": "dog",
                "prompt": "коричневая собака",
                "svg": '''<svg width="64" height="64">
  <rect x="22" y="35" width="20" height="15" rx="5" fill="#8B4513"/>
  <circle cx="32" cy="28" r="10" fill="#8B4513"/>
  <circle cx="28" cy="26" r="2" fill="#000000"/>
  <circle cx="36" cy="26" r="2" fill="#000000"/>
</svg>'''
            },
            # 7. Домик
            {
                "id": "house_01",
                "category": "house",
                "prompt": "домик с крышей",
                "svg": '''<svg width="64" height="64">
  <rect x="20" y="30" width="30" height="25" fill="#D2B48C"/>
  <polygon points="20,30 50,30 35,15" fill="#8B4513"/>
  <rect x="30" y="40" width="10" height="15" fill="#8B4513"/>
</svg>'''
            },
            # 8. Солнце
            {
                "id": "sun_01",
                "category": "sun",
                "prompt": "желтое солнце",
                "svg": '''<svg width="64" height="64">
  <circle cx="32" cy="32" r="20" fill="#FFD700"/>
  <line x1="32" y1="8" x2="32" y2="16" stroke="#FFD700" stroke-width="2"/>
  <line x1="32" y1="48" x2="32" y2="56" stroke="#FFD700" stroke-width="2"/>
  <line x1="8" y1="32" x2="16" y2="32" stroke="#FFD700" stroke-width="2"/>
  <line x1="48" y1="32" x2="56" y2="32" stroke="#FFD700" stroke-width="2"/>
</svg>'''
            },
            # 9. Цветок
            {
                "id": "flower_01",
                "category": "flower",
                "prompt": "красный цветок",
                "svg": '''<svg width="64" height="64">
  <circle cx="32" cy="32" r="10" fill="#FF4500"/>
  <circle cx="22" cy="22" r="8" fill="#FF4500"/>
  <circle cx="42" cy="22" r="8" fill="#FF4500"/>
  <circle cx="22" cy="42" r="8" fill="#FF4500"/>
  <circle cx="42" cy="42" r="8" fill="#FF4500"/>
  <circle cx="32" cy="52" r="8" fill="#228B22"/>
</svg>'''
            },
            # 10. Машина
            {
                "id": "car_01",
                "category": "car",
                "prompt": "красная машина",
                "svg": '''<svg width="64" height="64">
  <rect x="10" y="35" width="44" height="15" rx="5" fill="#FF0000"/>
  <rect x="15" y="25" width="20" height="15" fill="#FF0000"/>
  <circle cx="20" cy="50" r="5" fill="#000000"/>
  <circle cx="44" cy="50" r="5" fill="#000000"/>
</svg>'''
            }
        ]
    }

    # Сохраняем датасет
    with open("training_dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"✓ Создан датасет с {len(dataset['samples'])} примерами")
    return dataset


# ============================================================================
# 2. Токенизатор SVG
# ============================================================================

class SVGTokenizer:
    """Токенизатор для SVG команд"""

    def __init__(self):
        self.commands = {
            'rect': 0, 'circle': 1, 'ellipse': 2,
            'polygon': 3, 'path': 4, 'line': 5,
            'start_svg': 6, 'end_svg': 7
        }

        self.colors = {
            '#8B4513': 0, '#228B22': 1, '#87CEEB': 2, '#FFFFFF': 3,
            '#1E90FF': 4, '#808080': 5, '#D2691E': 6, '#000000': 7,
            '#D2B48C': 8, '#FFD700': 9, '#FF4500': 10, '#FF0000': 11,
            '#32CD32': 12
        }

        self.reverse_commands = {v: k for k, v in self.commands.items()}
        self.reverse_colors = {v: k for k, v in self.colors.items()}

    def parse_svg(self, svg_text: str):
        """Парсит SVG в последовательность токенов"""
        tokens = [[self.commands['start_svg'], 0, 0, 0, 0, 0, 0]]

        for line in svg_text.strip().split('\n'):
            line = line.strip()

            if '<rect' in line:
                tokens.append(self._parse_rect(line))
            elif '<circle' in line:
                tokens.append(self._parse_circle(line))
            elif '<ellipse' in line:
                tokens.append(self._parse_ellipse(line))

        tokens.append([self.commands['end_svg'], 0, 0, 0, 0, 0, 0])
        return tokens

    def _parse_rect(self, line):
        x = self._extract_number(line, 'x="', '"')
        y = self._extract_number(line, 'y="', '"')
        w = self._extract_number(line, 'width="', '"')
        h = self._extract_number(line, 'height="', '"')
        fill = self._extract_color(line)
        return [self.commands['rect'], int(x), int(y), int(w), int(h),
                self.colors.get(fill, 0), 0]

    def _parse_circle(self, line):
        cx = self._extract_number(line, 'cx="', '"')
        cy = self._extract_number(line, 'cy="', '"')
        r = self._extract_number(line, 'r="', '"')
        fill = self._extract_color(line)
        return [self.commands['circle'], int(cx), int(cy), int(r), 0,
                self.colors.get(fill, 0), 0]

    def _parse_ellipse(self, line):
        cx = self._extract_number(line, 'cx="', '"')
        cy = self._extract_number(line, 'cy="', '"')
        rx = self._extract_number(line, 'rx="', '"')
        ry = self._extract_number(line, 'ry="', '"')
        fill = self._extract_color(line)
        return [self.commands['ellipse'], int(cx), int(cy), int(rx), int(ry),
                self.colors.get(fill, 0), 0]

    def _extract_number(self, text, start, end):
        try:
            start_idx = text.find(start)
            if start_idx == -1: return 0
            start_idx += len(start)
            end_idx = text.find(end, start_idx)
            return float(text[start_idx:end_idx])
        except:
            return 0

    def _extract_color(self, text):
        try:
            start_idx = text.find('fill="')
            if start_idx == -1: return "#000000"
            start_idx += 6
            end_idx = text.find('"', start_idx)
            return text[start_idx:end_idx]
        except:
            return "#000000"

    def tokens_to_svg(self, tokens):
        """Конвертирует токены обратно в SVG"""
        svg_lines = ['<svg width="64" height="64">']

        for token in tokens:
            cmd = token[0]

            if cmd == self.commands['rect']:
                x, y, w, h, color, _ = token[1:7]
                svg_lines.append(
                    f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{self.reverse_colors.get(color, "#000000")}"/>')
            elif cmd == self.commands['circle']:
                cx, cy, r, _, color, _ = token[1:7]
                svg_lines.append(
                    f'  <circle cx="{cx}" cy="{cy}" r="{r}" fill="{self.reverse_colors.get(color, "#000000")}"/>')
            elif cmd == self.commands['ellipse']:
                cx, cy, rx, ry, color, _ = token[1:7]
                svg_lines.append(
                    f'  <ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="{self.reverse_colors.get(color, "#000000")}"/>')

        svg_lines.append('</svg>')
        return '\n'.join(svg_lines)


# ============================================================================
# 3. Датасет для PyTorch
# ============================================================================

class SVGDataset(Dataset):
    """Датасет SVG изображений"""

    def __init__(self):
        # Загружаем или создаем датасет
        self.dataset = self._load_or_create_dataset()
        self.tokenizer = SVGTokenizer()
        self.max_seq_len = 15

        # Создаем словарь промптов
        self._create_prompt_vocab()

    def _load_or_create_dataset(self):
        """Загружает датасет или создает новый"""
        dataset_file = "training_dataset.json"

        try:
            if Path(dataset_file).exists():
                with open(dataset_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"✓ Загружен датасет из {dataset_file}")
                    return data['samples']
            else:
                print("Датасет не найден")
                dataset_dict = create_training_dataset()
                return dataset_dict['samples']
        except Exception as e:
            print(f"Ошибка загрузки: {e}")
            print("Создаем новый датасет...")
            dataset_dict = create_training_dataset()
            return dataset_dict['samples']

    def _create_prompt_vocab(self):
        """Создает словарь из промптов"""
        all_words = set()
        for sample in self.dataset:
            all_words.update(sample['prompt'].lower().split())

        self.prompt_vocab = ['<PAD>', '<SOS>', '<EOS>', '<UNK>'] + sorted(all_words)
        self.prompt_to_idx = {word: i for i, word in enumerate(self.prompt_vocab)}
        self.idx_to_prompt = {i: word for word, i in self.prompt_to_idx.items()}

    def encode_prompt(self, prompt):
        """Кодирует промпт в тензор"""
        words = prompt.lower().split()[:10]
        indices = [self.prompt_to_idx['<SOS>']]
        indices += [self.prompt_to_idx.get(w, self.prompt_to_idx['<UNK>']) for w in words]
        indices += [self.prompt_to_idx['<EOS>']]

        # Дополняем до 15 токенов
        indices += [self.prompt_to_idx['<PAD>']] * (15 - len(indices))
        return torch.tensor(indices[:15], dtype=torch.long)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        # Кодируем промпт
        prompt_tensor = self.encode_prompt(sample['prompt'])

        # Парсим SVG
        tokens = self.tokenizer.parse_svg(sample['svg'])

        # Создаем последовательности
        input_seq = []
        target_seq = []

        for i in range(min(len(tokens), self.max_seq_len)):
            input_seq.append(tokens[i] if i < len(tokens) else [0] * 7)
            next_idx = i + 1 if i + 1 < len(tokens) else len(tokens) - 1
            target_seq.append(tokens[next_idx] if next_idx < len(tokens) else [0] * 7)

        # Дополняем
        while len(input_seq) < self.max_seq_len:
            input_seq.append([0] * 7)
            target_seq.append([0] * 7)

        return {
            'prompt': prompt_tensor,
            'input_seq': torch.tensor(input_seq[:self.max_seq_len], dtype=torch.float32),
            'target_seq': torch.tensor(target_seq[:self.max_seq_len], dtype=torch.float32),
            'original_svg': sample['svg'],
            'prompt_text': sample['prompt']
        }


# ============================================================================
# 4. Модель нейросети
# ============================================================================

class SVGGeneratorModel(nn.Module):
    """Модель для генерации SVG"""

    def __init__(self, prompt_vocab_size, hidden_size=64):
        super().__init__()

        self.hidden_size = hidden_size

        # Энкодер промпта
        self.prompt_embedding = nn.Embedding(prompt_vocab_size, 32)
        self.prompt_encoder = nn.LSTM(32, hidden_size, batch_first=True)

        # Энкодер команд
        self.command_encoder = nn.Linear(7, hidden_size)

        # Декодер
        self.decoder = nn.LSTM(hidden_size * 2, hidden_size, batch_first=True)

        # Выходные слои
        self.command_head = nn.Linear(hidden_size, 8)  # 8 типов команд
        self.param_head = nn.Linear(hidden_size, 6)  # параметры

    def forward(self, prompt, input_seq):
        batch_size = prompt.size(0)
        seq_len = input_seq.size(1)

        # Кодируем промпт
        prompt_emb = self.prompt_embedding(prompt)
        _, (hidden, _) = self.prompt_encoder(prompt_emb)
        prompt_context = hidden[-1].unsqueeze(1).expand(-1, seq_len, -1)

        # Кодируем команды
        command_emb = self.command_encoder(input_seq)

        # Объединяем и декодируем
        combined = torch.cat([command_emb, prompt_context], dim=-1)
        lstm_out, _ = self.decoder(combined)

        # Предсказания
        command_pred = self.command_head(lstm_out)
        param_pred = self.param_head(lstm_out)

        return command_pred, param_pred


# ============================================================================
# 5. Функции обучения
# ============================================================================

def train_model(model, train_loader, val_loader, epochs=30, device='cpu'):
    """Обучение модели"""

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    cmd_criterion = nn.CrossEntropyLoss()
    param_criterion = nn.MSELoss()

    best_loss = float('inf')

    print("\n" + "=" * 50)
    print("НАЧАЛО ОБУЧЕНИЯ")
    print("=" * 50)

    for epoch in range(epochs):
        # Обучение
        model.train()
        train_loss = 0

        for batch in train_loader:
            prompt = batch['prompt'].to(device)
            input_seq = batch['input_seq'].to(device)
            target_seq = batch['target_seq'].to(device)

            pred_cmd, pred_param = model(prompt, input_seq)

            loss_cmd = cmd_criterion(
                pred_cmd.reshape(-1, 8),
                target_seq[:, :, 0].long().reshape(-1)
            )
            loss_param = param_criterion(pred_param, target_seq[:, :, 1:])
            loss = loss_cmd + loss_param

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Валидация
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                prompt = batch['prompt'].to(device)
                input_seq = batch['input_seq'].to(device)
                target_seq = batch['target_seq'].to(device)

                pred_cmd, pred_param = model(prompt, input_seq)

                loss_cmd = cmd_criterion(
                    pred_cmd.reshape(-1, 8),
                    target_seq[:, :, 0].long().reshape(-1)
                )
                loss_param = param_criterion(pred_param, target_seq[:, :, 1:])
                val_loss += (loss_cmd + loss_param).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        # Прогресс
        print(f"Эпоха {epoch + 1:2d}/{epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        # Сохраняем лучшую модель
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'prompt_vocab': dataset.prompt_vocab,
                'prompt_to_idx': dataset.prompt_to_idx
            }, 'best_model.pth')
            print(f"  ✓ Модель сохранена (loss: {val_loss:.4f})")

    print("\n" + "=" * 50)
    print(f"ОБУЧЕНИЕ ЗАВЕРШЕНО! Лучшая loss: {best_loss:.4f}")
    print("=" * 50)

    return model


# ============================================================================
# 6. Генерация SVG
# ============================================================================

def generate_svg(model, prompt_text, dataset, device, max_len=10):
    """Генерирует SVG по текстовому промпту"""

    model.eval()

    # Кодируем промпт
    prompt_tensor = dataset.encode_prompt(prompt_text).unsqueeze(0).to(device)

    # Начальная последовательность
    current_seq = torch.tensor([[[dataset.tokenizer.commands['start_svg'], 0, 0, 0, 0, 0, 0]]],
                               dtype=torch.float32).to(device)

    generated_tokens = []

    with torch.no_grad():
        for _ in range(max_len):
            pred_cmd, pred_param = model(prompt_tensor, current_seq)

            # Берем последнее предсказание
            cmd_idx = torch.argmax(pred_cmd[0, -1]).item()
            params = [int(round(p.item())) for p in pred_param[0, -1]]

            # Добавляем токен
            new_token = [cmd_idx] + params[:6]
            generated_tokens.append(new_token)

            # Проверяем конец генерации
            if cmd_idx == dataset.tokenizer.commands['end_svg']:
                break

            # Обновляем последовательность
            new_token_tensor = torch.tensor([new_token], dtype=torch.float32).to(device)
            current_seq = torch.cat([current_seq, new_token_tensor.unsqueeze(0)], dim=1)

    # Конвертируем в SVG
    return dataset.tokenizer.tokens_to_svg(generated_tokens)


# ============================================================================
# 7. Функции меню
# ============================================================================

def print_header():
    """Печатает красивый заголовок"""
    os.system('cls' if os.name == 'nt' else 'clear')
    print("╔" + "═" * 58 + "╗")
    print("║            SVG ГЕНЕРАТОР - ПРОСТАЯ ВЕРСИЯ            ║")
    print("╚" + "═" * 58 + "╝")
    print()


def wait_for_enter():
    """Ждет нажатия Enter"""
    input("\nНажмите Enter для продолжения...")


def train_menu():
    """Меню обучения модели"""
    global model, dataset, device

    print_header()
    print("ОБУЧЕНИЕ МОДЕЛИ")
    print("-" * 60)

    # Проверяем устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")

    # Загружаем датасет
    print("\nЗагрузка датасета...")
    dataset = SVGDataset()

    # Разделяем на train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2)

    print(f"Тренировочных примеров: {train_size}")
    print(f"Валидационных примеров: {val_size}")

    # Создаем модель
    print("\nСоздание модели...")
    model = SVGGeneratorModel(
        prompt_vocab_size=len(dataset.prompt_vocab)
    ).to(device)

    # Параметры обучения
    print("\nПараметры обучения:")
    print("1. 10 эпох (быстро, для теста)")
    print("2. 30 эпох (рекомендуется)")
    print("3. 50 эпох (качественно, но долго)")

    choice = input("\nВыберите количество эпох (1-3): ").strip()

    if choice == '1':
        epochs = 10
    elif choice == '2':
        epochs = 30
    elif choice == '3':
        epochs = 50
    else:
        epochs = 30
        print("Выбрано по умолчанию: 30 эпох")

    # Подтверждение
    print(f"\nБудет выполнено обучение на {epochs} эпохах")
    confirm = input("Начать обучение? (y/n): ").strip().lower()

    if confirm == 'y':
        train_model(model, train_loader, val_loader, epochs, device)
        print("\n✓ Модель обучена и сохранена в best_model.pth")
    else:
        print("\nОбучение отменено")

    wait_for_enter()


def generate_menu():
    """Меню генерации SVG"""
    global model, dataset, device

    print_header()
    print("ГЕНЕРАЦИЯ SVG ПО ТЕКСТУ")
    print("-" * 60)

    # Проверяем наличие модели
    if model is None:
        print("Загрузка модели...")
        try:
            checkpoint = torch.load('best_model.pth', map_location=device)

            # Создаем датасет для словаря
            dataset = SVGDataset()
            dataset.prompt_vocab = checkpoint['prompt_vocab']
            dataset.prompt_to_idx = checkpoint['prompt_to_idx']

            # Создаем модель
            model = SVGGeneratorModel(
                prompt_vocab_size=len(dataset.prompt_vocab)
            ).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])

            print("✓ Модель загружена")
        except FileNotFoundError:
            print("✗ Модель не найдена! Сначала обучите модель.")
            wait_for_enter()
            return
        except Exception as e:
            print(f"✗ Ошибка загрузки модели: {e}")
            wait_for_enter()
            return

    # Ввод промпта
    print("\nДоступные категории: tree, sky, sea, cat, dog, house, sun, flower, car")
    print("Примеры: зеленое дерево, серый кот, красная машина")
    print()

    prompt = input("Введите описание: ").strip()

    if not prompt:
        print("Описание не может быть пустым")
        wait_for_enter()
        return

    print(f"\nГенерация SVG для: '{prompt}'")
    print("Это может занять несколько секунд...")

    try:
        svg = generate_svg(model, prompt, dataset, device)

        print("\n" + "=" * 60)
        print("СГЕНЕРИРОВАННЫЙ SVG:")
        print("=" * 60)
        print(svg)

        # Сохраняем в файл
        filename = f"generated_{prompt[:20].replace(' ', '_')}.svg"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(svg)

        print(f"\n✓ SVG сохранен в файл: {filename}")

    except Exception as e:
        print(f"✗ Ошибка генерации: {e}")

    wait_for_enter()


def dataset_menu():
    """Меню просмотра датасета"""

    print_header()
    print("ПРОСМОТР ДАТАСЕТА")
    print("-" * 60)

    # Загружаем датасет
    try:
        with open("training_dataset.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
            samples = data['samples']
    except:
        print("Датасет не найден, создаем новый...")
        dataset = SVGDataset()
        samples = dataset.dataset

    print(f"\nВсего примеров: {len(samples)}")
    print("\nСписок примеров:")
    print("-" * 40)

    # Группируем по категориям
    categories = {}
    for sample in samples:
        cat = sample.get('category', 'other')
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(sample)

    for cat, items in categories.items():
        print(f"\n{cat.upper()}:")
        for item in items:
            print(f"  • {item['prompt']} (id: {item['id']})")

    # Просмотр конкретного примера
    print("\n" + "-" * 40)
    choice = input("Введите ID примера для просмотра (или Enter для выхода): ").strip()

    if choice:
        for sample in samples:
            if sample['id'] == choice:
                print("\n" + "=" * 60)
                print(f"ПРИМЕР: {sample['prompt']}")
                print("=" * 60)
                print(sample['svg'])
                break
        else:
            print("Пример не найден")

    wait_for_enter()


def main_menu():
    """Главное меню программы"""
    global model, dataset, device

    # Инициализация
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = None
    dataset = None

    while True:
        print_header()

        # Статус
        print("СТАТУС:")
        if Path('best_model.pth').exists():
            print("  ✓ Модель обучена (best_model.pth)")
        else:
            print("  ✗ Модель не обучена")

        if Path('training_dataset.json').exists():
            print("  ✓ Датасет загружен")
        else:
            print("  ✗ Датасет не найден")

        print("\n" + "-" * 60)
        print("ГЛАВНОЕ МЕНЮ:")
        print("  1. 🚀 ОБУЧИТЬ МОДЕЛЬ")
        print("  2. 🎨 СГЕНЕРИРОВАТЬ SVG ПО ТЕКСТУ")
        print("  3. 📖 ПРОСМОТРЕТЬ ДАТАСЕТ")
        print("  4. ℹ  ИНФОРМАЦИЯ")
        print("  0. 🚪 ВЫХОД")
        print("-" * 60)

        choice = input("\nВаш выбор (0-4): ").strip()

        if choice == '1':
            train_menu()
        elif choice == '2':
            generate_menu()
        elif choice == '3':
            dataset_menu()
        elif choice == '4':
            info_menu()
        elif choice == '0':
            print("\nДо свидания!")
            break
        else:
            print("\nНеверный выбор!")
            wait_for_enter()


def info_menu():
    """Меню информации"""

    print_header()
    print("ИНФОРМАЦИЯ О ПРОГРАММЕ")
    print("-" * 60)
    print("""
SVG Генератор - простая нейросеть для создания SVG картинок
по текстовому описанию.

ВЕРСИЯ: 1.0 (упрощенная)

ВОЗМОЖНОСТИ:
• Обучение модели на 10 примерах
• Генерация SVG по тексту
• Просмотр датасета

КАТЕГОРИИ:
• tree (деревья)
• sky (небо)
• sea (море)
• cat (коты)
• dog (собаки)
• house (дома)
• sun (солнце)
• flower (цветы)
• car (машины)

ФАЙЛЫ ПРОГРАММЫ:
• training_dataset.json - база примеров
• best_model.pth - обученная модель
• generated_*.svg - сгенерированные файлы

ТРЕБОВАНИЯ:
• Windows/Linux/Mac
• 4GB RAM
• 500MB свободного места
    """)

    wait_for_enter()


# ============================================================================
# 8. Запуск программы
# ============================================================================

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\nПрограмма прервана пользователем")
    except Exception as e:
        print(f"\nКритическая ошибка: {e}")
        input("\nНажмите Enter для выхода...")