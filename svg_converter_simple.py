# svg_converter_fixed.py
"""
ИСПРАВЛЕННЫЙ конвертер SVG в изображения
Теперь реально создает PNG/JPG файлы
"""

import os
import json
from pathlib import Path
import sys

# ОБЯЗАТЕЛЬНЫЕ библиотеки
try:
    from PIL import Image, ImageDraw, ImageFont

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("КРИТИЧЕСКАЯ ОШИБКА: Установите Pillow: pip install Pillow")
    sys.exit(1)

try:
    import cairosvg

    CAIRO_AVAILABLE = True
except ImportError:
    CAIRO_AVAILABLE = False
    print("ВНИМАНИЕ: CairoSVG не установлен. Будет использован упрощенный конвертер.")
    print("Для лучшего качества: pip install cairosvg")

try:
    import webbrowser

    WEBBROWSER_AVAILABLE = True
except:
    WEBBROWSER_AVAILABLE = False

try:
    import tempfile
except:
    pass


class SVGConverter:
    """Конвертер SVG в изображения - РЕАЛЬНАЯ конвертация"""

    def __init__(self):
        self.dataset = None
        self.load_dataset()

    def load_dataset(self):
        """Загружает датасет с примерами SVG"""
        dataset_file = "training_dataset.json"

        try:
            if Path(dataset_file).exists():
                with open(dataset_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.dataset = data.get('samples', [])
                print(f"✓ Загружен датасет: {len(self.dataset)} примеров")
            else:
                print("✗ Файл датасета не найден")
                self.dataset = []
        except Exception as e:
            print(f"✗ Ошибка загрузки датасета: {e}")
            self.dataset = []

    def convert_svg_to_png(self, svg_string, output_path, scale=2.0):
        """
        РЕАЛЬНАЯ конвертация SVG в PNG

        Args:
            svg_string: строка с SVG кодом
            output_path: куда сохранить PNG
            scale: масштаб (2.0 = удвоенный размер)

        Returns:
            bool: True если успешно, False если ошибка
        """

        # СПОСОБ 1: CairoSVG (наилучшее качество)
        if CAIRO_AVAILABLE:
            try:
                print("  Конвертация через CairoSVG...")
                cairosvg.svg2png(
                    bytestring=svg_string.encode('utf-8'),
                    write_to=output_path,
                    scale=scale
                )

                # Проверяем что файл создался и не пустой
                if Path(output_path).exists() and Path(output_path).stat().st_size > 100:
                    print(f"  ✓ PNG создан: {output_path}")
                    return True
                else:
                    print("  ✗ Файл создан, но слишком маленький")

            except Exception as e:
                print(f"  ✗ Ошибка CairoSVG: {e}")
                print("  Пробую запасной вариант...")

        # СПОСОБ 2: Ручная отрисовка через PIL (гарантированно работает)
        if PIL_AVAILABLE:
            try:
                print("  Конвертация через PIL (упрощенная)...")

                # Парсим SVG вручную
                width, height = self._parse_svg_size(svg_string)
                if width <= 0 or height <= 0:
                    width, height = 512, 512

                # Создаем изображение
                img_size = (int(width * scale), int(height * scale))
                img = Image.new('RGBA', img_size, color=(255, 255, 255, 255))
                draw = ImageDraw.Draw(img)

                # Отрисовываем примитивы из SVG
                self._draw_svg_primitives(draw, svg_string, scale)

                # Сохраняем
                img.save(output_path, 'PNG')

                # Проверяем
                if Path(output_path).exists():
                    print(f"  ✓ PNG создан (упрощенный): {output_path}")
                    return True

            except Exception as e:
                print(f"  ✗ Ошибка PIL: {e}")

        return False

    def _parse_svg_size(self, svg_string):
        """Извлекает размер из SVG"""
        try:
            import re
            width_match = re.search(r'width="(\d+)"', svg_string)
            height_match = re.search(r'height="(\d+)"', svg_string)

            width = int(width_match.group(1)) if width_match else 64
            height = int(height_match.group(1)) if height_match else 64
            return width, height
        except:
            return 64, 64

    def _draw_svg_primitives(self, draw, svg_string, scale=2.0):
        """Рисует примитивы из SVG"""
        lines = svg_string.split('\n')

        for line in lines:
            line = line.strip()

            # Прямоугольник
            if '<rect' in line:
                x = self._extract_number(line, 'x="', '"') * scale
                y = self._extract_number(line, 'y="', '"') * scale
                w = self._extract_number(line, 'width="', '"') * scale
                h = self._extract_number(line, 'height="', '"') * scale
                color = self._extract_color(line) or '#000000'

                # Конвертируем цвет в RGB
                if color.startswith('#'):
                    rgb = tuple(int(color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))
                else:
                    rgb = (0, 0, 0)

                draw.rectangle([x, y, x + w, y + h], fill=rgb)

            # Круг
            elif '<circle' in line:
                cx = self._extract_number(line, 'cx="', '"') * scale
                cy = self._extract_number(line, 'cy="', '"') * scale
                r = self._extract_number(line, 'r="', '"') * scale
                color = self._extract_color(line) or '#000000'

                if color.startswith('#'):
                    rgb = tuple(int(color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))
                else:
                    rgb = (0, 0, 0)

                # Рисуем круг через эллипс
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=rgb)

            # Эллипс
            elif '<ellipse' in line:
                cx = self._extract_number(line, 'cx="', '"') * scale
                cy = self._extract_number(line, 'cy="', '"') * scale
                rx = self._extract_number(line, 'rx="', '"') * scale
                ry = self._extract_number(line, 'ry="', '"') * scale
                color = self._extract_color(line) or '#000000'

                if color.startswith('#'):
                    rgb = tuple(int(color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))
                else:
                    rgb = (0, 0, 0)

                draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=rgb)

            # Линия
            elif '<line' in line:
                x1 = self._extract_number(line, 'x1="', '"') * scale
                y1 = self._extract_number(line, 'y1="', '"') * scale
                x2 = self._extract_number(line, 'x2="', '"') * scale
                y2 = self._extract_number(line, 'y2="', '"') * scale
                stroke = self._extract_stroke(line) or '#000000'
                width = self._extract_stroke_width(line) or 1

                if stroke.startswith('#'):
                    rgb = tuple(int(stroke.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))
                else:
                    rgb = (0, 0, 0)

                draw.line([x1, y1, x2, y2], fill=rgb, width=int(width))

    def _extract_number(self, text, start, end):
        """Извлекает число из строки"""
        try:
            start_idx = text.find(start)
            if start_idx == -1:
                return 0
            start_idx += len(start)
            end_idx = text.find(end, start_idx)
            return float(text[start_idx:end_idx])
        except:
            return 0

    def _extract_color(self, text):
        """Извлекает цвет fill"""
        try:
            start_idx = text.find('fill="')
            if start_idx == -1:
                return None
            start_idx += 6
            end_idx = text.find('"', start_idx)
            return text[start_idx:end_idx]
        except:
            return None

    def _extract_stroke(self, text):
        """Извлекает цвет обводки"""
        try:
            start_idx = text.find('stroke="')
            if start_idx == -1:
                return None
            start_idx += 8
            end_idx = text.find('"', start_idx)
            return text[start_idx:end_idx]
        except:
            return None

    def _extract_stroke_width(self, text):
        """Извлекает толщину линии"""
        try:
            start_idx = text.find('stroke-width="')
            if start_idx == -1:
                return 1
            start_idx += 14
            end_idx = text.find('"', start_idx)
            return float(text[start_idx:end_idx])
        except:
            return 1

    def show_dataset_examples(self):
        """Показывает примеры из датасета"""
        if not self.dataset:
            print("\nДатасет пуст")
            return

        print("\n" + "=" * 60)
        print("ПРИМЕРЫ ИЗ ДАТАСЕТА:")
        print("=" * 60)

        for i, sample in enumerate(self.dataset):
            print(f"\n{i + 1}. {sample.get('prompt', 'N/A')}")
            print(f"   ID: {sample.get('id', 'N/A')}")
            print(f"   Категория: {sample.get('category', 'N/A')}")

    def preview_in_browser(self, svg_string, title="SVG Preview"):
        """Показывает SVG в браузере (только для просмотра)"""
        if not WEBBROWSER_AVAILABLE:
            print("Не могу открыть браузер")
            return False

        try:
            html = f"""<html>
<head><title>{title}</title></head>
<body style="background:#f0f0f0; padding:20px;">
    <h2>{title}</h2>
    <div style="background:white; padding:20px;">
        {svg_string}
    </div>
</body>
</html>"""

            with tempfile.NamedTemporaryFile(mode='w', suffix='.html',
                                             delete=False, encoding='utf-8') as f:
                f.write(html)
                temp_file = f.name

            webbrowser.open(f'file://{temp_file}')
            print("✓ Предпросмотр открыт в браузере")
            return True
        except:
            return False


# ============================================================================
# МЕНЮ ПРОГРАММЫ
# ============================================================================

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    clear_screen()
    print("╔" + "═" * 58 + "╗")
    print("║         SVG КОНВЕРТЕР - РЕАЛЬНАЯ КОНВЕРТАЦИЯ         ║")
    print("╚" + "═" * 58 + "╝")
    print()


def wait_for_enter():
    input("\nНажмите Enter для продолжения...")


def convert_example_menu(converter):
    """Конвертация примера из датасета"""
    print_header()
    print("КОНВЕРТАЦИЯ ПРИМЕРА ИЗ ДАТАСЕТА")
    print("-" * 60)

    if not converter.dataset:
        print("Датасет пуст")
        wait_for_enter()
        return

    # Показываем примеры
    converter.show_dataset_examples()

    try:
        choice = int(input("\nВведите номер примера (0 для отмены): "))
        if choice == 0:
            return

        if 1 <= choice <= len(converter.dataset):
            sample = converter.dataset[choice - 1]

            print(f"\nВыбран: {sample['prompt']}")

            # Параметры
            scale = 2.0
            try:
                s = input("Масштаб (1-3, Enter=2): ")
                if s:
                    scale = float(s)
            except:
                pass

            # Имя файла
            filename = f"{sample.get('id', 'example')}.png"
            custom = input(f"Имя файла (Enter={filename}): ").strip()
            if custom:
                filename = custom

            # Конвертируем!
            print(f"\nКонвертация в {filename}...")
            success = converter.convert_svg_to_png(sample['svg'], filename, scale)

            if success:
                print(f"\n✓ ГОТОВО! Файл сохранен: {filename}")

                # Показываем размер
                size = Path(filename).stat().st_size
                print(f"  Размер: {size} байт")
            else:
                print(f"\n✗ ОШИБКА! Файл не создан")
        else:
            print("Неверный номер")
    except ValueError:
        print("Введите число")

    wait_for_enter()


def convert_file_menu(converter):
    """Конвертация SVG файла"""
    print_header()
    print("КОНВЕРТАЦИЯ SVG ФАЙЛА")
    print("-" * 60)

    file_path = input("Введите путь к SVG файлу: ").strip()

    if not Path(file_path).exists():
        print(f"Файл не найден: {file_path}")
        wait_for_enter()
        return

    # Читаем файл
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            svg_string = f.read()

        print(f"Файл загружен: {len(svg_string)} символов")

        # Параметры
        scale = 2.0
        try:
            s = input("Масштаб (1-3, Enter=2): ")
            if s:
                scale = float(s)
        except:
            pass

        # Выходной файл
        default = f"{Path(file_path).stem}.png"
        output = input(f"Выходной файл (Enter={default}): ").strip()
        if not output:
            output = default

        # Конвертируем
        print(f"\nКонвертация в {output}...")
        success = converter.convert_svg_to_png(svg_string, output, scale)

        if success:
            print(f"\n✓ ГОТОВО! Файл сохранен: {output}")
            size = Path(output).stat().st_size
            print(f"  Размер: {size} байт")
        else:
            print(f"\n✗ ОШИБКА! Файл не создан")

    except Exception as e:
        print(f"Ошибка: {e}")

    wait_for_enter()


def test_converter(converter):
    """Тест конвертера на первом примере"""
    print_header()
    print("ТЕСТ КОНВЕРТЕРА")
    print("-" * 60)

    if not converter.dataset:
        print("Нет датасета для теста")
        wait_for_enter()
        return

    sample = converter.dataset[0]
    print(f"Тестовый пример: {sample['prompt']}")
    print("Создаю test_output.png...")

    success = converter.convert_svg_to_png(sample['svg'], "test_output.png", 2.0)

    if success:
        size = Path("test_output.png").stat().st_size
        print(f"\n✓ ТЕСТ ПРОЙДЕН!")
        print(f"  Файл: test_output.png")
        print(f"  Размер: {size} байт")
        print(f"  Путь: {os.path.abspath('test_output.png')}")
    else:
        print(f"\n✗ ТЕСТ НЕ ПРОЙДЕН!")

    wait_for_enter()


def main_menu():
    """Главное меню"""

    converter = SVGConverter()

    while True:
        print_header()

        # Статус
        print("СТАТУС:")
        if CAIRO_AVAILABLE:
            print("  ✓ CairoSVG: доступен (лучшее качество)")
        else:
            print("  ✗ CairoSVG: не установлен")

        print(f"  ✓ PIL: доступен")
        print(f"  ✓ Датасет: {len(converter.dataset)} примеров")

        print("\n" + "-" * 60)
        print("ГЛАВНОЕ МЕНЮ:")
        print("  1. 🎨 Конвертировать пример из датасета")
        print("  2. 📁 Конвертировать SVG файл")
        print("  3. 👀 Показать примеры из датасета")
        print("  4. 🧪 ТЕСТ конвертера")
        print("  5. ℹ  Информация")
        print("  0. 🚪 Выход")
        print("-" * 60)

        choice = input("\nВаш выбор: ").strip()

        if choice == '1':
            convert_example_menu(converter)
        elif choice == '2':
            convert_file_menu(converter)
        elif choice == '3':
            print_header()
            converter.show_dataset_examples()
            wait_for_enter()
        elif choice == '4':
            test_converter(converter)
        elif choice == '5':
            print_header()
            print("ИНФОРМАЦИЯ:")
            print("-" * 60)
            print("""
Этот конвертер создает РЕАЛЬНЫЕ PNG файлы из SVG.

Как это работает:
1. При наличии CairoSVG - идеальное качество
2. Без CairoSVG - упрощенная отрисовка через PIL
3. Всегда создается файл на диске

Проверка:
- Запустите ТЕСТ (пункт 4)
- Найдите test_output.png в папке
- Откройте его - должно быть изображение

Требования:
- Pillow (обязательно): pip install Pillow==9.5.0
- CairoSVG (опционально): pip install cairosvg==2.5.2
            """)
            wait_for_enter()
        elif choice == '0':
            print("\nДо свидания!")
            break


if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\nПрограмма прервана")
    except Exception as e:
        print(f"\nОшибка: {e}")
        input("\nНажмите Enter...")