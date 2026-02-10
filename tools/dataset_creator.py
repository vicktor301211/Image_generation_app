# create_txt_dataset.py
"""
Самый простой датасет в текстовом файле
Каждая строка: категория | описание | SVG
"""


def create_text_dataset():
    """Создает текстовый датасет"""

    lines = [
        "# SVG Dataset - формат: категория | описание | SVG",
        "",
        "tree|зеленое дерево|<svg width='64' height='64'><rect x='27' y='40' width='10' height='24' fill='#8B4513'/><circle cx='32' cy='30' r='20' fill='#228B22'/></svg>",
        "tree|дерево с листьями|<svg width='64' height='64'><rect x='29' y='35' width='6' height='29' fill='#654321'/><circle cx='32' cy='25' r='15' fill='#2E8B57'/><circle cx='20' cy='30' r='12' fill='#388E3C'/><circle cx='44' cy='30' r='12' fill='#388E3C'/></svg>",
        "sky|голубое небо|<svg width='64' height='64'><rect width='64' height='64' fill='#87CEEB'/></svg>",
        "sky|небо с облаками|<svg width='64' height='64'><rect width='64' height='64' fill='#1E90FF'/><circle cx='20' cy='30' r='8' fill='white'/><circle cx='30' cy='25' r='10' fill='white'/><circle cx='40' cy='30' r='8' fill='white'/></svg>",
        "sea|синее море|<svg width='64' height='64'><rect width='64' height='64' fill='#00008B'/></svg>",
        "cat|серый кот|<svg width='64' height='64'><circle cx='32' cy='40' r='15' fill='#808080'/><circle cx='32' cy='25' r='12' fill='#808080'/><circle cx='27' cy='23' r='3' fill='green'/><circle cx='37' cy='23' r='3' fill='green'/></svg>",
        "dog|коричневая собака|<svg width='64' height='64'><rect x='22' y='35' width='20' height='15' rx='5' fill='#8B4513'/><circle cx='32' cy='28' r='10' fill='#8B4513'/><circle cx='28' cy='26' r='2' fill='black'/><circle cx='36' cy='26' r='2' fill='black'/></svg>"
    ]

    with open("../svg_dataset.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Создан файл: svg_dataset.txt")
    print("Формат: категория | описание | SVG код")
    print("\nПример строки:")
    print(lines[2])


if __name__ == "__main__":
    create_text_dataset()