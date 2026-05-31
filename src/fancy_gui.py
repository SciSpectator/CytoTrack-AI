"""
CytoTrack AI — Professional Pygame GUI
=======================================
Light, clean, professional interface. No decorative animations or
floating cell graphics; just a programmatic logo, crisp panels, and
the workflow dialogs the rest of the app expects.

Public API (unchanged from the old module so main.py keeps working):

  APP_NAME, APP_VERSION, APP_FULL
  class FancyGUI(width, height) with:
      show_main_menu()
      show_message(title, message, buttons)
      show_input_dialog(title, fields, defaults)
      show_progress(title, total)         -> callable(pct, msg)
      show_folder_dialog(title, start_path=None)
      show_file_dialog(title, start_path=None, extensions=None)
      cleanup()
      running (bool), screen (pygame.Surface)
  show_splash_screen(duration)
  show_image_settings_preview(screen, files, settings) -> bool
  manual_cell_classification(screen, frame, detections, cell_types)
                                                   -> dict | None
"""

import os
import time

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False

APP_NAME = "CytoTrack AI"
APP_VERSION = "1.0"
APP_FULL = f"{APP_NAME} v{APP_VERSION}"

# ---------------------------------------------------------- light-theme palette
COLORS = {
    "bg":            (245, 247, 250),   # page background
    "bg_soft":       (237, 241, 246),   # subtle gradient end
    "panel":         (255, 255, 255),   # card background
    "panel_hover":   (240, 249, 255),
    "border":        (225, 229, 236),
    "border_strong": (206, 212, 222),
    "text":          (31,  41,  55),    # slate-800
    "text_soft":     (75,  85, 100),
    "text_dim":      (107, 114, 128),
    "accent":        (8,  145, 178),    # cyan-600
    "accent_light":  (6,  182, 212),    # cyan-500
    "accent_dark":   (14, 116, 144),    # cyan-700
    "accent_soft":   (207, 250, 254),   # cyan-100
    "primary_btn":   (8,  145, 178),
    "primary_btn_h": (14, 116, 144),
    "primary_txt":   (255, 255, 255),
    "danger":        (220, 38,  38),
    "danger_hover":  (185, 28,  28),
    "success":       (22, 163, 74),
    "shadow":        (15, 23,  42),
}


# ============================================================== LOGO (drawn)
def draw_logo(surface, center, size, mono=False):
    """
    Draw a programmatic, professional mark: an abstract 7-cell hexagonal
    cluster inside a thin outer ring. Reads as a microscopy / cell motif
    without cartoony features.
    """
    cx, cy = center
    outer_r = size
    ring_w = max(2, size // 14)
    accent = COLORS["accent"] if not mono else (140, 140, 140)
    light = COLORS["accent_light"] if not mono else (170, 170, 170)
    dark = COLORS["accent_dark"] if not mono else (100, 100, 100)
    fill = COLORS["accent_soft"] if not mono else (235, 235, 235)

    # outer soft disk
    disk = pygame.Surface((outer_r * 2 + 8, outer_r * 2 + 8), pygame.SRCALPHA)
    pygame.draw.circle(disk, (*fill, 180), (outer_r + 4, outer_r + 4), outer_r)
    surface.blit(disk, (cx - outer_r - 4, cy - outer_r - 4))

    # outer ring
    pygame.draw.circle(surface, accent, (cx, cy), outer_r, ring_w)

    # inner hexagonal cluster of 6 cells around a central one
    import math
    inner_r = size * 0.28
    cell_r = max(4, int(size * 0.22))

    # center cell
    pygame.draw.circle(surface, light, (cx, cy), cell_r)
    pygame.draw.circle(surface, dark, (cx, cy), cell_r, 2)

    for i in range(6):
        a = i * math.pi / 3.0 - math.pi / 6.0
        x = int(cx + math.cos(a) * inner_r * 2.1)
        y = int(cy + math.sin(a) * inner_r * 2.1)
        pygame.draw.circle(surface, COLORS["panel"], (x, y), cell_r + 2)
        pygame.draw.circle(surface, light, (x, y), cell_r)
        pygame.draw.circle(surface, dark, (x, y), cell_r, 2)


# ============================================================== helpers
def _draw_vertical_gradient(surface, top, bottom, rect=None):
    """Fill ``rect`` (or the whole surface) with a vertical gradient."""
    if rect is None:
        rect = surface.get_rect()
    h = rect.height
    for y in range(h):
        t = y / max(1, h - 1)
        r = int(top[0] * (1 - t) + bottom[0] * t)
        g = int(top[1] * (1 - t) + bottom[1] * t)
        b = int(top[2] * (1 - t) + bottom[2] * t)
        pygame.draw.line(surface, (r, g, b),
                         (rect.x, rect.y + y), (rect.right - 1, rect.y + y))


def _shadow_rect(surface, rect, radius=12, offset=4, alpha=40):
    shadow = pygame.Surface((rect.width + offset * 2,
                             rect.height + offset * 2), pygame.SRCALPHA)
    pygame.draw.rect(shadow, (*COLORS["shadow"], alpha),
                     shadow.get_rect(), border_radius=radius + 2)
    surface.blit(shadow, (rect.x - offset, rect.y - offset + 2))


def _wrap_text(font, text, max_width):
    lines = []
    for raw in text.splitlines():
        if not raw:
            lines.append("")
            continue
        words = raw.split(" ")
        line = ""
        for w in words:
            candidate = line + (" " if line else "") + w
            if font.size(candidate)[0] <= max_width:
                line = candidate
            else:
                if line:
                    lines.append(line)
                line = w
        if line:
            lines.append(line)
    return lines


# ============================================================== widgets
class Button:
    """Flat button — primary (filled) or secondary (outline)."""

    def __init__(self, x, y, w, h, text, kind="secondary"):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.kind = kind
        self.hover = False

    def update(self, mouse):
        self.hover = self.rect.collidepoint(mouse)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            return self.rect.collidepoint(event.pos)
        return False

    def draw(self, surface, font):
        r = self.rect
        if self.kind == "primary":
            color = COLORS["primary_btn_h"] if self.hover else COLORS["primary_btn"]
            pygame.draw.rect(surface, color, r, border_radius=8)
            pygame.draw.rect(surface, COLORS["accent_dark"], r, 1, border_radius=8)
            txt_color = COLORS["primary_txt"]
        elif self.kind == "danger":
            color = COLORS["danger_hover"] if self.hover else COLORS["danger"]
            pygame.draw.rect(surface, color, r, border_radius=8)
            txt_color = COLORS["primary_txt"]
        else:
            bg = COLORS["panel_hover"] if self.hover else COLORS["panel"]
            border = COLORS["accent"] if self.hover else COLORS["border_strong"]
            pygame.draw.rect(surface, bg, r, border_radius=8)
            pygame.draw.rect(surface, border, r, 1, border_radius=8)
            txt_color = COLORS["text"]

        txt = font.render(self.text, True, txt_color)
        surface.blit(txt, txt.get_rect(center=r.center))


class MenuButton:
    """Wider sidebar-style menu item with subtle chevron."""

    def __init__(self, x, y, w, h, text, hint=""):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.hint = hint
        self.hover = False

    def update(self, mouse):
        self.hover = self.rect.collidepoint(mouse)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            return self.rect.collidepoint(event.pos)
        return False

    def draw(self, surface, font_main, font_hint):
        r = self.rect
        bg = COLORS["accent_soft"] if self.hover else COLORS["panel"]
        border = COLORS["accent"] if self.hover else COLORS["border"]
        pygame.draw.rect(surface, bg, r, border_radius=10)
        pygame.draw.rect(surface, border, r, 1, border_radius=10)

        # Left accent bar on hover
        if self.hover:
            pygame.draw.rect(surface, COLORS["accent"],
                             (r.x, r.y, 4, r.height), border_radius=3)

        t = font_main.render(self.text, True, COLORS["text"])
        surface.blit(t, (r.x + 20, r.y + 10))
        if self.hint:
            h = font_hint.render(self.hint, True, COLORS["text_dim"])
            surface.blit(h, (r.x + 20, r.y + 34))

        # chevron ›
        ch = font_main.render("›", True,
                              COLORS["accent"] if self.hover else COLORS["text_dim"])
        surface.blit(ch, (r.right - 28, r.y + 8))


class TextInput:
    def __init__(self, x, y, w, h, placeholder="", default=""):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = default
        self.placeholder = placeholder
        self.focused = False
        self._blink = 0

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.focused = self.rect.collidepoint(event.pos)
        elif self.focused and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.unicode and event.unicode.isprintable():
                self.text += event.unicode

    def update(self):
        if self.focused:
            self._blink = (self._blink + 1) % 60

    def draw(self, surface, font):
        r = self.rect
        pygame.draw.rect(surface, COLORS["panel"], r, border_radius=6)
        border = COLORS["accent"] if self.focused else COLORS["border_strong"]
        pygame.draw.rect(surface, border, r, 1, border_radius=6)

        if self.text:
            t = font.render(self.text, True, COLORS["text"])
        else:
            t = font.render(self.placeholder, True, COLORS["text_dim"])
        surface.blit(t, (r.x + 12, r.centery - t.get_height() // 2))

        if self.focused and self._blink < 30:
            cx = r.x + 12 + font.size(self.text)[0]
            pygame.draw.line(surface, COLORS["text"],
                             (cx, r.y + 8), (cx, r.bottom - 8), 1)


class Slider:
    def __init__(self, x, y, w, h, min_val, max_val, value, label, step=1):
        self.rect = pygame.Rect(x, y, w, h)
        self.min_val = min_val
        self.max_val = max_val
        self.value = value
        self.label = label
        self.step = step
        self.dragging = False
        self.handle_r = h // 2 + 4

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if (self._handle_rect().collidepoint(event.pos) or
                    self.rect.collidepoint(event.pos)):
                self.dragging = True
                self._update_value(event.pos[0])
                return True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._update_value(event.pos[0])
            return True
        return False

    def _update_value(self, x):
        ratio = max(0.0, min(1.0, (x - self.rect.x) / max(1, self.rect.width)))
        raw = self.min_val + ratio * (self.max_val - self.min_val)
        self.value = round(raw / self.step) * self.step
        self.value = max(self.min_val, min(self.max_val, self.value))

    def _handle_rect(self):
        ratio = (self.value - self.min_val) / max(1e-9, self.max_val - self.min_val)
        hx = self.rect.x + ratio * self.rect.width
        return pygame.Rect(hx - self.handle_r, self.rect.centery - self.handle_r,
                           self.handle_r * 2, self.handle_r * 2)

    def draw(self, surface, font):
        pygame.draw.rect(surface, COLORS["border"], self.rect, border_radius=4)
        ratio = (self.value - self.min_val) / max(1e-9, self.max_val - self.min_val)
        fill = pygame.Rect(self.rect.x, self.rect.y,
                           int(self.rect.width * ratio), self.rect.height)
        pygame.draw.rect(surface, COLORS["accent"], fill, border_radius=4)

        hx = self.rect.x + ratio * self.rect.width
        pygame.draw.circle(surface, COLORS["panel"],
                           (int(hx), self.rect.centery), self.handle_r)
        pygame.draw.circle(surface, COLORS["accent"],
                           (int(hx), self.rect.centery), self.handle_r - 3)

        value_str = (f"{self.value:.2f}" if isinstance(self.value, float)
                     else str(int(self.value)))
        label = font.render(f"{self.label}: {value_str}", True, COLORS["text"])
        surface.blit(label, (self.rect.x, self.rect.y - 22))


# ============================================================== FileBrowser
class FileBrowser:
    def __init__(self, screen, start_path=None, select_folder=False,
                 extensions=None):
        self.screen = screen
        self.width = screen.get_width()
        self.height = screen.get_height()
        self.select_folder = select_folder
        self.extensions = extensions or []
        self.current_path = start_path or os.path.expanduser("~")
        self.entries = []
        self.selected = 0
        self.scroll = 0
        self.font = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 20)
        self._scan()

    def _scan(self):
        self.entries = []
        self.selected = 0
        self.scroll = 0
        try:
            items = sorted(
                os.listdir(self.current_path),
                key=lambda x: (not os.path.isdir(
                    os.path.join(self.current_path, x)), x.lower()),
            )
            for item in items:
                if item.startswith("."):
                    continue
                path = os.path.join(self.current_path, item)
                is_dir = os.path.isdir(path)
                if (is_dir or not self.extensions
                        or any(item.lower().endswith(e) for e in self.extensions)):
                    self.entries.append((item, is_dir, path))
        except Exception:
            pass

    def run(self):
        clock = pygame.time.Clock()
        title_txt = "Select Folder" if self.select_folder else "Select File"

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return None
                    elif event.key == pygame.K_UP:
                        self.selected = max(0, self.selected - 1)
                    elif event.key == pygame.K_DOWN:
                        self.selected = min(len(self.entries) - 1,
                                            self.selected + 1)
                    elif event.key == pygame.K_RETURN and self.entries:
                        _, is_dir, path = self.entries[self.selected]
                        if is_dir:
                            self.current_path = path
                            self._scan()
                        elif not self.select_folder:
                            return path
                    elif event.key == pygame.K_BACKSPACE:
                        parent = os.path.dirname(self.current_path)
                        if parent != self.current_path:
                            self.current_path = parent
                            self._scan()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 4:
                        self.scroll = max(0, self.scroll - 1)
                    elif event.button == 5:
                        self.scroll = min(max(0, len(self.entries) - 12),
                                          self.scroll + 1)
                    elif event.button == 1:
                        mx, my = event.pos
                        # list click
                        if 20 < mx < self.width - 20 and 110 < my < self.height - 90:
                            idx = (my - 110) // 32 + self.scroll
                            if 0 <= idx < len(self.entries):
                                if idx == self.selected:
                                    _, is_dir, path = self.entries[idx]
                                    if is_dir:
                                        self.current_path = path
                                        self._scan()
                                    elif not self.select_folder:
                                        return path
                                else:
                                    self.selected = idx
                        # up
                        if 20 < mx < 100 and 62 < my < 94:
                            parent = os.path.dirname(self.current_path)
                            if parent != self.current_path:
                                self.current_path = parent
                                self._scan()
                        # Select button
                        if (self.width - 140 < mx < self.width - 25
                                and self.height - 60 < my < self.height - 20):
                            if self.select_folder:
                                return self.current_path
                            if self.entries and not self.entries[self.selected][1]:
                                return self.entries[self.selected][2]
                        # Cancel
                        if (self.width - 275 < mx < self.width - 160
                                and self.height - 60 < my < self.height - 20):
                            return None

            self._draw(title_txt)
            pygame.display.flip()
            clock.tick(60)

    def _draw(self, title_txt):
        self.screen.fill(COLORS["bg"])

        # Header
        pygame.draw.rect(self.screen, COLORS["panel"],
                         (0, 0, self.width, 60))
        pygame.draw.line(self.screen, COLORS["border"],
                         (0, 60), (self.width, 60))
        t = self.font.render(title_txt, True, COLORS["text"])
        self.screen.blit(t, (24, 20))

        # Up button
        up_rect = pygame.Rect(20, 62, 80, 32)
        pygame.draw.rect(self.screen, COLORS["panel"], up_rect, border_radius=6)
        pygame.draw.rect(self.screen, COLORS["border_strong"], up_rect, 1,
                         border_radius=6)
        self.screen.blit(self.font_small.render("↑ Up", True, COLORS["text"]),
                         (42, 68))

        path_txt = self.font_small.render(self.current_path[-70:], True,
                                          COLORS["text_soft"])
        self.screen.blit(path_txt, (110, 68))

        # List
        list_rect = pygame.Rect(20, 110, self.width - 40, self.height - 200)
        pygame.draw.rect(self.screen, COLORS["panel"], list_rect,
                         border_radius=10)
        pygame.draw.rect(self.screen, COLORS["border"], list_rect, 1,
                         border_radius=10)

        visible = (list_rect.height - 10) // 32
        for i, (name, is_dir, _) in enumerate(
                self.entries[self.scroll:self.scroll + visible]):
            idx = i + self.scroll
            y = list_rect.y + 5 + i * 32
            if idx == self.selected:
                pygame.draw.rect(self.screen, COLORS["accent_soft"],
                                 (list_rect.x + 4, y,
                                  list_rect.width - 8, 30), border_radius=6)
            icon = "▸" if is_dir else "·"
            color = COLORS["text"]
            self.screen.blit(
                self.font_small.render(f"{icon}  {name}", True, color),
                (list_rect.x + 14, y + 7),
            )

        # Footer buttons
        cancel = pygame.Rect(self.width - 275, self.height - 60, 115, 40)
        select = pygame.Rect(self.width - 140, self.height - 60, 115, 40)

        pygame.draw.rect(self.screen, COLORS["panel"], cancel, border_radius=8)
        pygame.draw.rect(self.screen, COLORS["border_strong"], cancel, 1,
                         border_radius=8)
        ct = self.font_small.render("Cancel", True, COLORS["text"])
        self.screen.blit(ct, ct.get_rect(center=cancel.center))

        pygame.draw.rect(self.screen, COLORS["primary_btn"], select,
                         border_radius=8)
        st = self.font_small.render("Select", True, COLORS["primary_txt"])
        self.screen.blit(st, st.get_rect(center=select.center))


# ============================================================== FancyGUI
class FancyGUI:
    """Main app window. Light, professional, static."""

    def __init__(self, width=1000, height=700):
        if not HAS_PYGAME:
            raise RuntimeError("pygame required")

        pygame.init()
        pygame.display.set_caption(APP_FULL)
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()

        self.font_title = pygame.font.Font(None, 44)
        self.font_large = pygame.font.Font(None, 30)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 20)
        self.font_hint = pygame.font.Font(None, 18)

        self.running = True

    # --------------------------- generic backdrop (STATIC — no animations)
    def _draw_page_bg(self):
        _draw_vertical_gradient(self.screen, COLORS["bg"], COLORS["bg_soft"])

    def _draw_header(self, title, subtitle=""):
        # Card at top with logo + title
        header_rect = pygame.Rect(24, 22, self.width - 48, 96)
        pygame.draw.rect(self.screen, COLORS["panel"], header_rect,
                         border_radius=14)
        pygame.draw.rect(self.screen, COLORS["border"], header_rect, 1,
                         border_radius=14)

        draw_logo(self.screen, (header_rect.x + 56, header_rect.centery), 30)

        t = self.font_title.render(title, True, COLORS["text"])
        self.screen.blit(t, (header_rect.x + 110,
                             header_rect.y + 22))
        if subtitle:
            s = self.font_small.render(subtitle, True, COLORS["text_dim"])
            self.screen.blit(s, (header_rect.x + 112,
                                 header_rect.y + 62))

        # Version tag in header right
        v = self.font_hint.render(f"v{APP_VERSION}", True, COLORS["text_dim"])
        self.screen.blit(v, (header_rect.right - v.get_width() - 16,
                             header_rect.y + 14))

    # ------------------------------------------------------------ main menu
    def show_main_menu(self):
        menu = [
            ("Track Cells",         "Full migration-tracking pipeline"),
            ("Train Cell Line",     "Public data search or local files"),
            ("Train Phenotype",     "Same workflow for phenotypes"),
            ("Analyze Existing Data", "Replot from CSV summaries"),
            ("Help & About",        "Workflow & version info"),
            ("Exit",                "Quit the application"),
        ]

        menu_x = 80
        menu_w = self.width - 160
        menu_top = 160
        gap = 14
        height = 60

        buttons = [
            (MenuButton(menu_x, menu_top + i * (height + gap),
                        menu_w, height, text, hint), text)
            for i, (text, hint) in enumerate(menu)
        ]

        selected = None
        while self.running and selected is None:
            mouse = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "Exit"
                for b, text in buttons:
                    if b.handle_event(event):
                        selected = text

            for b, _ in buttons:
                b.update(mouse)

            self._draw_page_bg()
            self._draw_header(APP_NAME,
                              "Cell Migration Tracking & Analysis · AI-assisted")
            for b, _ in buttons:
                b.draw(self.screen, self.font_medium, self.font_hint)

            # Footer
            footer = self.font_hint.render(
                "Professional microscopy analytics · light theme",
                True, COLORS["text_dim"])
            self.screen.blit(footer, (24, self.height - 24))

            pygame.display.flip()
            self.clock.tick(60)

        return selected

    # ------------------------------------------------------------ message
    def show_message(self, title, message, buttons_list=None):
        if buttons_list is None:
            buttons_list = ["OK"]

        lines = _wrap_text(self.font_small, message, 520)
        dw = 600
        dh = max(200, 140 + 22 * len(lines) + 60)
        dx = (self.width - dw) // 2
        dy = (self.height - dh) // 2

        btns = []
        bw = 118
        tot = len(buttons_list) * bw + (len(buttons_list) - 1) * 12
        bx = dx + dw - tot - 24
        for i, text in enumerate(buttons_list):
            kind = "primary" if i == len(buttons_list) - 1 else "secondary"
            btns.append((Button(bx + i * (bw + 12), dy + dh - 58,
                                bw, 40, text, kind=kind), text))

        selected = None
        while self.running and selected is None:
            mouse = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.KEYDOWN and event.key in (
                        pygame.K_RETURN, pygame.K_ESCAPE):
                    return buttons_list[-1] if event.key == pygame.K_RETURN else buttons_list[0]
                for b, text in btns:
                    if b.handle_event(event):
                        selected = text

            for b, _ in btns:
                b.update(mouse)

            self._draw_page_bg()
            # Overlay dim
            dim = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            dim.fill((15, 23, 42, 80))
            self.screen.blit(dim, (0, 0))

            # Card
            card = pygame.Rect(dx, dy, dw, dh)
            _shadow_rect(self.screen, card, radius=14, offset=6, alpha=60)
            pygame.draw.rect(self.screen, COLORS["panel"], card,
                             border_radius=14)
            pygame.draw.rect(self.screen, COLORS["border"], card, 1,
                             border_radius=14)

            t = self.font_large.render(title, True, COLORS["text"])
            self.screen.blit(t, (dx + 24, dy + 22))
            pygame.draw.line(self.screen, COLORS["border"],
                             (dx + 24, dy + 60), (dx + dw - 24, dy + 60))

            for i, ln in enumerate(lines):
                tx = self.font_small.render(ln, True, COLORS["text_soft"])
                self.screen.blit(tx, (dx + 24, dy + 80 + i * 22))

            for b, _ in btns:
                b.draw(self.screen, self.font_small)

            pygame.display.flip()
            self.clock.tick(60)

        return selected

    # ------------------------------------------------------- input dialog
    def show_input_dialog(self, title, fields, defaults=None):
        if defaults is None:
            defaults = [""] * len(fields)

        dw = 540
        dh = 140 + len(fields) * 80 + 30
        dx = (self.width - dw) // 2
        dy = (self.height - dh) // 2

        inputs = []
        for i, (field, default) in enumerate(zip(fields, defaults)):
            inputs.append((field,
                           TextInput(dx + 24, dy + 100 + i * 80,
                                     dw - 48, 36, field, str(default))))

        ok = Button(dx + dw - 260, dy + dh - 58, 110, 40, "OK",
                    kind="primary")
        cancel = Button(dx + dw - 140, dy + dh - 58, 110, 40, "Cancel")

        result = None
        while self.running and result is None:
            mouse = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                for _, inp in inputs:
                    inp.handle_event(event)
                if ok.handle_event(event):
                    result = [inp.text for _, inp in inputs]
                if cancel.handle_event(event):
                    return None
                if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                    result = [inp.text for _, inp in inputs]
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return None

            for _, inp in inputs:
                inp.update()
            ok.update(mouse)
            cancel.update(mouse)

            self._draw_page_bg()
            dim = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            dim.fill((15, 23, 42, 80))
            self.screen.blit(dim, (0, 0))

            card = pygame.Rect(dx, dy, dw, dh)
            _shadow_rect(self.screen, card, radius=14, offset=6, alpha=60)
            pygame.draw.rect(self.screen, COLORS["panel"], card,
                             border_radius=14)
            pygame.draw.rect(self.screen, COLORS["border"], card, 1,
                             border_radius=14)

            self.screen.blit(self.font_large.render(title, True,
                                                    COLORS["text"]),
                             (dx + 24, dy + 22))
            pygame.draw.line(self.screen, COLORS["border"],
                             (dx + 24, dy + 60), (dx + dw - 24, dy + 60))

            for field, inp in inputs:
                self.screen.blit(
                    self.font_small.render(field, True, COLORS["text_soft"]),
                    (inp.rect.x, inp.rect.y - 22))
                inp.draw(self.screen, self.font_small)

            ok.draw(self.screen, self.font_small)
            cancel.draw(self.screen, self.font_small)

            pygame.display.flip()
            self.clock.tick(60)

        return result

    # ------------------------------------------------------- dialogs
    def show_folder_dialog(self, title="Select Folder", start_path=None):
        return FileBrowser(self.screen, start_path, select_folder=True).run()

    def show_file_dialog(self, title="Select File", start_path=None,
                         extensions=None):
        return FileBrowser(self.screen, start_path, select_folder=False,
                           extensions=extensions).run()

    # ------------------------------------------------------- progress
    def show_progress(self, title, total=100):
        bar = pygame.Rect(self.width // 2 - 220,
                          self.height // 2 + 10, 440, 18)

        def update(value, msg=""):
            self._draw_page_bg()
            dim = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            dim.fill((15, 23, 42, 120))
            self.screen.blit(dim, (0, 0))

            card = pygame.Rect(self.width // 2 - 300,
                               self.height // 2 - 80, 600, 200)
            _shadow_rect(self.screen, card, radius=14, offset=6, alpha=70)
            pygame.draw.rect(self.screen, COLORS["panel"], card,
                             border_radius=14)
            pygame.draw.rect(self.screen, COLORS["border"], card, 1,
                             border_radius=14)

            t = self.font_large.render(title, True, COLORS["text"])
            self.screen.blit(t, (card.x + 24, card.y + 20))

            pygame.draw.rect(self.screen, COLORS["border"], bar,
                             border_radius=9)
            progress = min(value / max(1, total), 1.0)
            fill_w = int((bar.width - 4) * progress)
            pygame.draw.rect(self.screen, COLORS["accent"],
                             (bar.x + 2, bar.y + 2, fill_w, bar.height - 4),
                             border_radius=7)

            pct = self.font_medium.render(f"{int(progress * 100)}%", True,
                                          COLORS["accent_dark"])
            self.screen.blit(pct, (bar.right + 10, bar.centery - pct.get_height() // 2))

            if msg:
                m = self.font_small.render(msg, True, COLORS["text_soft"])
                self.screen.blit(m, (card.x + 24, bar.bottom + 18))

            pygame.display.flip()
            self.clock.tick(60)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

        return update

    def cleanup(self):
        pygame.quit()


# ============================================================== splash
def show_splash_screen(duration=3):
    """Static, professional splash with logo + progress bar."""
    if not HAS_PYGAME:
        print("\n" + "=" * 50)
        print(f"  {APP_FULL}")
        print("=" * 50 + "\n")
        time.sleep(0.8)
        return

    pygame.init()
    W, H = 720, 440
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption(APP_FULL)
    clock = pygame.time.Clock()

    font_title = pygame.font.Font(None, 60)
    font_sub = pygame.font.Font(None, 26)
    font_small = pygame.font.Font(None, 20)

    start = time.time()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                pygame.quit()
                return

        if time.time() - start > duration:
            pygame.quit()
            return

        _draw_vertical_gradient(screen, COLORS["bg"], COLORS["bg_soft"])

        # Centered logo
        draw_logo(screen, (W // 2, H // 2 - 40), 70)

        # Title
        title_surf = font_title.render(APP_NAME, True, COLORS["text"])
        screen.blit(title_surf,
                    (W // 2 - title_surf.get_width() // 2, H // 2 + 50))

        sub = font_sub.render("Cell Migration · AI-assisted", True,
                              COLORS["text_dim"])
        screen.blit(sub, (W // 2 - sub.get_width() // 2, H // 2 + 98))

        # Progress
        prog = min((time.time() - start) / duration, 1.0)
        bw = 280
        bar = pygame.Rect(W // 2 - bw // 2, H - 70, bw, 6)
        pygame.draw.rect(screen, COLORS["border"], bar, border_radius=3)
        pygame.draw.rect(screen, COLORS["accent"],
                         (bar.x, bar.y, int(bw * prog), 6), border_radius=3)

        v = font_small.render(f"v{APP_VERSION}", True, COLORS["text_dim"])
        screen.blit(v, (W - v.get_width() - 20, H - 28))

        pygame.display.flip()
        clock.tick(60)


# ============================================================== image preview
def show_image_settings_preview(screen, files, settings):
    """Preview the first frames with adjustable brightness/contrast/gamma/filter."""
    import cv2
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from image_utils import apply_all_adjustments, FILTER_NAMES

    pygame.init()
    clock = pygame.time.Clock()
    W, H = screen.get_size()

    preview_count = min(10, len(files))
    frames = []
    for i in range(preview_count):
        img = cv2.imread(files[i])
        if img is not None:
            frames.append(img)
    if not frames:
        return False

    current_frame = 0
    panel_x = W - 330
    panel_y = 90
    panel_w = 310

    brightness_s = Slider(panel_x + 20, panel_y + 60, panel_w - 40, 10,
                          -100, 100, settings.brightness, "Brightness", 5)
    contrast_s = Slider(panel_x + 20, panel_y + 120, panel_w - 40, 10,
                        0.5, 3.0, settings.contrast, "Contrast", 0.1)
    gamma_s = Slider(panel_x + 20, panel_y + 180, panel_w - 40, 10,
                     0.1, 3.0, settings.gamma, "Gamma", 0.1)
    filter_s = Slider(panel_x + 20, panel_y + 240, panel_w - 40, 10,
                      0, len(FILTER_NAMES) - 1, settings.filter_mode,
                      "Filter", 1)
    sliders = [brightness_s, contrast_s, gamma_s, filter_s]

    font_title = pygame.font.Font(None, 30)
    font_medium = pygame.font.Font(None, 22)
    font_small = pygame.font.Font(None, 19)

    confirm = Button(W - 200, H - 68, 170, 44, "Start Tracking",
                     kind="primary")
    cancel = Button(W - 380, H - 68, 170, 44, "Cancel")
    reset = Button(panel_x + 20, panel_y + 290, 130, 32, "Reset")
    prev_btn = Button(24, H - 68, 100, 44, "‹ Prev")
    next_btn = Button(132, H - 68, 100, 44, "Next ›")

    running = True
    confirmed = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_RETURN:
                    confirmed = True
                    running = False
                elif event.key == pygame.K_LEFT:
                    current_frame = max(0, current_frame - 1)
                elif event.key == pygame.K_RIGHT:
                    current_frame = min(len(frames) - 1, current_frame + 1)

            if confirm.handle_event(event):
                confirmed = True
                running = False
            if cancel.handle_event(event):
                running = False
            if prev_btn.handle_event(event):
                current_frame = max(0, current_frame - 1)
            if next_btn.handle_event(event):
                current_frame = min(len(frames) - 1, current_frame + 1)
            if reset.handle_event(event):
                brightness_s.value = 0
                contrast_s.value = 1.0
                gamma_s.value = 1.0
                filter_s.value = 0
            for s in sliders:
                s.handle_event(event)

        mouse = pygame.mouse.get_pos()
        for b in (confirm, cancel, reset, prev_btn, next_btn):
            b.update(mouse)

        settings.brightness = int(brightness_s.value)
        settings.contrast = contrast_s.value
        settings.gamma = gamma_s.value
        settings.filter_mode = int(filter_s.value)

        processed = apply_all_adjustments(
            frames[current_frame], settings.brightness, settings.contrast,
            settings.gamma, settings.filter_mode)
        processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

        preview_w = panel_x - 60
        preview_h = H - 160
        img_h, img_w = processed_rgb.shape[:2]
        scale = min(preview_w / img_w, preview_h / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        processed_resized = cv2.resize(processed_rgb, (new_w, new_h))
        surf = pygame.surfarray.make_surface(processed_resized.swapaxes(0, 1))

        _draw_vertical_gradient(screen, COLORS["bg"], COLORS["bg_soft"])

        # Title strip
        pygame.draw.rect(screen, COLORS["panel"], (0, 0, W, 70))
        pygame.draw.line(screen, COLORS["border"], (0, 70), (W, 70))
        draw_logo(screen, (36, 35), 20)
        screen.blit(font_title.render("Image Settings Preview", True,
                                       COLORS["text"]), (72, 22))
        screen.blit(font_small.render(
            f"Frame {current_frame + 1} / {len(frames)}", True,
            COLORS["text_dim"]), (72, 46))

        # Preview card
        img_x = 30 + (preview_w - new_w) // 2
        img_y = 90 + (preview_h - new_h) // 2
        card = pygame.Rect(20, 90, panel_x - 50, H - 160)
        pygame.draw.rect(screen, COLORS["panel"], card, border_radius=10)
        pygame.draw.rect(screen, COLORS["border"], card, 1, border_radius=10)
        screen.blit(surf, (img_x, img_y))

        # Settings card
        panel_rect = pygame.Rect(panel_x, panel_y, panel_w, 350)
        pygame.draw.rect(screen, COLORS["panel"], panel_rect,
                         border_radius=10)
        pygame.draw.rect(screen, COLORS["border"], panel_rect, 1,
                         border_radius=10)
        screen.blit(font_medium.render("Adjust", True, COLORS["text"]),
                    (panel_x + 20, panel_y + 16))

        for s in sliders:
            s.draw(screen, font_small)
        screen.blit(font_small.render(
            FILTER_NAMES[settings.filter_mode], True, COLORS["text_dim"]),
            (panel_x + 20, panel_y + 260))
        reset.draw(screen, font_small)

        for b in (confirm, cancel, prev_btn, next_btn):
            b.draw(screen, font_small)

        screen.blit(font_small.render(
            "← → Navigate · Drag sliders · Enter confirm · Esc cancel",
            True, COLORS["text_dim"]),
            (24, H - 25))

        pygame.display.flip()
        clock.tick(60)

    return confirmed


# ============================================================== manual class.
def manual_cell_classification(screen, frame, detections, cell_types):
    """Let the user assign a class label to every detection."""
    import cv2

    pygame.init()
    clock = pygame.time.Clock()
    W, H = screen.get_size()

    font_title = pygame.font.Font(None, 30)
    font_medium = pygame.font.Font(None, 22)
    font_small = pygame.font.Font(None, 19)

    # Modern palette for type badges
    type_palette = [
        (8, 145, 178), (234, 88, 12), (22, 163, 74), (220, 38, 38),
        (147, 51, 234), (161, 98, 7), (219, 39, 119), (71, 85, 105),
    ]

    assigned = {}
    idx = 0
    total = len(detections)
    if total == 0:
        return {}

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    panel_x = W - 310
    panel_w = 280
    panel_y = 90

    type_buttons = []
    for i, ct in enumerate(cell_types):
        color = type_palette[i % len(type_palette)]
        r = pygame.Rect(panel_x + 15, panel_y + 70 + i * 58,
                        panel_w - 30, 48)
        type_buttons.append({"rect": r, "type": ct, "color": color, "num": i + 1})

    prev_btn = Button(panel_x + 15, panel_y + 70 + len(cell_types) * 58 + 20,
                      125, 38, "‹ Back")
    skip_btn = Button(panel_x + 155, panel_y + 70 + len(cell_types) * 58 + 20,
                      125, 38, "Skip ›")

    done_btn = Button(W - 300, H - 60, 130, 40, "Done", kind="primary")
    cancel_btn = Button(W - 160, H - 60, 130, 40, "Cancel")

    running = True
    result = None

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                elif event.key == pygame.K_LEFT:
                    idx = max(0, idx - 1)
                elif event.key == pygame.K_RIGHT:
                    idx = min(total - 1, idx + 1)
                elif event.key == pygame.K_RETURN and len(assigned) == total:
                    return assigned
                elif event.key in [getattr(pygame, f"K_{n}") for n in range(1, 9)]:
                    i = event.key - pygame.K_1
                    if i < len(cell_types):
                        assigned[idx] = cell_types[i]
                        if idx < total - 1:
                            idx += 1

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for tb in type_buttons:
                    if tb["rect"].collidepoint(event.pos):
                        assigned[idx] = tb["type"]
                        if idx < total - 1:
                            idx += 1
                        break

            if prev_btn.handle_event(event):
                idx = max(0, idx - 1)
            if skip_btn.handle_event(event):
                idx = min(total - 1, idx + 1)
            if done_btn.handle_event(event) and len(assigned) == total:
                return assigned
            if cancel_btn.handle_event(event):
                return None

        mouse = pygame.mouse.get_pos()
        for b in (prev_btn, skip_btn, done_btn, cancel_btn):
            b.update(mouse)

        # --- draw ---
        _draw_vertical_gradient(screen, COLORS["bg"], COLORS["bg_soft"])
        # header
        pygame.draw.rect(screen, COLORS["panel"], (0, 0, W, 70))
        pygame.draw.line(screen, COLORS["border"], (0, 70), (W, 70))
        draw_logo(screen, (36, 35), 20)
        screen.blit(font_title.render("Manual Cell Classification", True,
                                       COLORS["text"]), (72, 22))
        screen.blit(font_small.render(
            f"Cell {idx + 1} / {total}   ·   Assigned {len(assigned)} / {total}",
            True, COLORS["text_dim"]), (72, 46))

        # preview card on left
        card = pygame.Rect(20, 90, panel_x - 50, H - 160)
        pygame.draw.rect(screen, COLORS["panel"], card, border_radius=10)
        pygame.draw.rect(screen, COLORS["border"], card, 1, border_radius=10)

        # render frame with current cell highlighted
        det = detections[idx]
        x, y, w, h = det.bbox
        preview = frame_rgb.copy()
        # darken outside of selection, highlight rectangle
        import cv2 as _cv2
        overlay = preview.copy()
        _cv2.rectangle(overlay, (x, y), (x + w, y + h), (8, 145, 178), 3)
        # crop a zoom window
        pad = max(40, max(w, h))
        cx1, cy1 = max(0, x - pad), max(0, y - pad)
        cx2 = min(frame_rgb.shape[1], x + w + pad)
        cy2 = min(frame_rgb.shape[0], y + h + pad)

        preview_w = card.width - 20
        preview_h = card.height - 20
        fh, fw = overlay.shape[:2]
        scale = min(preview_w / fw, preview_h / fh)
        new_w, new_h = int(fw * scale), int(fh * scale)
        resized = _cv2.resize(overlay, (new_w, new_h))
        surf = pygame.surfarray.make_surface(resized.swapaxes(0, 1))
        screen.blit(surf,
                    (card.x + (card.width - new_w) // 2,
                     card.y + (card.height - new_h) // 2))

        # right panel: type list
        panel_h = 70 + len(cell_types) * 58 + 120
        panel_rect = pygame.Rect(panel_x, panel_y, panel_w, panel_h)
        pygame.draw.rect(screen, COLORS["panel"], panel_rect, border_radius=10)
        pygame.draw.rect(screen, COLORS["border"], panel_rect, 1,
                         border_radius=10)
        screen.blit(font_medium.render("Select type", True, COLORS["text"]),
                    (panel_x + 15, panel_y + 18))
        screen.blit(font_small.render(
            "Click a button, or press 1-8", True, COLORS["text_dim"]),
            (panel_x + 15, panel_y + 44))

        current_type = assigned.get(idx)
        for tb in type_buttons:
            r = tb["rect"]
            selected = current_type == tb["type"]
            bg = COLORS["panel_hover"] if r.collidepoint(mouse) else COLORS["panel"]
            if selected:
                bg = tuple(min(255, c + 20) for c in tb["color"])
            pygame.draw.rect(screen, bg, r, border_radius=8)
            border = tb["color"] if r.collidepoint(mouse) or selected else COLORS["border_strong"]
            pygame.draw.rect(screen, border, r, 2 if selected else 1,
                             border_radius=8)

            # color swatch
            sw = pygame.Rect(r.x + 12, r.y + 14, 20, 20)
            pygame.draw.rect(screen, tb["color"], sw, border_radius=5)

            txt_color = COLORS["primary_txt"] if selected else COLORS["text"]
            screen.blit(font_small.render(f"{tb['num']}. {tb['type']}",
                                          True, txt_color),
                        (r.x + 44, r.centery - 8))

        prev_btn.draw(screen, font_small)
        skip_btn.draw(screen, font_small)
        done_btn.draw(screen, font_small)
        cancel_btn.draw(screen, font_small)

        pygame.display.flip()
        clock.tick(60)

    return result
