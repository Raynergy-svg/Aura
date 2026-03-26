"""Aura Brand System — polished terminal UI inspired by Claude's design language.

Provides:
- AuraTheme: Named color palettes with 8+ switchable themes
- AuraBrand: Central brand engine — logo, banners, styled output primitives
- Theme switching via /theme command

Design philosophy: Aura is warm, intuitive, human-first. Her visual identity
uses soft purples, lavender, and rose tones — the complement to Buddy's
sharp cyan/green machine palette.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ─── ANSI Escape Helpers ────────────────────────────────────────────────────

def _rgb(r: int, g: int, b: int) -> str:
    """Return ANSI 24-bit foreground color escape."""
    return f"\033[38;2;{r};{g};{b}m"

def _bg_rgb(r: int, g: int, b: int) -> str:
    """Return ANSI 24-bit background color escape."""
    return f"\033[48;2;{r};{g};{b}m"

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
ITALIC = "\033[3m"
UNDERLINE = "\033[4m"
BLINK = "\033[5m"
INVERSE = "\033[7m"
STRIKETHROUGH = "\033[9m"


# ─── Theme System ───────────────────────────────────────────────────────────

@dataclass
class AuraTheme:
    """A named color theme for Aura's terminal output.

    All colors are (R, G, B) tuples for 24-bit terminal rendering.
    """

    name: str
    display_name: str
    description: str

    # Core identity
    primary: Tuple[int, int, int]       # Main brand color (logo, headers, Aura's name)
    secondary: Tuple[int, int, int]     # Accent / highlights
    accent: Tuple[int, int, int]        # Tertiary accent (sparkles, decorations)

    # Semantic colors
    success: Tuple[int, int, int]       # Positive / green
    warning: Tuple[int, int, int]       # Caution / amber
    error: Tuple[int, int, int]         # Critical / red
    info: Tuple[int, int, int]          # Informational / blue

    # Text colors
    text: Tuple[int, int, int]          # Primary text
    text_dim: Tuple[int, int, int]      # Secondary/muted text
    text_bright: Tuple[int, int, int]   # Emphasized text

    # UI chrome
    border: Tuple[int, int, int]        # Box borders, dividers
    bg_subtle: Tuple[int, int, int]     # Subtle background tint (for highlights)

    # Buddy relationship
    buddy_color: Tuple[int, int, int]   # Color for Buddy-related info

    def fg(self, color_name: str) -> str:
        """Get foreground ANSI code for a named color."""
        c = getattr(self, color_name, self.text)
        return _rgb(*c)

    def bg(self, color_name: str) -> str:
        """Get background ANSI code for a named color."""
        c = getattr(self, color_name, self.bg_subtle)
        return _bg_rgb(*c)


# ─── Built-in Themes ────────────────────────────────────────────────────────

THEMES: Dict[str, AuraTheme] = {}

def _register(theme: AuraTheme) -> AuraTheme:
    THEMES[theme.name] = theme
    return theme


# 1. Default — Lavender Eve (signature)
_register(AuraTheme(
    name="eve",
    display_name="Eve",
    description="Aura's signature — soft lavender and rose. Warm and intuitive.",
    primary=(183, 148, 244),      # Soft lavender
    secondary=(244, 164, 188),    # Rose pink
    accent=(255, 209, 148),       # Warm gold
    success=(134, 211, 155),      # Sage green
    warning=(255, 196, 112),      # Warm amber
    error=(255, 120, 120),        # Soft red
    info=(148, 196, 244),         # Sky blue
    text=(230, 225, 240),         # Warm white
    text_dim=(140, 135, 155),     # Muted lavender-gray
    text_bright=(255, 250, 255),  # Pure bright
    border=(120, 100, 160),       # Muted purple border
    bg_subtle=(40, 30, 55),       # Deep purple tint
    buddy_color=(100, 220, 200),  # Teal (Buddy's identity)
))

# 2. Midnight — deep blue and silver
_register(AuraTheme(
    name="midnight",
    display_name="Midnight",
    description="Deep twilight blues with silver accents. Focused and calm.",
    primary=(100, 140, 220),      # Steel blue
    secondary=(170, 180, 220),    # Silver blue
    accent=(220, 200, 255),       # Pale lilac
    success=(100, 200, 150),      # Mint
    warning=(230, 190, 100),      # Gold
    error=(220, 100, 110),        # Crimson
    info=(130, 170, 230),         # Cornflower
    text=(210, 215, 230),         # Cool white
    text_dim=(120, 125, 150),     # Slate gray
    text_bright=(240, 245, 255),  # Ice white
    border=(70, 80, 120),         # Midnight border
    bg_subtle=(20, 25, 45),       # Deep navy
    buddy_color=(80, 200, 180),   # Teal
))

# 3. Aurora — northern lights
_register(AuraTheme(
    name="aurora",
    display_name="Aurora",
    description="Northern lights — shifting greens, teals, and violet.",
    primary=(80, 220, 190),       # Aurora green
    secondary=(160, 120, 255),    # Electric violet
    accent=(255, 200, 80),        # Warm amber
    success=(120, 230, 160),      # Bright green
    warning=(255, 200, 100),      # Gold
    error=(255, 100, 130),        # Hot pink
    info=(100, 190, 255),         # Bright blue
    text=(220, 235, 230),         # Mint white
    text_dim=(120, 140, 135),     # Sage gray
    text_bright=(240, 255, 250),  # Bright mint
    border=(60, 120, 100),        # Deep teal border
    bg_subtle=(15, 35, 30),       # Deep forest
    buddy_color=(100, 200, 220),  # Sky blue
))

# 4. Ember — warm fire tones
_register(AuraTheme(
    name="ember",
    display_name="Ember",
    description="Warm embers and firelight. Cozy and grounded.",
    primary=(240, 150, 90),       # Warm orange
    secondary=(220, 120, 140),    # Dusty rose
    accent=(255, 220, 130),       # Golden
    success=(160, 210, 130),      # Leaf green
    warning=(255, 190, 80),       # Amber
    error=(240, 90, 90),          # Fire red
    info=(180, 170, 220),         # Soft purple
    text=(240, 230, 220),         # Warm white
    text_dim=(160, 145, 135),     # Warm gray
    text_bright=(255, 248, 240),  # Cream
    border=(140, 90, 60),         # Burnished copper
    bg_subtle=(45, 30, 20),       # Dark umber
    buddy_color=(100, 200, 180),  # Teal
))

# 5. Sakura — cherry blossom
_register(AuraTheme(
    name="sakura",
    display_name="Sakura",
    description="Cherry blossom pinks and soft greens. Gentle and serene.",
    primary=(255, 150, 180),      # Blossom pink
    secondary=(200, 170, 210),    # Wisteria
    accent=(255, 220, 200),       # Peach
    success=(150, 210, 160),      # Spring green
    warning=(240, 200, 120),      # Soft gold
    error=(230, 100, 120),        # Deep rose
    info=(160, 190, 230),         # Soft sky
    text=(240, 235, 240),         # Petal white
    text_dim=(170, 155, 165),     # Dusty pink-gray
    text_bright=(255, 245, 248),  # Snow
    border=(180, 120, 140),       # Blossom branch
    bg_subtle=(45, 30, 38),       # Deep plum
    buddy_color=(120, 210, 190),  # Sage teal
))

# 6. Void — monochrome minimalist
_register(AuraTheme(
    name="void",
    display_name="Void",
    description="Monochrome with a single accent. Minimal and focused.",
    primary=(200, 200, 210),      # Silver
    secondary=(160, 160, 170),    # Mid gray
    accent=(180, 140, 255),       # Single purple accent
    success=(160, 220, 170),      # Muted green
    warning=(220, 200, 130),      # Muted gold
    error=(220, 130, 130),        # Muted red
    info=(150, 180, 220),         # Muted blue
    text=(200, 200, 205),         # Silver text
    text_dim=(110, 110, 118),     # Dark gray
    text_bright=(240, 240, 245),  # White
    border=(80, 80, 88),          # Charcoal border
    bg_subtle=(25, 25, 28),       # Near black
    buddy_color=(130, 200, 190),  # Muted teal
))

# 7. Solaris — warm sunlit gold
_register(AuraTheme(
    name="solaris",
    display_name="Solaris",
    description="Sunlit gold and warm earth. Optimistic and energizing.",
    primary=(255, 200, 80),       # Solar gold
    secondary=(240, 170, 100),    # Warm tangerine
    accent=(255, 230, 160),       # Pale gold
    success=(140, 210, 130),      # Green
    warning=(255, 180, 70),       # Deep amber
    error=(240, 100, 100),        # Red
    info=(140, 190, 240),         # Blue
    text=(240, 235, 220),         # Warm white
    text_dim=(160, 150, 130),     # Sandy gray
    text_bright=(255, 250, 235),  # Cream white
    border=(160, 130, 60),        # Antique gold border
    bg_subtle=(40, 35, 20),       # Deep bronze
    buddy_color=(100, 210, 200),  # Teal
))

# 8. Claude — Anthropic-inspired (homage)
_register(AuraTheme(
    name="claude",
    display_name="Claude",
    description="Anthropic-inspired warmth — sandy tones and signature orange.",
    primary=(217, 119, 87),       # Anthropic orange #d97757
    secondary=(106, 155, 204),    # Anthropic blue #6a9bcc
    accent=(120, 140, 93),        # Anthropic green #788c5d
    success=(120, 140, 93),       # Green
    warning=(217, 180, 87),       # Warm gold
    error=(200, 90, 80),          # Deep red
    info=(106, 155, 204),         # Blue
    text=(250, 249, 245),         # Anthropic light #faf9f5
    text_dim=(176, 174, 165),     # Anthropic mid-gray #b0aea5
    text_bright=(250, 249, 245),  # Light
    border=(176, 174, 165),       # Mid gray border
    bg_subtle=(20, 20, 19),       # Near #141413
    buddy_color=(106, 155, 204),  # Blue
))

DEFAULT_THEME = "eve"


# ─── Logo & Visual Assets ───────────────────────────────────────────────────

# ─── Visual Identity ───────────────────────────────────────────────────────
# Aura is an intelligence, not a character. No faces, no sparkle overload.
# Design: clean, warm, minimal — like a premium CLI. Think Claude, not clippy.

# Wordmark — the only logo. Clean sans-style block letters.
WORDMARK = [
    "  ▄▀█ █ █ █▀█ ▄▀█",
    "  █▀█ █▄█ █▀▄ █▀█",
]

# Compact — single line for tight spaces
WORDMARK_INLINE = "aura"

TAGLINE = "Human Intelligence Engine"
CODENAME = "Eve"
VERSION_LINE = "v1.0"


# ─── Box Drawing Characters ─────────────────────────────────────────────────

class BoxStyle:
    """Unicode box-drawing character sets."""

    # Rounded (default — softer, more human)
    ROUNDED = {
        "tl": "╭", "tr": "╮", "bl": "╰", "br": "╯",
        "h": "─", "v": "│",
        "lj": "├", "rj": "┤", "tj": "┬", "bj": "┴", "cross": "┼",
    }

    # Sharp (for emphasis / warnings)
    SHARP = {
        "tl": "┌", "tr": "┐", "bl": "└", "br": "┘",
        "h": "─", "v": "│",
        "lj": "├", "rj": "┤", "tj": "┬", "bj": "┴", "cross": "┼",
    }

    # Heavy (for headers)
    HEAVY = {
        "tl": "┏", "tr": "┓", "bl": "┗", "br": "┛",
        "h": "━", "v": "┃",
        "lj": "┣", "rj": "┫", "tj": "┳", "bj": "┻", "cross": "╋",
    }

    # Double (for major sections)
    DOUBLE = {
        "tl": "╔", "tr": "╗", "bl": "╚", "br": "╝",
        "h": "═", "v": "║",
        "lj": "╠", "rj": "╣", "tj": "╦", "bj": "╩", "cross": "╬",
    }


# ─── Decorative Elements ────────────────────────────────────────────────────

SPARKLES = ["✦", "✧", "·", "⋆", "✴", "✹"]
DOTS = ["●", "○", "◉", "◎", "◌"]
ARROWS = ["→", "←", "↑", "↓", "⟶", "⟵", "⇒", "⇐"]
SYMBOLS = {
    "check": "✓",
    "cross": "✗",
    "dot": "●",
    "ring": "○",
    "star": "★",
    "star_empty": "☆",
    "diamond": "◆",
    "diamond_empty": "◇",
    "heart": "♥",
    "arrow_right": "→",
    "arrow_left": "←",
    "arrow_up": "↑",
    "arrow_down": "↓",
    "wave": "∿",
    "infinity": "∞",
    "warning": "⚠",
    "lightning": "⚡",
    "flame": "🔥",
    "eye": "◈",
    "brain": "⟡",
    "pulse": "⏣",
    "shield": "◊",
}

# Progress bar characters
PROGRESS_FULL = "█"
PROGRESS_PARTIAL = ["░", "▒", "▓"]
PROGRESS_EMPTY = "░"
PROGRESS_SMOOTH = ["▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"]


# ─── The Brand Engine ────────────────────────────────────────────────────────

class AuraBrand:
    """Central brand engine — applies theme to all output primitives.

    Usage:
        brand = AuraBrand()
        brand.print_startup_banner()
        brand.print_header("Session Active")
        brand.print_info("Processing your message...")
        brand.print_success("Readiness updated: 78/100")
    """

    def __init__(self, theme_name: Optional[str] = None, config_path: Optional[Path] = None):
        self._config_path = config_path or Path(".aura/brand_config.json")
        self._theme_name = theme_name or self._load_saved_theme() or DEFAULT_THEME
        self._theme = THEMES.get(self._theme_name, THEMES[DEFAULT_THEME])
        self._term_width = self._get_terminal_width()
        self._animations_enabled = True

    # ── Theme Management ──

    @property
    def theme(self) -> AuraTheme:
        return self._theme

    @property
    def theme_name(self) -> str:
        return self._theme_name

    def set_theme(self, name: str) -> bool:
        """Switch to a named theme. Returns True if successful."""
        if name not in THEMES:
            return False
        self._theme_name = name
        self._theme = THEMES[name]
        self._save_theme(name)
        return True

    def list_themes(self) -> List[Dict[str, str]]:
        """Return list of available themes with metadata."""
        return [
            {
                "name": t.name,
                "display_name": t.display_name,
                "description": t.description,
                "active": t.name == self._theme_name,
            }
            for t in THEMES.values()
        ]

    def _load_saved_theme(self) -> Optional[str]:
        """Load saved theme preference from config."""
        try:
            if self._config_path.exists():
                data = json.loads(self._config_path.read_text())
                return data.get("theme", None)
        except Exception:
            pass
        return None

    def _save_theme(self, name: str) -> None:
        """Persist theme preference to config."""
        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            if self._config_path.exists():
                try:
                    data = json.loads(self._config_path.read_text())
                except Exception:
                    pass
            data["theme"] = name
            self._config_path.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    def _get_terminal_width(self) -> int:
        """Get terminal width, defaulting to 80."""
        try:
            return shutil.get_terminal_size().columns
        except Exception:
            return 80

    # ── Color Helpers ──

    def c(self, color_name: str) -> str:
        """Get foreground color escape for a theme color."""
        return self._theme.fg(color_name)

    def bg(self, color_name: str) -> str:
        """Get background color escape for a theme color."""
        return self._theme.bg(color_name)

    @property
    def R(self) -> str:
        """Reset shorthand."""
        return RESET

    # ── Logo & Banner ──

    def render_wordmark(self) -> str:
        """Render the AURA wordmark in theme colors."""
        p = self.c("primary")
        b = BOLD
        r = RESET

        lines = []
        for line in WORDMARK:
            lines.append(f"  {p}{b}{line}{r}")
        return "\n".join(lines)

    def render_startup_banner(self) -> str:
        """Render the startup banner — clean, minimal, warm.

        Design: no boxes, no ASCII art faces. Just the wordmark,
        a tagline, and a quiet status line. Like starting a conversation.
        """
        p = self.c("primary")
        s = self.c("secondary")
        a = self.c("accent")
        d = self.c("text_dim")
        tb = self.c("text_bright")
        b = BOLD
        r = RESET

        w = min(self._term_width, 60)
        thin_line = f"{d}{'─' * w}{r}"

        lines = []
        lines.append("")
        lines.append(thin_line)
        lines.append("")

        # Wordmark
        for wl in WORDMARK:
            lines.append(f"  {p}{b}{wl}{r}")

        lines.append("")
        lines.append(f"  {d}{TAGLINE}  ·  {ITALIC}{CODENAME}{r}")
        lines.append("")
        lines.append(thin_line)

        # Status line: theme + version
        lines.append(f"  {d}{self._theme.display_name} theme  ·  {VERSION_LINE}  ·  /theme to change{r}")
        lines.append("")

        return "\n".join(lines)

    def print_startup_banner(self) -> None:
        """Print the full startup banner to stdout."""
        print(self.render_startup_banner())

    # ── Section Headers ──

    def render_header(self, text: str, style: str = "heavy") -> str:
        """Render a section header with border."""
        p = self.c("primary")
        a = self.c("accent")
        b = BOLD
        r = RESET
        sym = SYMBOLS["diamond"]

        if style == "heavy":
            return f"\n  {p}{b}━━━ {a}{sym} {p}{text} {a}{sym} {p}{'━' * max(0, 40 - len(text))}━━━{r}\n"
        elif style == "light":
            return f"\n  {p}─── {a}·{p} {text} {a}·{p} {'─' * max(0, 40 - len(text))}───{r}\n"
        else:  # minimal
            return f"\n  {a}▸ {p}{b}{text}{r}\n"

    def print_header(self, text: str, style: str = "heavy") -> None:
        print(self.render_header(text, style))

    # ── Status Messages ──

    def render_success(self, text: str) -> str:
        s = self.c("success")
        return f"  {s}{SYMBOLS['check']} {text}{RESET}"

    def render_warning(self, text: str) -> str:
        w = self.c("warning")
        return f"  {w}{SYMBOLS['warning']} {text}{RESET}"

    def render_error(self, text: str) -> str:
        e = self.c("error")
        return f"  {e}{SYMBOLS['cross']} {text}{RESET}"

    def render_info(self, text: str) -> str:
        i = self.c("info")
        return f"  {i}{SYMBOLS['diamond_empty']} {text}{RESET}"

    def render_dim(self, text: str) -> str:
        d = self.c("text_dim")
        return f"  {d}{text}{RESET}"

    def print_success(self, text: str) -> None:
        print(self.render_success(text))

    def print_warning(self, text: str) -> None:
        print(self.render_warning(text))

    def print_error(self, text: str) -> None:
        print(self.render_error(text))

    def print_info(self, text: str) -> None:
        print(self.render_info(text))

    def print_dim(self, text: str) -> None:
        print(self.render_dim(text))

    # ── Prompt ──

    def render_user_prompt(self) -> str:
        """Render the input prompt — soft arrow."""
        d = self.c("text_dim")
        return f"{d}→ {RESET}"

    def render_aura_prefix(self) -> str:
        """Render Aura's response prefix — eye glyph + name."""
        p = self.c("primary")
        a = self.c("accent")
        return f"\n{a}◈ {p}{BOLD}aura{RESET}"

    # ── Boxes ──

    def render_box(self, content: List[str], title: Optional[str] = None,
                   style: str = "rounded", width: Optional[int] = None,
                   color: str = "border") -> str:
        """Render content inside a box.

        Args:
            content: Lines of text to display inside box
            title: Optional title for the box header
            style: "rounded", "sharp", "heavy", or "double"
            width: Box width (auto-calculated if None)
            color: Theme color name for the border
        """
        box = getattr(BoxStyle, style.upper(), BoxStyle.ROUNDED)
        c = self.c(color)
        r = RESET

        # Calculate width
        if width is None:
            max_content = max((len(self._strip_ansi(line)) for line in content), default=0)
            if title:
                max_content = max(max_content, len(title) + 4)
            width = min(max_content + 4, self._term_width - 4)

        inner_w = width - 2

        lines = []

        # Top border with optional title
        if title:
            t_display = f" {title} "
            t_len = len(title) + 2
            remaining = inner_w - t_len
            lines.append(f"  {c}{box['tl']}{box['h'] * 2}{t_display}{'─' * max(0, remaining - 2)}{box['tr']}{r}")
        else:
            lines.append(f"  {c}{box['tl']}{box['h'] * inner_w}{box['tr']}{r}")

        # Content
        for line in content:
            stripped_len = len(self._strip_ansi(line))
            pad = max(0, inner_w - stripped_len - 2)
            lines.append(f"  {c}{box['v']}{r} {line}{' ' * pad} {c}{box['v']}{r}")

        # Bottom border
        lines.append(f"  {c}{box['bl']}{box['h'] * inner_w}{box['br']}{r}")

        return "\n".join(lines)

    def print_box(self, content: List[str], **kwargs) -> None:
        print(self.render_box(content, **kwargs))

    # ── Progress Bar ──

    def render_progress(self, value: float, max_value: float = 100, width: int = 30,
                        label: Optional[str] = None, color: Optional[str] = None) -> str:
        """Render a beautiful progress bar.

        Args:
            value: Current value
            max_value: Maximum value
            width: Bar width in characters
            label: Optional label to the right
            color: Auto-selected based on value if None
        """
        ratio = max(0.0, min(1.0, value / max_value if max_value > 0 else 0))
        filled = int(ratio * width)

        # Auto-color based on value
        if color is None:
            if ratio >= 0.7:
                color = "success"
            elif ratio >= 0.4:
                color = "warning"
            else:
                color = "error"

        c = self.c(color)
        d = self.c("text_dim")
        r = RESET

        bar = PROGRESS_FULL * filled + PROGRESS_EMPTY * (width - filled)
        pct = f"{ratio * 100:.0f}%"

        result = f"  {c}{bar}{r} {pct}"
        if label:
            result += f"  {d}{label}{r}"

        return result

    def print_progress(self, value: float, **kwargs) -> None:
        print(self.render_progress(value, **kwargs))

    # ── Readiness Gauge (signature Aura widget) ──

    def render_readiness_gauge(self, score: float, label: str = "Readiness") -> str:
        """Render the readiness gauge — clean bar, score, status."""
        d = self.c("text_dim")
        b = BOLD
        r = RESET

        # Color based on score
        if score >= 70:
            sc = self.c("success")
            status = "ready"
        elif score >= 40:
            sc = self.c("warning")
            status = "caution"
        else:
            sc = self.c("error")
            status = "not ready"

        # Build gauge — simple bar
        gauge_width = 20
        filled = int((score / 100) * gauge_width)
        gauge = "█" * filled + "░" * (gauge_width - filled)

        return f"  {d}{label}  {sc}{b}{score:.0f}{r}{d}/100  {sc}{gauge}{r}  {d}{status}{r}"

    def print_readiness_gauge(self, score: float, **kwargs) -> None:
        print(self.render_readiness_gauge(score, **kwargs))

    # ── Table Rendering ──

    def render_table(self, headers: List[str], rows: List[List[str]],
                     title: Optional[str] = None) -> str:
        """Render a pretty table."""
        p = self.c("primary")
        a = self.c("accent")
        d = self.c("text_dim")
        t = self.c("text")
        brd = self.c("border")
        b = BOLD
        r = RESET

        # Calculate column widths
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(self._strip_ansi(str(cell))))

        lines = []

        # Title
        if title:
            total_w = sum(col_widths) + (len(col_widths) - 1) * 3 + 4
            lines.append(f"  {a}▸ {p}{b}{title}{r}")
            lines.append("")

        # Header
        header_cells = []
        for i, h in enumerate(headers):
            header_cells.append(f"{p}{b}{h:<{col_widths[i]}}{r}")
        lines.append(f"  {'  │  '.join(header_cells)}")

        # Separator
        sep_parts = [f"{brd}{'─' * col_widths[i]}" for i in range(len(headers))]
        lines.append(f"  {'──┼──'.join(sep_parts)}{r}")

        # Rows
        for row in rows:
            cells = []
            for i, cell in enumerate(row):
                cell_str = str(cell)
                stripped_len = len(self._strip_ansi(cell_str))
                w = col_widths[i] if i < len(col_widths) else 10
                pad = max(0, w - stripped_len)
                cells.append(f"{t}{cell_str}{' ' * pad}{r}")
            lines.append(f"  {'  │  '.join(cells)}")

        return "\n".join(lines)

    def print_table(self, headers: List[str], rows: List[List[str]], **kwargs) -> None:
        print(self.render_table(headers, rows, **kwargs))

    # ── Dividers ──

    def render_divider(self, style: str = "dots") -> str:
        """Render a decorative divider."""
        p = self.c("primary")
        d = self.c("text_dim")
        a = self.c("accent")
        r = RESET
        w = min(self._term_width - 4, 60)

        if style == "dots":
            return f"  {d}{'· ' * (w // 2)}{r}"
        elif style == "line":
            return f"  {d}{'─' * w}{r}"
        elif style == "heavy":
            return f"  {p}{'━' * w}{r}"
        elif style == "sparkle":
            pattern = "✦ · · ✧ · · " * ((w // 12) + 1)
            return f"  {a}{pattern[:w]}{r}"
        else:
            return f"  {d}{'─' * w}{r}"

    def print_divider(self, style: str = "dots") -> None:
        print(self.render_divider(style))

    # ── Buddy Status Badge ──

    def render_buddy_badge(self, connected: bool, regime: str = "unknown",
                           pnl: Optional[float] = None) -> str:
        """Render Buddy connection status badge."""
        bc = self.c("buddy_color")
        sc = self.c("success") if connected else self.c("text_dim")
        d = self.c("text_dim")
        r = RESET

        status = f"{sc}● connected" if connected else f"{d}○ offline"
        result = f"  {bc}{BOLD}buddy{r} {status}{r}"

        if connected and pnl is not None:
            pnl_c = self.c("success") if pnl >= 0 else self.c("error")
            result += f"  {d}│{r}  {d}P/L: {pnl_c}{pnl:+.2f}{r}  {d}│{r}  {d}regime: {bc}{regime}{r}"

        return result

    # ── Theme Showcase ──

    def render_theme_list(self) -> str:
        """Render the list of available themes for /theme command."""
        p = self.c("primary")
        a = self.c("accent")
        d = self.c("text_dim")
        s = self.c("secondary")
        b = BOLD
        r = RESET

        lines = []
        lines.append(self.render_header("Available Themes", style="heavy"))
        lines.append("")

        for name, theme in THEMES.items():
            active = "  ✦ " if name == self._theme_name else "    "
            tc = _rgb(*theme.primary)
            sc = _rgb(*theme.secondary)
            ac = _rgb(*theme.accent)

            # Color swatches
            swatch = f"{tc}██{sc}██{ac}██{r}"

            active_label = f" {a}(active){r}" if name == self._theme_name else ""
            lines.append(f"  {active}{tc}{b}{theme.display_name:<12}{r} {swatch}  {d}{theme.description}{r}{active_label}")

        lines.append("")
        lines.append(f"  {d}Usage: /theme <name>  ·  e.g. /theme sakura{r}")

        return "\n".join(lines)

    def print_theme_list(self) -> None:
        print(self.render_theme_list())

    # ── Session Start/End ──

    def render_session_start(self, connected_to_buddy: bool = False) -> str:
        """Render session start — just a quiet hint."""
        d = self.c("text_dim")
        r = RESET

        return f"  {d}Just talk. /help if you need it.{r}\n"

    def render_session_end(self) -> str:
        """Render graceful session end — brief."""
        d = self.c("text_dim")
        r = RESET

        lines = []
        lines.append("")
        lines.append(f"  {d}Session saved.{r}")
        lines.append("")

        return "\n".join(lines)

    # ── Thinking / Processing Indicators ──
    #
    # Design: animated spinner like Claude Code — braille dots cycle
    # in a background thread while the pipeline computes. Clean erase when done.

    _ERASE_LINE = "\033[2K"  # ANSI: erase entire current line
    _thinking_thread: Any = None
    _thinking_stop: Any = None

    def print_thinking(self, label: str = "thinking") -> None:
        """Start an animated thinking spinner in a background thread.

        Cycles through braille dot frames (⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏) giving
        smooth animation like Claude Code's indicator.
        Call clear_thinking() to stop and erase.
        """
        import sys
        import threading

        # Stop any existing spinner first
        self.clear_thinking()

        stop_event = threading.Event()
        self._thinking_stop = stop_event

        d = self.c("text_dim")
        frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

        def _animate():
            i = 0
            while not stop_event.is_set():
                frame = frames[i % len(frames)]
                sys.stdout.write(f"\r{self._ERASE_LINE}  {d}{frame} {ITALIC}{label}...{RESET}")
                sys.stdout.flush()
                i += 1
                stop_event.wait(0.08)

        t = threading.Thread(target=_animate, daemon=True)
        self._thinking_thread = t
        t.start()

    def clear_thinking(self) -> None:
        """Stop the spinner and erase the line."""
        import sys

        if self._thinking_stop is not None:
            self._thinking_stop.set()
        if self._thinking_thread is not None:
            self._thinking_thread.join(timeout=0.5)
            self._thinking_thread = None
            self._thinking_stop = None

        sys.stdout.write(f"\r{self._ERASE_LINE}")
        sys.stdout.flush()

    # ── Response Formatting ──

    def format_response(self, text: str) -> str:
        """Format Aura's response text — word-wrapped, indented, clean.

        Handles multi-line responses with proper indentation so text
        flows like Claude's output — left-aligned under the prefix.
        """
        import textwrap

        max_w = min(self._term_width - 6, 72)
        lines = text.split("\n")
        formatted = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                formatted.append("")
                continue

            # Preserve lines that look like data/tables/headers (contain ═, ─, │, etc.)
            if any(ch in stripped for ch in "═─│┼━┃╔╗╚╝╭╮╰╯├┤┬┴"):
                formatted.append(f"  {line}")
                continue

            # Preserve short lines and lines starting with symbols
            if len(stripped) < max_w or stripped[0] in "•·▸→✓✗⚠◇●○■□":
                formatted.append(f"  {stripped}")
                continue

            # Word-wrap long prose lines
            wrapped = textwrap.fill(stripped, width=max_w)
            for wl in wrapped.split("\n"):
                formatted.append(f"  {wl}")

        return "\n".join(formatted)

    # ── Signal Dashboard Panel ──

    def render_signal_panel(self, readiness: Any = None, signals: Any = None,
                            active_stressors: List[str] = None) -> Optional[str]:
        """Render a compact signal dashboard — only when something is off.

        The panel stays hidden during calm, normal conversation. It only
        surfaces when readiness drops dangerously low, biases spike,
        tilt/fatigue are elevated, overrides are detected, or stress is high.
        This keeps Aura feeling like a quiet presence, not a monitoring station.
        """
        items: List[str] = []
        d = self.c("text_dim")
        r = RESET

        if readiness is not None:
            score = getattr(readiness, "readiness_score", None)

            if score is not None:
                # Only show readiness gauge when it's actually concerning
                if score < 30:
                    sc = self.c("error")
                    gauge_w = 16
                    filled = int((score / 100) * gauge_w)
                    gauge = "█" * filled + "░" * (gauge_w - filled)
                    items.append(f"{d}readiness  {sc}{BOLD}{score:.0f}{r}{d}/100  {sc}{gauge}{r}  {sc}not ready{r}")

                # Tilt — only at meaningful levels
                tilt = getattr(readiness, "tilt_score", 0)
                if tilt > 0.45:
                    wc = self.c("warning")
                    items.append(f"{wc}⚠ tilt risk  {BOLD}{tilt:.0%}{r}")

                # Decision fatigue — only when it matters
                fatigue = getattr(readiness, "fatigue_score", 0)
                if fatigue > 0.55:
                    wc = self.c("warning")
                    items.append(f"{wc}⚠ fatigue  {BOLD}{fatigue:.0%}{r}")

                # Cognitive load — only when high or overloaded
                cog = getattr(readiness, "cognitive_load", None)
                if cog and cog in ("high", "overloaded"):
                    wc = self.c("warning")
                    items.append(f"{wc}cognitive load  {d}{cog}{r}")

            # Biases — only strong ones (>0.5)
            biases = getattr(readiness, "bias_scores", {})
            if biases:
                active = [(k, v) for k, v in sorted(biases.items(), key=lambda x: x[1], reverse=True) if v > 0.5][:3]
                if active:
                    wc = self.c("warning")
                    parts = [f"{name} {val:.0%}" for name, val in active]
                    items.append(f"{wc}⚠ biases  {d}{', '.join(parts)}{r}")

        if signals is not None:
            # Only flag genuinely distressed emotional states
            valence = getattr(signals, "affect_valence", 0)
            arousal = getattr(signals, "affect_arousal", 0)
            if valence < -0.4 and arousal > 0.6:
                ec = self.c("error")
                items.append(f"{ec}⚠ high stress detected{r}")

            # Override — always important
            override = getattr(signals, "override_mentioned", False)
            if override:
                ec = self.c("error")
                items.append(f"{ec}{BOLD}⚠ override detected{r}")

        if active_stressors and len(active_stressors) >= 2:
            wc = self.c("warning")
            items.append(f"{wc}stressors  {d}{', '.join(active_stressors)}{r}")

        # Only render if there's something truly worth surfacing
        if not items:
            return None

        return self.render_box(items, title="signals", style="rounded", color="text_dim")

    # ── Utility ──

    @staticmethod
    def _strip_ansi(text: str) -> str:
        """Remove ANSI escape sequences for length calculation."""
        import re
        return re.sub(r'\033\[[0-9;]*m', '', text)

    def wrap_text(self, text: str, width: Optional[int] = None, indent: int = 2) -> str:
        """Word-wrap text to terminal width with indent."""
        import textwrap
        w = (width or self._term_width) - indent
        wrapped = textwrap.fill(text, width=w)
        prefix = " " * indent
        return "\n".join(prefix + line for line in wrapped.split("\n"))


# ─── Singleton Access ────────────────────────────────────────────────────────

_brand_instance: Optional[AuraBrand] = None

def get_brand(theme_name: Optional[str] = None) -> AuraBrand:
    """Get or create the global AuraBrand instance."""
    global _brand_instance
    if _brand_instance is None:
        _brand_instance = AuraBrand(theme_name=theme_name)
    return _brand_instance

def reset_brand(theme_name: Optional[str] = None) -> AuraBrand:
    """Reset and recreate the global AuraBrand instance."""
    global _brand_instance
    _brand_instance = AuraBrand(theme_name=theme_name)
    return _brand_instance
