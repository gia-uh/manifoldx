//! Native widget painting. Layout boxes and clips are in UI pixels.

use std::collections::HashMap;
use std::sync::Arc;

use crate::gui::{Box2, Placement};
use crate::renderer::srgb_to_linear;

pub type Color = [f32; 4];
type PushQuad<'a> = dyn FnMut(Box2, Color, Color, f32, f32, Box2, Source) + 'a;

#[derive(Clone, Debug, PartialEq)]
pub struct Text {
    pub value: String,
    pub font: String,
    pub size: f32,
}

#[derive(Clone, Debug)]
pub enum Widget {
    Panel,
    Text(Text, f32),
    Button(Text),
    Slider(f32),
    Toggle(Text, bool),
    Joystick([f32; 2]),
    Image(u32, [f32; 2], bool, Box2),
}

#[derive(Clone, Debug)]
pub struct Style {
    pub bg: Color,
    pub fg: Color,
    pub border_color: Color,
    pub border: f32,
    pub radius: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    Solid,
    Text,
    Image(u32),
}

#[derive(Clone, Debug)]
pub struct Quad {
    pub rect: Box2,
    pub clip: Box2,
    pub fill: Color,
    pub border_color: Color,
    pub uv: Box2,
    pub radius: f32,
    pub border: f32,
    pub exact: bool,
    pub source: Source,
}

#[derive(Default)]
pub struct Frame {
    pub quads: Vec<Quad>,
    pub atlas: Option<Arc<Atlas>>,
}

pub struct Atlas {
    pub size: [u32; 2],
    pub alpha: Vec<u8>,
}

#[derive(Default)]
pub struct Painter {
    fonts: HashMap<String, fontdue::Font>,
    lines: Vec<(Text, Box2)>,
    atlas: Option<Arc<Atlas>>,
    scale: f32,
}

impl Painter {
    pub fn font(&mut self, id: String, data: &[u8]) -> Result<(), &'static str> {
        let font = fontdue::Font::from_bytes(data, fontdue::FontSettings::default())?;
        self.fonts.insert(id, font);
        self.lines.clear();
        Ok(())
    }

    pub fn has_font(&self, id: &str) -> bool {
        self.fonts.contains_key(id)
    }

    fn font_for(&self, text: &Text) -> &fontdue::Font {
        &self.fonts[&text.font]
    }

    fn size(&self, text: &Text) -> [f32; 2] {
        let font = self.font_for(text);
        let height = font
            .horizontal_line_metrics(text.size)
            .map_or(text.size, |m| m.new_line_size);
        let width = text
            .value
            .chars()
            .map(|c| font.metrics(c, text.size).advance_width)
            .sum::<f32>();
        [width.ceil(), height.ceil()]
    }

    pub fn intrinsic(&self, widget: &Widget) -> [f32; 2] {
        match widget {
            Widget::Panel => [0.0; 2],
            Widget::Slider(_) => [160.0, 16.0],
            Widget::Joystick(_) => [96.0; 2],
            Widget::Image(_, size, _, _) => *size,
            Widget::Text(text, min_width) => {
                let [w, h] = self.size(text);
                [w.max(*min_width), h]
            }
            Widget::Button(text) => {
                let [w, h] = self.size(text);
                [w + 24.0, h + 8.0]
            }
            Widget::Toggle(text, _) => {
                let [w, h] = self.size(text);
                [w + 20.0, h.max(14.0)]
            }
        }
    }

    fn prepare_atlas(&mut self, widgets: &[(Widget, Style)], scale: f32) {
        let mut unique = Vec::new();
        for (widget, _) in widgets {
            if let Widget::Text(text, _) | Widget::Button(text) | Widget::Toggle(text, _) = widget {
                if !text.value.is_empty() && !unique.contains(&text) {
                    unique.push(text);
                }
            }
        }
        if self.scale == scale
            && unique
                .iter()
                .copied()
                .eq(self.lines.iter().map(|(text, _)| text))
        {
            return;
        }
        self.scale = scale;
        self.lines.clear();
        if unique.is_empty() {
            self.atlas = None;
            return;
        }
        const SIDE: usize = 2048;
        let mut alpha = vec![0; SIDE * SIDE];
        let (mut left, mut top, mut row) = (1, 1, 0);
        for text in unique {
            let [w, h] = self
                .size(text)
                .map(|extent| (extent * scale).ceil() as usize);
            if w == 0 || h == 0 || w + 2 > SIDE || h + 2 > SIDE {
                continue;
            }
            if left + w >= SIDE {
                top += row + 1;
                left = 1;
                row = 0;
            }
            if top + h >= SIDE {
                break;
            }
            let font = self.font_for(text);
            let px = text.size * scale;
            let ascent = font
                .horizontal_line_metrics(px)
                .map_or(px, |m| m.ascent)
                .ceil() as i32;
            let mut cursor = 0.0_f32;
            for c in text.value.chars() {
                let (metrics, bitmap) = font.rasterize(c, px);
                let x = cursor.round() as i32 + metrics.xmin;
                let y = ascent - metrics.ymin - metrics.height as i32;
                for (r, pixels) in bitmap.chunks(metrics.width.max(1)).enumerate() {
                    let yy = y + r as i32;
                    if !(0..h as i32).contains(&yy) {
                        continue;
                    }
                    for (col, &coverage) in pixels.iter().enumerate() {
                        let xx = x + col as i32;
                        if (0..w as i32).contains(&xx) {
                            let destination =
                                &mut alpha[(top + yy as usize) * SIDE + left + xx as usize];
                            *destination = (*destination).max(coverage);
                        }
                    }
                }
                cursor += metrics.advance_width;
            }
            self.lines.push((
                text.clone(),
                [left as f32, top as f32, (left + w) as f32, (top + h) as f32],
            ));
            left += w + 1;
            row = row.max(h);
        }
        let height = (top + row + 1).min(SIDE);
        alpha.truncate(SIDE * height);
        for (_, uv) in &mut self.lines {
            for (i, coordinate) in uv.iter_mut().enumerate() {
                *coordinate /= if i % 2 == 0 { SIDE } else { height } as f32;
            }
        }
        self.atlas = Some(Arc::new(Atlas {
            size: [SIDE as u32, height as u32],
            alpha,
        }));
    }

    pub fn paint(
        &mut self,
        widgets: &[(Widget, Style)],
        placements: &[Placement],
        scale: f32,
        exact: bool,
    ) -> Frame {
        self.prepare_atlas(widgets, scale);
        let mut frame = Frame {
            quads: Vec::new(),
            atlas: self.atlas.clone(),
        };
        for ((widget, style), placement) in widgets.iter().zip(placements) {
            let [x, y, w, h] = placement.rect;
            let clip = placement.clip.map(|v| v * scale);
            let mut push = |rect: Box2, fill, outline, border, radius, uv, source| {
                if rect[2] > 0.0 && rect[3] > 0.0 && clip[2] > 0.0 && clip[3] > 0.0 {
                    frame.quads.push(Quad {
                        rect: rect.map(|v| v * scale),
                        clip,
                        fill: linear(fill, exact || source != Source::Solid),
                        border_color: linear(outline, exact),
                        uv,
                        radius: radius * scale,
                        border: border * scale,
                        exact,
                        source,
                    });
                }
            };
            let no_color = [0.0; 4];
            let no_uv = [0.0; 4];
            let paint_text = |text: &Text, left, top, push: &mut PushQuad<'_>| {
                if let Some((_, uv)) = self.lines.iter().find(|(item, _)| item == text) {
                    let [tw, th] = self.size(text);
                    push(
                        [left, top, tw, th],
                        style.fg,
                        no_color,
                        0.0,
                        0.0,
                        *uv,
                        Source::Text,
                    );
                }
            };
            match widget {
                Widget::Panel | Widget::Button(_) | Widget::Slider(_) | Widget::Joystick(_) => {
                    let radius = if matches!(widget, Widget::Joystick(_)) {
                        w.min(h) * 0.5
                    } else {
                        style.radius
                    };
                    push(
                        placement.rect,
                        style.bg,
                        style.border_color,
                        style.border,
                        radius,
                        no_uv,
                        Source::Solid,
                    );
                }
                _ => {}
            }
            match widget {
                Widget::Panel => {}
                Widget::Text(text, _) => paint_text(text, x, y, &mut push),
                Widget::Button(text) => {
                    let [tw, th] = self.size(text);
                    paint_text(text, x + (w - tw) * 0.5, y + (h - th) * 0.5, &mut push);
                }
                Widget::Slider(value) => {
                    push(
                        [x, y, w * value.clamp(0.0, 1.0), h],
                        style.fg,
                        no_color,
                        0.0,
                        style.radius,
                        no_uv,
                        Source::Solid,
                    );
                }
                Widget::Toggle(text, value) => {
                    let top = y + (h - 14.0) * 0.5;
                    push(
                        [x, top, 14.0, 14.0],
                        style.bg,
                        style.fg,
                        1.0,
                        2.0,
                        no_uv,
                        Source::Solid,
                    );
                    if *value {
                        push(
                            [x + 3.0, top + 3.0, 8.0, 8.0],
                            style.fg,
                            no_color,
                            0.0,
                            1.0,
                            no_uv,
                            Source::Solid,
                        );
                    }
                    let [_, th] = self.size(text);
                    paint_text(text, x + 20.0, y + (h - th) * 0.5, &mut push);
                }
                Widget::Joystick([vx, vy]) => {
                    let r = w.min(h) * 0.5;
                    let knob = 0.8 * r;
                    let cx = x + w * 0.5 + vx * (r - knob * 0.5);
                    let cy = y + h * 0.5 - vy * (r - knob * 0.5);
                    push(
                        [cx - knob * 0.5, cy - knob * 0.5, knob, knob],
                        style.fg,
                        no_color,
                        0.0,
                        knob * 0.5,
                        no_uv,
                        Source::Solid,
                    );
                }
                Widget::Image(id, [iw, ih], flex, uv) => {
                    if *iw > 0.0 && *ih > 0.0 {
                        let scale = (w / iw).min(h / ih);
                        let scale = if *flex { scale } else { scale.min(1.0) };
                        let (iw, ih) = (iw * scale, ih * scale);
                        push(
                            [x + (w - iw) * 0.5, y + (h - ih) * 0.5, iw, ih],
                            style.fg,
                            no_color,
                            0.0,
                            style.radius * scale,
                            *uv,
                            Source::Image(*id),
                        );
                    }
                }
            }
        }
        frame
    }
}

fn linear(mut rgba: Color, exact: bool) -> Color {
    if exact {
        let [r, g, b, _] = srgb_to_linear([rgba[0], rgba[1], rgba[2]]);
        rgba[..3].copy_from_slice(&[r, g, b]);
    }
    rgba
}
