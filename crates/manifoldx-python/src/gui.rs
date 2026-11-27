use std::sync::Arc;

use manifoldx_render::gui::{Box2, Gui, Node, Pointer};
use manifoldx_render::gui_paint::{Frame, Painter, Style, Text, Widget};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

type Layout = (
    [i8; 2],
    [f32; 2],
    bool,
    Box2,
    f32,
    [Option<f32>; 2],
    Option<f32>,
);
type Appearance = (Box2, Box2, Box2, f32, f32);
type Label = (String, String, f32);
type Image = (u32, [f32; 2], bool, Box2);
type Content = (u8, Option<Label>, f32, [f32; 2], Option<Image>);
pub(crate) type Declaration = (u64, Option<usize>, Layout, Appearance, Content);
pub(crate) type Hit = Option<(u64, Box2)>;

#[derive(Default)]
pub(crate) struct GuiState {
    layout: Gui,
    painter: Painter,
    pub(crate) frame: Arc<Frame>,
    pub(crate) before_post: bool,
}

impl GuiState {
    pub(crate) fn font(&mut self, key: String, data: &[u8]) -> PyResult<()> {
        self.painter.font(key, data).map_err(PyValueError::new_err)
    }

    pub(crate) fn update(
        &mut self,
        rows: Vec<Declaration>,
        viewport: [f32; 2],
        scale: f32,
        exact: bool,
    ) -> PyResult<()> {
        if !scale.is_finite() || scale <= 0.0 || rows.len() > 4096 {
            return Err(PyValueError::new_err("invalid GUI scale or widget count"));
        }
        let mut nodes = Vec::with_capacity(rows.len());
        let mut widgets = Vec::with_capacity(rows.len());
        for (id, parent, layout, appearance, content) in rows {
            let (anchor, offset, horizontal, padding, gap, size, flex) = layout;
            let (bg, fg, border_color, border, radius) = appearance;
            let (kind, label, value, stick, image) = content;
            let invalid = || PyValueError::new_err("invalid native GUI widget");
            let label = label.map(|(value, font, size)| Text { value, font, size });
            let widget = match kind {
                0 => Widget::Panel,
                1 => Widget::Text(label.ok_or_else(invalid)?, value),
                2 => Widget::Button(label.ok_or_else(invalid)?),
                3 => Widget::Slider(value),
                4 => Widget::Toggle(label.ok_or_else(invalid)?, value != 0.0),
                5 => Widget::Joystick(stick),
                6 => {
                    let (texture, size, flex, uv) = image.ok_or_else(invalid)?;
                    Widget::Image(texture, size, flex, uv)
                }
                _ => return Err(invalid()),
            };
            if let Widget::Text(text, _) | Widget::Button(text) | Widget::Toggle(text, _) = &widget
            {
                if !self.painter.has_font(&text.font) || !(1.0..=512.0).contains(&text.size) {
                    return Err(PyValueError::new_err("invalid native GUI font or size"));
                }
            }
            nodes.push(Node {
                id,
                parent,
                anchor,
                offset,
                horizontal,
                padding,
                gap,
                size,
                flex,
                intrinsic: self.painter.intrinsic(&widget),
                interactive: matches!(kind, 2..=5),
                captures: matches!(kind, 3 | 5),
            });
            widgets.push((
                widget,
                Style {
                    bg,
                    fg,
                    border_color,
                    border,
                    radius,
                },
            ));
        }
        self.layout
            .set_tree(nodes, viewport.map(|v| v / scale))
            .map_err(PyValueError::new_err)?;
        self.frame = Arc::new(
            self.painter
                .paint(&widgets, self.layout.placements(), scale, exact),
        );
        Ok(())
    }

    pub(crate) fn pointer(
        &mut self,
        id: u64,
        phase: &str,
        point: [f32; 2],
    ) -> PyResult<(bool, Hit)> {
        let phase = match phase {
            "down" => Pointer::Down,
            "move" => Pointer::Move,
            "up" => Pointer::Up,
            "wheel" => Pointer::Wheel,
            _ => return Err(PyValueError::new_err("unknown GUI pointer phase")),
        };
        let route = self.layout.route(id, phase, point);
        Ok((route.consumed, route.target))
    }

    pub(crate) fn fixture_omit_capture(&mut self, enabled: bool) {
        self.layout.fixture_omit_capture(enabled);
    }

    pub(crate) fn focus(&self) -> Option<u64> {
        self.layout.focus()
    }

    pub(crate) fn reset(&mut self) {
        self.layout.reset_input();
    }

    pub(crate) fn boxes(&self) -> Vec<(Box2, Box2)> {
        self.layout
            .placements()
            .iter()
            .map(|place| (place.rect, place.clip))
            .collect()
    }
}
