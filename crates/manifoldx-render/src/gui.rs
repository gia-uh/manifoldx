//! Layout and interaction for the declarative desktop GUI.

use std::collections::{HashMap, HashSet};

pub type Box2 = [f32; 4];

#[derive(Clone, Debug)]
pub struct Node {
    pub id: u64,
    pub parent: Option<usize>,
    pub anchor: [i8; 2],
    pub offset: [f32; 2],
    pub horizontal: bool,
    pub padding: Box2,
    pub gap: f32,
    pub size: [Option<f32>; 2],
    pub flex: Option<f32>,
    pub intrinsic: [f32; 2],
    pub interactive: bool,
    pub captures: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Placement {
    pub rect: Box2,
    pub clip: Box2,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Pointer {
    Down,
    Move,
    Up,
    Wheel,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Route {
    pub consumed: bool,
    pub target: Option<(u64, Box2)>,
}

#[derive(Default)]
pub struct Gui {
    nodes: Vec<Node>,
    placements: Vec<Placement>,
    by_id: HashMap<u64, usize>,
    children: Vec<Vec<usize>>,
    roots: Vec<usize>,
    captures: HashMap<u64, u64>,
    focus: Option<u64>,
    omit_capture: bool,
}

impl Gui {
    pub fn set_tree(&mut self, nodes: Vec<Node>, viewport: [f32; 2]) -> Result<(), &'static str> {
        let mut by_id = HashMap::with_capacity(nodes.len());
        let mut children = vec![Vec::new(); nodes.len()];
        let mut roots = Vec::new();
        for (i, node) in nodes.iter().enumerate() {
            if by_id.insert(node.id, i).is_some() {
                return Err("duplicate GUI node id");
            }
            match node.parent {
                Some(parent) if parent < i => children[parent].push(i),
                Some(_) => return Err("GUI parent must precede its children"),
                None => roots.push(i),
            }
        }
        let live: HashSet<_> = nodes
            .iter()
            .filter(|node| node.interactive)
            .map(|node| node.id)
            .collect();
        self.captures.retain(|_, id| live.contains(id));
        self.focus = self.focus.filter(|id| live.contains(id));
        self.placements.resize(nodes.len(), Placement::default());
        self.nodes = nodes;
        self.by_id = by_id;
        self.children = children;
        self.roots = roots;
        let screen = [0.0, 0.0, viewport[0], viewport[1]];
        let mut content = vec![[0.0; 2]; self.nodes.len()];
        for i in (0..self.nodes.len()).rev() {
            let node = &self.nodes[i];
            let children = &self.children[i];
            let mut dimensions = node.intrinsic;
            if !children.is_empty() {
                let main = usize::from(!node.horizontal);
                let cross = 1 - main;
                dimensions = [
                    node.padding[1] + node.padding[3],
                    node.padding[0] + node.padding[2],
                ];
                dimensions[main] += children
                    .iter()
                    .map(|child| content[*child][main])
                    .sum::<f32>()
                    + node.gap * (children.len() - 1) as f32;
                dimensions[cross] += children
                    .iter()
                    .map(|child| content[*child][cross])
                    .fold(0.0, f32::max);
            }
            content[i] = std::array::from_fn(|axis| node.size[axis].unwrap_or(dimensions[axis]));
        }
        for &i in &self.roots {
            let node = &self.nodes[i];
            let fill = node.anchor == [-1, -1];
            let size: [f32; 2] = std::array::from_fn(|axis| {
                node.size[axis].unwrap_or(if fill {
                    viewport[axis]
                } else {
                    content[i][axis]
                })
            });
            let position: [f32; 2] = std::array::from_fn(|axis| match node.anchor[axis] {
                -1 => node.offset[axis],
                1 => viewport[axis] - size[axis] - node.offset[axis],
                _ => (viewport[axis] - size[axis]) * 0.5 + node.offset[axis],
            });
            self.placements[i] = Placement {
                rect: [position[0], position[1], size[0], size[1]],
                clip: screen,
            };
        }
        for (i, node) in self.nodes.iter().enumerate() {
            let children = &self.children[i];
            if children.is_empty() {
                continue;
            }
            let [x, y, w, h] = self.placements[i].rect;
            let [top, right, bottom, left] = node.padding;
            let inner = [
                x + left,
                y + top,
                (w - left - right).max(0.0),
                (h - top - bottom).max(0.0),
            ];
            let clip = intersection(self.placements[i].rect, self.placements[i].clip);
            let main = usize::from(!node.horizontal);
            let mut free = inner[main + 2] - node.gap * (children.len() - 1) as f32;
            let mut total = 0.0;
            for &child in children {
                let item = &self.nodes[child];
                match item.flex {
                    Some(weight) => total += weight,
                    None => free -= item.size[main].unwrap_or(item.intrinsic[main]),
                }
            }
            let mut cursor = inner[main];
            for &child in children {
                let item = &self.nodes[child];
                let extent = match item.flex {
                    Some(weight) if total > 0.0 => free.max(0.0) * weight / total,
                    Some(_) => 0.0,
                    None => item.size[main].unwrap_or(item.intrinsic[main]),
                };
                let mut rect = inner;
                rect[main] = cursor;
                rect[main + 2] = extent;
                self.placements[child] = Placement { rect, clip };
                cursor += extent + node.gap;
            }
        }
        Ok(())
    }

    pub fn placements(&self) -> &[Placement] {
        &self.placements
    }

    pub fn focus(&self) -> Option<u64> {
        self.focus
    }

    pub fn fixture_omit_capture(&mut self, enabled: bool) {
        self.omit_capture = enabled;
    }

    pub fn reset_input(&mut self) {
        self.captures.clear();
        self.focus = None;
    }

    pub fn route(&mut self, pointer: u64, phase: Pointer, point: [f32; 2]) -> Route {
        if phase == Pointer::Down {
            self.captures.remove(&pointer);
        } else if let Some(&id) = self.captures.get(&pointer) {
            if phase == Pointer::Up {
                self.captures.remove(&pointer);
            }
            return Route {
                consumed: true,
                target: (phase != Pointer::Wheel)
                    .then_some((id, self.placements[self.by_id[&id]].rect)),
            };
        }
        let hit = self.roots.iter().rev().find_map(|&i| self.hit(i, point));
        if phase == Pointer::Down {
            self.focus = hit.map(|i| self.nodes[i].id);
            if let Some(i) = hit.filter(|&i| self.nodes[i].captures && !self.omit_capture) {
                self.captures.insert(pointer, self.nodes[i].id);
            }
        }
        Route {
            consumed: hit.is_some(),
            target: hit
                .filter(|_| matches!(phase, Pointer::Down | Pointer::Up))
                .map(|i| (self.nodes[i].id, self.placements[i].rect)),
        }
    }

    fn hit(&self, i: usize, point: [f32; 2]) -> Option<usize> {
        let placement = self.placements[i];
        if !contains(intersection(placement.rect, placement.clip), point) {
            return None;
        }
        self.children[i]
            .iter()
            .find_map(|&child| self.hit(child, point))
            .or_else(|| self.nodes[i].interactive.then_some(i))
    }
}

fn contains([x, y, w, h]: Box2, [px, py]: [f32; 2]) -> bool {
    w > 0.0 && h > 0.0 && px >= x && px <= x + w && py >= y && py <= y + h
}

fn intersection(a: Box2, b: Box2) -> Box2 {
    let x = a[0].max(b[0]);
    let y = a[1].max(b[1]);
    [
        x,
        y,
        (a[0] + a[2]).min(b[0] + b[2]).max(x) - x,
        (a[1] + a[3]).min(b[1] + b[3]).max(y) - y,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn panel(id: u64, parent: Option<usize>) -> Node {
        Node {
            id,
            parent,
            anchor: [1, 1],
            offset: [5.0, 5.0],
            horizontal: false,
            padding: [0.0; 4],
            gap: 0.0,
            size: [Some(20.0), Some(10.0)],
            flex: None,
            intrinsic: [0.0; 2],
            interactive: false,
            captures: false,
        }
    }

    #[test]
    fn topmost_capture_survives_exit_and_resize_then_clears_on_removal() {
        let mut slider = panel(3, Some(1));
        slider.interactive = true;
        slider.captures = true;
        let nodes = vec![panel(1, None), panel(2, None), slider];
        let mut gui = Gui::default();
        gui.set_tree(nodes.clone(), [100.0, 100.0]).unwrap();
        assert_eq!(
            gui.route(0, Pointer::Down, [80.0, 90.0]).target,
            Some((3, [75.0, 85.0, 20.0, 10.0]))
        );
        assert_eq!(gui.focus(), Some(3));
        gui.set_tree(nodes, [200.0, 100.0]).unwrap();
        assert_eq!(
            gui.route(0, Pointer::Move, [0.0, 0.0]).target,
            Some((3, [175.0, 85.0, 20.0, 10.0]))
        );
        assert!(!gui.route(1, Pointer::Move, [0.0, 0.0]).consumed);
        gui.set_tree(vec![], [200.0, 100.0]).unwrap();
        assert_eq!(gui.focus(), None);
        assert!(!gui.route(0, Pointer::Up, [0.0, 0.0]).consumed);
    }
}
