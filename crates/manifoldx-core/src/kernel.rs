//! Per-entity evaluation of validated kernel IR on the frame thread.

use crate::ecs::{ElementType, World};

pub const VERSION: u32 = 1;
const MAX_VALUES: usize = 256;
pub const MAX_CODE: usize = 512;

#[derive(Clone, Copy)]
enum Arithmetic {
    Add,
    Sub,
    Mul,
}

#[derive(Clone, Copy)]
enum Compare {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Clone, Copy)]
enum Instruction {
    Literal(usize, f32),
    Load(usize, usize, usize),
    Store(usize, usize, usize),
    Copy(usize, usize),
    Neg(usize, usize, usize),
    Scalar(Arithmetic, usize, usize, usize),
    Vector(Arithmetic, usize, usize, usize, bool, bool),
    Clamp(usize, usize, usize, usize),
    Compare(Compare, usize, usize, usize),
    Branch(usize, usize),
    Jump(usize),
}

/// Private lowering of the typed IR. The wire format is internal to this wheel.
pub struct Program {
    name: String,
    component: String,
    width: usize,
    code: Vec<Instruction>,
    writes: bool,
}

// op, destination, operands (or field offset / jump address), and literal.
pub type Word = (String, usize, usize, usize, usize, f32);

impl Program {
    pub fn decode(
        version: u32,
        name: String,
        component: String,
        width: usize,
        words: Vec<Word>,
    ) -> Result<Self, &'static str> {
        if version != VERSION {
            return Err("unsupported kernel IR version");
        }
        if !(1..=256).contains(&width) || words.len() > MAX_CODE {
            return Err("kernel schema or code exceeds limits");
        }
        let mut code = Vec::with_capacity(words.len());
        let mut writes = false;
        let len = words.len();
        for (pc, (op, out, a, b, c, literal)) in words.into_iter().enumerate() {
            if out >= MAX_VALUES {
                return Err("kernel register index exceeds limits");
            }
            let reg = |i| {
                if i < MAX_VALUES {
                    Ok(i)
                } else {
                    Err("kernel register index exceeds limits")
                }
            };
            let size = |n| {
                if n == 1 || n == 3 {
                    Ok(n)
                } else {
                    Err("kernel value must be f32 or vec3")
                }
            };
            let field = |offset: usize, n| {
                if offset <= width && size(n)? <= width - offset {
                    Ok(offset)
                } else {
                    Err("kernel field is outside its schema")
                }
            };
            let jump = |target| {
                if target > pc && target <= len {
                    Ok(target)
                } else {
                    Err("kernel jump must point forward within the code")
                }
            };
            if out == 0 && !matches!(op.as_str(), "store" | "branch" | "jump") {
                return Err("kernel uniform is read-only");
            }
            code.push(match op.as_str() {
                "literal" if literal.is_finite() => Instruction::Literal(out, literal),
                "load" => Instruction::Load(out, field(a, b)?, b),
                "store" => {
                    writes = true;
                    Instruction::Store(field(a, b)?, b, reg(out)?)
                }
                "copy" => Instruction::Copy(out, reg(a)?),
                "neg" => Instruction::Neg(out, reg(a)?, size(b)?),
                "add" | "sub" | "mul" => {
                    let arithmetic = match op.as_str() {
                        "add" => Arithmetic::Add,
                        "sub" => Arithmetic::Sub,
                        _ => Arithmetic::Mul,
                    };
                    match c {
                        0 => Instruction::Scalar(arithmetic, out, reg(a)?, reg(b)?),
                        1..=3 if c == 3 || op == "mul" => {
                            Instruction::Vector(arithmetic, out, reg(a)?, reg(b)?, c != 1, c != 2)
                        }
                        _ => return Err("invalid kernel arithmetic widths"),
                    }
                }
                "clamp" => Instruction::Clamp(out, reg(a)?, reg(b)?, reg(c)?),
                "eq" | "ne" | "lt" | "le" | "gt" | "ge" => {
                    let comparison = match op.as_str() {
                        "eq" => Compare::Eq,
                        "ne" => Compare::Ne,
                        "lt" => Compare::Lt,
                        "le" => Compare::Le,
                        "gt" => Compare::Gt,
                        _ => Compare::Ge,
                    };
                    Instruction::Compare(comparison, out, reg(a)?, reg(b)?)
                }
                "branch" => Instruction::Branch(reg(a)?, jump(b)?),
                "jump" => Instruction::Jump(jump(a)?),
                _ => return Err("invalid kernel instruction"),
            });
        }
        validate_registers(&code)?;
        Ok(Self {
            name,
            component,
            width,
            code,
            writes,
        })
    }

    pub fn validate_column(&self, world: &World, column: usize) -> Result<(), &'static str> {
        let schema = world.schema(column).ok_or("unknown kernel column")?;
        if schema.name != self.component
            || schema.width as usize != self.width
            || schema.element != ElementType::F32
        {
            return Err("kernel column differs from the compiled schema");
        }
        Ok(())
    }

    pub fn dispatch(
        &self,
        world: &mut World,
        column: usize,
        dt: f32,
    ) -> Result<usize, &'static str> {
        self.validate_column(world, column)?;
        if !dt.is_finite() || dt < 0.0 {
            return Err("kernel dt must be finite and nonnegative");
        }
        let mut values = [[0.0; 3]; MAX_VALUES];
        values[0][0] = dt;
        world.visit_kernel_rows(column, self.writes, |row| {
            self.evaluate(row, &mut values);
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    fn evaluate(&self, row: &mut [f32], values: &mut [[f32; 3]; MAX_VALUES]) {
        let mut pc = 0;
        while pc < self.code.len() {
            match self.code[pc] {
                Instruction::Literal(out, value) => values[out][0] = value,
                Instruction::Load(out, offset, n) => {
                    values[out][..n].copy_from_slice(&row[offset..offset + n]);
                }
                Instruction::Store(offset, n, src) => {
                    row[offset..offset + n].copy_from_slice(&values[src][..n]);
                }
                Instruction::Copy(out, src) => values[out] = values[src],
                Instruction::Neg(out, src, n) => {
                    let source = values[src];
                    for (target, value) in values[out][..n].iter_mut().zip(source) {
                        *target = -value;
                    }
                }
                Instruction::Scalar(op, out, a, b) => {
                    values[out][0] = op.apply(values[a][0], values[b][0]);
                }
                Instruction::Vector(op, out, a, b, a_vector, b_vector) => {
                    let a = if a_vector {
                        values[a]
                    } else {
                        [values[a][0]; 3]
                    };
                    let b = if b_vector {
                        values[b]
                    } else {
                        [values[b][0]; 3]
                    };
                    values[out] = [
                        op.apply(a[0], b[0]),
                        op.apply(a[1], b[1]),
                        op.apply(a[2], b[2]),
                    ];
                }
                Instruction::Clamp(out, src, lo, hi) => {
                    values[out][0] = values[src][0].max(values[lo][0]).min(values[hi][0]);
                }
                Instruction::Compare(op, out, a, b) => {
                    values[out][0] = u8::from(op.test(values[a][0], values[b][0])) as f32;
                }
                Instruction::Branch(src, target) if values[src][0] == 0.0 => {
                    pc = target;
                    continue;
                }
                Instruction::Branch(..) => {}
                Instruction::Jump(target) => {
                    pc = target;
                    continue;
                }
            }
            pc += 1;
        }
    }
}

// A register must have one type on every incoming path before it can be
// read. Branches are forward-only, so each predecessor is visited once.
fn validate_registers(code: &[Instruction]) -> Result<(), &'static str> {
    const UNSET: u8 = 0;
    const F32: u8 = 1;
    const BOOL: u8 = 2;
    const VEC3: u8 = 3;
    let mut incoming = vec![None; code.len() + 1];
    let mut entry = [UNSET; MAX_VALUES];
    entry[0] = F32;
    incoming[0] = Some(entry);
    for (pc, op) in code.iter().enumerate() {
        let Some(mut types) = incoming[pc].take() else {
            return Err("unreachable kernel instruction");
        };
        let operand = |index: usize, expected: u8| {
            if types[index] == expected {
                Ok(())
            } else {
                Err("kernel operand is uninitialized or has the wrong type")
            }
        };
        let mut edge = |target: usize, types: [u8; MAX_VALUES]| {
            if let Some(current) = &mut incoming[target] {
                for (current, value) in current.iter_mut().zip(types) {
                    if *current != value {
                        *current = UNSET;
                    }
                }
            } else {
                incoming[target] = Some(types);
            }
        };
        match *op {
            Instruction::Literal(out, _) => types[out] = F32,
            Instruction::Load(out, _, n) => types[out] = n as u8,
            Instruction::Store(_, n, src) => operand(src, n as u8)?,
            Instruction::Copy(out, src) => {
                if types[src] == UNSET {
                    return Err("kernel copy source is uninitialized");
                }
                types[out] = types[src];
            }
            Instruction::Neg(out, src, n) => {
                operand(src, n as u8)?;
                types[out] = n as u8;
            }
            Instruction::Scalar(_, out, a, b) | Instruction::Compare(_, out, a, b) => {
                operand(a, F32)?;
                operand(b, F32)?;
                types[out] = if matches!(op, Instruction::Compare(..)) {
                    BOOL
                } else {
                    F32
                };
            }
            Instruction::Vector(_, out, a, b, va, vb) => {
                operand(a, if va { VEC3 } else { F32 })?;
                operand(b, if vb { VEC3 } else { F32 })?;
                types[out] = VEC3;
            }
            Instruction::Clamp(out, src, lo, hi) => {
                operand(src, F32)?;
                operand(lo, F32)?;
                operand(hi, F32)?;
                types[out] = F32;
            }
            Instruction::Branch(src, target) => {
                operand(src, BOOL)?;
                edge(target, types);
            }
            Instruction::Jump(target) => {
                edge(target, types);
                continue;
            }
        }
        edge(pc + 1, types);
    }
    Ok(())
}

impl Arithmetic {
    #[inline]
    fn apply(self, a: f32, b: f32) -> f32 {
        match self {
            Self::Add => a + b,
            Self::Sub => a - b,
            Self::Mul => a * b,
        }
    }
}

impl Compare {
    fn test(self, a: f32, b: f32) -> bool {
        match self {
            Self::Eq => a == b,
            Self::Ne => a != b,
            Self::Lt => a < b,
            Self::Le => a <= b,
            Self::Gt => a > b,
            Self::Ge => a >= b,
        }
    }
}
