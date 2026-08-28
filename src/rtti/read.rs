//! The **read interpreter** — schema over a live data file → a [`Value`] tree, with no
//! compiled-in types. The non-recursive counterpart of a typed block read: it follows
//! owning edges into child blocks and stops (recording just the offset) at non-owning
//! ones, which also breaks reference cycles.

use std::collections::HashMap;
use std::io;

use bstack::BStack;

use crate::primitives::{EightCC, WidePtr};
use crate::util::{io_error, read_u64};

use super::walk::{
    budget_exceeded, checked_vec_len, disc_mask, option_present, pop_n, pop_named, read_disc,
};
use super::{
    AnyRef, BYTEVEC_HEADER, RttiBody, RttiOrdinal, RttiRegistry, RttiType, Shape, Value, add_off,
    mul_off, unknown_tag,
};

/// One step of the non-recursive walk. The interpreter runs a `work` stack of these
/// against a `results` value stack: leaf steps push a [`Value`]; an `Assemble*` step
/// pops the `n` values its children pushed and combines them into one.
pub(in crate::rtti) enum Op {
    /// Read the block of type `ord` at `block_off` (its whole `OnDisk`, header + fields).
    Block {
        ord: RttiOrdinal,
        block_off: u64,
    },
    /// Interpret one shape at an absolute data offset.
    Shape {
        shape: Shape,
        offset: u64,
    },
    /// Pop `n` field values (child-first order) and assemble a struct block.
    MakeBlock {
        tag: EightCC,
        names: Box<[String]>,
    },
    /// Pop `n` field values and assemble an enum block.
    MakeEnum {
        tag: EightCC,
        variant: String,
        names: Box<[String]>,
    },
    /// Pop `n` values into an array / vec / tuple.
    MakeArray(usize),
    MakeVec(usize),
    MakeTuple(usize),
    /// Pop one value and wrap it in `Some`.
    MakeSome,
}

impl RttiRegistry {
    /// Read a structure of type `ordinal` at `block_off` in `data` into a [`Value`]
    /// tree — the core RTTI operation: interpret an on-disk structure with no
    /// compiled-in Rust type.
    ///
    /// The walk is **non-recursive** (an explicit `work` stack), so arbitrarily deep
    /// or self-referential data cannot blow the call stack. It **follows** owning
    /// edges (`owned` / `strong` / `embed`) into child blocks, and **stops** at
    /// non-owning ones (`weak` / `ref` / `foreign`), recording just their offset —
    /// which also breaks any reference cycle. A node budget guards against a corrupt
    /// file describing an unterminated walk.
    pub fn read_value(
        &self,
        data: &BStack,
        ordinal: RttiOrdinal,
        block_off: u64,
    ) -> io::Result<Value> {
        self.run_read(
            data,
            vec![Op::Block {
                ord: ordinal,
                block_off,
            }],
        )
    }

    /// The read machine: run a `work` stack of [`Op`]s to a single [`Value`]. Seeded
    /// with a `Block` op by [`read_value`](Self::read_value) (a whole block) or a
    /// `Shape` op by [`get`](Self::get) (one field).
    pub(in crate::rtti) fn run_read(&self, data: &BStack, initial: Vec<Op>) -> io::Result<Value> {
        let mut cache: HashMap<RttiOrdinal, RttiType> = HashMap::new();
        let mut work: Vec<Op> = initial;
        let mut results: Vec<Value> = Vec::new();
        // Bounds the total nodes visited: a corrupt schema/data pair (or a strong
        // cycle) can otherwise loop forever.
        let mut budget: u64 = 4_000_000;

        while let Some(op) = work.pop() {
            budget = budget.checked_sub(1).ok_or_else(|| {
                io_error!(
                    InvalidData,
                    "[BSTACK0807] RTTI interpret budget exceeded (corrupt data or a cycle?)"
                )
            })?;
            match op {
                Op::Block { ord, block_off } => {
                    if let std::collections::hash_map::Entry::Vacant(e) = cache.entry(ord) {
                        e.insert(self.load_type(ord)?);
                    }
                    let ty = &cache[&ord];
                    match &ty.body {
                        RttiBody::Struct(fields) => {
                            let names = fields.iter().map(|f| f.name.clone()).collect();
                            // Assemble marker first (popped last), then fields in
                            // order (so they pop child-first into the marker).
                            let field_ops: Vec<Op> = fields
                                .iter()
                                .map(|f| -> io::Result<Op> {
                                    Ok(Op::Shape {
                                        shape: f.shape.clone(),
                                        offset: add_off(block_off, f.offset as u64)?,
                                    })
                                })
                                .collect::<io::Result<Vec<Op>>>()?;
                            work.push(Op::MakeBlock { tag: ty.tag, names });
                            work.extend(field_ops);
                        }
                        RttiBody::Enum(e) => {
                            let raw = read_disc(
                                data,
                                add_off(block_off, e.disc_off as u64)?,
                                e.disc_width,
                            )?;
                            let mask = disc_mask(e.disc_width);
                            let variant = e
                                .variants
                                .iter()
                                .find(|v| (v.disc_value as u64) & mask == raw)
                                .ok_or_else(|| {
                                    io_error!(
                                        InvalidData,
                                        format!(
                                            "[BSTACK0808] no RTTI variant for discriminant {raw}"
                                        )
                                    )
                                })?;
                            let names = variant.fields.iter().map(|f| f.name.clone()).collect();
                            let payload_base = add_off(block_off, e.payload_off as u64)?;
                            let field_ops: Vec<Op> = variant
                                .fields
                                .iter()
                                .map(|f| -> io::Result<Op> {
                                    Ok(Op::Shape {
                                        shape: f.shape.clone(),
                                        offset: add_off(payload_base, f.offset as u64)?,
                                    })
                                })
                                .collect::<io::Result<Vec<Op>>>()?;
                            work.push(Op::MakeEnum {
                                tag: ty.tag,
                                variant: variant.name.clone(),
                                names,
                            });
                            work.extend(field_ops);
                        }
                    }
                }

                Op::Shape { shape, offset } => match shape {
                    Shape::Pod { width } => {
                        // `width` is an untrusted record field; bound it against the
                        // stack before sizing an allocation with it (the read after
                        // would fail anyway — this fails first, without the alloc).
                        if width as u64 > data.len()?.saturating_sub(offset) {
                            return Err(io_error!(
                                InvalidData,
                                "[BSTACK0800] RTTI POD width runs past the end of the data stack",
                            ));
                        }
                        let mut buf = vec![0u8; width as usize];
                        data.get_into(offset, &mut buf)?;
                        results.push(Value::Pod(buf.into()));
                    }
                    Shape::Class { value, .. } => {
                        // A class variable's value is schema-side, not per-instance.
                        results.push(Value::Class(value));
                    }
                    Shape::Owned(tag) | Shape::Strong(tag) => {
                        let child = read_u64(data, offset)?;
                        if child == 0 {
                            results.push(Value::Null);
                        } else {
                            let ord = self.ordinal_of(tag).ok_or_else(unknown_tag)?;
                            work.push(Op::Block {
                                ord,
                                block_off: child,
                            });
                        }
                    }
                    Shape::Embed(tag) => {
                        // The child's whole OnDisk is inline at this slot (no pointer).
                        let ord = self.ordinal_of(tag).ok_or_else(unknown_tag)?;
                        work.push(Op::Block {
                            ord,
                            block_off: offset,
                        });
                    }
                    Shape::Weak(tag) | Shape::Ref(tag) => {
                        results.push(Value::Ref {
                            tag,
                            offset: read_u64(data, offset)?,
                        });
                    }
                    Shape::Foreign { tag, kind } => {
                        // WidePtr { file_id:u32 @0, type_index:u32 @4, offset:u64 @8 }.
                        // The target is in another file — recorded, not followed.
                        let __wp = WidePtr::read_from_stack(data, offset)?;
                        let (file_id, off) = (__wp.file_id(), __wp.offset().get());
                        results.push(Value::Foreign {
                            tag,
                            kind,
                            file_id,
                            offset: off,
                        });
                    }
                    Shape::Option(inner) => {
                        // Niche location depends on the inner shape (a `Foreign`'s is
                        // its offset word @8, not the leading word).
                        if option_present(data, &inner, offset)? {
                            work.push(Op::MakeSome);
                            work.push(Op::Shape {
                                shape: *inner,
                                offset,
                            });
                        } else {
                            results.push(Value::Null);
                        }
                    }
                    Shape::Array { n, inner } => {
                        // Charge the budget for all elements up front, as the `Vec`
                        // arm does — `n` comes off an untrusted record, and the ops
                        // are materialized eagerly, so an absurd count must fail
                        // cleanly rather than pre-allocate past the budget.
                        budget = budget.checked_sub(n as u64).ok_or_else(budget_exceeded)?;
                        let stride = self.shape_stride(&inner, &mut cache)?;
                        let elem_ops: Vec<Op> = (0..n as u64)
                            .map(|i| -> io::Result<Op> {
                                Ok(Op::Shape {
                                    shape: (*inner).clone(),
                                    offset: add_off(offset, mul_off(i, stride)?)?,
                                })
                            })
                            .collect::<io::Result<Vec<Op>>>()?;
                        work.push(Op::MakeArray(n as usize));
                        work.extend(elem_ops);
                    }
                    Shape::Vec(inner) => {
                        let data_off = read_u64(data, offset)?; // VecDesc.data_off @0
                        if data_off == 0 {
                            results.push(Value::Vec(Box::default()));
                        } else {
                            // `@0` is the byte length, validated against the block size
                            // (`VecDesc.data_size` @8) so a forged length can't drive an
                            // out-of-block read / petabyte allocation.
                            let data_size = read_u64(data, add_off(offset, 8)?)?;
                            let base = add_off(data_off, BYTEVEC_HEADER)?;
                            let stride = self.shape_stride(&inner, &mut cache)?;
                            let byte_len = read_u64(data, data_off)?;
                            let len = checked_vec_len(byte_len, data_size, stride)?;
                            // Charge the budget for all elements up front — the ops are
                            // materialized eagerly, so a huge (but in-block) length must
                            // fail cleanly rather than pre-allocate past the budget.
                            budget = budget.checked_sub(len).ok_or_else(budget_exceeded)?;
                            let elem_ops: Vec<Op> = (0..len)
                                .map(|i| -> io::Result<Op> {
                                    Ok(Op::Shape {
                                        shape: (*inner).clone(),
                                        offset: add_off(base, mul_off(i, stride)?)?,
                                    })
                                })
                                .collect::<io::Result<Vec<Op>>>()?;
                            work.push(Op::MakeVec(len as usize));
                            work.extend(elem_ops);
                        }
                    }
                    Shape::Tuple(items) => {
                        let mut elem_ops: Vec<Op> = Vec::with_capacity(items.len());
                        let mut off = offset;
                        for it in &items {
                            elem_ops.push(Op::Shape {
                                shape: it.clone(),
                                offset: off,
                            });
                            off = add_off(off, self.shape_stride(it, &mut cache)?)?;
                        }
                        work.push(Op::MakeTuple(items.len()));
                        work.extend(elem_ops);
                    }
                },

                Op::MakeBlock { tag, names } => {
                    let fields = pop_named(&mut results, &names)?;
                    results.push(Value::Block {
                        tag,
                        fields: fields.into(),
                    });
                }
                Op::MakeEnum {
                    tag,
                    variant,
                    names,
                } => {
                    let fields = pop_named(&mut results, &names)?;
                    results.push(Value::Enum {
                        tag,
                        variant,
                        fields: fields.into(),
                    });
                }
                Op::MakeArray(n) => {
                    let v = pop_n(&mut results, n)?;
                    results.push(Value::Array(v.into()));
                }
                Op::MakeVec(n) => {
                    let v = pop_n(&mut results, n)?;
                    results.push(Value::Vec(v.into()));
                }
                Op::MakeTuple(n) => {
                    let v = pop_n(&mut results, n)?;
                    results.push(Value::Tuple(v.into()));
                }
                Op::MakeSome => {
                    let inner = results.pop().ok_or_else(|| {
                        io_error!(InvalidData, "[BSTACK0809] RTTI interpret stack underflow")
                    })?;
                    results.push(Value::Some(Box::new(inner)));
                }
            }
        }

        match results.len() {
            1 => Ok(results.pop().unwrap()),
            n => Err(io_error!(
                InvalidData,
                format!("[BSTACK0809] RTTI interpret produced {n} values (expected 1)")
            )),
        }
    }

    /// Read the structure a typed [`WidePtr`] points at, within `data`. The
    /// pointer must be **typed** (carry an RTTI ordinal), and `data` must be the file
    /// it targets. This resolves within a single file by design — for a cross-file
    /// pointer, resolve its `file_id` through the [`registry`](crate::registry) first
    /// and call this against that file's stack.
    pub fn read_ptr(&self, data: &BStack, ptr: WidePtr) -> io::Result<Value> {
        let ord = self.resolve_ptr(ptr).ok_or_else(|| {
            io_error!(
                InvalidData,
                "[BSTACK080A] cannot read an untyped / out-of-range RTTI pointer"
            )
        })?;
        self.read_value(data, ord, ptr.offset().get())
    }

    /// The runtime-typed [`AnyRef`] a **typed** pointer denotes — its registry tag
    /// (resolved from the pointer's `type_index`) plus offset. `None` for an untyped
    /// (`type_index == 0`) or out-of-range pointer, so a stray pointer can never
    /// masquerade as a registered type. Downcast the result with
    /// [`AnyRef::downcast`], or read it generically with [`read_any`](Self::read_any).
    pub fn any_ref(&self, ptr: WidePtr) -> Option<AnyRef> {
        let ord = self.resolve_ptr(ptr)?;
        let tag = self.tag_of(ord)?;
        // SAFETY: the tag is registry-authoritative for the pointer's type_index,
        // and the offset is the typed pointer's own target.
        Some(unsafe { AnyRef::new(tag, ptr.offset().get()) })
    }

    /// Interpret the structure an [`AnyRef`] points at into a [`Value`] tree — the
    /// generic fallback when [`AnyRef::downcast`] does not match a compiled-in type.
    /// Errors if the reference's tag is not a registered type.
    pub fn read_any(&self, data: &BStack, any: &AnyRef) -> io::Result<Value> {
        let ord = self.ordinal_of(any.tag()).ok_or_else(unknown_tag)?;
        self.read_value(data, ord, any.offset())
    }
}
