//! 有序数组保存规则顺序；数字槽定位当前实体，布局在分支间共享，历史快照不占槽。
use crate::model::{State, Unit};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EntityHandle(usize);
#[derive(Clone)]
struct Slot {
    id: String,
    index: usize,
    landmark: bool,
}
#[derive(Clone, Default)]
struct Layout {
    slots: Vec<Option<Slot>>,
    ids: HashMap<String, EntityHandle>,
    count: Option<(usize, usize)>,
}
#[derive(Clone, Default)]
pub struct EntityIndex(RefCell<Rc<Layout>>);
impl EntityIndex {
    fn current<'a>(s: &'a State, slot: &Slot) -> Option<&'a Unit> {
        let unit = if slot.landmark {
            s.landmarks().get(slot.index)
        } else {
            s.units.get(slot.index)
        };
        unit.filter(|u| u.id == slot.id)
    }
    fn sync(&self, s: &State, force: bool) {
        let count = (s.units.len(), s.landmarks().len());
        if !force && self.0.borrow().count == Some(count) {
            return;
        }
        let mut binding = self.0.borrow_mut();
        let layout = Rc::make_mut(&mut binding);
        layout.slots.fill(None);
        for (landmark, units) in [(false, s.units.as_slice()), (true, s.landmarks())] {
            for (index, u) in units.iter().enumerate() {
                let h = *layout.ids.entry(u.id.clone()).or_insert_with(|| {
                    let h = EntityHandle(layout.slots.len());
                    layout.slots.push(None);
                    h
                });
                if layout.slots[h.0].is_none() {
                    layout.slots[h.0] = Some(Slot {
                        id: u.id.clone(),
                        index,
                        landmark,
                    });
                }
            }
        }
        layout.count = Some(count);
    }
    pub fn handle(&self, s: &State, id: &str) -> Option<EntityHandle> {
        #[cfg(feature = "kernel-profile")]
        let _profile = crate::profile::scope(crate::profile::Phase::Identity);
        self.sync(s, false);
        let locate = || {
            let t = self.0.borrow();
            t.ids.get(id).copied().filter(|h| {
                t.slots[h.0]
                    .as_ref()
                    .and_then(|slot| Self::current(s, slot))
                    .is_some()
            })
        };
        if let Some(h) = locate() {
            return Some(h);
        }
        // 不存在的目标（包括基地与已经死亡的来源）不是索引失效，不能反复重建整张布局。
        // 同长度替换确实引入了该 ID 时才重建，保留时钟及兼容入口的可观察语义。
        if !s.units.iter().chain(s.landmarks()).any(|u| u.id == id) {
            return None;
        }
        self.sync(s, true);
        locate()
    }
    pub fn at<'a>(&self, s: &'a State, h: EntityHandle) -> Option<&'a Unit> {
        self.sync(s, false);
        let locate = || {
            let t = self.0.borrow();
            t.slots
                .get(h.0)
                .and_then(Option::as_ref)
                .and_then(|slot| Self::current(s, slot))
        };
        if let Some(u) = locate() {
            return Some(u);
        }
        self.sync(s, true);
        locate()
    }
    pub fn location(&self, s: &State, h: EntityHandle) -> Option<(bool, usize)> {
        self.at(s, h)?;
        let t = self.0.borrow();
        t.slots[h.0]
            .as_ref()
            .map(|slot| (slot.landmark, slot.index))
    }
}
