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
    ids: Rc<HashMap<String, EntityHandle>>,
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
        // 与 TS 相同：只在新增身份时复制身份表；已有槽保留字符串，更新位置而非重新生成。
        let mut slots = vec![None; layout.slots.len()];
        for (landmark, units) in [(false, s.units.as_slice()), (true, s.landmarks())] {
            for (index, u) in units.iter().enumerate() {
                let h = layout.ids.get(&u.id).copied().unwrap_or_else(|| {
                    let h = EntityHandle(slots.len());
                    Rc::make_mut(&mut layout.ids).insert(u.id.clone(), h);
                    slots.push(None);
                    h
                });
                if slots[h.0].is_none() {
                    let mut slot = layout
                        .slots
                        .get_mut(h.0)
                        .and_then(Option::take)
                        .unwrap_or_else(|| Slot {
                            id: u.id.clone(),
                            index,
                            landmark,
                        });
                    slot.index = index;
                    slot.landmark = landmark;
                    slots[h.0] = Some(slot);
                }
            }
        }
        layout.slots = slots;
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

#[cfg(test)]
mod tests {
    #[test]
    fn changed_layout_and_shared_ids_preserve_first_match_and_snapshot_handles() {
        let mut catalog = None;
        crate::handle(
            serde_json::from_slice(crate::records::RULES).unwrap(),
            &mut catalog,
            &mut vec![],
            &mut crate::Resident::default(),
        )
        .unwrap();
        let catalog = catalog.unwrap();
        let mut source = crate::runtime::create(8137, false, &catalog).unwrap();
        for x in [3.0, 4.0, 5.0] {
            crate::resolution::add_unit(
                &mut source,
                "1",
                1,
                crate::model::Point { x, y: 6.0 },
                &catalog,
                &mut crate::resolution::Resolution::default(),
            )
            .unwrap();
        }
        let ids: Vec<_> = source.units.iter().map(|u| u.id.clone()).collect();
        let handles: Vec<_> = ids
            .iter()
            .map(|id| source.entities.handle(&source, id).unwrap())
            .collect();
        let mut branch = source.fork();
        branch.units[0].id = "replacement".into();
        let replacement = branch.entities.handle(&branch, "replacement").unwrap();
        assert_ne!(replacement, handles[0]);
        assert!(branch.entity(handles[0]).is_none());
        assert_eq!(source.entity(handles[0]).unwrap().id, ids[0]);
        branch.units.reverse();
        assert_eq!(branch.entity(handles[1]).unwrap().id, ids[1]);
        assert_eq!(branch.entity(replacement).unwrap().id, "replacement");
        let mut duplicate = branch.units[0].clone();
        duplicate.hp = 1.0;
        branch.units.push(duplicate);
        assert_ne!(branch.unit(&ids[2]).unwrap().hp, 1.0);
        branch.units.remove(0);
        assert_eq!(branch.unit(&ids[2]).unwrap().hp, 1.0);
        assert_ne!(source.unit(&ids[2]).unwrap().hp, 1.0);
    }
}
