//! 运行命令种类是编号枚举；字符串仅保留在协议边界和未知命令的错误兼容入口。
use crate::model::{Catalog, Command, Failure, State, Unit};
use crate::{combat, movement, preparation, reactions};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::ops::Deref;

macro_rules! command_kinds {
    ($($variant:ident => $wire:literal),+ $(,)?) => {
        #[derive(Clone, Debug, PartialEq, Eq)]
        pub enum CommandKind { $($variant,)+ Unknown(String) }
        pub const COMMANDS: &[&str] = &[$($wire,)+];
        impl CommandKind {
            pub fn as_str(&self) -> &str {
                match self { $(Self::$variant => $wire,)+ Self::Unknown(s) => s }
            }
        }
        impl From<&str> for CommandKind {
            fn from(value: &str) -> Self {
                match value { $($wire => Self::$variant,)+ _ => Self::Unknown(value.into()) }
            }
        }
    };
}
command_kinds! {
    Move => "move", FinishMode => "finish-mode", Attack => "attack", React => "react",
    Summon => "summon", ExtraSummon => "extra-summon", ChooseSummons => "choose-summons",
    Deploy => "deploy", Charge => "charge", Equip => "equip", ActivateAura => "activate-aura",
    Reroll => "reroll", Begin => "begin", SkipSynthesis => "skip-synthesis", End => "end",
    Synthesize => "synthesize", Craft => "craft", ChooseShrine => "choose-shrine",
    FinishShrineSetup => "finish-shrine-setup", Skill => "skill", Cast => "cast",
    Clock => "clock", Shatter => "shatter",
}
impl Default for CommandKind {
    fn default() -> Self {
        Self::Unknown(String::new())
    }
}
impl Deref for CommandKind {
    type Target = str;
    fn deref(&self) -> &str {
        self.as_str()
    }
}
impl PartialEq<&str> for CommandKind {
    fn eq(&self, rhs: &&str) -> bool {
        self.as_str() == *rhs
    }
}
impl Serialize for CommandKind {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.as_str())
    }
}
impl<'de> Deserialize<'de> for CommandKind {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let value = String::deserialize(d)?;
        Ok(Self::from(value.as_str()))
    }
}

/// 已完成公共阶段检查；预检与正式连续运行共用同一个规则内核。
pub fn execute(s: &State, c: &Command, catalog: &Catalog, preview: bool) -> Result<State, Failure> {
    #[cfg(feature = "kernel-profile")]
    let _profile = crate::profile::scope(if preview {
        crate::profile::Phase::Preview
    } else {
        crate::profile::Phase::Settlement
    });
    let mut next = match c.kind.as_str() {
        "attack" | "react" => combat::apply(s, c, catalog, preview),
        "move"
            if s.unit(c.unit_id.as_deref().unwrap_or(""))
                .is_some_and(Unit::runner) =>
        {
            reactions::move_runner(s, c, catalog, preview)
        }
        "move" | "finish-mode" => movement::apply(s, c, catalog),
        _ if COMMANDS.contains(&c.kind.as_str()) => preparation::apply(s, c, catalog, preview),
        _ => Err(Failure::Unsupported("command-kind")),
    }?;
    movement::refresh_transit(&mut next, c, catalog);
    Ok(next)
}
pub fn apply_runtime(s: &State, c: &Command, catalog: &Catalog) -> Result<State, Failure> {
    movement::stage(s, c, catalog)?;
    execute(s, c, catalog, false)
}
