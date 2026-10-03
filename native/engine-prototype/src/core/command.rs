//! 运行命令种类是编号枚举；字符串仅保留在协议边界和未知命令的错误兼容入口。
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
