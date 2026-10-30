#[path = "commands/abilities.rs"]
mod abilities;
#[path = "ai/mod.rs"]
mod ai;
#[path = "core/allocation.rs"]
pub mod allocation;
#[path = "core/canonical.rs"]
mod canonical;
#[path = "setup/catalog.rs"]
pub mod catalog;
#[path = "core/chance.rs"]
mod chance;
#[path = "commands/combat.rs"]
mod combat;
#[path = "core/command.rs"]
mod command;
#[path = "commands/damage.rs"]
mod damage;
mod development;
#[path = "core/entities.rs"]
mod entities;
#[path = "core/geometry.rs"]
pub mod geometry;
mod identity;
#[path = "commands/inspection.rs"]
pub mod inspection;
#[path = "learning/mod.rs"]
mod learning;
#[path = "commands/lifecycle.rs"]
mod lifecycle;
pub mod model;
#[path = "commands/movement.rs"]
pub mod movement;
#[path = "commands/options.rs"]
pub mod options;
#[path = "commands/preparation.rs"]
mod preparation;
#[path = "commands/reactions.rs"]
mod reactions;
#[path = "core/resolution.rs"]
pub mod resolution;
#[path = "setup/runtime.rs"]
pub mod runtime;
#[path = "core/shared.rs"]
mod shared;
#[path = "setup/shrines.rs"]
mod shrines;
#[path = "commands/spells.rs"]
pub mod spells;
#[path = "core/stats.rs"]
mod stats;
#[path = "setup/synthesis.rs"]
pub mod synthesis;

use std::io;

pub fn run_cli() -> io::Result<()> {
    match std::env::args().nth(1).as_deref() {
        Some("--version" | "--capabilities") => {
            println!("{}", identity::describe());
            Ok(())
        }
        None | Some("--development") => development::serve(),
        Some(_) => Err(io::Error::other("use --version or --development")),
    }
}
