#[path = "commands/abilities.rs"]
mod abilities;
#[path = "training/actions.rs"]
mod actions;
#[path = "core/allocation.rs"]
mod allocation;
#[path = "training/boundary.rs"]
mod boundary;
#[path = "training/canonical.rs"]
mod canonical;
#[path = "setup/catalog.rs"]
mod catalog;
#[path = "commands/combat.rs"]
mod combat;
#[path = "core/command.rs"]
mod command;
#[path = "commands/damage.rs"]
mod damage;
mod development;
#[path = "training/encoding.rs"]
mod encoding;
#[path = "core/entities.rs"]
mod entities;
#[path = "training/exercise.rs"]
mod exercise;
#[path = "core/geometry.rs"]
mod geometry;
mod identity;
#[path = "commands/inspection.rs"]
mod inspection;
#[path = "commands/lifecycle.rs"]
mod lifecycle;
mod model;
#[path = "commands/movement.rs"]
mod movement;
#[path = "training/policy.rs"]
mod policy;
#[path = "commands/preparation.rs"]
mod preparation;
#[cfg(feature = "kernel-profile")]
#[path = "core/profile.rs"]
mod profile;
#[path = "commands/reactions.rs"]
mod reactions;
#[path = "training/records.rs"]
mod records;
#[path = "core/resolution.rs"]
mod resolution;
#[path = "setup/runtime.rs"]
mod runtime;
#[path = "training/sampler.rs"]
mod sampler;
#[path = "core/shared.rs"]
mod shared;
#[path = "setup/shrines.rs"]
mod shrines;
#[path = "commands/spells.rs"]
mod spells;
#[path = "core/stats.rs"]
mod stats;
#[path = "setup/synthesis.rs"]
mod synthesis;
#[path = "training/host.rs"]
mod training_host;
#[path = "training/tree.rs"]
mod tree;

use std::io;

fn main() -> io::Result<()> {
    match std::env::args().nth(1).as_deref() {
        Some("--training") => training_host::serve().map_err(io::Error::other),
        Some("--version" | "--capabilities") => {
            println!("{}", identity::describe());
            Ok(())
        }
        None | Some("--development") => development::serve(),
        Some(_) => Err(io::Error::other(
            "use --version, --training or --development",
        )),
    }
}
