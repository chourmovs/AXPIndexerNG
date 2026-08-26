mod cli;
mod crawler;
mod extractors;
use anyhow::Result;
use axpindexer_core::Database;
use clap::Parser;
use cli::{Cli, Command};
fn main() -> Result<()> {
    tracing_subscriber::fmt().with_target(false).init();
    match Cli::parse().command {
        Command::Health { db } => {
            let h = Database::open(db)?.health()?;
            println!(
                "AXPIndexerDaemon health OK\nSQLite: {}\nSchema: {}\nFTS5: OK",
                h.sqlite, h.schema
            )
        }
        Command::Status { db } => {
            let d = Database::open(db)?;
            println!("documents: {}", d.count()?)
        }
        Command::Scan { root, db } => {
            let s = crawler::scan(&root, &db)?;
            println!(
                "supported: {}\nnew: {}\nupdated: {}\nunchanged: {}\ndeleted: {}\nfailed: {}\nduration_ms: {}",
                s.supported, s.new, s.updated, s.unchanged, s.deleted, s.failed, s.duration_ms
            )
        }
    }
    Ok(())
}
